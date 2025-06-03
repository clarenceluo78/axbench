from .model import Model
import torch, einops
import torch.nn as nn
import torch.nn.functional as F

from tqdm.auto import tqdm
import os
import pandas as pd
from pyvene import (
    IntervenableConfig,
    IntervenableModel
)
from .interventions import (
    TopKReLUSubspaceIntervention,
    SimpleAdditionIntervention,
    SubspaceIntervention,
    SamplingAdditionIntervention,
    ThresholdingIntervention
)
from ..utils.data_utils import make_data_module
from ..utils.constants import EXAMPLE_TAG
from torch.utils.data import DataLoader
from ..utils.model_utils import (
    set_decoder_norm_to_unit_norm,
    remove_gradient_parallel_to_decoder_directions,
    gather_residual_activations, 
    get_lr,
    calculate_l1_losses
)
from transformers import get_scheduler, AutoTokenizer, AutoConfig
from transformers import set_seed
from ..scripts.inference import prepare_df
from .hypernet.configuration_hypernet import HypernetConfig
from .hypernet.modeling_hypernet import HypernetModel
import torch.distributed as dist
import json

from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

EARLY_STOPPING_PATIENCE = 5
MAX_NUMBER_OF_CHECKPOINTS = 3

def load_concept_id_to_desecription(metadata_path):
    """
    Load flatten metadata from a JSON lines file.
    """
    concept_id_to_description = {}
    with open(metadata_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            concept_id, description = data["concept_id"], data["concept"]
            concept_id_to_description[concept_id] = description
    return concept_id_to_description
            

def partition_df(df, n):
    """
    Partition a df into n approximately equal slices.

    Args:
        df (pd.Dataframe): The dataframe to partition.
        n (int): The number of partitions.

    Returns:
        list of dfs: A list containing n sub-dfs.
    """
    # Calculate the size of each partition
    total_rows = len(df)
    partition_size = total_rows // n
    remainder = total_rows % n
    
    # Initialize list to store partitions
    partitions = []
    
    # Start index for slicing
    start_idx = 0
    
    # Create each partition
    for i in range(n):
        # If there's a remainder, distribute it among the first 'remainder' partitions
        # by adding 1 extra row to each
        current_size = partition_size + (1 if i < remainder else 0)
        
        # Calculate end index for the current partition
        end_idx = start_idx + current_size
        
        # Slice the dataframe and add to partitions
        partitions.append(df.iloc[start_idx:end_idx].copy())
        
        # Update the start index for the next partition
        start_idx = end_idx
    
    return partitions

    
class HyperSteer(Model):
    """HyperSteer with Cross-Attention"""
    def __str__(self):
        return 'HyperSteer'

    def make_model(self, **kwargs):
        mode = kwargs.get("mode", "latent")
        intervention_type = kwargs.get("intervention_type", "addition")
        
        intervention_type = kwargs.get("intervention_type", "addition")
        if intervention_type == "addition":
            ax = SimpleAdditionIntervention(
                embed_dim=self.model.config.hidden_size, 
                low_rank_dimension=kwargs.get("low_rank_dimension", 1),
            )
        else:
            raise NotImplementedError(f"{intervention_type} not implemented for CrossAttnHyperReFT in {mode} mode.")

        layers = self.steering_layers if self.steering_layers else [self.layer]
        self.ax = ax.to(self.device)
        self.ax.train()
        ax_config = IntervenableConfig(representations=[{
            "layer": l,
            "component": f"model.layers[{l}].output",
            "low_rank_dimension": kwargs.get("low_rank_dimension", 1),
            "intervention": self.ax} for l in layers])
        ax_model = IntervenableModel(ax_config, self.model)
        ax_model.set_device(self.device)
        self.ax_model = ax_model
        
        hypernet_name_or_path = kwargs.get("hypernet_name_or_path", "google/gemma-2-2b")
        
        # Load the interpreting model
        self.hypernet_tokenizer = AutoTokenizer.from_pretrained(hypernet_name_or_path, model_max_length=512)
        self.hypernet_tokenizer.padding_side = "left"
        if self.hypernet_tokenizer.pad_token is None:
            self.hypernet_tokenizer.pad_token = self.hypernet_tokenizer.eos_token
            self.hypernet_tokenizer.pad_token_id = self.hypernet_tokenizer.eos_token_id
        
        if self.hypernet_tokenizer.unk_token == None and self.hypernet_tokenizer.pad_token == None:
            # raw llama3
            print("adding a special padding token...")
            self.hypernet_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            need_resize = True
        else:
            need_resize = False
        if need_resize:
            self.hypernet_tokenizer.resize_token_embeddings(len(self.hypernet_tokenizer))
            
        num_hidden_layers = kwargs.get("num_hidden_layers", 2)
        print(f"num_hidden_layers: {num_hidden_layers}")
                
        hypernet_config = HypernetConfig.from_pretrained(
            pretrained_model_name_or_path=hypernet_name_or_path,
            num_hidden_layers=num_hidden_layers,
            torch_dtype=torch.bfloat16,
            use_target_model_embedding=False,
        )
        
        use_pretrained_parameter = kwargs.get("hypernet_initialize_from_pretrained", True)
        if use_pretrained_parameter:
            print(f"Loading pretrained hypernet model from {hypernet_name_or_path}")
            self.concept_embedding = HypernetModel.from_pretrained(
                hypernet_name_or_path,
                config=hypernet_config
            ) 
        else:
            self.concept_embedding = HypernetModel(config=hypernet_config)
            
        self.concept_embedding = self.concept_embedding.to(self.device, dtype=torch.bfloat16)
            
        # To easily test concept embedding through logit lens
        meta_data = kwargs.get("metadata", None)
        if meta_data is not None:
            self.concept_id_to_text = {}
            for d in meta_data:
                self.concept_id_to_text[d['concept_id']] = d['concept']     
                        
    def _setup_model(rank, model):
        # Move model to the appropriate device
        device = torch.device(f"cuda:{rank}")
        model = model.to(device)
        
        # Wrap the model with DDP
        model = DDP(model, device_ids=[rank])
        return model

    def train(self, examples, **kwargs):
        
        rank = torch.distributed.get_rank()
        world_size = kwargs.get("world_size", 1)
                 
        train_dataloader, train_sampler = self.make_dataloader(
            examples, rank=rank, concept_tokenizer=self.hypernet_tokenizer,
            distributed=True, **kwargs
        )
                        
        torch.cuda.empty_cache()
        
        embedding_model = self.concept_embedding if world_size == 1 else DDP(self.concept_embedding, device_ids=[rank], find_unused_parameters=True)
        
        # Optimizer and lr
        optimizer = torch.optim.AdamW(
            embedding_model.parameters(), 
            lr=self.training_args.lr, weight_decay=self.training_args.weight_decay
        )
            
        num_training_steps = self.training_args.n_epochs * (len(train_dataloader) // self.training_args.gradient_accumulation_steps)
        
        lr_scheduler = get_scheduler(
            "linear", optimizer=optimizer,
            num_warmup_steps=0, num_training_steps=num_training_steps)
                            
        # Main training loop.        
        progress_bar, curr_step, losses = tqdm(range(num_training_steps), position=rank, leave=True), 0, []
            
        embedding_model.train()        
        
        for epoch in range(self.training_args.n_epochs):
            
            for step, batch in enumerate(train_dataloader):
                                
                train_sampler.set_epoch(epoch)
                
                # prepare input
                inputs = {k: v.to(self.device) for k, v in batch.items()}
                                
                unit_locations={"sources->base": (
                    None,
                    inputs["intervention_locations"].permute(1, 0, 2).tolist()
                )}
                subspaces = [{
                    "k": self.training_args.topk
                }]
                
                concept_inputs_embeds = None
                
                base_intervention_mask = inputs["labels"] == -100                
                base_intervention_mask = base_intervention_mask & inputs["attention_mask"]
                
                base_hidden_state = self.model(
                    input_ids=inputs["input_ids"],
                    attention_mask=base_intervention_mask,
                    output_hidden_states=True,
                ).hidden_states[self.layer]
                
                v = embedding_model(
                    input_ids=inputs["concept_input_ids"],
                    inputs_embeds=concept_inputs_embeds,
                    attention_mask=inputs["concept_attention_mask"],
                    base_encoder_hidden_states=base_hidden_state,
                    base_encoder_attention_mask=base_intervention_mask,
                    output_hidden_states=False,
                ).last_hidden_state
                                                                
                self.ax._update_v(v)
                                
                # forward
                _, cf_outputs = self.ax_model(
                    base={
                        "input_ids": inputs["input_ids"],
                        "attention_mask": inputs["attention_mask"]
                    }, unit_locations=unit_locations, labels=inputs["labels"],
                    subspaces=subspaces, use_cache=False)
                                
                steering_loss = cf_outputs.loss
                
                loss = steering_loss
                loss = loss.mean()
                loss /= self.training_args.gradient_accumulation_steps
                # steering_loss += coeff * self.training_args.coeff_latent_l1_loss * l1_loss
                # steering_loss = steering_loss.mean()
                # steering_loss /= self.training_args.gradient_accumulation_steps
                # grads
                    
                loss.backward()
                
                # clear the steering vector generated for this batch
                self.ax._reset_v()
            
                # Perform optimization step every gradient_accumulation_steps
                if (step + 1) % self.training_args.gradient_accumulation_steps == 0 or (step + 1) == len(train_dataloader):
                    torch.nn.utils.clip_grad_norm_(embedding_model.parameters(), 1.0)
                    # set_decoder_norm_to_unit_norm(self.ax)
                    
                    # TODO: need to be implimented for concept_embedding
                    # remove_gradient_parallel_to_decoder_directions(self.ax)
                    
                    curr_step += 1
                    losses.append(loss.item())
                    curr_lr = get_lr(optimizer)
                    # optim
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    progress_bar.update(1)
                    progress_bar.set_description(
                        "lr %.6f || loss %.6f" % (curr_lr, loss))
             
        progress_bar.close()
        
        
    
    @torch.no_grad()
    def predict_latent(self, examples, **kwargs):
        pass
    
    @torch.no_grad()
    def predict_latents(self, examples, **kwargs):
        pass
        
    def make_dataloader(self, examples, rank, world_size, shuffle=True, distributed=False, concept_tokenizer=None, **kwargs):
        
        if distributed:
            sampler = DistributedSampler(
                examples, 
                num_replicas=world_size,
                rank=rank,
                shuffle=shuffle
            )
            data_module = make_data_module(self.tokenizer, examples, concept_tokenizer=self.hypernet_tokenizer, **kwargs)
            g = torch.Generator()
            g.manual_seed(self.seed)
            train_dataloader = DataLoader(
                data_module["train_dataset"],
                batch_size=self.training_args.batch_size, 
                collate_fn=data_module["data_collator"],
                sampler=sampler,
                generator=g)
            return train_dataloader, sampler
        else:
            data_module = make_data_module(self.tokenizer, examples, concept_tokenizer=self.hypernet_tokenizer, **kwargs)
            g = torch.Generator()
            g.manual_seed(self.seed)
            train_dataloader = DataLoader(
                data_module["train_dataset"],
                batch_size=self.training_args.batch_size, 
                collate_fn=data_module["data_collator"],
                shuffle=shuffle,
                generator=g)
            return train_dataloader
    
    def save(self, dump_dir, **kwargs):
        weight_file = os.path.join(dump_dir, "hyperreft")
        self.concept_embedding.save_pretrained(weight_file)
        self.hypernet_tokenizer.save_pretrained(weight_file)

    def load(self, dump_dir=None, **kwargs):
        weight_file = os.path.join(dump_dir, f"hyperreft")
        kwargs["hypernet_name_or_path"] = weight_file
        self.make_model(**kwargs)
        
    def get_logits(self, concept_id, example, metadata=None, k=10):
        top_logits, neg_logits = [None], [None]
        
        # Split the example and only keep the '<start_of_turn>user ... <start_of_turn>model' part
        W_U = self.model.lm_head.weight.T
        W_U = W_U * (self.model.model.norm.weight +
                    torch.ones_like(self.model.model.norm.weight))[:, None]
        W_U -= einops.reduce(
            W_U, "d_model d_vocab -> 1 d_vocab", "mean"
        )
            
        if metadata is not None:
            concept_text = None
            for d in metadata:
                if d['concept_id'] == concept_id:
                    concept_text = d['concept']
                    break
            if concept_text is None:
                raise ValueError("Concept ID not found in metadata.")
            
        else:
            concept_text = self.concept_id_to_text[concept_id]
        
        concept_input = self.hypernet_tokenizer(
            concept_text, return_tensors="pt", 
            add_special_tokens=True, padding=True, truncation=True).to(self.device)
        
        text_input = self.tokenizer(
            example, return_tensors="pt", 
            add_special_tokens=True, padding=True, truncation=True).to(self.device)
        # set tokenizer padding to left
        
        concept_inputs_embeds = self.model.model.embed_tokens(concept_input["input_ids"])
                
        base_hidden_state = self.model(
            input_ids=text_input["input_ids"],
            attention_mask=text_input["attention_mask"],
            output_hidden_states=True,
        ).hidden_states[self.layer]
        
        #print(self.concept_embedding.regression_head.weight.grad)
        
        concept_subspace = self.concept_embedding(
            input_ids=None,
            inputs_embeds=concept_inputs_embeds,
            attention_mask=concept_input["attention_mask"],
            base_encoder_hidden_states=base_hidden_state,
            base_encoder_attention_mask=text_input["attention_mask"],
            output_hidden_states=False,
        ).last_hidden_state      
        
        vocab_logits = concept_subspace @ W_U
        top_values, top_indices = vocab_logits.topk(k=k, sorted=True)
        top_tokens = self.tokenizer.batch_decode(top_indices)

        top_logits = [list(zip(top_tokens, top_values.tolist()))]

        neg_values, neg_indices = vocab_logits.topk(k=k, largest=False, sorted=True)
        neg_tokens = self.tokenizer.batch_decode(neg_indices)
        neg_logits = [list(zip(neg_tokens, neg_values.tolist()))]

        return top_logits, neg_logits

    @torch.no_grad()
    def predict_steer(self, examples, **kwargs):
        self.ax.eval()
        
        return_vector = kwargs.get("return_vector", False)
        
        rank = torch.distributed.get_rank()
        # set tokenizer padding to left
        self.tokenizer.padding_side = "left"
        # depending on the model, we use different concept id columns
        concept_id_col = "sae_id" if "sae" in self.__str__().lower() and not kwargs.get("disable_neuronpedia_max_act", False) else "concept_id"
        use_synergy = kwargs.get("use_synergy", False)

        # iterate rows in batch
        batch_size = kwargs.get("batch_size", 64)
        eval_output_length = kwargs.get("eval_output_length", 128)
        temperature = kwargs.get("temperature", 1.0)
        all_generations = []
        all_perplexities = []
        all_strenghts = []
        all_magnitudes = []
        # Main training loop.
        progress_bar = tqdm(range(0, len(examples), batch_size), position=rank, leave=True)
        for i in range(0, len(examples), batch_size):
            batch_examples = examples.iloc[i:i+batch_size]
            
            if use_synergy:
                # print("Using steered prompt to evaluate synergy of prompt and lsreft.")
                input_strings = batch_examples['steered_input'].tolist()
            else:
                input_strings = batch_examples['input'].tolist()
            mag = torch.tensor(batch_examples['factor'].tolist()).to(self.device)
            idx = torch.tensor(batch_examples["concept_id"].tolist()).to(self.device)
                        
            # tokenize input_strings
            inputs = self.tokenizer(
                input_strings, return_tensors="pt", padding=True, truncation=True
            ).to(self.device)
            
            input_concept = batch_examples["input_concept"].tolist()
        
            concept_inputs = self.hypernet_tokenizer(
                input_concept, return_tensors="pt",
                add_special_tokens=True, padding=True, truncation=True).to(self.device)
            
            concept_inputs_embeds = None
                
            base_hidden_state = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                output_hidden_states=True,
            ).hidden_states[self.layer]
            
            #print(self.concept_embedding.regression_head.weight.grad)            
            v = self.concept_embedding(
                input_ids=concept_inputs["input_ids"],
                inputs_embeds=concept_inputs_embeds,
                attention_mask=concept_inputs["attention_mask"],
                base_encoder_hidden_states=base_hidden_state,
                base_encoder_attention_mask=inputs["attention_mask"],
                output_hidden_states=False,
            ).last_hidden_state
            
            vector_magnitude = torch.norm(v, p=2, dim=-1).tolist()
            all_magnitudes += vector_magnitude

            self.ax._update_v(v)
            
            _, generations = self.ax_model.generate(
                inputs, 
                unit_locations=None, intervene_on_prompt=True, 
                subspaces=[{"idx": idx, "mag": mag, "prefix_length": kwargs["prefix_length"]}] * self.num_of_layers,
                max_new_tokens=eval_output_length, do_sample=True, 
                temperature=temperature,
            )
            
            self.ax._reset_v()
            
            # Decode and print only the generated text without prompt tokens
            input_lengths = [len(input_ids) for input_ids in inputs.input_ids]
            generated_texts = [
                self.tokenizer.decode(generation[input_length:], skip_special_tokens=True)
                for generation, input_length in zip(generations, input_lengths)
            ]
            all_generations += generated_texts

            # Calculate perplexity for each sequence
            unpruned_generated_texts = [
                self.tokenizer.decode(generation, skip_special_tokens=True)
                for generation in generations
            ]
            batch_input_ids = self.tokenizer(
                unpruned_generated_texts, return_tensors="pt", padding=True, truncation=True).input_ids.to(self.device)
            batch_attention_mask = (batch_input_ids != self.tokenizer.pad_token_id).float()
            
            # Forward pass without labels to get logits
            outputs = self.model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
            
            logits = outputs.logits[:, :-1, :].contiguous()  # Remove last token prediction
            target_ids = batch_input_ids[:, 1:].contiguous()  # Shift right by 1
            
            # Calculate loss for each token
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            token_losses = loss_fct(logits.view(-1, logits.size(-1)), target_ids.view(-1))
            
            # Reshape losses and mask
            token_losses = token_losses.view(batch_input_ids.size(0), -1)
            mask = batch_attention_mask[:, 1:].contiguous()
            
            # Calculate perplexity for each sequence
            seq_lengths = mask.sum(dim=1)
            seq_losses = (token_losses * mask).sum(dim=1) / seq_lengths
            seq_perplexities = torch.exp(seq_losses).tolist()
            all_perplexities.extend(seq_perplexities)
            all_strenghts.extend((mag).tolist())
            progress_bar.update(1)
            
        return_dict = {
            "steered_generation": all_generations,
            "perplexity": all_perplexities,
            "strength": all_strenghts,
            "vector_magnitude": all_magnitudes,
        }
        
        if return_vector:
            return_dict["steering_vector"] = v.cpu()
        
        return return_dict