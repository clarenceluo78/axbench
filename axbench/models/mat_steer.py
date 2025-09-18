import logging

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import wandb

from transformers import get_scheduler
from .model import Model
from .probe import DataCollator, make_data_module
from pyvene import (
    IntervenableConfig,
    IntervenableModel
)
from .interventions import (
    GatedAdditionIntervention,
    AdditionIntervention,
    SubspaceIntervention,
    ProbeIntervention,
    SparseProbeIntervention
)
from ..utils.model_utils import gather_residual_activations

logger = logging.getLogger(__name__)


def _gaussian_kernel(x: torch.Tensor, y: torch.Tensor, sigma: float) -> torch.Tensor:
    if x.numel() == 0 or y.numel() == 0:
        return x.new_zeros(1)
    pairwise_dist = torch.cdist(x, y, p=2) ** 2
    return torch.exp(-pairwise_dist / (2 * sigma ** 2))


def _unbiased_mmd(x: torch.Tensor, y: torch.Tensor, sigma: float) -> torch.Tensor:
    m = x.size(0)
    n = y.size(0)
    if m == 0 or n == 0:
        return x.new_tensor(0.0)

    # k_xx = _gaussian_kernel(x, x, sigma)
    # k_yy = _gaussian_kernel(y, y, sigma)
    # k_xy = _gaussian_kernel(x, y, sigma)

    # if m > 1:
    #     xx = (k_xx.sum() - k_xx.diag().sum()) / (m * (m - 1))
    # else:
    #     xx = x.new_tensor(0.0)

    # if n > 1:
    #     yy = (k_yy.sum() - k_yy.diag().sum()) / (n * (n - 1))
    # else:
    #     yy = x.new_tensor(0.0)

    # xy = k_xy.mean()

    xx = _gaussian_kernel(x, x, sigma).mean()
    yy = _gaussian_kernel(y, y, sigma).mean()
    xy = _gaussian_kernel(x, y, sigma).mean()

    return xx + yy - 2 * xy


class MATSteer(Model):
    """Per-step gated intervention with per-concept vectors."""

    def __str__(self):
        return "MATSteer"

    def make_model(self, **kwargs):
        mode = kwargs.get("mode", "train")
        low_rank_dimension = kwargs.get("low_rank_dimension", 1)
        layers = self.steering_layers if self.steering_layers else [self.layer]

        # TODO: assume always addition intervention for now
        self.ax = GatedAdditionIntervention(
            embed_dim=self.model.config.hidden_size,
            low_rank_dimension=low_rank_dimension,
        ).to(self.device)
        self.ax.train()
        ax_config = IntervenableConfig(representations=[{
            "layer": l,
            "component": f"model.layers[{l}].output",
            "low_rank_dimension": kwargs.get("low_rank_dimension", 1),
            "intervention": self.ax} for l in layers])
        ax_model = IntervenableModel(ax_config, self.model)
        ax_model.set_device(self.device)
        self.ax_model = ax_model

    def make_dataloader(self, examples, **kwargs):
        data_module = make_data_module(self.tokenizer, self.model, examples)
        train_dataloader = DataLoader(
            data_module["train_dataset"], shuffle=True, batch_size=self.training_args.batch_size, 
            collate_fn=data_module["data_collator"])
        return train_dataloader

    def train(self, examples, **kwargs):
        concept_name = kwargs.get("concept_name", "concept")

        # Initialize wandb if enabled
        if self.use_wandb:
            logging_metadata = kwargs.get("logging_metadata", {})
            run_name = f"MATSteer_{concept_name}_l{logging_metadata.get('layer', 'unknown')}"
            wandb_proj = kwargs.get("wandb_project", None)
            # wandb_name = kwargs.get("wandb_name", None)
            run = wandb.init(
                project=f"{wandb_proj}", 
                # entity=wandb_name,
                name=run_name,
            )

        train_dataloader = self.make_dataloader(examples, **kwargs)
        torch.cuda.empty_cache()

        # Optimizer and lr
        optimizer = torch.optim.AdamW(
            self.ax.parameters(), lr=self.training_args.lr, 
            weight_decay=self.training_args.weight_decay)
        num_training_steps = self.training_args.n_epochs * len(train_dataloader)
        lr_scheduler = get_scheduler(
            "linear", optimizer=optimizer,
            num_warmup_steps=0, num_training_steps=num_training_steps)

        # Loss weights (hyperparameters)
        lambda_pos = kwargs.get("lambda_pos", 0.8)  # positive preservation weight
        lambda_sparse = kwargs.get("lambda_sparse", 0.8)  # sparsity weight
        sigma = kwargs.get("mmd_sigma", 2.0)  # RBF kernel bandwidth for MMD

        # Main training loop
        rank = torch.distributed.get_rank()
        progress_bar, curr_step = tqdm(range(num_training_steps), position=rank, leave=True), 0

        for epoch in range(self.training_args.n_epochs):
            for batch in train_dataloader:
                # Prepare input
                inputs = {k: v.to(self.device) for k, v in batch.items()}
                
                # Get base activations at the intervention layer (detach to avoid gradients)
                activations = gather_residual_activations(
                    self.model, self.layer, 
                    {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]}
                ).detach()  # [batch_size, seq_len, hidden_size]
                
                # Skip BOS tokens
                prefix_length = kwargs.get("prefix_length", 1)
                nonbos_mask = inputs["attention_mask"][:, prefix_length:]
                activations = activations[:, prefix_length:]  # [batch_size, seq_len-prefix, hidden_size]
                
                # Get labels for each token
                labels = inputs["labels"].unsqueeze(1).repeat(1, activations.shape[1])  # [batch_size, seq_len-prefix]
                
                # Apply mask to get valid activations and labels
                valid_mask = nonbos_mask.bool()
                valid_activations = activations[valid_mask]  # [n_valid_tokens, hidden_size]
                valid_labels = labels[valid_mask]  # [n_valid_tokens]
                
                # Separate positive and negative activations
                positive_activations = valid_activations[valid_labels == 1]  # [n_pos, hidden_size]
                negative_activations = valid_activations[valid_labels != 1]  # [n_neg, hidden_size]
                
                if len(positive_activations) == 0 or len(negative_activations) == 0:
                    continue  # Skip if no positive or negative samples
                
                # Apply gated intervention to negative activations
                # G_c(a_i) = sigmoid(w_c * a_i + b_c)
                gate_values_neg = torch.sigmoid(
                    torch.matmul(negative_activations, self.ax.gate_linear.weight[0].unsqueeze(0).T) + 
                    self.ax.gate_linear.bias[0]
                ).squeeze(-1)  # [n_neg]
                
                # Get steering vector theta_c
                steering_vec = self.ax.proj.weight[0]  # [hidden_size]
                
                # Apply gated addition: f_c(a) = a + G_c(a) * theta_c
                adjusted_negative = negative_activations + gate_values_neg.unsqueeze(-1) * steering_vec.unsqueeze(0)
                
                # Norm preservation (like the original normalize_activations function)
                neg_norm = torch.norm(negative_activations, dim=-1, keepdim=True)
                adjusted_norm = torch.norm(adjusted_negative, dim=-1, keepdim=True) + 1e-8
                adjusted_negative = adjusted_negative * (neg_norm / adjusted_norm)
                
                # Loss 1: MMD loss - push adjusted negatives toward original positives
                # This matches the original: compare adjusted negatives to original positives
                mmd_loss = _unbiased_mmd(adjusted_negative, positive_activations, sigma)
                
                # Loss 2: Positive preservation - drive gates to 0 on positives
                gate_values_pos = torch.sigmoid(
                    torch.matmul(positive_activations, self.ax.gate_linear.weight[0].unsqueeze(0).T) + 
                    self.ax.gate_linear.bias[0]
                ).squeeze(-1)  # [n_pos]
                pos_loss = torch.sum(gate_values_pos ** 2)
                
                # Loss 3: Sparsity on negatives - encourage sparse gates
                sparse_loss = torch.sum(torch.abs(gate_values_neg))
                
                # Total loss
                total_loss = mmd_loss + lambda_pos * pos_loss + lambda_sparse * sparse_loss
                
                # Backprop
                total_loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                curr_step += 1
                progress_bar.update(1)
                progress_bar.set_description(
                    f"MMD: {mmd_loss.item():.4f} | Pos: {pos_loss.item():.4f} | Sparse: {sparse_loss.item():.4f}"
                )
                
                # Log to wandb
                if self.use_wandb:
                    metrics = {
                        'loss/total': total_loss.item(),
                        'loss/mmd': mmd_loss.item(),
                        'loss/positive_preservation': pos_loss.item(),
                        'loss/sparsity': sparse_loss.item(),
                        'training/learning_rate': optimizer.param_groups[0]['lr'],
                        'training/epoch': epoch,
                        'training/step': curr_step,
                        'stats/num_positives': len(positive_activations),
                        'stats/num_negatives': len(negative_activations),
                        'stats/mean_gate_negative': gate_values_neg.mean().item(),
                        'stats/mean_gate_positive': gate_values_pos.mean().item(),
                    }
                    wandb.log(metrics, step=curr_step)
        
        progress_bar.close()
        
        # Finish wandb run
        if self.use_wandb:
            run.finish()

    @torch.no_grad()
    def predict_latent(self, examples, **kwargs):
        if not hasattr(self, "ax"):
            raise RuntimeError("Call make_model(mode='train') or load() before predict_latent().")

        self.ax.eval()

        batch_size = kwargs.get("batch_size", 32)
        prefix_length = kwargs.get("prefix_length", 1)
        return_max_act_only = kwargs.get("return_max_act_only", False)
        is_chat_model = kwargs.get("is_chat_model", False)
        eager_prepare_df = kwargs.get("eager_prepare_df", False)
        overwrite_concept_id = kwargs.get("overwrite_concept_id", None)

        all_acts, all_max_act, all_max_act_idx = [], [], []
        all_max_token, all_tokens = [], []

        if eager_prepare_df:
            from ..scripts.inference import prepare_df

        for start in range(0, len(examples), batch_size):
            batch = examples.iloc[start:start + batch_size]
            if eager_prepare_df:
                batch = prepare_df(batch, self.tokenizer, is_chat_model)

            inputs = self.tokenizer(
                batch["input"].tolist(),
                return_tensors="pt",
                padding=True,
                truncation=True,
                add_special_tokens=True,
            ).to(self.device)

            # Get base activations
            full_activations = gather_residual_activations(self.model, self.layer, inputs)
            activations = full_activations[:, prefix_length:]
            attention_trim = inputs["attention_mask"][:, prefix_length:]

            # Compute gate values directly: G_c(a_i) = sigmoid(w_c * a_i + b_c)
            gate_values = torch.sigmoid(
                torch.matmul(activations, self.ax.gate_linear.weight[0].unsqueeze(0).T) + 
                self.ax.gate_linear.bias[0]
            ).squeeze(-1).float().cpu()  # [batch_size, seq_len]

            seq_lens = attention_trim.sum(dim=1).cpu()
            for seq_idx, row in enumerate(batch.itertuples()):
                concept_id = overwrite_concept_id if overwrite_concept_id is not None else row.concept_id
                seq_len = int(seq_lens[seq_idx].item())
                concept_id = int(concept_id)

                gate_values_seq = gate_values[seq_idx, :seq_len].tolist()
                gate_values_seq = [float(round(g, 6)) for g in gate_values_seq]
                max_act = max(gate_values_seq) if gate_values_seq else 0.0
                all_max_act.append(max_act)

                if not return_max_act_only:
                    max_indices = [i for i, g in enumerate(gate_values_seq) if g == max_act]
                    max_idx = max_indices[0] if max_indices else 0
                    tokens = self.tokenizer.tokenize(row.input)[prefix_length - 1:][:seq_len]
                    max_token = tokens[max_idx] if tokens else ""

                    all_acts.append(gate_values_seq)
                    all_max_act_idx.append(max_idx)
                    all_max_token.append(max_token)
                    all_tokens.append(tokens)

            del inputs, full_activations, activations, gate_values
            torch.cuda.empty_cache()

        if return_max_act_only:
            return {"max_act": all_max_act}

        return {
            "acts": all_acts,
            "max_act": all_max_act,
            "max_act_idx": all_max_act_idx,
            "max_token": all_max_token,
            "tokens": all_tokens,
        }

    @torch.no_grad()
    def predict_latents(self, examples, **kwargs):
        if not hasattr(self, "ax"):
            raise RuntimeError("Call make_model(mode='train') or load() before predict_latents().")

        self.ax.eval()
        batch_size = kwargs.get("batch_size", 32)
        prefix_length = kwargs.get("prefix_length", 1)

        all_max_act = []

        for start in range(0, len(examples), batch_size):
            batch = examples.iloc[start:start + batch_size]

            inputs = self.tokenizer(
                batch["input"].tolist(),
                return_tensors="pt",
                padding=True,
                truncation=True,
                add_special_tokens=True,
            ).to(self.device)

            full_activations = gather_residual_activations(self.model, self.layer, inputs)
            activations = full_activations[:, prefix_length:]
            attention_trim = inputs["attention_mask"][:, prefix_length:]

            # Compute gate values for all concepts
            gate_values = torch.sigmoid(
                torch.matmul(activations, self.ax.gate_linear.weight.T) + 
                self.ax.gate_linear.bias
            ).float().cpu()  # [batch_size, seq_len, num_concepts]
            
            seq_lens = attention_trim.sum(dim=1).cpu()

            for seq_idx in range(gate_values.size(0)):
                seq_len = int(seq_lens[seq_idx].item())
                gate_seq = gate_values[seq_idx, :seq_len]  # [seq_len, num_concepts]
                if seq_len > 0:
                    max_per_concept = gate_seq.max(dim=0).values  # [num_concepts]
                else:
                    max_per_concept = torch.zeros(gate_values.size(-1))
                all_max_act.append([float(round(v.item(), 6)) for v in max_per_concept])

            del inputs, full_activations, activations, gate_values
            torch.cuda.empty_cache()

        return {"max_act": all_max_act}

    def save(self, dump_dir, **kwargs):
        model_name = kwargs.get("model_name", self.__str__())
        
        # Save steering vector weights and biases
        weight_file = dump_dir / f"{model_name}_weight.pt"
        weight = self.ax.proj.weight.data.cpu()
        if weight_file.exists():
            weight = torch.cat([torch.load(weight_file), weight], dim=0)
        torch.save(weight, weight_file)
        
        bias_file = dump_dir / f"{model_name}_bias.pt"
        bias = self.ax.proj.bias.data.cpu()
        if bias_file.exists():
            bias = torch.cat([torch.load(bias_file), bias], dim=0)
        torch.save(bias, bias_file)
        
        # Save gate weights and biases
        gate_weight_file = dump_dir / f"{model_name}_gate_weight.pt"
        gate_weight = self.ax.gate_linear.weight.data.cpu()
        if gate_weight_file.exists():
            gate_weight = torch.cat([torch.load(gate_weight_file), gate_weight], dim=0)
        torch.save(gate_weight, gate_weight_file)
        
        gate_bias_file = dump_dir / f"{model_name}_gate_bias.pt"
        gate_bias = self.ax.gate_linear.bias.data.cpu()
        if gate_bias_file.exists():
            gate_bias = torch.cat([torch.load(gate_bias_file), gate_bias], dim=0)
        torch.save(gate_bias, gate_bias_file)

    def load(self, dump_dir=None, **kwargs):
        priority_mode = kwargs.get("priority_mode", "compute_priority")
        self.priority_mode = priority_mode
        model_name = kwargs.get("model_name", self.__str__())
        
        if priority_mode == "mem_priority":
            # Load only specific concept for memory efficiency
            concept_id = kwargs.get("concept_id")
            
            # Load steering parameters
            weight = torch.load(
                f"{dump_dir}/{model_name}_weight.pt",
                map_location=torch.device("cpu"),
                mmap=True
            )
            bias = torch.load(
                f"{dump_dir}/{model_name}_bias.pt",
                map_location=torch.device("cpu"),
                mmap=True
            )
            
            # Load gate parameters
            gate_weight = torch.load(
                f"{dump_dir}/{model_name}_gate_weight.pt",
                map_location=torch.device("cpu"),
                mmap=True
            )
            gate_bias = torch.load(
                f"{dump_dir}/{model_name}_gate_bias.pt",
                map_location=torch.device("cpu"),
                mmap=True
            )
            
            # Select only the needed concept
            weight_rank_1 = weight[concept_id].unsqueeze(0)
            bias_rank_1 = bias[concept_id].unsqueeze(0)
            gate_weight_rank_1 = gate_weight[concept_id].unsqueeze(0)
            gate_bias_rank_1 = gate_bias[concept_id].unsqueeze(0)
            
            # Initialize model with single concept
            kwargs["low_rank_dimension"] = 1
            self.make_model(**kwargs)
            
            # Load parameters
            self.ax.proj.weight.data = weight_rank_1.to(self.device)
            self.ax.proj.bias.data = bias_rank_1.to(self.device)
            self.ax.gate_linear.weight.data = gate_weight_rank_1.to(self.device)
            self.ax.gate_linear.bias.data = gate_bias_rank_1.to(self.device)
            
        elif priority_mode == "compute_priority":
            # Load all concepts for compute efficiency
            print(f"Loading {model_name} from {dump_dir}.")
            
            # Load steering parameters
            weight = torch.load(
                f"{dump_dir}/{model_name}_weight.pt",
                map_location=torch.device("cpu"),
            )
            bias = torch.load(
                f"{dump_dir}/{model_name}_bias.pt",
                map_location=torch.device("cpu"),
            )
            
            # Load gate parameters
            gate_weight = torch.load(
                f"{dump_dir}/{model_name}_gate_weight.pt",
                map_location=torch.device("cpu"),
            )
            gate_bias = torch.load(
                f"{dump_dir}/{model_name}_gate_bias.pt",
                map_location=torch.device("cpu"),
            )
            
            # Override low_rank_dimension based on saved weights
            kwargs["low_rank_dimension"] = weight.shape[0]
            self.make_model(**kwargs)
            
            # Load all parameters
            self.ax.proj.weight.data = weight.to(self.device)
            self.ax.proj.bias.data = bias.to(self.device)
            self.ax.gate_linear.weight.data = gate_weight.to(self.device)
            self.ax.gate_linear.bias.data = gate_bias.to(self.device)
