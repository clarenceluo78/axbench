<div align="center">
  <a align="center"><img src="https://github.com/user-attachments/assets/661f78cf-4044-4c46-9a71-1316bb2c69a5" width="100" height="100" /></a>
  <h1 align="center">AxBench <sub>by <a href="https://github.com/stanfordnlp/pyvene">pyvene</a></sub></h1>
  <a href="https://arxiv.org/abs/2501.17148"><strong>Read our paper »</strong></a>
</div>     

<br>

**AxBench** is a a scalable benchmark that evaluates interpretability techniques on two axes: *concept detection* and *model steering*. This repo includes all benchmarking code, including data generation, training, evaluation, and analysis.

We introduced **supervised dictionary learning** (SDL) on synthetic data as an analogue to SAEs. You can access pretrained SDLs and our training/eval datasets here:

- 🤗 **HuggingFace**: [**AxBench Collections**](https://huggingface.co/collections/pyvene/axbench-release-6787576a14657bb1fc7a5117)  
- 🤗 **ReFT-r1 Live Demo**: [**Steering ChatLM**](https://huggingface.co/spaces/pyvene/AxBench-ReFT-r1-16K)
- 🤗 **ReFT-cr1 Live Demo**: [**Conditional Steering ChatLM**](https://huggingface.co/spaces/pyvene/AxBench-ReFT-cr1-16K)
- 📚 **Feature Visualizer**: [**Visualize LM Activations**](https://nlp.stanford.edu/~wuzhengx/axbench/index.html)
- 🔍 **Subspace Gazer**: [**Visualize Subspaces via UMAP**](https://nlp.stanford.edu/~wuzhengx/axbench/visualization_UMAP.html)
- [<img align="center" src="https://colab.research.google.com/assets/colab-badge.svg" />](https://colab.research.google.com/github/stanfordnlp/axbench/blob/main/axbench/examples/tutorial.ipynb) **Tutorial of using our dictionary via [pyvene](https://github.com/stanfordnlp/pyvene)**


## 🏆 Steering leaderboard

| Method                     | 2B L10 | 2B L20 | 9B L20 | 9B L31 |  Avg |
|----------------------------|-------:|-------:|-------:|-------:|-----:|
| Prompt                     | 0.698 | 0.731 | **1.075** | **1.072** | **0.894** |
| LoReFT$_{RePS}$            | 0.758 | **0.805** | 0.757 | 0.759 | 0.770 |
| LoReFT$_{\text{Lang.}}$    | 0.768 | 0.790 | 0.722 | 0.725 | 0.751 |
| SV$_{RePS}$                | 0.756 | 0.606 | 0.892 | 0.624 | 0.720 |
| LoRA$_{RePS}$              | **0.798** | 0.793 | 0.631 | 0.633 | 0.714 |
| SFT                        | 0.637 | 0.714 |   —   |   —   | 0.676* |
| SV$_{\text{Lang.}}$        | 0.663 | 0.568 | 0.788 | 0.580 | 0.650 |
| LoRA$_{\text{Lang.}}$      | 0.710 | 0.723 | 0.578 | 0.549 | 0.640 |
| ReFT-r1                    | 0.633 | 0.509 | 0.630 | 0.401 | 0.543 |
| DiffMean                   | 0.297 | 0.178 | 0.322 | 0.158 | 0.239 |
| SV$_{\text{BiPO}}$         | 0.199 | 0.173 | 0.217 | 0.179 | 0.192 |
| LoRA$_{\text{BiPO}}$       | 0.149 | 0.156 | 0.209 | 0.188 | 0.176 |
| SAE                        | 0.177 | 0.151 | 0.191 | 0.140 | 0.165 |
| SAE-A                      | 0.166 | 0.132 | 0.186 | 0.143 | 0.157 |
| LAT                        | 0.117 | 0.130 | 0.127 | 0.134 | 0.127 |
| PCA                        | 0.107 | 0.083 | 0.128 | 0.104 | 0.106 |
| Probe                      | 0.095 | 0.091 | 0.108 | 0.099 | 0.098 |
| LoReFT$_{\text{BiPO}}$     | 0.077 | 0.067 | 0.075 | 0.084 | 0.076 |
| SSV                        | 0.072 | 0.001 | 0.024 | 0.008 | 0.026 |

* SFT average is over the two available scores (2B L10/L20).


## 🔥 New releases

- 05/25: Steering eval on feature suppression / many-shot jailbreaking are added.
- 05/25: New steering method RePS from [improved representation steering for language models](link-goes-here).


## 🎯 Highlights

1. **Scalabale evaluation harness**: Framework for generating synthetic training + eval data from concept lists (e.g. GemmaScope SAE labels).
2. **Comprehensive implementations**: 10+ interpretability methods evaluated, along with finetuning and prompting baselines.
2. **16K concept training data**: Full-scale datasets for **supervised dictionary learning (SDL)**.  
3. **Two pretrained SDL models**: Drop-in replacements for standard SAEs.  
4. **LLM-in-the-loop training**: Generate your own datasets for less than \$0.01 per concept.


## Additional experiments

We include exploratory notebooks under `axbench/examples`, such as:

| Experiment                              | Description                                                                   |
|----------------------------------------|-------------------------------------------------------------------------------|
| `basics.ipynb`                         | Analyzes basic geometry of learned dictionaries.                              |
| `subspace_gazer.ipynb`                | Visualizes learned subspaces.                                                 |
| `lang>subspace.ipynb`                 | Fine-tunes a hyper-network to map natural language to subspaces or steering vectors. |
| `platonic.ipynb`                      | Explores the platonic representation hypothesis in subspace learning.         |

---

## Instructions for AxBenching your methods

### Installation

We highly suggest using `uv` for your Python virtual environment, but you can use any venv manager.

```bash
git clone git@github.com:stanfordnlp/axbench.git
cd axbench
uv sync # if using uv
```

Set up your API keys for OpenAI and Neuronpedia:

```python
import os
os.environ["OPENAI_API_KEY"] = "your_openai_api_key_here"
os.environ["NP_API_KEY"] = "your_neuronpedia_api_key_here"
```

Download the necessary datasets to `axbench/data`:

```bash
uv run axbench/data/download-seed-sentences.py
cd axbench/data
bash download-2b.sh
bash download-9b.sh
bash download-alpaca.sh
```

### Try a simple demo.

To run a complete demo with a single config file:

```bash
bash axbench/demo/demo.sh
```

## Data generation

(If using our pre-generated data, you can skip this.)

**Generate training data:**

```bash
uv run axbench/scripts/generate.py --config axbench/demo/sweep/simple.yaml --dump_dir axbench/demo
```

**Generate inference data:**

```bash
uv run axbench/scripts/generate_latent.py --config axbench/demo/sweep/simple.yaml --dump_dir axbench/demo
```

To modify the data generation process, edit `simple.yaml`.

## Training

Train and save your methods:

```bash
uv run torchrun --nproc_per_node=$gpu_count axbench/scripts/train.py \
  --config axbench/demo/sweep/simple.yaml \
  --dump_dir axbench/demo
```

(Replace `$gpu_count` with the number of GPUs to use.)

For additional config:

```bash
torchrun --nproc_per_node=$gpu_count axbench/scripts/train.py \
  --config axbench/sweep/wuzhengx/2b/l10/no_grad.yaml \
  --dump_dir axbench/results/prod_2b_l10_concept500_no_grad \
  --overwrite_data_dir axbench/concept500/prod_2b_l10_v1/generate
```

where `--dump_dir` is the output directory, and `--overwrite_data_dir` is where the training data resides.

## Inference

### Concept detection

Run inference:

```bash
uv run torchrun --nproc_per_node=$gpu_count axbench/scripts/inference.py \
  --config axbench/demo/sweep/simple.yaml \
  --dump_dir axbench/demo \
  --mode latent
```

For additional config using custom directories:

```bash
uv run torchrun --nproc_per_node=$gpu_count axbench/scripts/inference.py \
  --config axbench/sweep/wuzhengx/2b/l10/no_grad.yaml \
  --dump_dir axbench/results/prod_2b_l10_concept500_no_grad \
  --overwrite_metadata_dir axbench/concept500/prod_2b_l10_v1/generate \
  --overwrite_inference_data_dir axbench/concept500/prod_2b_l10_v1/inference \
  --mode latent
```

#### Imbalanced concept detection

For real-world scenarios with fewer than 1% positive examples, we upsample negatives (100:1) and re-evaluate. Use:

```bash
uv run torchrun --nproc_per_node=$gpu_count axbench/scripts/inference.py \
  --config axbench/sweep/wuzhengx/2b/l10/no_grad.yaml \
  --dump_dir axbench/results/prod_2b_l10_concept500_no_grad \
  --overwrite_metadata_dir axbench/concept500/prod_2b_l10_v1/generate \
  --overwrite_inference_data_dir axbench/concept500/prod_2b_l10_v1/inference \
  --mode latent_imbalance
```

### Model steering

For steering experiments:

```bash
uv run torchrun --nproc_per_node=$gpu_count axbench/scripts/inference.py \
  --config axbench/demo/sweep/simple.yaml \
  --dump_dir axbench/demo \
  --mode steering
```

Or a custom run:

```bash
uv run torchrun --nproc_per_node=$gpu_count axbench/scripts/inference.py \
  --config axbench/sweep/wuzhengx/2b/l10/no_grad.yaml \
  --dump_dir axbench/results/prod_2b_l10_concept500_no_grad \
  --overwrite_metadata_dir axbench/concept500/prod_2b_l10_v1/generate \
  --overwrite_inference_data_dir axbench/concept500/prod_2b_l10_v1/inference \
  --mode steering
```

## Evaluation

### Concept detection

To evaluate concept detection results:

```bash
uv run axbench/scripts/evaluate.py \
  --config axbench/demo/sweep/simple.yaml \
  --dump_dir axbench/demo \
  --mode latent
```

Enable wandb logging:

```bash
uv run axbench/scripts/evaluate.py \
  --config axbench/demo/sweep/simple.yaml \
  --dump_dir axbench/demo \
  --mode latent \
  --report_to wandb \
  --wandb_entity "your_wandb_entity"
```

Or evaluate using your custom config:

```bash
uv run axbench/scripts/evaluate.py \
  --config axbench/sweep/wuzhengx/2b/l10/no_grad.yaml \
  --dump_dir axbench/results/prod_2b_l10_concept500_no_grad \
  --mode latent
```

### Model steering on evaluation set

To evaluate steering:

```bash
uv run axbench/scripts/evaluate.py \
  --config axbench/demo/sweep/simple.yaml \
  --dump_dir axbench/demo \
  --mode steering
```

Or a custom config:

```bash
uv run axbench/scripts/evaluate.py \
  --config axbench/sweep/wuzhengx/2b/l10/no_grad.yaml \
  --dump_dir axbench/results/prod_2b_l10_concept500_no_grad \
  --mode steering
```

### Model steering on test set
Note that the commend above is for evaluation. We select the best factor by using the results on the evaluation set. After that you will do the evaluation on the test set.

```bash
uv run axbench/scripts/evaluate.py \
  --config axbench/sweep/wuzhengx/2b/l10/no_grad.yaml \
  --dump_dir axbench/results/prod_2b_l10_concept500_no_grad \
  --mode steering_test
```

## Analyses
Once you finished evaluation, you can do the analyses with our provided notebook in `axbench/scripts/analyses.ipynb`. All of our results in the paper are produced by this notebook.

You need to point revelant directories to your own results by modifying the notebook. If you introduce new models, datasets, or new evaluation metrics, you can add your own analysis by following the notebook.

## Reproducing our results.

Please see `axbench/experiment_commands.txt` for detailed commands and configurations.
