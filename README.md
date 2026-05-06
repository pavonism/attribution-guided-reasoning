# Attribution-Guided Reasoning in Vision Language Models

This repository contains the official implementation of the training, data generation, and evaluation pipelines for our Attribution-Guided Reinforcement Fine-Tuning framework.

## 🌳 Project Structure
```bash
│   .gitignore            # Files to ignore
│   README.md             # This file
├───scripts               # Execution scripts for training, eval, and data gen
├───singularity           # Singularity recipe files for reproducible environments
├───slurm                 # Sanitized HPC cluster submission scripts
├───src                   # Core source code (algorithms, structures)
└───tests                 # Unit tests
```

## 🛠️ Setup & Environments

To ensure strict reproducibility, we use [Singularity/Apptainer](https://apptainer.org/) containers. Before running any scripts, build the appropriate container using the `recipe.def` files located in the `./singularity` directory.

* **`virft` container:** Use this for training, data generation, and Hugging Face evaluations.
* **`vllm` container:** Use this for vLLM evaluation scripts (provides the newest version of the `vllm` package).

You can build the containers using the following command: 
```bash
# Example: Building the vLLM container
singularity build ./containers/vllm.sif ./singularity/vllm/recipe.def

# Example: Building the standard training/eval container
singularity build ./containers/virft.sif ./singularity/virft/recipe.def
```

## 🚀 Usage Examples

The arguments for all scripts are defined using the standard `argparse` Python package. Use `--help` on any script to see the full list of available parameters.

HPC Cluster Execution: The examples below utilize sbatch wrappers for SLURM-managed clusters. If you are running locally without a scheduler, you can execute the underlying singularity exec commands found directly inside these shell scripts.

*(Note: If you are accessing gated models like Gemma, ensure your `HF_TOKEN` environment variable is set before executing the container).*

### 1. Evaluation
To run an evaluation script with the `vllm` container, use the following command: 
```bash
sbatch ./slurm/evaluate-vllm.sh \
--model google/gemma-4-E4B-it \
--dataset mehrankazemi/ReMI \
--output_file ./outputs/evaluate/remi/google/gemma-4-E4B-it.json \
--batch_size 64 \
--max_pixels 1568000
```

### 2. Training (Reinforcement Fine-Tuning)
To run the training, use the `virft` container:

```bash
sbatch ./slurm/virft-array.sh \
--model_name_or_path Qwen/Qwen3-VL-4B-Instruct \
--train_split ./datasets/bongard-merged-concat-hf \
--output_dir ./models/qwen3-vl-4B-virft-bmerged-concat-bow-8k-KL-0_005-MAX_PIX-1.5M \
--run_name qwen3-vl-4B-virft-bmerged-concat-bow-8k-KL-0_005-MAX_PIX-1.5M \
--reward_funcs format bow \
--resume
```
*(Adjust the training arguments above to match your actual `grpo.py` flags).*

## 📊 Data
Scripts for generating the entities and attribution masks (via SAM3) can be found in the `scripts/` directory.