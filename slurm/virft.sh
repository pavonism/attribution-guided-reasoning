#!/usr/bin/env bash

#SBATCH --partition=<YOUR_PARTITION>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --gpus=4
#SBATCH --time=1-00:00:00
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --array=1-10%1

export HOME="<YOUR_HOME_DIRECTORY>"
export STORAGE="<YOUR_STORAGE_DIRECTORY>"
export WANDB_API_KEY="<YOUR_WANDB_API_KEY>"
export WANDB_PROJECT="<YOUR_WANDB_PROJECT_NAME>"
export PROJECT_DIR="<YOUR_PROJECT_DIRECTORY>"

export SLURM_RUNNING_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
export JOB_HF_HOME="${STORAGE}/huggingface/tmp/${SLURM_JOB_ID}"
export JOB_TMPDIR="${STORAGE}/tmp/${SLURM_JOB_ID}"

date
echo "SLURMD_NODENAME: ${SLURMD_NODENAME}"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "SLURM_RUNNING_GPUS: ${SLURM_RUNNING_GPUS}"
echo "singularity version: $(singularity version)"
echo "nvidia-container-toolkit version: $(nvidia-container-toolkit -version)"
echo "nvidia-container-cli info: $(nvidia-container-cli info)"
echo "JOB_HF_HOME: ${JOB_HF_HOME}"
echo "JOB_TMPDIR: ${JOB_TMPDIR}"
echo "STORAGE: ${STORAGE}"

echo "Args: ${@}"

mkdir ${JOB_HF_HOME}
mkdir ${JOB_TMPDIR}

module load singularity

singularity exec \
  --nv \
  --env HF_HUB_CACHE=/app/huggingface/hub \
  --env HF_HOME="/app/huggingface/tmp/${SLURM_JOB_ID}" \
  --env TMPDIR="/app/tmp/${SLURM_JOB_ID}" \
  --env TRITON_CACHE_DIR="/app/tmp/${SLURM_JOB_ID}/.triton" \
  --env TORCH_HOME="/app/tmp/${SLURM_JOB_ID}/.torch" \
  --env TORCH_EXTENSIONS_DIR="/app/tmp/${SLURM_JOB_ID}/.torch_extensions" \
  --env PYTHONPATH=/app \
  --env VLLM_LOGGING_LEVEL=DEBUG \
  --env WANDB_API_KEY="${WANDB_API_KEY}" \
  --env WANDB_PROJECT="${WANDB_PROJECT}" \
  --env WANDB_RESUME="allow" \
  --env PYTHONDONTWRITEBYTECODE=1 \
  --home $HOME \
  --pwd /app \
  --workdir /app \
  --bind $PROJECT_DIR:/app:rw \
  --bind $STORAGE/huggingface:/app/huggingface:rw \
  --bind $STORAGE/tmp:/app/tmp:rw \
  --bind $STORAGE/datasets:/app/datasets:rw \
  --bind $STORAGE/models:/app/models:rw \
  $STORAGE/containers/virft.sif \
  torchrun --nproc_per_node="${SLURM_RUNNING_GPUS}" \
    --nnodes="${SLURM_NNODES}" \
    --node_rank="${SLURM_NODEID}" \
    --master-port=29502 \
    src/virft/grpo.py \
    --deepspeed src/virft/deepspeed.json \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 true \
    --report_to wandb \
    --gradient_checkpointing true \
    --max_pixels 1568000 \
    --learning_rate 1e-6 \
    --lr_scheduler_type constant \
    --attn_implementation flash_attention_2 \
    --save_steps 100 \
    --save_only_model false \
    --num_generations 4 \
    --eval_strategy no \
    --max_steps_per_job 100 \
    --reward_funcs format bow \
    --max_completion_length 8192 \
    --beta 0.005 \
    --format vision-r1 \
    --job_id ${SLURM_ARRAY_TASK_ID} \
    --job_count ${SLURM_ARRAY_TASK_COUNT} \
    $@

date

rm -r ${JOB_HF_HOME}
rm -r ${JOB_TMPDIR}
