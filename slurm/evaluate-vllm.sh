#!/usr/bin/env bash

#SBATCH --partition=<YOUR_PARTITION>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --gpus=1
#SBATCH --time=1-00:00:00
#SBATCH --account=<YOUR_ACCOUNT>

export PROJECT_DIR="<YOUR_PROJECT_DIRECTORY>"
export STORAGE="<YOUR_STORAGE_DIRECTORY>"
export JOB_HF_HOME="$STORAGE/huggingface/tmp/${SLURM_JOB_ID}"
export JOB_TMPDIR="$STORAGE/tmp/${SLURM_JOB_ID}"

date
echo "SLURMD_NODENAME: ${SLURMD_NODENAME}"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "singularity version: $(singularity version)"
echo "nvidia-container-toolkit version: $(nvidia-container-toolkit -version)"
echo "nvidia-container-cli info: $(nvidia-container-cli info)"
echo "JOB_HF_HOME: ${JOB_HF_HOME}"
echo "JOB_TMPDIR: ${JOB_TMPDIR}"
echo "PROJECT_DIR: ${PROJECT_DIR}"
echo "STORAGE: ${STORAGE}"

echo "Args: ${@}"

mkdir ${JOB_HF_HOME}
mkdir ${JOB_TMPDIR}

module load singularity

singularity exec \
  --nv \
  --env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  --env HF_HUB_CACHE=/app/huggingface/hub \
  --env HF_HOME="/app/huggingface/tmp/${SLURM_JOB_ID}" \
  --env HF_TOKEN_PATH=/app/huggingface/token \
  --env TMPDIR="/app/tmp/${SLURM_JOB_ID}" \
  --env PYTHONPATH=/app \
  --pwd /app \
  --workdir /app \
  --bind $PROJECT_DIR:/app:rw \
  --bind $STORAGE/huggingface:/app/huggingface:rw \
  --bind $STORAGE/tmp:/app/tmp \
  --bind $STORAGE/models:/app/models \
  --bind $STORAGE/outputs:/app/outputs \
  --bind $STORAGE/datasets:/app/datasets \
  $STORAGE/containers/vllm.sif \
  python3 ./scripts/evaluate-vllm.py \
    --tensor_parallel_size 1 \
    $@

date

rm -r ${JOB_HF_HOME}
rm -r ${JOB_TMPDIR}

