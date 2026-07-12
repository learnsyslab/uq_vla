#!/bin/bash
#SBATCH --job-name=pretrain
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=rtx_4090:1
#SBATCH --output=slurm/pretrain-%j.out
#SBATCH --error=slurm/pretrain-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}}"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
    echo "Python executable not found or not executable: $PYTHON_BIN" >&2
    exit 1
fi

if [ $# -ne 2 ]; then
    echo "Usage: sbatch cluster_pretrain.sbatch <config_path> <seed>" >&2
    exit 1
fi

CONFIG_PATH="$1"
SEED="$2"

export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"

source ".env"

STORAGE_ROOT="${STORAGE_ROOT:?STORAGE_ROOT environment variable must be set}"
OUTPUT_DIR="${STORAGE_ROOT}/outputs/pretrain/$(basename "$CONFIG_PATH" .yaml)/seed_${SEED}"

echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Host: $(hostname)"
echo "Config: $CONFIG_PATH"
echo "Seed: $SEED"
echo "Output: $OUTPUT_DIR"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "HF_HOME: ${HF_HOME:-unset}"
date


WANDB_ENABLE="${WANDB_ENABLE:-false}"
WANDB_PROJECT="${WANDB_PROJECT:-active-vla}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_MODE="${WANDB_MODE:-disabled}"

WANDB_ARGS=(--wandb.enable "$WANDB_ENABLE" --wandb.mode "$WANDB_MODE")
if [ -n "$WANDB_PROJECT" ]; then
    WANDB_ARGS+=(--wandb.project "$WANDB_PROJECT")
fi
if [ -n "$WANDB_ENTITY" ]; then
    WANDB_ARGS+=(--wandb.entity "$WANDB_ENTITY")
fi

srun "$PYTHON_BIN" src/lerobot/scripts/lerobot_train.py \
    --config "$CONFIG_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --seed "$SEED" \
    --job_name "pretrain_$(basename "$CONFIG_PATH" .yaml)_seed${SEED}" \
    "${WANDB_ARGS[@]}" \
    --policy.push_to_hub false
