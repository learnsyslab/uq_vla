#!/bin/bash
#SBATCH --job-name=active_learning_vlas_seed
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=rtx_4090:2
#SBATCH --output=slurm/active_learning-seed-%j.out
#SBATCH --error=slurm/active_learning-seed-%j.err

set -euo pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
    echo "Python executable not found or not executable: $PYTHON_BIN" >&2
    exit 1
fi

if [ $# -ne 2 ]; then
    echo "Usage: sbatch cluster_train_multi_seed_child.sbatch <config_path> <seed>" >&2
    exit 1
fi
CONFIG_PATH="$1"
SEED="$2"

export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"

source ".env"

echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Host: $(hostname)"
echo "Working directory: $SCRIPT_DIR"
echo "Python: $PYTHON_BIN"
echo "Config: $CONFIG_PATH"
echo "Seed: $SEED"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "HF_HOME: ${HF_HOME:-unset}"
date


no_proxy=",${no_proxy:-},"
no_proxy="${no_proxy//,api.wandb.ai,/,}"
export no_proxy="${no_proxy#,}"
export no_proxy="${no_proxy%,}"

NO_PROXY=",${NO_PROXY:-},"
NO_PROXY="${NO_PROXY//,api.wandb.ai,/,}"
export NO_PROXY="${NO_PROXY#,}"
export NO_PROXY="${NO_PROXY%,}"

srun "$PYTHON_BIN" -m iterative_fine_tuning.run_multi_seed_child \
    --config-path="$CONFIG_PATH" \
    --seed="$SEED"
