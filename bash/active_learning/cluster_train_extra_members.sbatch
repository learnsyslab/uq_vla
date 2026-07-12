#!/bin/bash
#SBATCH --job-name=train_extra_members
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=rtx_4090:1
#SBATCH --output=slurm/train_extra_members-%j.out
#SBATCH --error=slurm/train_extra_members-%j.err

set -euo pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
    echo "Python executable not found or not executable: $PYTHON_BIN" >&2
    exit 1
fi

if [ $# -lt 1 ]; then
    echo "Usage: sbatch cluster_train_extra_members.sbatch <train_extra_al_members args...>" >&2
    exit 1
fi

export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"
source ".env"

echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Host: $(hostname)"
echo "Working directory: $SCRIPT_DIR"
echo "Python: $PYTHON_BIN"
echo "Args: $*"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
date

srun --cpu-bind=none "$PYTHON_BIN" scripts/maintenance/train_extra_al_members.py "$@"
