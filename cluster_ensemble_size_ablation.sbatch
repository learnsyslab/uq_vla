#!/bin/bash
#SBATCH --job-name=ensemble_size_ablation
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=6G
#SBATCH --gpus=rtx_4090:1
#SBATCH --output=slurm/ensemble_size_ablation-%j.out
#SBATCH --error=slurm/ensemble_size_ablation-%j.err

set -euo pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
    echo "Python executable not found or not executable: $PYTHON_BIN" >&2
    exit 1
fi

if [ $# -lt 1 ]; then
    echo "Usage: sbatch cluster_ensemble_size_ablation.sbatch <args passed to ensemble_size_ablation.py>" >&2
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

srun --cpu-bind=none "$PYTHON_BIN" scripts/calibration/ensemble_size_ablation.py "$@"
