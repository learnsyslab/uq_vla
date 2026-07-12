#!/bin/bash
#SBATCH --job-name=active_learning_vlas
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=rtx_4090:2
#SBATCH --output=slurm/active_learning-%j.out
#SBATCH --error=slurm/active_learning-%j.err

set -euo pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
    echo "Python executable not found or not executable: $PYTHON_BIN" >&2
    exit 1
fi

if [ $# -lt 1 ]; then
    echo "Usage: sbatch cluster_train.sbatch <config_path> [seed_a] [seed_b] [extra draccus overrides...]" >&2
    echo "  seed_a / seed_b: model seed indices (e.g. 0 1, 2 3, 4 5)." >&2
    echo "  selection.seed is set to seed_a so each seed pair gets different selections." >&2
    exit 1
fi
CONFIG_PATH="$1"
export SEED_A="${2:-0}"
export SEED_B="${3:-1}"
shift 3 2>/dev/null || shift $#
EXTRA_ARGS=("$@")

export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"

source ".env"

echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Host: $(hostname)"
echo "Working directory: $SCRIPT_DIR"
echo "Python: $PYTHON_BIN"
echo "Config: $CONFIG_PATH"
echo "SEED_A: $SEED_A  SEED_B: $SEED_B"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "HF_HOME: ${HF_HOME:-unset}"
date


srun --cpu-bind=none "$PYTHON_BIN" -m iterative_fine_tuning.main \
    --config_path="$CONFIG_PATH" \
    --selection.seed="$SEED_A" \
    "${EXTRA_ARGS[@]}"
