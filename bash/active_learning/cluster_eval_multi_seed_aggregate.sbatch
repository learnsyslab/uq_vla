#!/bin/bash
#SBATCH --job-name=active_learning_seed_aggregate
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --output=slurm/active_learning-seed-aggregate-%j.out
#SBATCH --error=slurm/active_learning-seed-aggregate-%j.err

set -euo pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
    echo "Python executable not found or not executable: $PYTHON_BIN" >&2
    exit 1
fi

if [ $# -ne 1 ]; then
    echo "Usage: sbatch cluster_eval_multi_seed_aggregate.sbatch <config_path>" >&2
    exit 1
fi
CONFIG_PATH="$1"

export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"

source ".env"

echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Host: $(hostname)"
echo "Working directory: $SCRIPT_DIR"
echo "Python: $PYTHON_BIN"
echo "Config: $CONFIG_PATH"
date

srun "$PYTHON_BIN" -m iterative_fine_tuning.summarize_multi_seed_run --config-path="$CONFIG_PATH"
