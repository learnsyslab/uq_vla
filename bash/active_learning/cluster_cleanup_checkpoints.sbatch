#!/bin/bash
#SBATCH --job-name=cleanup_ckpts
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --output=slurm/cleanup_checkpoints-%j.out
#SBATCH --error=slurm/cleanup_checkpoints-%j.err

set -euo pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}"
cd "$SCRIPT_DIR"

if [ $# -ne 1 ]; then
    echo "Usage: sbatch cluster_cleanup_checkpoints.sbatch <run_name>" >&2
    exit 1
fi

RUN_NAME="$1"

source ".env"

if [ -z "${STORAGE_ROOT:-}" ]; then
    echo "STORAGE_ROOT is not set." >&2
    exit 1
fi

CHECKPOINT_RUN_DIR="${STORAGE_ROOT}/outputs/iterative_fine_tuning/${RUN_NAME}"

echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Host: $(hostname)"
echo "Run name: $RUN_NAME"
echo "Checkpoint run directory: $CHECKPOINT_RUN_DIR"
date

if [ ! -d "$CHECKPOINT_RUN_DIR" ]; then
    echo "Checkpoint run directory does not exist; nothing to delete."
    exit 0
fi

find "$CHECKPOINT_RUN_DIR" -type d -name checkpoints -prune -print -exec rm -rf {} +

echo "Checkpoint cleanup finished."
date
