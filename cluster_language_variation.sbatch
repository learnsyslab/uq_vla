#!/bin/bash
#SBATCH --job-name=language_variation
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=6G
#SBATCH --gpus=rtx_4090:1
#SBATCH --output=slurm/language_variation-%j.out
#SBATCH --error=slurm/language_variation-%j.err

set -euo pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
    echo "Python executable not found or not executable: $PYTHON_BIN" >&2
    exit 1
fi

if [ $# -lt 1 ]; then
    echo "Usage: sbatch cluster_language_variation.sbatch --run_dir <dir> --round <N> [extra args...]" >&2
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

mkdir -p slurm

RUN_ROOT_SCRATCH="${STORAGE_ROOT}/outputs/iterative_fine_tuning"
RUN_ROOT_HOME="${SCRIPT_DIR}/outputs/iterative_fine_tuning"
ARGS=()
for arg in "$@"; do
    if [[ "$arg" == "$RUN_ROOT_SCRATCH"* ]]; then
        ARGS+=("${RUN_ROOT_HOME}${arg#"$RUN_ROOT_SCRATCH"}")
    else
        ARGS+=("$arg")
    fi
done

srun --cpu-bind=none "$PYTHON_BIN" scripts/calibration/language_variation.py "${ARGS[@]}"
