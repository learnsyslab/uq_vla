#!/bin/bash
#SBATCH --job-name=eval_calibration
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=6G
#SBATCH --gpus=rtx_4090:1
#SBATCH --output=slurm/eval_calibration-%j.out
#SBATCH --error=slurm/eval_calibration-%j.err

set -euo pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
    echo "Python executable not found or not executable: $PYTHON_BIN" >&2
    exit 1
fi

if [ $# -lt 1 ] || [ $# -gt 2 ]; then
    echo "Usage: sbatch cluster_eval.sbatch <config_path|run_name> [training_job_id]" >&2
    exit 1
fi
TARGET="$1"
WATCH_JOB_ID="${2:-}"

export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"

source ".env"

# If TARGET is a run name (not a file), extract SEED_A/SEED_B from the _sXY suffix
# so ${SEED_A}/${SEED_B} in the saved config snapshot can be resolved.
if [ ! -f "$TARGET" ] && [[ "$TARGET" =~ _s([0-9])([0-9])$ ]]; then
    export SEED_A="${BASH_REMATCH[1]}"
    export SEED_B="${BASH_REMATCH[2]}"
    echo "Extracted SEED_A=$SEED_A SEED_B=$SEED_B from run name."
fi

# When watching a run by name, wait for the train job to create the config snapshot.
if [ -n "$WATCH_JOB_ID" ] && [ ! -f "$TARGET" ]; then
    SNAPSHOT="${SCRIPT_DIR}/outputs/iterative_fine_tuning/${TARGET}/iterative_fine_tuning_config.json"
    echo "Waiting for config snapshot: $SNAPSHOT"
    until [ -f "$SNAPSHOT" ]; do sleep 30; done
    echo "Config snapshot found, starting evaluation."
fi
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Host: $(hostname)"
echo "Working directory: $SCRIPT_DIR"
echo "Python: $PYTHON_BIN"
echo "Target: $TARGET"
echo "Watched training job: ${WATCH_JOB_ID:-unset}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "HF_HOME: ${HF_HOME:-unset}"
date


EVAL_ARGS=(
    --member-devices cuda:0
    --workers-per-device 2
    --n-rollouts 30
    --eval-batch-size 4
    --sync-envs
    --skip-existing
)

if [ -f "$TARGET" ]; then
    EVAL_ARGS+=(--config-path="$TARGET" --watch)
else
    EVAL_ARGS+=(--run-name="$TARGET" --runs-root="${SCRIPT_DIR}/outputs/iterative_fine_tuning")
fi

if [ -n "$WATCH_JOB_ID" ]; then
    EVAL_ARGS+=(--watch --watch-job-id="$WATCH_JOB_ID")
fi

if [ "${CLEANUP_CHECKPOINTS_WHEN_SAFE:-0}" = "1" ]; then
    EVAL_ARGS+=(--cleanup-checkpoints-when-safe)
fi

srun --cpu-bind=none "$PYTHON_BIN" -m iterative_fine_tuning.evaluate_run "${EVAL_ARGS[@]}"
