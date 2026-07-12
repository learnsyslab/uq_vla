#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

if [ $# -lt 1 ]; then
    echo "Usage: $0 <config_path> [seed_a] [seed_b] [extra draccus overrides...]" >&2
    exit 1
fi

CONFIG_INPUT="$1"
if [ ! -f "$CONFIG_INPUT" ]; then
    echo "Config file not found: $CONFIG_INPUT" >&2
    exit 1
fi

CONFIG_PATH="$(cd "$(dirname "$CONFIG_INPUT")" && pwd)/$(basename "$CONFIG_INPUT")"

mkdir -p slurm

CLEANUP_CHECKPOINTS_AFTER_EVAL=false
TRAIN_ARGS=()
RUN_NAME=""
for arg in "${@:2}"; do
    case "$arg" in
        --delete-checkpoints-after-eval)
            CLEANUP_CHECKPOINTS_AFTER_EVAL=true
            ;;
        *)
            TRAIN_ARGS+=("$arg")
            if [[ "$arg" == --paths.run_name=* ]]; then
                RUN_NAME="${arg#--paths.run_name=}"
            fi
            ;;
    esac
done

SBATCH_JOB_ARGS=()
if [ -n "$RUN_NAME" ]; then
    SBATCH_JOB_ARGS+=(--job-name="$RUN_NAME")
fi

TRAIN_SUBMISSION="$(sbatch --parsable "${SBATCH_JOB_ARGS[@]}" bash/active_learning/cluster_train.sbatch "$CONFIG_PATH" "${TRAIN_ARGS[@]}")"
TRAIN_JOB_ID="${TRAIN_SUBMISSION%%;*}"

EVAL_TARGET="${RUN_NAME:-$CONFIG_PATH}"

EVAL_SUBMISSION="$(
    EVAL_ENV_ARGS=()
    if [ "$CLEANUP_CHECKPOINTS_AFTER_EVAL" = true ]; then
        EVAL_ENV_ARGS+=(--export=ALL,CLEANUP_CHECKPOINTS_WHEN_SAFE=1)
    fi
    sbatch --parsable "${SBATCH_JOB_ARGS[@]}" "${EVAL_ENV_ARGS[@]}" --dependency=after:"$TRAIN_JOB_ID" \
        bash/active_learning/cluster_eval.sbatch "$EVAL_TARGET" "$TRAIN_JOB_ID"
)"
EVAL_JOB_ID="${EVAL_SUBMISSION%%;*}"

CLEANUP_JOB_ID=""
if [ "$CLEANUP_CHECKPOINTS_AFTER_EVAL" = true ]; then
    if [ -z "$RUN_NAME" ]; then
        echo "Cannot schedule checkpoint cleanup without an explicit --paths.run_name override." >&2
        exit 1
    fi
    CLEANUP_SUBMISSION="$(
        sbatch --parsable --dependency=afterok:"$EVAL_JOB_ID" \
            bash/active_learning/cluster_cleanup_checkpoints.sbatch "$RUN_NAME"
    )"
    CLEANUP_JOB_ID="${CLEANUP_SUBMISSION%%;*}"
fi

echo "Config:              $CONFIG_PATH"
echo "Train job submitted: $TRAIN_JOB_ID"
echo "Eval job submitted:  $EVAL_JOB_ID  (target: $EVAL_TARGET)"
if [ -n "$CLEANUP_JOB_ID" ]; then
    echo "Cleanup job submitted: $CLEANUP_JOB_ID  (after successful eval)"
fi
