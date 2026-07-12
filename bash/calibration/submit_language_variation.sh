#!/bin/bash
# Submit one language_variation job per seed directory.
#
# Usage:
#   bash submit_language_variation.sh <output_dir> <run_dir> [run_dir ...] [-- extra args]
#
# Extra args (after --) are forwarded to language_variation.py, e.g. --round 19 --n_rollouts 10
#
# Example:
#   bash bash/calibration/submit_language_variation.sh \
#       plots/language_variation/leak3_history05_steps2000 \
#       ${STORAGE_ROOT}/outputs/iterative_fine_tuning/uniform_leak3_history05_steps2000_s01 \
#       ${STORAGE_ROOT}/outputs/iterative_fine_tuning/uniform_leak3_history05_steps2000_s23 \
#       ${STORAGE_ROOT}/outputs/iterative_fine_tuning/uniform_leak3_history05_steps2000_s45 \
#       -- --round 19 --n_rollouts 10

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

if [ $# -lt 2 ]; then
    echo "Usage: $0 <output_dir> <run_dir> [run_dir ...] [-- extra args]" >&2
    exit 1
fi

OUTPUT_DIR="$1"
shift

# Split positional args and extra args on --
RUN_DIRS=()
EXTRA_ARGS=()
in_extra=false
for arg in "$@"; do
    if [ "$arg" = "--" ]; then
        in_extra=true
    elif $in_extra; then
        EXTRA_ARGS+=("$arg")
    else
        RUN_DIRS+=("$arg")
    fi
done

if [ ${#RUN_DIRS[@]} -eq 0 ]; then
    echo "Error: no run directories provided." >&2
    exit 1
fi

mkdir -p slurm

for RUN_DIR in "${RUN_DIRS[@]}"; do
    JOB_ID="$(sbatch --parsable cluster_language_variation.sbatch \
        --run_dir "$RUN_DIR" --output "$OUTPUT_DIR" "${EXTRA_ARGS[@]}")"
    echo "Submitted job $JOB_ID for $RUN_DIR"
done
