#!/bin/bash
# Submit one calibration_comparison job per seed, then an aggregation job once all finish.
#
# Usage:
#   bash submit_calibration_comparison.sh <output_dir> <run_dir> [run_dir ...] [-- extra args]
#
# Extra args (after --) are forwarded to calibration_comparison.py, e.g. --rounds 0 1 2
#
# Example:
# bash bash/calibration/submit_calibration_comparison.sh \
#     plots/calibration_comparison/libero/uniform_leak3_weighted_pools_lr_schedule_history05_steps2000 \
#     outputs/iterative_fine_tuning/uniform_leak3_weighted_pools_lr_schedule_history05_steps2000_s01 \
#     outputs/iterative_fine_tuning/uniform_leak3_weighted_pools_lr_schedule_history05_steps2000_s23 \
#     outputs/iterative_fine_tuning/uniform_leak3_weighted_pools_lr_schedule_history05_steps2000_s45 \
#       -- --rounds 0 1 2 3 4 5 6 7 8 9

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

# Submit one job per seed
SEED_JOB_IDS=()
for RUN_DIR in "${RUN_DIRS[@]}"; do
    JOB_ID="$(sbatch --parsable cluster_methods_comparison.sbatch \
        --run_dirs "$RUN_DIR" --output "$OUTPUT_DIR" "${EXTRA_ARGS[@]}")"
    echo "Submitted seed job $JOB_ID for $RUN_DIR"
    SEED_JOB_IDS+=("$JOB_ID")
done

# Submit aggregation job with dependency on all seed jobs
DEPENDENCY="afterok:$(IFS=:; echo "${SEED_JOB_IDS[*]}")"
AGG_JOB_ID="$(sbatch --parsable --dependency="$DEPENDENCY" \
    cluster_methods_comparison.sbatch \
    --run_dirs "${RUN_DIRS[@]}" --output "$OUTPUT_DIR" "${EXTRA_ARGS[@]}")"
echo "Submitted aggregation job $AGG_JOB_ID (depends on ${SEED_JOB_IDS[*]})"
