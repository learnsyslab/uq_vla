#!/bin/bash
# Submit a multi-seed SAVE iterative fine-tuning sweep.
#
# For each config, launches one (train -> eval) job chain per seed declared in the
# config's multi_seed block, then an aggregation job that summarizes across seeds.
# Training runs main's iterative loop per child seed; evaluation runs evaluate_run
# per child; aggregation runs summarize_multi_seed_run.
#
# Usage:
#   bash bash/active_learning/submit_multi_seed.sh <config.yaml> [config.yaml ...]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

if [ $# -lt 1 ]; then
    echo "Usage: $0 <config.yaml> [config.yaml ...]" >&2
    exit 1
fi

mkdir -p slurm
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"

TRAIN_SBATCH="bash/active_learning/cluster_train_multi_seed_child.sbatch"
EVAL_SBATCH="bash/active_learning/cluster_eval_multi_seed_child.sbatch"
AGG_SBATCH="bash/active_learning/cluster_eval_multi_seed_aggregate.sbatch"

read_seeds() {
    # Print the configured seeds (one per line) from a config's multi_seed block.
    "$PYTHON_BIN" - "$1" <<'PY'
import sys, yaml
data = yaml.safe_load(open(sys.argv[1])) or {}
ms = data.get("multi_seed") or {}
runs = ms.get("runs") or []
if runs:
    for run in runs:
        print(int(run["seed"]))
else:
    for seed in (ms.get("seeds") or []):
        print(int(seed))
PY
}

for config_rel in "$@"; do
    if [ ! -f "$config_rel" ]; then
        echo "Config not found: $config_rel" >&2
        exit 1
    fi
    config_abs="$(cd "$(dirname "$config_rel")" && pwd)/$(basename "$config_rel")"
    echo "Config: $config_abs"

    mapfile -t SEEDS < <(read_seeds "$config_abs")
    if [ ${#SEEDS[@]} -eq 0 ]; then
        echo "  No multi_seed seeds configured; skipping." >&2
        continue
    fi

    eval_job_ids=()
    for seed in "${SEEDS[@]}"; do
        train_submission="$(sbatch --parsable "$TRAIN_SBATCH" "$config_abs" "$seed")"
        train_job_id="${train_submission%%;*}"
        eval_submission="$(sbatch --parsable --dependency=afterok:"$train_job_id" "$EVAL_SBATCH" "$config_abs" "$seed")"
        eval_job_id="${eval_submission%%;*}"
        eval_job_ids+=("$eval_job_id")
        echo "  Seed $seed: train $train_job_id, eval $eval_job_id"
    done

    aggregate_dependency="$(IFS=:; echo "${eval_job_ids[*]}")"
    aggregate_submission="$(sbatch --parsable --dependency=afterok:"$aggregate_dependency" "$AGG_SBATCH" "$config_abs")"
    echo "  Aggregate: ${aggregate_submission%%;*}"
    echo
done
