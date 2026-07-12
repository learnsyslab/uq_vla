#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

SEED_PAIRS=("0 1" "2 3" "4 5")
TOTAL_ROUNDS=2
EPISODES_PER_ROUND=50
TEMPERATURES=(4.0 0.0)

CONFIG="configs/iterative_fine_tuning/active_learning_experiment/task_weighted_leak3.yaml"

for SEEDS in "${SEED_PAIRS[@]}"; do
    read -r SEED_A SEED_B <<< "$SEEDS"
    for TEMP in "${TEMPERATURES[@]}"; do
        TEMP_LABEL="${TEMP%.*}"
        RUN_NAME="task_weighted_leak3_history05_T${TEMP_LABEL}_R${TOTAL_ROUNDS}_E${EPISODES_PER_ROUND}_s${SEED_A}${SEED_B}"
        bash "$SCRIPT_DIR/cluster_all.sh" "$CONFIG" "$SEED_A" "$SEED_B" \
            --selection.task_selection_temperature="$TEMP" \
            --iteration.total_rounds="$TOTAL_ROUNDS" \
            --iteration.episodes_per_round="$EPISODES_PER_ROUND" \
            --replay.history_fraction=0.5 \
            --replay.new_fraction=0.5 \
            --paths.run_name="$RUN_NAME" \
            --job_name="$RUN_NAME"
    done
done
