#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

if [ $# -lt 1 ]; then
    echo "Usage: $0 <config_path>" >&2
    echo "  e.g. $0 configs/iterative_fine_tuning/calibration_experiment/uniform_leak3.yaml" >&2
    exit 1
fi

CONFIG="$1"

if [ ! -f "$CONFIG" ]; then
    echo "Config file not found: $CONFIG" >&2
    exit 1
fi

# Derive pretrain name from filename: uniform_<pretrain>.yaml -> <pretrain>
BASENAME="$(basename "$CONFIG" .yaml)"
PRETRAIN="${BASENAME#uniform_}"

SEED_PAIRS=("0 1" "2 3" "4 5")
HISTORY_FRACTIONS=("0.5")
STEPS=(2000)
TASK_SELECTION_TEMPERATURES=("0")
SWEEP_TASK_SELECTION_TEMPERATURE=false

if [[ "$BASENAME" = "uniform_metaworld_leak3" || "$BASENAME" = "uniform_metaworld_leak3easier_topk" ]]; then
    SWEEP_TASK_SELECTION_TEMPERATURE=true
    TASK_SELECTION_TEMPERATURES=("0" "1" "1.5" "2")
fi

DELETE_CHECKPOINTS_AFTER_EVAL=false
if [ "$BASENAME" = "uniform_metaworld_leak3easier_topk" ]; then
    DELETE_CHECKPOINTS_AFTER_EVAL=true
fi

temperature_label() {
    local temperature="$1"
    echo "T${temperature/./p}"
}

run_is_active() {
    local run_name="$1"
    squeue -h -u "$USER" -o "%j %R" | grep -Fq "$run_name"
}

for SEEDS in "${SEED_PAIRS[@]}"; do
    read -r SEED_A SEED_B <<< "$SEEDS"
    for HF in "${HISTORY_FRACTIONS[@]}"; do
        for S in "${STEPS[@]}"; do
            for T in "${TASK_SELECTION_TEMPERATURES[@]}"; do
                HF_LABEL="${HF/./}"
                T_LABEL="$(temperature_label "$T")"
                if [ "$SWEEP_TASK_SELECTION_TEMPERATURE" = true ]; then
                    RUN_NAME="uniform_${PRETRAIN}_${T_LABEL}_history${HF_LABEL}_steps${S}_s${SEED_A}${SEED_B}"
                else
                    RUN_NAME="uniform_${PRETRAIN}_history${HF_LABEL}_steps${S}_s${SEED_A}${SEED_B}"
                fi
                NF=$(printf "%.1f" "$(echo "1 - $HF" | bc)")
                EXTRA_ARGS=()
                if [ "$SWEEP_TASK_SELECTION_TEMPERATURE" = true ]; then
                    EXTRA_ARGS+=(--selection.task_selection_temperature="$T")
                fi
                if { [ "$SWEEP_TASK_SELECTION_TEMPERATURE" = true ] && [ "$T" != "0" ]; } || [ "$DELETE_CHECKPOINTS_AFTER_EVAL" = true ]; then
                    EXTRA_ARGS+=(--delete-checkpoints-after-eval)
                fi
                if run_is_active "$RUN_NAME"; then
                    echo "Skipping active run: $RUN_NAME"
                    continue
                fi
                bash bash/active_learning/cluster_all.sh "$CONFIG" "$SEED_A" "$SEED_B" \
                    --replay.history_fraction="$HF" \
                    --replay.new_fraction="$NF" \
                    --steps="$S" \
                    --paths.run_name="$RUN_NAME" \
                    --job_name="$RUN_NAME" \
                    "${EXTRA_ARGS[@]}"
            done
        done
    done
done
