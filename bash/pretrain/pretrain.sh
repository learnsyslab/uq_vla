#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ $# -lt 1 ] || [ $# -gt 2 ]; then
    echo "Usage: $0 <config_path> [num_seeds]" >&2
    exit 1
fi

BASE_CONFIG="$1"
NUM_SEEDS="${2:-2}"
if [ ! -f "$BASE_CONFIG" ]; then
    echo "Config file not found: $BASE_CONFIG" >&2
    exit 1
fi

if [ -f ".env" ]; then
    source ".env"
fi

STORAGE_ROOT="${STORAGE_ROOT:?STORAGE_ROOT environment variable must be set}"
OUTPUT_ROOT="${STORAGE_ROOT}/outputs/pretrain/$(basename "$BASE_CONFIG" .yaml)"
LOG_DIR="${STORAGE_ROOT}/logs/pretrain/$(basename "$BASE_CONFIG" .yaml)"
mkdir -p "$LOG_DIR"

WANDB_ENABLE="${WANDB_ENABLE:-false}"
WANDB_PROJECT="${WANDB_PROJECT:-active-vla}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_MODE="${WANDB_MODE:-disabled}"

WANDB_ARGS=(--wandb.enable "$WANDB_ENABLE" --wandb.mode "$WANDB_MODE")
if [ -n "$WANDB_PROJECT" ]; then
    WANDB_ARGS+=(--wandb.project "$WANDB_PROJECT")
fi
if [ -n "$WANDB_ENTITY" ]; then
    WANDB_ARGS+=(--wandb.entity "$WANDB_ENTITY")
fi

export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"

echo "========================================================"
echo "Config:  $BASE_CONFIG"
echo "Output:  $OUTPUT_ROOT"
echo "Logs:    $LOG_DIR"
echo "========================================================"

for SEED in $(seq 0 $((NUM_SEEDS - 1))); do
    OUT_DIR="${OUTPUT_ROOT}/seed_${SEED}"
    LOG_FILE="${LOG_DIR}/train_seed${SEED}.log"

    if [ -d "$OUT_DIR/checkpoints/last/pretrained_model" ]; then
        echo "SKIP: seed_${SEED} already trained ($OUT_DIR)"
        continue
    fi

    echo "Training seed ${SEED}... (log: $LOG_FILE)"
    "$PYTHON_BIN" src/lerobot/scripts/lerobot_train.py \
        --config "$BASE_CONFIG" \
        --output_dir "$OUT_DIR" \
        --seed $SEED \
        --job_name "pretrain_$(basename "$BASE_CONFIG" .yaml)_seed${SEED}" \
        "${WANDB_ARGS[@]}" \
        --policy.push_to_hub false \
        2>&1 | tee "$LOG_FILE"
done

echo "Done. Models saved to $OUTPUT_ROOT/seed_0 and seed_1"
