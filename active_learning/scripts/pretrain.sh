#!/usr/bin/env bash
# Pretrain the ensemble members of one recipe, one seed after another.
#
#   bash scripts/pretrain.sh <recipe> <seed> [<seed> ...] [-- <lerobot_train overrides>]
#
# <recipe> is a config in configs/pretrain/ without ".yaml". The paper's ensembles:
#   fm_pusht              seeds 0-19 (ten Push-T seed pairs s01 ... s1819)
#   smolvla_libero        seeds 0-5  (seed pairs s01, s23, s45)
#   xvla_libero           seeds 0-5
#   fastwam_libero        seeds 0-5  (needs the UMT5 text contexts, see README)
#   smolvla_libero_all40  seeds 0-5  (LIBERO-Plus failure detection)
# Each member is written to outputs/pretrain/<recipe>/seed_<k>. A finished member is skipped and an
# interrupted one is resumed from its last checkpoint. FastWAM trains data-parallel on NUM_PROCESSES
# GPUs (default 4) through accelerate; every other recipe uses one GPU.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .env ] && { set -a; . ./.env; set +a; }
export PYTHONPATH=src${PYTHONPATH:+:$PYTHONPATH}

[ $# -ge 2 ] || { sed -n '2,15p' "$0"; exit 1; }
RECIPE="$1"; shift
SEEDS=(); while [ $# -gt 0 ] && [ "$1" != -- ]; do SEEDS+=("$1"); shift; done
[ "${1:-}" = -- ] && shift
OVERRIDES=("$@")
CONFIG="configs/pretrain/${RECIPE}.yaml"
[ -f "$CONFIG" ] || { echo "no such recipe: $CONFIG" >&2; exit 1; }

if [ "$RECIPE" = fastwam_libero ]; then
    LAUNCH=(python -m accelerate.commands.launch --num_machines 1 --num_processes "${NUM_PROCESSES:-4}"
            --mixed_precision no --dynamo_backend no -m lerobot.scripts.lerobot_train)
else
    LAUNCH=(python -m lerobot.scripts.lerobot_train)
fi

for SEED in "${SEEDS[@]}"; do
    OUT="outputs/pretrain/${RECIPE}/seed_${SEED}"
    LAST="${OUT}/checkpoints/last/pretrained_model"
    STEPS=$(sed -n 's/^steps: *//p' "$CONFIG")
    for o in "${OVERRIDES[@]}"; do case "$o" in --steps=*) STEPS="${o#--steps=}" ;; esac; done
    if [ "$(readlink "${OUT}/checkpoints/last" 2>/dev/null)" = "$(printf %06d "$STEPS")" ]; then
        echo "[${RECIPE} seed ${SEED}] already complete"; continue
    fi
    echo "[${RECIPE} seed ${SEED}] -> ${OUT}"
    if [ -f "${LAST}/train_config.json" ]; then
        "${LAUNCH[@]}" --config_path="${LAST}/train_config.json" --resume=true "${OVERRIDES[@]}"
    else
        "${LAUNCH[@]}" --config_path="$CONFIG" --seed="$SEED" --output_dir="$OUT" "${OVERRIDES[@]}"
    fi
done
