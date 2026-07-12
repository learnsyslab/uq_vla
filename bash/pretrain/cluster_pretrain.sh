#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CALL_DIR="$(pwd)"
cd "$SCRIPT_DIR"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [ $# -lt 1 ] || [ $# -gt 3 ]; then
    echo "Usage: $0 <config_path> [num_seeds] [start_seed]" >&2
    exit 1
fi

CONFIG_PATH="$1"
if [[ "$CONFIG_PATH" != /* ]]; then
    CONFIG_PATH="$CALL_DIR/$CONFIG_PATH"
fi
if [ ! -f "$CONFIG_PATH" ]; then
    echo "Config file not found: $CONFIG_PATH" >&2
    exit 1
fi
CONFIG_PATH="$(cd "$(dirname "$CONFIG_PATH")" && pwd)/$(basename "$CONFIG_PATH")"

NUM_SEEDS="${2:-2}"
START_SEED="${3:-0}"

mkdir -p slurm

for SEED in $(seq "$START_SEED" $((START_SEED + NUM_SEEDS - 1))); do
    JOB_ID="$(sbatch --parsable --export=ALL,REPO_ROOT="$REPO_ROOT" cluster_pretrain.sbatch "$CONFIG_PATH" "$SEED")"
    echo "Seed $SEED submitted: job $JOB_ID"
done
