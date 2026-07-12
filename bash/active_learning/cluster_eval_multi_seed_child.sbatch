#!/bin/bash
#SBATCH --job-name=active_learning_seed_eval
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=24G
#SBATCH --gpus=rtx_3090:2
#SBATCH --output=slurm/active_learning-seed-eval-%j.out
#SBATCH --error=slurm/active_learning-seed-eval-%j.err

set -euo pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
    echo "Python executable not found or not executable: $PYTHON_BIN" >&2
    exit 1
fi

if [ $# -ne 2 ]; then
    echo "Usage: sbatch cluster_eval_multi_seed_child.sbatch <config_path> <seed>" >&2
    exit 1
fi
CONFIG_PATH="$1"
SEED="$2"

export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"

source ".env"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Host: $(hostname)"
echo "Working directory: $SCRIPT_DIR"
echo "Python: $PYTHON_BIN"
echo "Config: $CONFIG_PATH"
echo "Seed: $SEED"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "HF_HOME: ${HF_HOME:-unset}"
date


no_proxy=",${no_proxy:-},"
no_proxy="${no_proxy//,api.wandb.ai,/,}"
export no_proxy="${no_proxy#,}"
export no_proxy="${no_proxy%,}"

NO_PROXY=",${NO_PROXY:-},"
NO_PROXY="${NO_PROXY//,api.wandb.ai,/,}"
export NO_PROXY="${NO_PROXY#,}"
export NO_PROXY="${NO_PROXY%,}"

CHILD_RUN_DIR="$("$PYTHON_BIN" - "$CONFIG_PATH" "$SEED" <<'PY'
import sys
from pathlib import Path
from iterative_fine_tuning.config import IterativeFineTuningConfig
from iterative_fine_tuning.env import resolve_local_path
from iterative_fine_tuning.main import _build_child_cfg

config_path = resolve_local_path(sys.argv[1], base_dir=Path.cwd(), must_exist=True)
seed = int(sys.argv[2])
cfg = IterativeFineTuningConfig.from_pretrained(str(config_path), cli_args=[])
cfg.validate()
for run_spec in cfg.multi_seed.iter_run_specs():
    if int(run_spec.seed) == seed:
        child_cfg = _build_child_cfg(cfg, parent_dir=cfg.run_dir(), run_spec=run_spec)
        print(child_cfg.run_dir())
        break
else:
    raise SystemExit(f"Seed {seed} is not configured in {config_path}")
PY
)"

EVAL_ARGS=(
    "--run-dir=$CHILD_RUN_DIR"
    --member-devices cuda:0,cuda:1
    --workers-per-device 2
    --n-rollouts 30
    --eval-batch-size 15
    --sync-envs
    --skip-existing
)

srun "$PYTHON_BIN" -m iterative_fine_tuning.evaluate_run "${EVAL_ARGS[@]}"
