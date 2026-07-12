#!/usr/bin/env bash
# Reproduce the FIPER LIBERO-10 plot from checkpoints only.
#
# Final target:
#   ${FIPER_ROOT}/data/results_by_seed/plots/w1_ct_quantile_accuracy_detection_time_seed_mean.pdf
#
# This script records LIBERO-10 rollouts, scores them with the SmolVLA ensemble,
# computes FIPER metrics, merges the vfd results, and generates
# the paper plot.
#
# Override ACTIVE_ROOT / FIPER_ROOT / PYTHON_BIN to point at your local checkouts
# and Python interpreter.

set -euo pipefail

ACTIVE_ROOT="${ACTIVE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
FIPER_ROOT="${FIPER_ROOT:-${ACTIVE_ROOT}/../fiper}"
PYTHON_BIN="${PYTHON_BIN:-python}"

# Checkpoint roots. Each must contain:
#   round_015/training/member_00/checkpoints/last/pretrained_model
#   round_015/training/member_01/checkpoints/last/pretrained_model
RUN_S01="${RUN_S01:-${ACTIVE_ROOT}/outputs/iterative_fine_tuning/uniform_leak3fixed_weighted_pools_lr_schedule_history05_steps2000_s01}"
RUN_S23="${RUN_S23:-${ACTIVE_ROOT}/outputs/iterative_fine_tuning/uniform_leak3fixed_weighted_pools_lr_schedule_history05_steps2000_s23}"
RUN_S45="${RUN_S45:-${ACTIVE_ROOT}/outputs/iterative_fine_tuning/uniform_leak3fixed_weighted_pools_lr_schedule_history05_steps2000_s45}"

SEEDS="${SEEDS:-s01 s23 s45}"
TASK_IDS="${TASK_IDS:-0 1 2 3 4 5 6 7 8 9}"
GPU_IDS="${GPU_IDS:-0}"
MAX_PARALLEL="${MAX_PARALLEL:-5}"
SEED_BASE="${SEED_BASE:-20118}"

export PYTHON_BIN GPU_IDS MAX_PARALLEL TASK_IDS SEED_BASE

seed_run_root() {
  case "$1" in
    s01) echo "${RUN_S01}" ;;
    s23) echo "${RUN_S23}" ;;
    s45) echo "${RUN_S45}" ;;
    *) echo "Unknown seed '$1'" >&2; exit 2 ;;
  esac
}

echo "== 1. Record LIBERO-10 rollouts and score vfd_oneway/vfd =="
cd "${ACTIVE_ROOT}"
for seed in ${SEEDS}; do
  run_root="$(seed_run_root "${seed}")/round_015/training"
  export POLICY_CHECKPOINT="${run_root}/member_00/checkpoints/last/pretrained_model"
  export ENSEMBLE_MEMBER_00="${run_root}/member_00/checkpoints/last/pretrained_model"
  export ENSEMBLE_MEMBER_01="${run_root}/member_01/checkpoints/last/pretrained_model"
  export EXPERIMENT_NAME="smolvla_libero10_all_tasks_5eps_parallel_${seed}"

  for path in "${POLICY_CHECKPOINT}" "${ENSEMBLE_MEMBER_00}" "${ENSEMBLE_MEMBER_01}"; do
    if [[ ! -d "${path}" ]]; then
      echo "Missing checkpoint directory: ${path}" >&2
      exit 1
    fi
  done

  echo "-- ${seed}: recording + one-way scoring"
  scripts/fiper/run_libero10_seed_5eps_data.sh "${seed}"

  echo "-- ${seed}: rescoring with vfd"
  scripts/fiper/run_libero10_seed_2way_scoring.sh "${seed}"
done

echo "== 2. Compute FIPER metrics for existing methods =="
cd "${FIPER_ROOT}"
for seed in ${SEEDS}; do
  echo "-- ${seed}: all baseline/FIPER methods"
  PATH="$(dirname "${PYTHON_BIN}"):${PATH}" scripts/run_smolvla_libero10_seed_results.sh "${seed}"
done

echo "== 3. Compute FIPER metrics for vfd =="
for seed in ${SEEDS}; do
  echo "-- ${seed}: Bayesian vfd"
  active_scored="${ACTIVE_ROOT}/outputs/fiper_rollout_scoring/smolvla_libero10_all_tasks_5eps_parallel_${seed}/libero_10"
  target="${FIPER_ROOT}/data/libero_10"
  results_dir="${FIPER_ROOT}/data/results"
  seed_results_dir="${FIPER_ROOT}/data/results_by_seed_bayesian_2way/${seed}"

  rm -f "${target}"
  ln -s "${active_scored}" "${target}"
  if [[ -e "${results_dir}" ]]; then
    mv "${results_dir}" "${results_dir}.before_${seed}_bayesian_2way_scalar_$(date +%Y%m%d_%H%M%S)"
  fi

  CUDA_VISIBLE_DEVICES="" PATH="$(dirname "${PYTHON_BIN}"):${PATH}" python scripts/pipeline.py \
    --config-name default_smolvla_libero10_all_tasks_5eps_bayesian_2way \
    'suite_to_task_ids.libero_10=[0,1,2,3,4,5,6,7,8,9]' \
    'eval.scoring_metrics=[vfd]' \
    'eval.required_tensors=[ensemble_vfd_scores]' \
    results.overwrite_data=true

  rm -rf "${seed_results_dir}"
  mkdir -p "$(dirname "${seed_results_dir}")"
  cp -a "${results_dir}" "${seed_results_dir}"
done

echo "== 4. Merge existing method results with vfd results =="
"${PYTHON_BIN}" - <<'PY'
from pathlib import Path
import pandas as pd

old_base = Path("data/results_by_seed")
new_base = Path("data/results_by_seed_bayesian_2way")
merged_base = Path("data/results_by_seed_with_2way_s01_s23_s45")
seeds = ["s01", "s23", "s45"]
old_methods = ["entropy", "tc", "rnd_oe", "bayesian_ensemble_vfd_oneway"]
new_methods = ["bayesian_ensemble_vfd"]

for seed in seeds:
    out_dir = merged_base / seed
    out_dir.mkdir(parents=True, exist_ok=True)
    old = pd.read_csv(old_base / seed / "complete_results.csv")
    old = old[old["Method"].isin(old_methods)].copy()
    new = pd.read_csv(new_base / seed / "complete_results.csv")
    new = new[new["Method"].isin(new_methods)].copy()
    merged = pd.concat([old, new], ignore_index=True)
    merged.to_csv(out_dir / "complete_results.csv", index=False)
    print(seed, len(old), len(new), len(merged))
PY

echo "== 5. Generate final paper plot =="
"${PYTHON_BIN}" scripts/plot_seed_mean_accuracy_detection.py \
  --results-by-seed data/results_by_seed_with_2way_s01_s23_s45 \
  --seeds s01 s23 s45 \
  --window 1 \
  --threshold ct_quantile \
  --methods entropy tc rnd_oe bayesian_ensemble_vfd_oneway bayesian_ensemble_vfd \
  --output data/results_by_seed/plots/w1_ct_quantile_accuracy_detection_time_seed_mean.png

echo "Done:"
echo "${FIPER_ROOT}/data/results_by_seed/plots/w1_ct_quantile_accuracy_detection_time_seed_mean.pdf"
