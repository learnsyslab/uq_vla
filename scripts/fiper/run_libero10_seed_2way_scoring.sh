#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <s01|s23|s45>" >&2
  exit 2
fi

SEED_NAME="$1"
RUN_NAME="uniform_leak3fixed_weighted_pools_lr_schedule_history05_steps2000_${SEED_NAME}"
RUN_ROOT="outputs/iterative_fine_tuning/${RUN_NAME}/round_015/training"

POLICY_CHECKPOINT="${POLICY_CHECKPOINT:-${RUN_ROOT}/member_00/checkpoints/last/pretrained_model}"
ENSEMBLE_MEMBER_00="${ENSEMBLE_MEMBER_00:-${RUN_ROOT}/member_00/checkpoints/last/pretrained_model}"
ENSEMBLE_MEMBER_01="${ENSEMBLE_MEMBER_01:-${RUN_ROOT}/member_01/checkpoints/last/pretrained_model}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-smolvla_libero10_all_tasks_5eps_parallel_${SEED_NAME}}"
SEED_BASE="${SEED_BASE:-20118}"

for path in "${POLICY_CHECKPOINT}" "${ENSEMBLE_MEMBER_00}" "${ENSEMBLE_MEMBER_01}"; do
  if [[ ! -d "${path}" ]]; then
    echo "Checkpoint directory does not exist: ${path}" >&2
    exit 1
  fi
done

export POLICY_CHECKPOINT
export ENSEMBLE_MEMBER_00
export ENSEMBLE_MEMBER_01
export EXPERIMENT_NAME
export SEED_BASE

scripts/fiper/run_libero10_parallel_2way_scoring.sh
