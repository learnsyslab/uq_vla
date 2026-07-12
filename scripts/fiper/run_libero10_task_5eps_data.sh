#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <task_id>" >&2
  exit 2
fi

TASK_ID="$1"
TASK_ID_PADDED="$(printf "%02d" "${TASK_ID}")"

POLICY_CHECKPOINT="${POLICY_CHECKPOINT:-outputs/iterative_fine_tuning/uniform_leak3fixed_weighted_pools_lr_schedule_history05_steps2000_s01/round_015/training/member_00/checkpoints/last/pretrained_model}"
ENSEMBLE_MEMBER_00="${ENSEMBLE_MEMBER_00:-outputs/iterative_fine_tuning/uniform_leak3fixed_weighted_pools_lr_schedule_history05_steps2000_s01/round_015/training/member_00/checkpoints/last/pretrained_model}"
ENSEMBLE_MEMBER_01="${ENSEMBLE_MEMBER_01:-outputs/iterative_fine_tuning/uniform_leak3fixed_weighted_pools_lr_schedule_history05_steps2000_s01/round_015/training/member_01/checkpoints/last/pretrained_model}"
ENSEMBLE_MODEL_PATHS="${ENSEMBLE_MODEL_PATHS:-[\"${ENSEMBLE_MEMBER_00}\",\"${ENSEMBLE_MEMBER_01}\"]}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-smolvla_libero10_all_tasks_5eps_parallel}"
RECORD_CONFIG="${RECORD_CONFIG:-configs/smolvla/record_fiper_rollout/libero10_all_tasks_5eps.yaml}"
SCORE_CONFIG="${SCORE_CONFIG:-configs/smolvla/score_fiper_rollout/libero10_all_tasks_5eps.yaml}"

RECORD_ROOT="${RECORD_ROOT:-outputs/fiper_rollout_recording/${EXPERIMENT_NAME}}"
SCORE_ROOT="${SCORE_ROOT:-outputs/fiper_rollout_scoring/${EXPERIMENT_NAME}/libero_10}"
SEED_BASE="${SEED_BASE:-20118}"
TASK_SEED="$((SEED_BASE + TASK_ID))"

echo "[task${TASK_ID_PADDED}] recording -> ${RECORD_ROOT}"
PYTHONPATH=src python src/lerobot/scripts/fiper_data_generation/record_fiper_rollout.py \
  --config_path="${RECORD_CONFIG}" \
  --policy.path="${POLICY_CHECKPOINT}" \
  --env.task_ids="[${TASK_ID}]" \
  --seed="${TASK_SEED}" \
  --job_name="${EXPERIMENT_NAME}_task${TASK_ID_PADDED}" \
  --output_dir="${RECORD_ROOT}"

echo "[task${TASK_ID_PADDED}] scoring -> ${SCORE_ROOT}"
PYTHONPATH=src python src/lerobot/scripts/fiper_data_generation/score_fiper_rollout.py \
  --config_path="${SCORE_CONFIG}" \
  --policy.path="${POLICY_CHECKPOINT}" \
  --env.task_ids="[${TASK_ID}]" \
  --seed="${TASK_SEED}" \
  --job_name="${EXPERIMENT_NAME}_task${TASK_ID_PADDED}_scoring" \
  --input_dir="${RECORD_ROOT}/libero_10" \
  --output_dir="${SCORE_ROOT}" \
  --fiper_rollout_scorer.ensemble_model_paths="${ENSEMBLE_MODEL_PATHS}" \
  --allow_existing_output_dir=true

echo "[task${TASK_ID_PADDED}] done"
