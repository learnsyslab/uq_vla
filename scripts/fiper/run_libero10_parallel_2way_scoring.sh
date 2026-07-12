#!/usr/bin/env bash
set -euo pipefail

TASK_IDS="${TASK_IDS:-0 1 2 3 4 5 6 7 8 9}"
GPU_IDS="${GPU_IDS:-0}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-smolvla_libero10_all_tasks_5eps_parallel}"
LOG_DIR="${LOG_DIR:-logs/fiper/${EXPERIMENT_NAME}_2way_scoring}"

mkdir -p "${LOG_DIR}"

read -r -a GPU_ID_ARRAY <<< "${GPU_IDS}"
if [[ "${#GPU_ID_ARRAY[@]}" -eq 0 ]]; then
  echo "GPU_IDS must contain at least one GPU id." >&2
  exit 2
fi

running_jobs=0
task_index=0

for task_id in ${TASK_IDS}; do
  while [[ "${running_jobs}" -ge "${MAX_PARALLEL}" ]]; do
    wait -n
    running_jobs=$((running_jobs - 1))
  done

  gpu_id="${GPU_ID_ARRAY[$((task_index % ${#GPU_ID_ARRAY[@]}))]}"
  task_id_padded="$(printf "%02d" "${task_id}")"
  log_file="${LOG_DIR}/task${task_id_padded}.log"

  echo "[launcher] task${task_id_padded} on GPU ${gpu_id}; log=${log_file}"
  (
    export CUDA_VISIBLE_DEVICES="${gpu_id}"
    scripts/fiper/run_libero10_task_2way_scoring.sh "${task_id}"
  ) >"${log_file}" 2>&1 &

  running_jobs=$((running_jobs + 1))
  task_index=$((task_index + 1))
done

while [[ "${running_jobs}" -gt 0 ]]; do
  wait -n
  running_jobs=$((running_jobs - 1))
done

echo "[launcher] all scoring tasks finished"
