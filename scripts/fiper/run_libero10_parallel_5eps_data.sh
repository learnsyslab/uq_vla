#!/usr/bin/env bash
set -euo pipefail

TASK_IDS_STR="${TASK_IDS:-0 1 2 3 4 5 6 7 8 9}"
GPU_IDS_STR="${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-0}}"
MAX_PARALLEL="${MAX_PARALLEL:-}"
LOG_DIR="${LOG_DIR:-logs/fiper/${EXPERIMENT_NAME:-smolvla_libero10_all_tasks_5eps_parallel}}"

IFS=', ' read -r -a GPU_IDS <<< "${GPU_IDS_STR}"
if [[ "${#GPU_IDS[@]}" -eq 0 || -z "${GPU_IDS[0]}" ]]; then
  GPU_IDS=(0)
fi

if [[ -z "${MAX_PARALLEL}" ]]; then
  MAX_PARALLEL="${#GPU_IDS[@]}"
fi

mkdir -p "${LOG_DIR}"

running=0
slot=0
failed=0

for task_id in ${TASK_IDS_STR}; do
  gpu="${GPU_IDS[$((slot % ${#GPU_IDS[@]}))]}"
  task_id_padded="$(printf "%02d" "${task_id}")"
  log_file="${LOG_DIR}/task${task_id_padded}.log"

  echo "[launcher] task${task_id_padded} on GPU ${gpu}; log=${log_file}"
  (
    CUDA_VISIBLE_DEVICES="${gpu}" \
      scripts/fiper/run_libero10_task_5eps_data.sh "${task_id}"
  ) >"${log_file}" 2>&1 &

  running=$((running + 1))
  slot=$((slot + 1))

  if [[ "${running}" -ge "${MAX_PARALLEL}" ]]; then
    if ! wait -n; then
      failed=1
    fi
    running=$((running - 1))
  fi
done

while [[ "${running}" -gt 0 ]]; do
  if ! wait -n; then
    failed=1
  fi
  running=$((running - 1))
done

if [[ "${failed}" -ne 0 ]]; then
  echo "[launcher] one or more task shards failed; inspect ${LOG_DIR}" >&2
  exit 1
fi

echo "[launcher] all task shards completed"
