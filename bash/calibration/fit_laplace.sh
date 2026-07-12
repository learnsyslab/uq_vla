#!/bin/bash

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

source .env

EXTRA_ARGS=("$@")
mkdir -p slurm

fit_run() {
    local run_dir="$1"
    local job_id

    job_id="$(sbatch --parsable cluster_fit_laplace.sbatch --run_dir "$run_dir" "${EXTRA_ARGS[@]}")"
    echo "Submitted fit_laplace job ${job_id} for ${run_dir}"
    echo
}

fit_run "${STORAGE_ROOT}/outputs/iterative_fine_tuning/uniform_leak3_history05_steps2000_s01"
fit_run "${STORAGE_ROOT}/outputs/iterative_fine_tuning/uniform_leak3_history05_steps2000_s23"
fit_run "${STORAGE_ROOT}/outputs/iterative_fine_tuning/uniform_leak3_history05_steps2000_s45"
