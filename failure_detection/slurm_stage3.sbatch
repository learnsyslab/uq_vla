#!/bin/bash
# Generic SLURM template for one failure-detection run (one ensemble, one environment). Adapt the resource
# lines and the environment activation to your cluster.
#
#   sbatch slurm_stage3.sbatch pusht       <scored-rollout dir> <output dir>
#   sbatch slurm_stage3.sbatch libero_plus <scored-rollout dir> <output dir>
#SBATCH --job-name=failure_detection
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=16:00:00
set -euo pipefail
# source <your conda/venv activation>
cd "${SLURM_SUBMIT_DIR:-.}"
bash run_stage3.sh "$@"
