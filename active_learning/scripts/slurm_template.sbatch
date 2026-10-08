#!/bin/bash
# Generic SLURM wrapper: runs any command of the README inside this folder.
#
#   sbatch --gpus=2 --cpus-per-task=16 --mem=128G --time=24:00:00 scripts/slurm_template.sbatch \
#       bash scripts/run_active_learning.sh configs/active_learning/smolvla/vfd.yaml s01
#
# Request resources on the sbatch command line (see the README's compute table) and add the
# partition/account options your cluster needs. Activate the Python environment in ENV_SETUP, e.g.
# ENV_SETUP="conda activate vfd" sbatch ... . Long runs are resumable: resubmit the same command.
#SBATCH --job-name=vfd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p logs
if [ -n "${ENV_SETUP:-}" ]; then eval "$ENV_SETUP"; fi
echo "[$(date -Is)] $(hostname): $*"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
srun "$@"
