#!/bin/bash
#SBATCH --job-name=calibration_comparison
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=6G
#SBATCH --gpus=rtx_4090:1
#SBATCH --output=slurm/calibration_comparison-%j.out
#SBATCH --error=slurm/calibration_comparison-%j.err

set -euo pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
    echo "Python executable not found or not executable: $PYTHON_BIN" >&2
    exit 1
fi

if [ $# -lt 1 ]; then
    echo "Usage: sbatch cluster_calibration_comparison.sbatch <args passed to calibration_comparison.py>" >&2
    echo "  e.g.: --run_dirs <dir> [dir ...] --output <output_dir> [--rounds N ...]" >&2
    exit 1
fi

export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"

source ".env"

echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Host: $(hostname)"
echo "Working directory: $SCRIPT_DIR"
echo "Python: $PYTHON_BIN"
echo "Args: $*"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
date


RUN_ROOT_SCRATCH="${STORAGE_ROOT}/outputs/iterative_fine_tuning"
RUN_ROOT_HOME="${SCRIPT_DIR}/outputs/iterative_fine_tuning"
ARGS=()
for arg in "$@"; do
    if [[ "$arg" == "$RUN_ROOT_SCRATCH"* ]]; then
        ARGS+=("${RUN_ROOT_HOME}${arg#"$RUN_ROOT_SCRATCH"}")
    else
        ARGS+=("$arg")
    fi
done

srun --cpu-bind=none "$PYTHON_BIN" scripts/calibration/calibration_comparison.py "${ARGS[@]}"

# Generate summary bar plots when all seeds are present (aggregation step)
RUN_DIRS_SUMMARY=()
OUTPUT_SUMMARY=""
FRAME_INDEX_SUMMARY="0"
ENV_SUMMARY="libero"
METHODS_SUMMARY=()
i=0
while [ $i -lt ${#ARGS[@]} ]; do
    if [ "${ARGS[$i]}" = "--run_dirs" ]; then
        i=$((i+1))
        while [ $i -lt ${#ARGS[@]} ] && [[ "${ARGS[$i]}" != --* ]]; do
            RUN_DIRS_SUMMARY+=("${ARGS[$i]}")
            i=$((i+1))
        done
    elif [ "${ARGS[$i]}" = "--output" ]; then
        i=$((i+1))
        OUTPUT_SUMMARY="${ARGS[$i]}"
        i=$((i+1))
    elif [ "${ARGS[$i]}" = "--frame_index" ]; then
        i=$((i+1))
        FRAME_INDEX_SUMMARY="${ARGS[$i]}"
        i=$((i+1))
    elif [ "${ARGS[$i]}" = "--env" ]; then
        i=$((i+1))
        ENV_SUMMARY="${ARGS[$i]}"
        i=$((i+1))
    elif [ "${ARGS[$i]}" = "--methods" ]; then
        i=$((i+1))
        while [ $i -lt ${#ARGS[@]} ] && [[ "${ARGS[$i]}" != --* ]]; do
            METHODS_SUMMARY+=("${ARGS[$i]}")
            i=$((i+1))
        done
    else
        i=$((i+1))
    fi
done

if [ ${#RUN_DIRS_SUMMARY[@]} -gt 1 ] && [ -n "$OUTPUT_SUMMARY" ]; then
    echo "Generating calibration summary plots..."
    SUMMARY_CACHE_DIR="calibration_comparison"
    if [ "$FRAME_INDEX_SUMMARY" != "0" ]; then
        SUMMARY_CACHE_DIR="calibration_comparison_frame${FRAME_INDEX_SUMMARY}"
    fi
    SUMMARY_METHOD_ARGS=()
    if [ ${#METHODS_SUMMARY[@]} -gt 0 ]; then
        SUMMARY_METHOD_ARGS=(--methods "${METHODS_SUMMARY[@]}")
    fi
    "$PYTHON_BIN" scripts/calibration/calibration_summary.py \
        --run_dirs "${RUN_DIRS_SUMMARY[@]}" \
        --cache_dir_name "$SUMMARY_CACHE_DIR" \
        --env "$ENV_SUMMARY" \
        "${SUMMARY_METHOD_ARGS[@]}" \
        --output "${OUTPUT_SUMMARY}/calibration_summary.png"
    "$PYTHON_BIN" scripts/calibration/calibration_summary.py \
        --run_dirs "${RUN_DIRS_SUMMARY[@]}" \
        --cache_dir_name "$SUMMARY_CACHE_DIR" \
        --env "$ENV_SUMMARY" \
        "${SUMMARY_METHOD_ARGS[@]}" \
        --max_round 15 \
        --output "${OUTPUT_SUMMARY}/calibration_summary_15rounds.png"
fi
