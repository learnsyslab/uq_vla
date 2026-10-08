#!/usr/bin/env bash
# Failure-detection data for LIBERO-Plus (tables/failure_prediction_liberoplus, the LIBERO-Plus column of
# tables/failure_prediction_summary), for the SmolVLA ensembles trained on all 40 LIBERO tasks.
#
#   bash scripts/failure_detection/run_libero_plus.sh [s01 s23 s45]
#
# Needs the pretrained ensembles (bash scripts/pretrain.sh smolvla_libero_all40 0 1 2 3 4 5) and the
# LIBERO-Plus benchmark (bash scripts/setup_libero_plus.sh). Per seed pair and LIBERO-10 task:
#   1. in-distribution LIBERO-10 rollouts of member A: 20 successful calibration rollouts and 30 test
#      rollouts (configs/failure_detection/record_libero.yaml), scored with the two-member ensemble;
#   2. 6 perturbation families x 30 LIBERO-Plus test rollouts (record_libero_plus.py), scored the same way;
#   3. the in-distribution rollouts merged into the LIBERO-Plus run (episode numbers from 1000);
#   4. embeddings of the policy's training demonstrations (every 10th frame) for logpZO / RND-OE.
# Result: outputs/fiper_rollout_scoring/smolvla_libero_plus_<pair>/libero_10, the input of
# ../failure_detection. GPU_IDS (default "0") lists the GPUs the per-task / per-shard jobs are spread over.
# For a quick test, RECORD_ARGS / PLUS_ARGS add flags to the two recorders, e.g.
# RECORD_ARGS="--n_calib_episodes=1 --n_test_episodes=1" PLUS_ARGS="--limit 2".
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -f .env ] && { set -a; . ./.env; set +a; }
export PYTHONPATH=src${PYTHONPATH:+:$PYTHONPATH}
export MUJOCO_GL="${MUJOCO_GL:-egl}" PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
LIBERO_PLUS_ENV="third_party/libero_plus/env.sh"
[ -f "$LIBERO_PLUS_ENV" ] || { echo "run scripts/setup_libero_plus.sh first" >&2; exit 1; }
IFS=', ' read -r -a GPUS <<< "${GPU_IDS:-0}"
TASKS=(0 1 2 3 4 5 6 7 8 9)
NUM_SHARDS="${NUM_SHARDS:-${#GPUS[@]}}"

# Run "$@" once per item of the list in $ITEMS, at most one job per GPU at a time.
parallel_over_gpus() {
    local i=0 failed=0 item pid pids=()
    for item in $ITEMS; do
        ( CUDA_VISIBLE_DEVICES="${GPUS[$((i % ${#GPUS[@]}))]}" "$@" "$item" ) &
        pids+=($!); i=$((i + 1))
        if (( i % ${#GPUS[@]} == 0 )); then
            for pid in "${pids[@]}"; do wait "$pid" || failed=1; done; pids=()
        fi
    done
    for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
    return $failed
}

for PAIR in "${@:-s01 s23 s45}"; do
    case "$PAIR" in s01) A=0; B=1 ;; s23) A=2; B=3 ;; s45) A=4; B=5 ;;
        *) echo "unknown seed pair $PAIR" >&2; exit 1 ;; esac
    MEMBER_A="outputs/pretrain/smolvla_libero_all40/seed_${A}/checkpoints/last/pretrained_model"
    MEMBER_B="outputs/pretrain/smolvla_libero_all40/seed_${B}/checkpoints/last/pretrained_model"
    ENSEMBLE="[\"${MEMBER_A}\",\"${MEMBER_B}\"]"
    VAN_REC="outputs/fiper_rollout_recording/smolvla_libero_${PAIR}"
    VAN_SCO="outputs/fiper_rollout_scoring/smolvla_libero_${PAIR}/libero_10"
    PLUS_REC="outputs/fiper_rollout_recording/smolvla_libero_plus_${PAIR}"
    PLUS_SCO="outputs/fiper_rollout_scoring/smolvla_libero_plus_${PAIR}/libero_10"

    record_vanilla() {  # <task>
        python src/lerobot/scripts/fiper_data_generation/record_fiper_rollout.py \
            --config_path=configs/failure_detection/record_libero.yaml --policy.path="$MEMBER_A" \
            --env.task_ids="[$1]" --seed=$((20118 + $1)) --job_name="smolvla_libero_${PAIR}_task$1" \
            --output_dir="$VAN_REC" ${RECORD_ARGS:-}
    }
    score() {  # <input dir> <output dir> <task>
        python src/lerobot/scripts/fiper_data_generation/score_fiper_rollout.py \
            --config_path=configs/failure_detection/score_libero.yaml --policy.path="$MEMBER_A" \
            --env.task_ids="[$3]" --seed=$((20118 + $3)) --job_name="smolvla_libero_${PAIR}_score_task$3" \
            --input_dir="$1" --output_dir="$2" \
            --fiper_rollout_scorer.ensemble_model_paths="$ENSEMBLE" --allow_existing_output_dir=true
    }
    score_vanilla() { score "${VAN_REC}/libero_10" "$VAN_SCO" "$1"; }
    score_plus() { score "${PLUS_REC}/libero_10" "$PLUS_SCO" "$1"; }
    record_plus() {  # <shard>
        ( . "$LIBERO_PLUS_ENV"
          PYTHONPATH="${LIBERO_PLUS_SHIM}:${PYTHONPATH}" python \
            src/lerobot/scripts/fiper_data_generation/record_libero_plus.py \
            --policy_path "$MEMBER_A" --output_dir "$PLUS_REC" \
            --vanilla_task_map configs/failure_detection/libero10_vanilla_task_map.json \
            --seed $((20118 + A)) --num_uncertainty_sequences 16 --n_per_cell 30 \
            --shard "$1" --num_shards "$NUM_SHARDS" ${PLUS_ARGS:-} )
    }

    echo "=== $(date -Is) ${PAIR}: in-distribution LIBERO-10 rollouts -> ${VAN_REC}"
    ITEMS="${TASKS[*]}" parallel_over_gpus record_vanilla
    ITEMS="${TASKS[*]}" parallel_over_gpus score_vanilla
    echo "=== $(date -Is) ${PAIR}: LIBERO-Plus rollouts -> ${PLUS_REC}"
    ITEMS="$(seq -s ' ' 0 $((NUM_SHARDS - 1)))" parallel_over_gpus record_plus
    ITEMS="${TASKS[*]}" parallel_over_gpus score_plus
    echo "=== $(date -Is) ${PAIR}: merge -> ${PLUS_SCO}"
    python scripts/failure_detection/merge_libero_into_plus.py "$VAN_SCO" "$PLUS_SCO" --k 16
    echo "=== $(date -Is) ${PAIR}: demonstration embeddings"
    python src/lerobot/scripts/fiper_data_generation/extract_demo_embeddings.py \
        --policy_path "$MEMBER_A" --repo_id HuggingFaceVLA/libero --frame_stride 10 \
        --pad_kv_tokens 5,64,32,1 --output "${PLUS_SCO}/demo_embeddings/smolvla_all40_stride10_obs_embedding.pt"
done
echo "=== $(date -Is) done"
