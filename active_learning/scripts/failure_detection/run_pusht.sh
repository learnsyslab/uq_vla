#!/usr/bin/env bash
# Failure-detection data for Push-T (tables/failure_prediction_pusht, the Push-T column of
# tables/failure_prediction_summary): rollouts of the 50-episode pretrained FM-Policy ensembles,
# scored with their two-member ensemble, plus the demonstration embeddings the logpZO / RND-OE
# baselines need.
#
#   bash scripts/failure_detection/run_pusht.sh [s01 s23 s45]
#
# Needs the pretrained ensembles (bash scripts/pretrain.sh fm_pusht 0 1 2 3 4 5). Writes
# outputs/fiper_rollout_scoring/pusht_pre50_<pair>/{rollouts/{calibration,test},demo_embeddings}, the
# input of ../failure_detection (README there). Member A of a pair generates the rollouts; 70 successful
# calibration rollouts and 200 test rollouts per pair. RECORD_ARGS adds recorder flags, e.g. for a quick
# test: RECORD_ARGS="--n_calib_episodes=1 --n_test_episodes=2".
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -f .env ] && { set -a; . ./.env; set +a; }
export PYTHONPATH=src${PYTHONPATH:+:$PYTHONPATH} SDL_VIDEODRIVER=dummy

for PAIR in "${@:-s01 s23 s45}"; do
    case "$PAIR" in s01) A=0; B=1 ;; s23) A=2; B=3 ;; s45) A=4; B=5 ;;
        *) echo "unknown seed pair $PAIR" >&2; exit 1 ;; esac
    MEMBER_A="outputs/pretrain/fm_pusht/seed_${A}/checkpoints/last/pretrained_model"
    MEMBER_B="outputs/pretrain/fm_pusht/seed_${B}/checkpoints/last/pretrained_model"
    NAME="pusht_pre50_${PAIR}"
    RECORD="outputs/fiper_rollout_recording/${NAME}"
    SCORE="outputs/fiper_rollout_scoring/${NAME}"
    SEED=$((40118 + A))

    echo "=== $(date -Is) ${PAIR}: record -> ${RECORD}"
    python src/lerobot/scripts/fiper_data_generation/record_fiper_rollout.py \
        --config_path=configs/failure_detection/record_pusht.yaml --policy.path="$MEMBER_A" \
        --seed="$SEED" --job_name="$NAME" --output_dir="$RECORD" ${RECORD_ARGS:-}

    echo "=== $(date -Is) ${PAIR}: score -> ${SCORE}"
    python src/lerobot/scripts/fiper_data_generation/score_fiper_rollout.py \
        --config_path=configs/failure_detection/score_pusht.yaml --policy.path="$MEMBER_A" \
        --seed="$SEED" --job_name="${NAME}_scoring" --input_dir="$RECORD" --output_dir="$SCORE" \
        --fiper_rollout_scorer.ensemble_model_paths="[\"${MEMBER_A}\",\"${MEMBER_B}\"]" \
        --allow_existing_output_dir=true

    echo "=== $(date -Is) ${PAIR}: demonstration embeddings (the pair's 50 pretraining episodes)"
    EPISODES=$(python -c "import sys, yaml; print(','.join(map(str, yaml.safe_load(open('configs/pretrain/fm_pusht.yaml'))['seed_pairs'][sys.argv[1]]['episodes'])))" "$PAIR")
    python src/lerobot/scripts/fiper_data_generation/extract_demo_embeddings.py \
        --policy_path "$MEMBER_A" --repo_id lerobot/pusht --episodes "$EPISODES" \
        --output "${SCORE}/demo_embeddings/member_00_global_cond.pt"
done
echo "=== $(date -Is) done"
