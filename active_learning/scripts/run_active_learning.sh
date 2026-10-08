#!/usr/bin/env bash
# Run one iterative fine-tuning experiment (active learning or calibration) for one seed pair, then
# evaluate the pretrained ensemble and every round.
#
#   bash scripts/run_active_learning.sh <config.yaml> <seed_pair> [draccus overrides ...]
#
#   bash scripts/run_active_learning.sh configs/active_learning/smolvla/vfd.yaml s01
#   bash scripts/run_active_learning.sh configs/calibration/xvla.yaml s23
#
# The seed pairs of a setup are listed at the bottom of its config (s01, s23, s45; Push-T s01 ...
# s1819). Both stages are resumable: rerunning skips finished rounds and finished evaluations.
#
# Evaluation follows the protocol of the published numbers (30 rollouts per LIBERO-10 task and member,
# env seeds 100..; 100 Push-T rollouts per member): SmolVLA active-learning runs evaluate ensemble
# member 0, every other setup both members. Active-learning runs delete a round's checkpoints once the
# next round is trained and the round is evaluated; calibration runs keep them, since the calibration
# scripts re-score every round. Override the evaluation with EVAL_ARGS="..." (e.g. other GPUs).
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .env ] && { set -a; . ./.env; set +a; }
export PYTHONPATH=src${PYTHONPATH:+:$PYTHONPATH}
export MUJOCO_GL="${MUJOCO_GL:-egl}" PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

[ $# -ge 2 ] || { sed -n '2,18p' "$0"; exit 1; }
CONFIG="$1"; PAIR="$2"; shift 2

case "$CONFIG" in
    *configs/active_learning/smolvla/*) KIND=al; POLICY=smolvla ;;
    *configs/active_learning/xvla/*)    KIND=al; POLICY=xvla ;;
    *configs/active_learning/fastwam/*) KIND=al; POLICY=fastwam ;;
    *configs/active_learning/pusht/*)   KIND=al; POLICY=pusht ;;
    *configs/calibration/*)             KIND=cal; POLICY=$(basename "$CONFIG" .yaml) ;;
    *) echo "unrecognised config path: $CONFIG" >&2; exit 1 ;;
esac

RUN_DIR=$(python scripts/active_learning/print_run_dir.py "$CONFIG" "$PAIR" "$@")
echo "=== ${CONFIG} [${PAIR}] -> ${RUN_DIR}"

echo "=== $(date -Is) select + train"
python -m iterative_fine_tuning.main --config_path="$CONFIG" --seed_pair="$PAIR" "$@"

case "$POLICY" in
    smolvla) DEFAULT_EVAL="--member-devices cuda:0,cuda:1 --workers-per-device 2 --n-rollouts 30 --eval-batch-size 15 --sync-envs" ;;
    xvla)    DEFAULT_EVAL="--member-devices cuda:0 --workers-per-device 2 --n-rollouts 30 --eval-batch-size 4 --sync-envs" ;;
    fastwam) DEFAULT_EVAL="--member-devices cuda:0,cuda:1,cuda:2,cuda:3 --n-rollouts 30 --eval-batch-size 15" ;;
    pusht)   DEFAULT_EVAL="--device cuda:0 --n-rollouts 100 --sync-envs" ;;
esac
MEMBERS=all
[ "$KIND/$POLICY" = al/smolvla ] && MEMBERS=0
CLEANUP=""
[ "$KIND" = al ] && CLEANUP="--cleanup-checkpoints-when-safe"

echo "=== $(date -Is) evaluate"
# shellcheck disable=SC2086
python -m iterative_fine_tuning.evaluate_run --run-dir "$RUN_DIR" --rounds all --members "$MEMBERS" \
    --seed-start 100 --max-steps 520 --skip-existing $CLEANUP ${EVAL_ARGS:-$DEFAULT_EVAL}
echo "=== $(date -Is) done: ${RUN_DIR}"
