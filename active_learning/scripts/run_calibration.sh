#!/usr/bin/env bash
# Calibration experiment (tab:calibration_summary / tab:calibration_full) for one policy and seed pair:
# iterative fine-tuning with random data selection, evaluation of every round, then offline scoring of
# every round's checkpoints with every uncertainty method.
#
#   bash scripts/run_calibration.sh <smolvla|xvla|fastwam> <seed_pair> [--extras]
#
# --extras (SmolVLA only) adds the appendix studies on the same runs:
#   ensemble size (fig:ensemble_size_ablation): two extra members per round, then VFD with 2/3/4 members;
#   ensemble vs. Laplace (fig:ensemble_vs_laplace): a last-layer Laplace posterior on member 0, rounds 0-4;
#   language variation (fig:language_variation): five paraphrases of every prompt on the round-15 policy.
# Every step caches its results and skips work that is already done.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .env ] && { set -a; . ./.env; set +a; }
export PYTHONPATH=src${PYTHONPATH:+:$PYTHONPATH}
export MUJOCO_GL="${MUJOCO_GL:-egl}" PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

[ $# -ge 2 ] || { sed -n '2,14p' "$0"; exit 1; }
POLICY="$1"; PAIR="$2"; EXTRAS="${3:-}"
RUN_DIR="outputs/calibration/${POLICY}/random_${PAIR}"
ROUNDS=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14)

bash scripts/run_active_learning.sh "configs/calibration/${POLICY}.yaml" "$PAIR"

case "$POLICY" in
    smolvla) METHODS=(action_l2 ace decu ensemble_terminal_variance inter_vel_diff_2way vlm_token_entropy vlm_perplexity) ;;
    *)       METHODS=(action_l2 ace decu ensemble_terminal_variance inter_vel_diff_2way) ;;
esac
echo "=== $(date -Is) score every round with: ${METHODS[*]}"
python scripts/calibration/calibration_comparison.py --run_dirs "$RUN_DIR" --rounds "${ROUNDS[@]}" \
    --methods "${METHODS[@]}" --num_action_samples 5 --device cuda:0 \
    --output "plots/calibration_comparison/${POLICY}"
if [ "$POLICY" != smolvla ]; then
    # DECU saturates on X-VLA and FastWAM; break exact ties (see the script's docstring).
    python scripts/calibration/jitter_degenerate_decu.py --run-dirs "$RUN_DIR"
fi

[ "$EXTRAS" = --extras ] || exit 0
[ "$POLICY" = smolvla ] || { echo "--extras is for SmolVLA only" >&2; exit 1; }

# Pretraining seeds of the two extra ensemble members of each seed pair.
case "$PAIR" in
    s01) EXTRA_SEEDS=(2 4) ;;
    s23) EXTRA_SEEDS=(5 1) ;;
    s45) EXTRA_SEEDS=(0 3) ;;
    *) echo "unknown seed pair $PAIR" >&2; exit 1 ;;
esac
echo "=== $(date -Is) ensemble size: train members 2 and 3 on every round's data"
python scripts/calibration/train_extra_members.py --metadata-run-dir "$RUN_DIR" --checkpoint-run-dir "$RUN_DIR" \
    --pretrain-root outputs/pretrain/smolvla_libero --pretrained-seeds "${EXTRA_SEEDS[@]}" \
    --member-indices 2 3 --devices cuda:0 cuda:1 --rounds "${ROUNDS[@]}"
python scripts/calibration/ensemble_size_ablation.py --run_dirs "$RUN_DIR" --rounds "${ROUNDS[@]}" \
    --ensemble_sizes 2 3 4 --compute_only --output plots/ensemble_size_ablation/smolvla

echo "=== $(date -Is) Laplace posterior on member 0, rounds 0-4"
python scripts/calibration/fit_laplace.py --run_dir "$RUN_DIR" --rounds 0 1 2 3 4 --scope action_out_proj
python scripts/calibration/calibration_comparison.py --run_dirs "$RUN_DIR" --rounds 0 1 2 3 4 \
    --methods inter_vel_diff_2way_laplace --num_action_samples 5 --device cuda:0 \
    --output plots/calibration_comparison/smolvla_laplace

echo "=== $(date -Is) language variation on the round-15 policy"
python scripts/calibration/language_variation.py --run_dir "$RUN_DIR" --round 14 --n_rollouts 32 \
    --output outputs/calibration/smolvla/language_variation
echo "=== $(date -Is) done"
