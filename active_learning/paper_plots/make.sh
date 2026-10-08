#!/usr/bin/env bash
# Every calibration / active-fine-tuning figure and table of the paper, from the evaluation results and
# uncertainty caches under outputs/ (no GPU needed). Each artefact is copied to paper_plots/ under the
# file name the paper uses.
#
#   bash paper_plots/make.sh                      # all targets
#   bash paper_plots/make.sh <target> [args...]   # one target
#
# Targets:
#   calibration_table          tab:calibration_summary, tab:calibration_full      -> calibration_table.tex
#   success_rate_comparisons   tab:active_learning_comparison(_summary), tab:active_learning_comparison_full
#                              (hyperparameter table), fig:al_success_rate_comparison
#                                                      -> active_learning_comparison.tex, success_rate_tables.txt,
#                                                         al_success_rate_comparison.pdf, ...
#   statistical_significance   tab:significance                                    -> statistical_significance.tex
#   ensemble_size_ablation     fig:ensemble_size_ablation                          -> ensemble_size_ablation.pdf
#   ensemble_vs_laplace        fig:ensemble_vs_laplace                             -> ensemble_vs_laplace.pdf
#   language_variation         fig:language_variation                              -> language_variation.pdf
#   success_share_entropy      fig:success_and_uncertainty_share                   -> success_share_entropy.pdf,
#                                                                                     exploration_vs_success_small.pdf
#   pareto_and_selection       fig:exploration-success-pareto, fig:selection_uncertainty
#                                                      -> exploration_exploitation_entropy.pdf,
#                                                         selection_uncertainty_heatmap.pdf
# The figures use LaTeX for their text (text.usetex, Times): they need tueplots and a LaTeX installation.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH=src${PYTHONPATH:+:$PYTHONPATH}
PY="${PY:-python}"
OUT="$ROOT/paper_plots"
CAL_RUNS=(outputs/calibration/smolvla/random_s01 outputs/calibration/smolvla/random_s23 outputs/calibration/smolvla/random_s45)

calibration_table() (
set -euo pipefail
"$PY" scripts/calibration/calibration_table.py "$@" | tee "$OUT/calibration_table.tex"
)

success_rate_comparisons() (
set -uo pipefail
LOG="$OUT/success_rate_tables.txt"
: > "$LOG"
echo "=== learning curves (fig:al_success_rate_comparison) ===" | tee -a "$LOG"
"$PY" scripts/plots/success_rate_paper.py --panels "$@" 2>&1 | tee -a "$LOG" || exit 1
echo "=== compact LIBERO figure ===" | tee -a "$LOG"
"$PY" scripts/plots/success_rate_paper.py --compact --benchmark libero 2>&1 | tee -a "$LOG" || exit 1
echo "=== sample-efficiency tables (tab:active_learning_comparison) ===" | tee -a "$LOG"
"$PY" scripts/plots/success_rate_paper.py --efficiency-table "$@" > "$OUT/active_learning_comparison.tex" || exit 1
tee -a "$LOG" < "$OUT/active_learning_comparison.tex"
echo "=== hyperparameter table, SmolVLA (tab:active_learning_comparison_full) ===" | tee -a "$LOG"
"$PY" scripts/plots/success_rate_paper.py --ablation-table 2>&1 | tee -a "$LOG" || exit 1
echo "=== cross-benchmark table (tab:active_learning_comparison_summary) ===" | tee -a "$LOG"
"$PY" scripts/plots/success_rate_paper.py --tables "$@" 2>&1 | tee -a "$LOG" || exit 1
for f in al_success_rate_comparison.pdf libero/libero_success_rates_small.pdf; do
    [ -f "plots/success_rate_comparison/$f" ] && cp "plots/success_rate_comparison/$f" "$OUT/$(basename "$f")"
done
echo "-> $LOG"
)

statistical_significance() (
set -euo pipefail
"$PY" scripts/plots/significance_table.py "$@" 2>&1 | grep -v Warning | tee "$OUT/statistical_significance.tex"
)

ensemble_size_ablation() (
set -euo pipefail
DIR=plots/ensemble_size_ablation/smolvla
"$PY" scripts/calibration/ensemble_size_ablation.py --run_dirs "${CAL_RUNS[@]}" --output "$DIR" \
    --rounds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 --ensemble_sizes 2 3 4 --aggregate_only
cp "$DIR/aggregated/spearman_pearson_across_rounds.pdf" "$OUT/ensemble_size_ablation.pdf"
)

ensemble_vs_laplace() (
set -euo pipefail
"$PY" scripts/calibration/ensemble_vs_laplace.py --run_dirs "${CAL_RUNS[@]}" --output plots/ensemble_vs_laplace
cp plots/ensemble_vs_laplace/ensemble_vs_laplace.pdf "$OUT/ensemble_vs_laplace.pdf"
)

language_variation() (
set -euo pipefail
"$PY" scripts/calibration/language_variation_aggregate.py --run_dirs "${CAL_RUNS[@]}" --round 14 \
    --input_root outputs/calibration/smolvla/language_variation --output_root plots/language_variation
cp plots/language_variation/round_014_success_rates_summary.pdf "$OUT/language_variation.pdf"
)

success_share_entropy() (
set -euo pipefail
"$PY" scripts/plots/success_share_entropy.py --output plots/success_rate_comparison/success_share_entropy.pdf \
    --standalone-right plots/success_rate_comparison/exploration_vs_success_small.pdf
cp plots/success_rate_comparison/success_share_entropy.pdf plots/success_rate_comparison/exploration_vs_success_small.pdf "$OUT/"
)

pareto_and_selection() (
set -euo pipefail
"$PY" scripts/plots/pareto_and_selection.py --out-dir plots/success_rate_comparison
cp plots/success_rate_comparison/exploration_exploitation_entropy.pdf \
   plots/success_rate_comparison/selection_uncertainty_heatmap.pdf "$OUT/"
)

TARGETS=(calibration_table success_rate_comparisons statistical_significance ensemble_size_ablation
         ensemble_vs_laplace language_variation success_share_entropy pareto_and_selection)
if [ $# -eq 0 ] || [ "$1" = all ]; then
    rc=0
    for t in "${TARGETS[@]}"; do
        echo; echo "######## $t"
        "$t"; s=$?
        [ $s -eq 0 ] || { echo "!! $t failed (exit $s)"; rc=1; }
    done
    exit $rc
fi
case " ${TARGETS[*]} " in
    *" $1 "*) t="$1"; shift; "$t" "$@" ;;
    *) echo "unknown target '$1'; one of: ${TARGETS[*]}" >&2; exit 2 ;;
esac
