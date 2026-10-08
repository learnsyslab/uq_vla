#!/usr/bin/env bash
# Failure detection for all ensembles of the paper, then the tables.
#
#   bash run_all.sh [<scored-rollout root>]
#
# <scored-rollout root> defaults to ../active_learning/outputs/fiper_rollout_scoring, where active_learning's
# rollout recording and scoring writes pusht_pre50_<ensemble>/ and smolvla_libero_plus_<ensemble>/.
# Results go to results/{pusht,libero_plus}/<ensemble>/, tables to tables/out/. Each run needs one GPU
# (logpZO/RND-OE training); LIBERO-Plus fits one logpZO model per ensemble (several hours), which is cached
# in the scored-rollout directory and re-used by later runs.
set -euo pipefail
FD=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCORED=$(realpath "${1:-$FD/../active_learning/outputs/fiper_rollout_scoring}")
PY=${PYTHON:-python}
for e in s01 s23 s45; do
  bash "$FD/run_stage3.sh" pusht "$SCORED/pusht_pre50_$e" "$FD/results/pusht/$e"
  bash "$FD/run_stage3.sh" libero_plus "$SCORED/smolvla_libero_plus_$e" "$FD/results/libero_plus/$e"
done
"$PY" "$FD/tables/make_tables.py" --results "$FD/results" --out "$FD/tables/out"
