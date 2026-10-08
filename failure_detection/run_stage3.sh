#!/usr/bin/env bash
# Failure detection (calibration, detector training, rollout scoring, metrics) for one policy ensemble.
#
#   bash run_stage3.sh pusht       <scored-rollout dir> <output dir>
#   bash run_stage3.sh libero_plus <scored-rollout dir> <output dir>
#
# <scored-rollout dir> is the output of active_learning's rollout scoring for one ensemble:
#   pusht:       <dir>/rollouts/{calibration,test}/*.pkl + <dir>/demo_embeddings/
#   libero_plus: <dir>/libero_10/taskNN/rollouts/{calibration,test}/*.pkl + <dir>/libero_10/demo_embeddings/
# The fitted logpZO / RND-OE models are cached inside it (logpzo_models/, rnd_models/, pooled_models/), so a
# second run re-uses them. <output dir> receives complete_results.csv (all metrics, every threshold, window and
# quantile), method_results/ (per-rollout scores) and, for libero_plus, test_rollouts.txt (the test rollouts in
# the pipeline's order, which the per-family table needs). Extra arguments are passed to pipeline.py as Hydra
# overrides.
set -euo pipefail
ENV=${1:?environment: pusht or libero_plus}; SRC=$(realpath "${2:?scored-rollout dir}"); OUT=${3:?output dir}
shift 3
FD=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PY=${PYTHON:-python}

ROOT=$(mktemp -d "${TMPDIR:-/tmp}/fd_stage3.XXXXXX")
trap 'rm -rf "$ROOT"' EXIT
case $ENV in
  pusht)       ln -s "$SRC" "$ROOT/push_t" ;;
  libero_plus) ln -s "$SRC/libero_10" "$ROOT/libero_10" ;;
  *) echo "unknown environment '$ENV' (pusht or libero_plus)" >&2; exit 2 ;;
esac

FD_DATA_ROOT=$ROOT MPLBACKEND=Agg "$PY" "$FD/pipeline.py" --config-name "$ENV" "$@"

mkdir -p "$OUT"
rm -rf "$OUT/complete_results.csv" "$OUT/method_results" "$OUT/summaries"
cp -a "$ROOT/results/." "$OUT/"
if [ "$ENV" = libero_plus ]; then
  (cd "$FD" && "$PY" - "$SRC/libero_10" "$OUT/test_rollouts.txt" <<'PY'
import os, sys
from shared_utils.data_management import _get_filenames
suite, out = sys.argv[1], sys.argv[2]
with open(out, "w") as f:
    for task in sorted(d for d in os.listdir(suite) if d.startswith("task")):
        for name in _get_filenames(os.path.join(suite, task, "rollouts", "test"), keywords=["episode", "eps", "rollout"]):
            f.write(f"{task} {name}\n")
PY
  )
fi
echo "-> $OUT"
