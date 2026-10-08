#!/usr/bin/env bash
# The SmolVLA hyperparameter sweep of the appendix (tables/active_learning_ablation, the temperature
# figures): every setting that is not one of the main runs in configs/active_learning/smolvla/.
#
#   bash scripts/run_smolvla_sweep.sh <seed_pair> [<seed_pair> ...]      # s01 s23 s45
#
# Runs are written next to the main runs as outputs/active_learning/smolvla/<setting>_<seed_pair>.
set -euo pipefail
cd "$(dirname "$0")/.."
[ $# -ge 1 ] || { sed -n '2,8p' "$0"; exit 1; }
C=configs/active_learning/smolvla
for PAIR in "$@"; do
    # Task-sampling temperature tau of the three uncertainty-guided rules (main runs: GU tau=1,
    # Action-L2 tau=1.5, VFD tau=2.5).
    for METHOD_BEST in gu:1 action_l2:1.5 vfd:2.5; do
        METHOD=${METHOD_BEST%%:*}; BEST=${METHOD_BEST##*:}
        for TAU in 0 1 1.5 2 2.5; do
            if [ "$TAU" != "$BEST" ]; then
                bash scripts/run_active_learning.sh "$C/${METHOD}.yaml" "$PAIR" \
                    --selection.task_selection_temperature="$TAU" --paths.run_name="${METHOD}_t${TAU}_${PAIR}"
            fi
        done
    done
    # VFD task sampling with a uniformly random episode inside each sampled task.
    bash scripts/run_active_learning.sh $C/vfd.yaml "$PAIR" \
        --selection.task_weighted_episode_selection=uniform_random --paths.run_name=vfd_uniform_t2.5_${PAIR}
    # AMF noise scale sigma (main run: 1e-2).
    for SIGMA in 1e-4 1e-3 1e-1; do
        bash scripts/run_active_learning.sh $C/amf.yaml "$PAIR" \
            --selection.amf.noise=$SIGMA --paths.run_name=amf_sigma${SIGMA}_${PAIR}
    done
done
