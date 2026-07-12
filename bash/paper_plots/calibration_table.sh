#!/bin/bash
# Paper Table 1 (calibration analysis): regenerate the calibration summary bar chart
# from cached uncertainties. Cache-only, no GPU. Caches must already exist (compute
# them first with bash/calibration/submit_calibration_comparison.sh).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

.venv/bin/python scripts/calibration/calibration_summary.py \
  --run_dirs \
    outputs/iterative_fine_tuning/uniform_leak3_weighted_pools_lr_schedule_history05_steps2000_s01 \
    outputs/iterative_fine_tuning/uniform_leak3_weighted_pools_lr_schedule_history05_steps2000_s23 \
    outputs/iterative_fine_tuning/uniform_leak3_weighted_pools_lr_schedule_history05_steps2000_s45 \
  --output plots/calibration_comparison/libero/uniform_leak3_weighted_pools_lr_schedule_history05_steps2000/calibration_summary_15rounds.png \
  --max_round 15 \
  --methods action_l2 ace decu ensemble_terminal_variance vfd vlm_token_entropy vlm_perplexity
