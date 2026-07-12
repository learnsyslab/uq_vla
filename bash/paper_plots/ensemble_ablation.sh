.venv/bin/python scripts/calibration/ensemble_size_ablation.py \
  --run_dirs \
    outputs/iterative_fine_tuning/uniform_leak3_weighted_pools_lr_schedule_history05_steps2000_s01 \
    outputs/iterative_fine_tuning/uniform_leak3_weighted_pools_lr_schedule_history05_steps2000_s23 \
    outputs/iterative_fine_tuning/uniform_leak3_weighted_pools_lr_schedule_history05_steps2000_s45 \
  --output plots/ensemble_size_ablation/libero/uniform_leak3_weighted_pools_lr_schedule_history05_steps2000 \
  --rounds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 \
  --ensemble_sizes 2 3 4 \
  --env libero \
  --aggregate_only && \
stat -c '%y %s %n' plots/ensemble_size_ablation/libero/uniform_leak3_weighted_pools_lr_schedule_history05_steps2000/aggregated/spearman_across_rounds.pdf