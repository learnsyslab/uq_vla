PYTHONPATH=src .venv/bin/python scripts/calibration/language_variation_aggregate.py \
  --run_dirs \
    outputs/iterative_fine_tuning/uniform_leak3_weighted_pools_lr_schedule_history05_steps2000_s01 \
    outputs/iterative_fine_tuning/uniform_leak3_weighted_pools_lr_schedule_history05_steps2000_s23 \
    outputs/iterative_fine_tuning/uniform_leak3_weighted_pools_lr_schedule_history05_steps2000_s45 \
  --round 14 \
  --input_root plots/language_variation/leak3fixed_weighted_pools_lr_schedule_history05_steps2000 \
  --output_root plots/language_variation/leak3fixed_weighted_pools_lr_schedule_history05_steps2000/aggregated && \
stat -c '%y %s %n' plots/language_variation/leak3fixed_weighted_pools_lr_schedule_history05_steps2000/aggregated/round_014_success_rates_summary.png

bash bash/calibration/submit_language_variation.sh \
    plots/language_variation/leak3fixed_weighted_pools_lr_schedule_history05_steps2000 \
    outputs/iterative_fine_tuning/uniform_leak3_weighted_pools_lr_schedule_history05_steps2000_s01 \
    outputs/iterative_fine_tuning/uniform_leak3_weighted_pools_lr_schedule_history05_steps2000_s23 \
    outputs/iterative_fine_tuning/uniform_leak3_weighted_pools_lr_schedule_history05_steps2000_s45 \
    -- --round 14
