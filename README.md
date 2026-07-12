# Uncertainty Quantification for Flow-Based Vision-Language-Action Models

[Ralf Römer](https://ralfroemer.com)<sup>1</sup>,
[Maximilian Seeliger](https://www.linkedin.com/in/maximilian-seeliger/)<sup>2</sup>,
[Saida Liu](https://saidaliu27.github.io/)<sup>1</sup>,
[Ben Sturgis](https://www.linkedin.com/in/ben-sturgis/?locale=de)<sup>1</sup>,
[Marco Bagatella](https://marbaga.github.io/)<sup>2,3</sup>,
[Daniel Marta](https://daniellsm.github.io/)<sup>2</sup>,
[Andreas Krause](https://las.inf.ethz.ch/krausea)<sup>2</sup>,
[Angela P. Schoellig](https://www.dynsyslab.org/prof-angela-schoellig/)<sup>1</sup>

<sup>1</sup>Technical University of Munich &nbsp;&nbsp;
<sup>2</sup>ETH Zurich &nbsp;&nbsp;
<sup>3</sup>MPI for Intelligent Systems

[![arXiv](https://img.shields.io/badge/arXiv-2606.18043-red)](https://arxiv.org/abs/2606.18043)
[![Website](https://img.shields.io/badge/Website-UQ--VLA-blue)](https://tum-lsy.github.io/uq_vla/)
[![PyTorch](https://img.shields.io/badge/Python-PyTorch-orange.svg)](https://www.pytorch.org)

The official code repository for *"Uncertainty Quantification for Flow-Based Vision-Language-Action Models"*.

> **Abstract:** Vision-language-action models (VLAs) combine vision-language backbones with
> expressive generative action heads trained via flow matching on large-scale robotic
> datasets. Despite their strong empirical performance in robotic manipulation, VLAs lack
> mechanisms to quantify confidence in their predictions and to detect when their actions
> may be unreliable. This presents a critical limitation for real-world deployment in
> non-stationary environments, where models inevitably encounter scenarios outside their
> pretraining distribution and may fail without warning. To address this, we derive an
> efficient method for quantifying epistemic uncertainty in flow-matching models by
> leveraging velocity-field disagreement (VFD) across a small ensemble. We successfully use
> this uncertainty estimate for failure detection during deployment and active fine-tuning
> of flow-based VLAs. To this end, we propose SAVE, a framework for uncertainty-guided
> active multitask fine-tuning that reduces the number of costly expert demonstrations
> required to adapt VLAs to new tasks. Through extensive experiments on the LIBERO
> benchmark, we demonstrate that VFD yields better-calibrated uncertainty estimates
> predictive of downstream performance, that VFD achieves strong performance in detecting
> failures, and that uncertainty-guided data acquisition with SAVE requires at least 22%
> fewer samples than baselines. In summary, our work shows that quantifying epistemic
> uncertainty in flow-based VLAs improves both failure awareness and adaptation.

---

This Readme describes how to reproduce the results reported in the paper.

# Installation

### Conda

```bash
conda create -y -n uq_vla python=3.10
conda activate uq_vla
conda install ffmpeg=7.1.1 -c conda-forge
pip install -e ".[smolvla,libero,uncertainty]"
```

### uv (no conda; e.g. on a cluster login node)

```bash
uv venv --python 3.10
uv pip install "cmake<4"
uv pip install -e ".[smolvla,libero,uncertainty]"
uv pip install seaborn
source .venv/bin/activate
```

## Environment

Create a `.env` file at the repo root with at least:

```bash
STORAGE_ROOT=/path/to/storage           
HF_HOME=/path/to/huggingface_cache
```

Before running any script in the rest of this README:

```bash
source .venv/bin/activate
export PYTHONPATH=src
source .env
```

# Calibration

All calibration figures use one iterative fine-tuning run with random episode selection, across 3 seeds and the first 15 rounds (`--max_round 15`):

```
uniform_leak3_weighted_pools_lr_schedule_history05_steps2000_s01
uniform_leak3_weighted_pools_lr_schedule_history05_steps2000_s23
uniform_leak3_weighted_pools_lr_schedule_history05_steps2000_s45
```

The calibration pipeline is split into a GPU compute layer and a cache-only plot layer.

- **`bash/calibration/*` — Compute (GPU, cluster).** Run inference and write
  per-method, per-round caches into the run tree
  (`round_*/calibration_comparison/<method>_episode_uncertainties.json`).
- **`bash/paper_plots/*` — Plot (cache-only, CPU).** Re-read the caches and
  regenerate the paper figures. No GPU, runs in seconds.

## Stage 1 — Generate the calibration runs (GPU, cluster)

Iterative fine-tuning with uniform/random selection, swept over seed pairs:

```bash
bash bash/calibration/calibration_all.sh \
  configs/iterative_fine_tuning/calibration_experiment/uniform_leak3_weighted_pools_lr_schedule.yaml
```

This calls `bash/active_learning/cluster_all.sh` per seed pair, submitting a train +
eval SLURM job per round. Output lands in
`outputs/iterative_fine_tuning/<run>_sXX/round_*/` with checkpoints and
`round_evaluation.json` (per-task success rates) per round.

Prerequisite: a pretrained SmolVLA ensemble (see `bash/pretrain/`).

## Stage 2 — Compute uncertainty caches (GPU, cluster)

```bash
bash bash/calibration/submit_calibration_comparison.sh \
  plots/calibration_comparison/libero/uniform_leak3_weighted_pools_lr_schedule_history05_steps2000 \
  outputs/iterative_fine_tuning/uniform_leak3_weighted_pools_lr_schedule_history05_steps2000_s01 \
  outputs/iterative_fine_tuning/uniform_leak3_weighted_pools_lr_schedule_history05_steps2000_s23 \
  outputs/iterative_fine_tuning/uniform_leak3_weighted_pools_lr_schedule_history05_steps2000_s45 \
  -- --rounds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14
```

Submits one `calibration_comparison.py` job per seed (`cluster_methods_comparison.sbatch`)
plus an aggregation job. Caches are written per round under
`round_*/calibration_comparison/`. The aggregation job also creates per-round scatter
plots and the Table 1 summary chart.

Extra caches needed for some figures:

- **Ensemble size ablation:** requires ensemble members `00..03` in each round
  (train extras with `bash/active_learning/cluster_train_extra_members.sbatch`),
  then compute the per-size caches:
  ```bash
  sbatch cluster_ensemble_size_ablation.sbatch \
    --run_dirs <s01> <s23> <s45> \
    --rounds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 \
    --ensemble_sizes 2 3 4 --env libero
  ```
- **Language prompt variation:** GPU rollouts under prompt perturbations:
  ```bash
  bash bash/calibration/submit_language_variation.sh \
    plots/language_variation/leak3fixed_weighted_pools_lr_schedule_history05_steps2000 \
    <s01> <s23> <s45> -- --round 14
  ```
- **Ensemble vs Laplace:** fit the last-layer Laplace posterior per
  round (`bash/calibration/fit_laplace.sh`), then run `calibration_comparison.py`
  including the `vfd_laplace` method.

## Stage 3 — Generate figures (cache-only, no GPU)

Once the caches exist, every figure regenerates in seconds:

```bash
bash bash/paper_plots/calibration_table.sh    
bash bash/paper_plots/ensemble_ablation.sh      
bash bash/paper_plots/language_variation.sh     
bash bash/paper_plots/ensemble_vs_laplace.sh    
```

# Active Fine-Tuning (SAVE)

An M = 2 SmolVLA ensemble is iteratively fine-tuned on LIBERO-10 over R = 15 rounds.
Each round prioritizes tasks by their mean VFD uncertainty (via a task-sampling
temperature τ), requests n_e = 5 expert demonstrations, and fine-tunes with a
50/50 replay of pretraining data (λ = 0.5). All runs use 3 seeds.

## Paper setup (Appendix B.6)

| Setting | Value |
| --- | --- |
| Benchmark | LIBERO-10 (K = 10 tasks) |
| Ensemble | SmolVLA, M = 2 members (pretrained on Goal/Spatial/Object + 3 LIBERO-10 "leak" tasks) |
| Rounds R | 15 |
| Demonstrations per round n_e | 5 (75 total) |
| Gradient steps per round | 4000, batch size 32, cosine LR 5e-5 → 5e-6 (200 warmup) |
| Replay ratio λ | 0.5 |
| Task-sampling temperature τ | sweep {0, 1, 1.5, 2, 2.5, 3} (τ = 0 is uniform) |
| Seeds | 3 (1000, 1001, 1002) |
| Per-round evaluation | 30 rollouts, first ensemble member, ≤ 520 steps |

These match the imported configs under `configs/iterative_fine_tuning/iter_al*/`
(`total_rounds: 15`, `episodes_per_round: 5`, `steps: 4000`, replay `0.5/0.5`,
two `ensemble_model_paths`, three `multi_seed` runs).

## Config → paper method mapping

Each directory holds the temperature sweep `t0, t1, t1.5, t2, t2.5, t3`.

## How to run (cluster)

The launcher submits, per config, a `train → eval → aggregate` chain for every
seed declared in that config's `multi_seed` block:

```bash
# best VFD temperature
bash bash/active_learning/submit_multi_seed.sh \
  configs/iterative_fine_tuning/iter_al/t2.5_w50_eps5_step4000_lr.yaml

# full VFD temperature sweep
bash bash/active_learning/submit_multi_seed.sh configs/iterative_fine_tuning/iter_al/*.yaml
```

Per seed it submits a training job
(`cluster_train_multi_seed_child.sbatch` → `iterative_fine_tuning.run_multi_seed_child`)
and a dependent evaluation job
(`cluster_eval_multi_seed_child.sbatch` → `iterative_fine_tuning.evaluate_run`);
once all seeds finish, one aggregation job
(`cluster_eval_multi_seed_aggregate.sbatch` → `iterative_fine_tuning.summarize_multi_seed_run`)
summarizes success rates across seeds into `multi_seed_evaluation.json`.

## Diversity baseline (`predefined_ranking`)

The `iter_al_diversity` configs use a `predefined_ranking` selection strategy:
each round simply pops the next `n_e` not-yet-selected episodes from a
precomputed ranking file. The paper's diversity baseline ranks all
candidate initial observations by SigLIP k-center-greedy; that ranking is
computed offline, written to JSON, and referenced from the config via
`selection.predefined_ranking_path`.

### Ranking file format

A JSON list of objects, ordered most-diverse-first. Each entry must have at
least `episode_id`; optional fields (`task_group`, `task_id`, `instruction`,
`frame_index`) are recorded into the per-round selection manifest for
downstream analysis.

```json
[
  {"episode_id": 23, "task_group": "libero_10", "task_id": 2, "instruction": "..."},
  {"episode_id": 11, "task_group": "libero_10", "task_id": 7, "instruction": "..."}
]
```

The default path referenced by the diversity configs is
`${STORAGE_ROOT}/outputs/active_learning/diversity_ranking_libero10.json`.

### Producing the ranking

Use [`scripts/diversity/k_greedy.py`](scripts/diversity/k_greedy.py)
(SigLIP `so400m-patch14-384` + k-center-greedy over the candidate task pool); see
that script's `--help` for arguments. Its output should be written to the path
the config expects, in the format above.

### Implementation

`predefined_ranking` is implemented in
[`src/iterative_fine_tuning/selection.py`](src/iterative_fine_tuning/selection.py)
via `_run_predefined_ranking_selection`. It short-circuits at the top of
`run_selection_round`, skipping the policy / ensemble / uncertainty path, and
writes the usual `SelectionManifest` so the rest of the iterative loop (training
and evaluation) is unchanged. The companion config changes (allowing empty
`ensemble_model_paths`, adding the `predefined_ranking_path` field) live in
[`src/iterative_fine_tuning/config.py`](src/iterative_fine_tuning/config.py).

## Implementation note (thin multi-seed wrapper)

The multi-seed orchestration is a thin layer over main's existing single-run loop:
`iterative_fine_tuning.main.run()` dispatches to `_run_multi_seed` when a config
declares `multi_seed` seeds/runs, which builds a per-seed child config and calls
the unchanged single-run loop (`_run_single`). Evaluation reuses `evaluate_run.py`.
main's selection / training / evaluation behavior is unchanged, so the calibration
and single-seed results above are unaffected.

# Failure Prediction

Reproduce the runtime **failure-detection** experiment. At each policy-inference timestep we compute the VFD epistemic
uncertainty score and flag the rollout as *Fail* once the score exceeds a
per-task conformal threshold calibrated on successful rollouts. The paper reports
Accuracy, True-Positive-Rate (TPR), True-Negative-Rate (TNR), normalized
Detection Time, and Timestep-Wise Accuracy (TWA), and — following the failure
prediction framework — averages over the conformal quantiles `0.90, 0.91, …, 0.99`
rather than cherry-picking one confidence level.

| Code name | Paper name |
| --- | --- |
| `vfd` | VFD (ours) |

> Baselines (Entropy, TC, RND-OE, ACE) are computed downstream — see
> [Downstream FIPER metrics/plots](#downstream-fiper-metricsplots) below.

## Pipeline overview

The reproduction has two layers. This repo covers the first (record + score);
the FIPER metric computation and the final plot live in the separate
[FIPER repo](https://github.com/learnsyslab/fiper).

1. **Record** LIBERO-10 rollouts, caching the intermediate
   ODE-trajectory tensors needed for scoring
   (`src/lerobot/scripts/fiper_data_generation/record_fiper_rollout.py`).
2. **Score** each recorded rollout with the M = 2 ensemble, producing the
   per-timestep `vfd_oneway` and `vfd` (VFD) uncertainty
   scores (`src/lerobot/scripts/fiper_data_generation/score_fiper_rollout.py`).

The `vfd` metric is implemented in
[`src/lerobot/fiper_data_generator/fiper_rollout_scorer.py`](src/lerobot/fiper_data_generator/fiper_rollout_scorer.py)
(`compute_ensemble_vfd_scores`, gated by the
`bayesian_ensemble: [..., vfd]` entry in the score config).

## Prerequisite

A trained M = 2 SmolVLA ensemble. The configs/scripts default to the round-15
checkpoints of the SAVE runs from
[Active Fine-Tuning (SAVE)](#active-fine-tuning-save), per seed `s01 / s23 / s45`:

```
outputs/iterative_fine_tuning/uniform_leak3fixed_weighted_pools_lr_schedule_history05_steps2000_<seed>/round_015/training/member_0{0,1}/checkpoints/last/pretrained_model
```

Override via the `POLICY_CHECKPOINT`, `ENSEMBLE_MEMBER_00`, `ENSEMBLE_MEMBER_01`
environment variables (see the scripts below).

## Configs

The paper pipeline uses a single record/score config pair:

| Config pair (`record_fiper_rollout/`, `score_fiper_rollout/`) | Purpose |
| --- | --- |
| `libero10_all_tasks_5eps.yaml` | LIBERO-10, scores `vfd_oneway` + `vfd` |

Episode counts, ODE evaluation times, and the ensemble paths are set inside the
YAML — see `configs/smolvla/score_fiper_rollout/libero10_all_tasks_5eps.yaml`.

## How to run

`paper_plots/all_commands.sh` is the end-to-end orchestrator. Its **Step 1** is
this repo's portion; per seed it runs:

```bash
# record + one-way (vfd_oneway) scoring, sharded over the 10 tasks
scripts/fiper/run_libero10_seed_5eps_data.sh s01
# re-score the recorded rollouts with vfd (VFD)
scripts/fiper/run_libero10_seed_2way_scoring.sh s01
```

The script layering is
`run_libero10_seed_*` → `run_libero10_parallel_*` (per-GPU task fan-out) →
`run_libero10_task_*` (one task → `record_fiper_rollout.py` / `score_fiper_rollout.py`).
Useful env vars: `TASK_IDS`, `GPU_IDS`, `MAX_PARALLEL`, `SEED_BASE`,
`EXPERIMENT_NAME`. Scored output lands in
`outputs/fiper_rollout_scoring/<experiment>/libero_10/`. For a quick single-task
check, run one task shard directly:
`scripts/fiper/run_libero10_task_5eps_data.sh 0`.

## Downstream FIPER metrics/plots

Steps 2–5 of `paper_plots/all_commands.sh` — computing the FIPER detection metrics
for all methods, merging the `vfd` results, and rendering
Figure 12 via `scripts/plot_seed_mean_accuracy_detection.py` — run against the
separate [FIPER repository](https://github.com/learnsyslab/fiper). The
`all_commands.sh` reference (with `FIPER_ROOT`-relative paths) is included as a
record of the full cross-repo workflow; integrating those steps into this repo is
a follow-up.

# Citation

If you find this work useful, please consider citing our paper:

```bibtex
@article{romer2026uq_vla,
  title={Uncertainty Quantification for Flow-Based Vision-Language-Action Models},
  author={Ralf R{\"o}mer and Maximilian Seeliger and Saida Liu and Ben Sturgis and Marco Bagatella and Daniel Marta and Andreas Krause and Angela P. Schoellig},
  journal={arXiv preprint arXiv:2606.18043},
  year={2026}
}
```

# Acknowledgments

This work builds upon:
- [LeRobot](https://github.com/huggingface/lerobot) by Hugging Face
- [SmolVLA](https://huggingface.co/blog/smolvla) by Hugging Face
- [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) by Bo Liu et al.

We thank the authors of these projects for their open-source contributions.
