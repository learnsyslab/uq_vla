# Calibration and active fine-tuning

This part covers ensemble pretraining, the calibration experiments (Table 1 and the related appendix
figures), the active fine-tuning experiments (tables, learning curves, SmolVLA sweep, significance test)
and the rollout recording for failure detection.

`src/lerobot` is a modified copy of [LeRobot](https://github.com/huggingface/lerobot) 0.4.3, stripped down to
the policies (SmolVLA, X-VLA, FastWAM and the flow-matching policy for Push-T) and environments (LIBERO,
Push-T) we use. The uncertainty estimators live in `src/lerobot/uncertainty`, the rollout recorder and scorer
in `src/lerobot/fiper_data_generator`. The active fine-tuning loop itself is in `src/iterative_fine_tuning`.

All models start from public checkpoints (`HuggingFaceTB/SmolVLM2-500M-Video-Instruct`, `lerobot/xvla-base`,
`lerobot/fastwam_base`) and are trained on `HuggingFaceVLA/libero` or `lerobot/pusht`.

## Installation

We used Python 3.10 and PyTorch 2.7 with CUDA 12.8.

```bash
conda create -n vfd python=3.10 && conda activate vfd
conda install -c conda-forge ffmpeg
pip install -e ".[all]"
cp .env.example .env
```

LIBERO is rendered headless with EGL (`MUJOCO_GL=egl`, set in `.env`). The plotting scripts use LaTeX for
text, so `paper_plots/make.sh` needs a TeX installation with `dvipng`.

LIBERO-Plus is only needed for the failure-detection rollouts. The setup script clones the LIBERO-Plus fork
at a fixed commit into `third_party/libero_plus`, downloads its assets (6.4 GB) and makes the fork importable
alongside hf-libero:

```bash
pip install -e ".[libero_plus]"
bash scripts/setup_libero_plus.sh
```

## Overview

Configs are in `configs/`:

- `pretrain/`: one recipe per ensemble (`fm_pusht`, `smolvla_libero`, `xvla_libero`, `fastwam_libero`,
  `smolvla_libero_all40`)
- `active_learning/<policy>/`: one config per acquisition rule (`random`, `diversity`, `amf`, `action_l2`,
  `gu`, `vfd`; no AMF for Push-T)
- `calibration/`: one config per policy
- `failure_detection/`: rollout recording and scoring
- `diversity_rankings/`: precomputed rankings for the Diversity baseline

The shell scripts in `scripts/` are the entry points; the Python scripts they call are in its subfolders.
Everything is written to `outputs/` (pretrained models in `outputs/pretrain`, runs in
`outputs/active_learning/<policy>/<rule>_<pair>` and `outputs/calibration/<policy>/random_<pair>`).

All experiments use ensembles of two models. A seed pair names the pretraining seeds of the two members, e.g.
`s01` for seeds 0 and 1. The LIBERO experiments use `s01`, `s23` and `s45`. Push-T uses ten pairs (`s01` to
`s1819`), and each pair is pretrained on its own random subset of 50 of the 206 demonstrations. The seed
pairs, together with the training and selection seeds, are listed at the end of every config.

## Pretraining

```bash
bash scripts/pretrain.sh fm_pusht 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
bash scripts/pretrain.sh smolvla_libero 0 1 2 3 4 5
bash scripts/pretrain.sh xvla_libero 0 1 2 3 4 5
bash scripts/pretrain.sh fastwam_libero 0 1 2 3 4 5        # 4 GPUs per model
bash scripts/pretrain.sh smolvla_libero_all40 0 1 2 3 4 5  # only for the LIBERO-Plus experiment
```

The LIBERO models are trained for 30k steps on LIBERO-Spatial, -Object and -Goal and on tasks 0-2 of
LIBERO-10 (1437 demonstrations). The other LIBERO-10 tasks are only seen during fine-tuning. Before training
FastWAM, compute the text embeddings of the task instructions once:

```bash
python scripts/fastwam/precompute_task_contexts.py --out outputs/pretrain/fastwam_text_contexts/libero.pt
```

## Active fine-tuning

Each run trains one acquisition rule on one seed pair for 15 rounds (30 for Push-T) and then evaluates every
round:

```bash
bash scripts/run_active_learning.sh configs/active_learning/smolvla/vfd.yaml s01
```

We ran all rules with all seed pairs. The remaining settings of the SmolVLA hyperparameter sweep (other task
temperatures and AMF noise scales) are run with

```bash
bash scripts/run_smolvla_sweep.sh s01 s23 s45
```

In each round, 5 demonstrations are selected from the dataset, and both ensemble members are fine-tuned on
the selected demonstrations mixed 50/50 with pretraining data (70/30 for Push-T).

After training, the pretrained ensemble and every round are evaluated, with the same settings as in the paper:
30 rollouts per LIBERO-10 task (environment seeds starting at 100, at most 520 steps) or 100 Push-T rollouts
per ensemble member. For SmolVLA we only evaluated the first member in the active fine-tuning runs; all other
runs evaluate both. Training and evaluation can be resumed by rerunning the same command. Checkpoints of
earlier rounds are deleted once they are no longer needed.

## Calibration

```bash
bash scripts/run_calibration.sh smolvla s01 --extras
bash scripts/run_calibration.sh xvla s01
bash scripts/run_calibration.sh fastwam s01
```

(likewise for `s23` and `s45`). A calibration run fine-tunes the ensemble for 15 rounds with randomly
selected data and keeps all checkpoints. After evaluating each round, it computes the uncertainty of the
first frame of every LIBERO-10 demonstration with each method: Action-L2, ACE, DECU, GU and VFD, plus Entropy
and Perplexity for SmolVLA. The calibration table then correlates the mean uncertainty of each task with its
success rate.

`--extras` runs the three SmolVLA studies from the appendix on the same runs: ensembles of size 3 and 4 (two
extra members per round), a last-layer Laplace approximation instead of the second member (rounds 0-4), and
the language variations (five paraphrases per instruction, evaluated on the final policy).

## Failure-detection rollouts

```bash
bash scripts/failure_detection/run_pusht.sh s01 s23 s45
GPU_IDS=0,1,2,3 bash scripts/failure_detection/run_libero_plus.sh s01 s23 s45
```

On Push-T, one member of each ensemble records 70 successful calibration rollouts and 200 test rollouts.
For LIBERO-Plus, we use the SmolVLA ensembles trained on all 40 LIBERO tasks and record, for every LIBERO-10
task, 20 successful calibration rollouts and 30 test rollouts without perturbations, plus 30 test rollouts for
each of the six LIBERO-Plus perturbation types (camera, robot initial state, sensor noise, background,
lighting, layout). The rollouts are then scored with the ensemble, and the training demonstrations are
embedded for the logpZO and RND-OE baselines. `../failure_detection` reads the results from
`outputs/fiper_rollout_scoring/`.

## Figures and tables

```bash
bash paper_plots/make.sh
```

This only reads from `outputs/` and does not need a GPU. Single targets can be built with
`bash paper_plots/make.sh <target>`; the list of targets is at the top of the script.

## Compute

We used one 32 GB GPU for Push-T, one GPU with at least 40 GB per ensemble member for SmolVLA, one 80 GB GPU
for X-VLA and four 141 GB GPUs for FastWAM. A Push-T round takes a few minutes, a FastWAM run about a day per
seed pair. `scripts/slurm_template.sbatch` is a minimal wrapper for running any of the commands on a SLURM
cluster.

## Notes

The configs contain the hyperparameters and seeds of the runs in the paper. Training and simulation are not
deterministic on GPUs, so rerunning gives similar but not identical numbers.

The Diversity baseline uses k-center-greedy rankings over SigLIP embeddings
(`scripts/active_learning/k_greedy*.py`). Since the first element is chosen at random, the SmolVLA runs
ended up with a different ranking than X-VLA and FastWAM; both rankings are in `configs/diversity_rankings/`.

The FastWAM results in the paper use the first 10 rounds.
