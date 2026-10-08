# Uncertainty Quantification for Flow-Based Generalist Robot Policies

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
[![Website](<https://img.shields.io/badge/Website-project%20page-blue>)](https://tum-lsy.github.io/uq_generalist_policies/)
[![PyTorch](https://img.shields.io/badge/Python-PyTorch-orange.svg)](https://www.pytorch.org)

The official code repository for *"Uncertainty Quantification for Flow-Based Generalist Robot Policies"*.

## News

- **2026-10:** Refactored code release for the updated paper. It adds failure detection experiments
  (Push-T and LIBERO-Plus), and the X-VLA and FastWAM backbones. The previous code is on the
  [`old_structure`](https://github.com/learnsyslab/uq_vla/tree/old_structure) branch.
- **2026-07:** Initial code release.

> **Abstract:** Generalist robot policies, such as vision-language-action models (VLAs) and world-action
> models (WAMs), combine powerful pretrained backbones with expressive generative action heads trained via
> flow matching on large-scale robotic datasets. Despite their strong empirical performance in robotic
> manipulation, these policies lack mechanisms to quantify confidence in their predictions and to detect when
> their actions may be unreliable. This presents a critical limitation for real-world deployment in
> non-stationary environments, where models inevitably encounter scenarios outside their pretraining
> distribution and may fail without warning. To address this, we derive an efficient method to quantify
> epistemic uncertainty in flow-matching models by leveraging velocity-field disagreement (VFD) across a small
> ensemble. We successfully use this uncertainty estimate for detecting failures during deployment and active
> fine-tuning of flow-based generalist policies. For the latter, we propose SAVE, a simple yet effective
> method for uncertainty-guided active multitask fine-tuning that reduces the number of costly expert
> demonstrations required to adapt generalist policies to new tasks. We conduct experiments in simulation and
> the real world, across VLAs and a WAM. VFD yields better-calibrated uncertainty estimates predictive of
> downstream performance and detects failures with 8 pp higher overall accuracy than existing methods. Across
> three real-world tasks, SAVE improves final average success from 39% to 47% with a fixed demonstration
> budget. Our results show that measuring epistemic uncertainty with VFD enhances both failure awareness and
> adaptation of generalist robot policies.

---

This repository contains the code for the simulation experiments in the paper.

## Overview

The repository has two parts that share one Python environment.

`active_learning/` trains the policy ensembles and runs every experiment that involves a policy:

- calibration of VFD and the baseline uncertainty estimates for SmolVLA, X-VLA and FastWAM on LIBERO-Long,
  including the ablations over ensemble size, Laplace approximation and language variations,
- active fine-tuning with SAVE and the baselines (random, diversity, AMF, Action-L2, GU) on Push-T and
  LIBERO-Long, including the SmolVLA hyperparameter sweep and the Push-T significance test,
- recording and scoring of the rollouts used for failure detection on Push-T and LIBERO-Plus.

It contains a modified copy of [LeRobot](https://github.com/huggingface/lerobot) with the uncertainty
estimators added, and the iterative fine-tuning pipeline (`src/iterative_fine_tuning`).

`failure_detection/` evaluates VFD against STAC, ACE, logpZO and RND-OE as runtime failure detectors on the
scored rollouts, based on [FIPER](https://github.com/utiasDSL/fiper).

The real-robot experiments are not part of this repository. We do not provide checkpoints; all ensembles are
trained from public base models (`HuggingFaceTB/SmolVLM2-500M-Video-Instruct`, `lerobot/xvla-base`,
`lerobot/fastwam_base`) and datasets (`HuggingFaceVLA/libero`, `lerobot/pusht`). The READMEs in the two
folders describe each step in more detail.

## Installation

We used Python 3.10 and PyTorch 2.7 with CUDA 12.8.

```bash
conda create -n vfd python=3.10 && conda activate vfd
conda install -c conda-forge ffmpeg
pip install -e "active_learning[all]"
cp active_learning/.env.example active_learning/.env
```

The `.env` file sets the Hugging Face cache and headless rendering for LIBERO. The plotting scripts use LaTeX,
so generating the figures also needs a TeX installation. For the LIBERO-Plus experiment, additionally run
`pip install -e "active_learning[libero_plus]"` and `bash active_learning/scripts/setup_libero_plus.sh`,
which downloads the benchmark and its assets (6.4 GB).

## Running the experiments

Unless noted otherwise, run the commands from `active_learning/`. Every experiment uses ensembles of two
models. A seed pair names the pretraining seeds of the two members (`s01` for seeds 0 and 1). The LIBERO
experiments use the pairs `s01`, `s23` and `s45`; Push-T uses ten pairs, each pretrained on its own random
subset of 50 demonstrations. All results are written to `active_learning/outputs/`.

**Pretraining.** Train the ensemble members for each policy:

```bash
bash scripts/pretrain.sh fm_pusht 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
bash scripts/pretrain.sh smolvla_libero 0 1 2 3 4 5
bash scripts/pretrain.sh xvla_libero 0 1 2 3 4 5
bash scripts/pretrain.sh fastwam_libero 0 1 2 3 4 5
bash scripts/pretrain.sh smolvla_libero_all40 0 1 2 3 4 5
```

The LIBERO ensembles are trained on LIBERO-Spatial, -Object and -Goal and three of the ten LIBERO-Long tasks.
`smolvla_libero_all40` is trained on all 40 LIBERO tasks and is only used for the LIBERO-Plus failure
detection experiment. FastWAM needs precomputed text embeddings, see `active_learning/README.md`.

**Calibration.** Each run fine-tunes an ensemble for 15 rounds with randomly selected demonstrations, keeps
all checkpoints and then scores every round with all uncertainty methods. `--extras` adds the SmolVLA
ablations from the appendix.

```bash
bash scripts/run_calibration.sh smolvla s01 --extras
bash scripts/run_calibration.sh xvla s01
bash scripts/run_calibration.sh fastwam s01
```

**Active fine-tuning.** Each run fine-tunes an ensemble with one acquisition rule (`random`, `diversity`,
`amf`, `action_l2`, `gu` or `vfd`) on one seed pair and evaluates every round. The SmolVLA sweep covers the
remaining task temperatures and AMF noise scales.

```bash
bash scripts/run_active_learning.sh configs/active_learning/smolvla/vfd.yaml s01
bash scripts/run_smolvla_sweep.sh s01 s23 s45
```

**Failure detection.** Record and score the rollouts in `active_learning/`, then evaluate the detectors in
`failure_detection/`:

```bash
bash scripts/failure_detection/run_pusht.sh
bash scripts/failure_detection/run_libero_plus.sh
cd ../failure_detection && bash run_all.sh
```

**Figures and tables.** `bash paper_plots/make.sh` in `active_learning/` generates all calibration and
active fine-tuning figures and tables from the results in `outputs/` (no GPU needed); `run_all.sh` writes the
failure detection tables.

The runs in the paper used one 32 GB GPU for Push-T, one GPU with at least 40 GB per ensemble member for
SmolVLA, one 80 GB GPU for X-VLA and four 141 GB GPUs for FastWAM. `active_learning/scripts/slurm_template.sbatch`
can be used to run the commands on a SLURM cluster. Since training and simulation are not deterministic on
GPUs, rerunning the experiments gives similar but not identical numbers.

## Citation

If you find this work useful, please consider citing our paper:

```bibtex
@article{romer2026uq_vla,
  title={Uncertainty Quantification for Flow-Based Generalist Robot Policies},
  author={Ralf R{\"o}mer and Maximilian Seeliger and Saida Liu and Ben Sturgis and Marco Bagatella and Daniel Marta and Andreas Krause and Angela P. Schoellig},
  journal={arXiv preprint arXiv:2606.18043},
  year={2026}
}
```

## Acknowledgments

This work builds on [LeRobot](https://github.com/huggingface/lerobot), [SmolVLA](https://huggingface.co/blog/smolvla),
[X-VLA](https://huggingface.co/lerobot/xvla-base), [FastWAM](https://huggingface.co/lerobot/fastwam_base),
[LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO), [LIBERO-Plus](https://github.com/sylvestf/LIBERO-plus)
and [FIPER](https://github.com/utiasDSL/fiper). We thank the authors for making their work available.

## License

This repository is released under the Apache 2.0 license (`LICENSE`). `active_learning/src/lerobot` is a
modified copy of LeRobot (Apache 2.0, `active_learning/LICENSE_LEROBOT`), and `failure_detection/` builds on
FIPER (MIT, `failure_detection/LICENSE_FIPER`).
