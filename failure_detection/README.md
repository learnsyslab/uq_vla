# Failure detection

Code for the simulation failure-detection experiments: VFD compared with STAC, ACE, logpZO and RND-OE on
Push-T and LIBERO-Plus. It produces the Push-T and LIBERO-Plus columns of the failure-detection table in the
main text and the two appendix tables for these environments, including the LIBERO-Plus breakdown by
perturbation family. The Real World column and the average over all three environments need the robot and
are not reproduced here.

There is no policy code in this part. It works on scored rollouts, which are recorded in `../active_learning`.

## Producing the scored rollouts

Run from `../active_learning`:

```bash
bash scripts/pretrain.sh fm_pusht 0 1 2 3 4 5
bash scripts/pretrain.sh smolvla_libero_all40 0 1 2 3 4 5
bash scripts/failure_detection/run_pusht.sh
bash scripts/setup_libero_plus.sh          # once
bash scripts/failure_detection/run_libero_plus.sh
```

We use three two-member ensembles per environment, `s01`, `s23` and `s45` (seeds 0 and 1, 2 and 3, 4 and 5).
For each of them the runners write a directory to `../active_learning/outputs/fiper_rollout_scoring/`.
For Push-T that is `pusht_pre50_<pair>/`, with `rollouts/{calibration,test}/` and the embedded training
demonstrations in `demo_embeddings/member_00_global_cond.pt`. For LIBERO-Plus it is
`smolvla_libero_plus_<pair>/libero_10/`, with `taskNN/rollouts/{calibration,test}/` and
`demo_embeddings/smolvla_all40_stride10_obs_embedding.pt`.

Calibration directories contain successful rollouts only, and the `_s_` or `_f_` in a file name marks success
or failure. In LIBERO-Plus the test episode number encodes the perturbation family. Episode `30k + j` belongs
to family `k`, in the order camera, robot initialization, sensor noise, background, lighting and layout.
The unperturbed test rollouts are numbered from 1000.

## Running failure detection

The code runs in the same environment as `../active_learning` and needs a GPU to train logpZO and RND-OE.

```bash
bash run_all.sh ../active_learning/outputs/fiper_rollout_scoring
```

This runs all six ensembles and then writes the tables to `tables/out/`. A single ensemble can be run with

```bash
bash run_stage3.sh pusht       <scored dir>/pusht_pre50_s01         results/pusht/s01
bash run_stage3.sh libero_plus <scored dir>/smolvla_libero_plus_s01 results/libero_plus/s01
python tables/make_tables.py --results results
```

and `slurm_stage3.sbatch` wraps one such call for a SLURM cluster.

A run calibrates the thresholds, trains logpZO and RND-OE, scores all test rollouts with the five detectors
and computes the metrics. The output directory gets `complete_results.csv`, with the metrics for every
threshold type, window size and quantile, and `method_results/` with the per-rollout scores. The trained
logpZO and RND-OE models are cached in the scored-rollout directory (`logpzo_models/` and `rnd_models/` for
Push-T, `pooled_models/` for LIBERO-Plus) and reused by later runs. A Push-T run takes minutes. On LIBERO-Plus,
training logpZO takes several hours per ensemble on one GPU.

## Setting

The threshold is the 0.90 quantile of the maximum scores of the calibration rollouts, with window size 1. We
use 20 successful calibration rollouts: the first 20 on Push-T and 20 unperturbed rollouts per task on
LIBERO-Plus. On Push-T we test on 200 rollouts. On LIBERO-Plus we test on 30 rollouts per task and
perturbation family and report the perturbed ones only. The 30 unperturbed test rollouts per task are
recorded but not reported. logpZO and RND-OE are trained on the policy's training demonstrations, with one
model per policy on LIBERO-Plus. VFD uses the flow times 0.2, 0.4, 0.6, 0.8 and 0.9 with 8 action samples on
Push-T, and all recorded flow times with 16 action samples on LIBERO-Plus.

Metrics are averaged over tasks and reported as mean and standard deviation over the three ensembles. A task
with fewer than two test rollouts of either outcome is left out of that ensemble's average. The settings are
in `configs/pusht.yaml` and `configs/libero_plus.yaml`, the detector hyperparameters in `configs/eval/`.

## Code

`pipeline.py` is the entry point (`python pipeline.py --config-name pusht`, with the data root in
`FD_DATA_ROOT`). Rollouts are loaded by `tasks/` and `rollout_data/`. The detectors, threshold calibration
and metrics are in `evaluation/`, RND-OE training is in `rnd/`, and `shared_utils/` holds the training data
for the learned detectors and other utilities. `tables/make_tables.py` builds the tables.
