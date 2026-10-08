"""Episode-level split of the calibration rollouts into train and threshold halves.

Every detector that *trains* on the calibration rollouts (RND-*, logpZO) must not also
calibrate its threshold on the episodes it was fit to: its scores there are in-sample
and low, which drags the threshold down and inflates the false-alarm rate. Untrained
scorers (VFD, ACE, STAC) have nothing to leak and keep the full calibration set.

The split is episode-level, not timestep-level, because timesteps within one rollout are
highly correlated -- a per-timestep split would leak almost as much as no split at all.

Configuration lives in `eval/base.yaml` as

    calibration_split:
      train_fraction: 0.5
      seed: 1

and therefore applies to every method by default. `rnd_oe.yaml` additionally carries the
older `rnd_train.calibration_train_fraction` / `rnd_train.calibration_split_seed` keys,
which take precedence when present; they are kept there on purpose because
`hparams.model.rnd_train` feeds RND checkpoint identification, so removing them would
invalidate every existing `rnd_models/**/*.ckpt`. Both paths resolve to the same 0.5 /
seed 1, so RND-OE is unaffected either way.
"""

import numpy as np


def get_calibration_split(cfg) -> tuple[float, int]:
    """Resolve `(train_fraction, seed)` for the calibration split.

    Prefers the legacy `rnd_train.*` keys when a config defines them (see module
    docstring), else the global `calibration_split` block, else no split.
    """
    rnd_train_cfg = cfg.get("rnd_train", {}) or {}
    if "calibration_train_fraction" in rnd_train_cfg:
        train_fraction = float(rnd_train_cfg["calibration_train_fraction"])
        seed = rnd_train_cfg.get("calibration_split_seed", rnd_train_cfg.get("seed", 0))
        return train_fraction, int(seed)

    split_cfg = cfg.get("calibration_split", {}) or {}
    train_fraction = float(split_cfg.get("train_fraction", 1.0))
    seed = split_cfg.get("seed", cfg.get("seed", 0))
    return train_fraction, int(seed)


def splits_calibration_set(cfg) -> bool:
    """Whether the resolved configuration actually holds episodes back for calibration."""
    train_fraction, _ = get_calibration_split(cfg)
    return train_fraction < 1.0


def get_calibration_split_indices(dataset, cfg) -> tuple[set[int], set[int]]:
    """Return train/threshold episode indices relative to the calibration episodes.

    Indices are positions in `dataset.iterate_episodes(subset="calibration")` order, the
    same order both the trainers and `BaseEvalClass._process_rollouts` enumerate, so the
    two halves stay consistent between training and thresholding.

    With no split configured (or fewer than two calibration episodes) both returned sets
    are the full index set, i.e. train on everything and calibrate on everything.
    """
    train_fraction, seed = get_calibration_split(cfg)

    metadata = dataset.data["metadata"]
    num_calibration_episodes = int(np.sum(metadata["calibration_rollout_labels"]))
    all_indices = np.arange(num_calibration_episodes)

    if num_calibration_episodes <= 1 or train_fraction >= 1.0:
        all_index_set = set(all_indices.tolist())
        return all_index_set, all_index_set

    if train_fraction <= 0.0:
        raise ValueError("calibration_split.train_fraction must be in (0, 1].")

    rng = np.random.default_rng(seed)
    shuffled_indices = rng.permutation(all_indices)

    num_train = int(np.floor(num_calibration_episodes * train_fraction))
    num_train = min(max(num_train, 1), num_calibration_episodes - 1)

    train_indices = set(shuffled_indices[:num_train].tolist())
    threshold_indices = set(shuffled_indices[num_train:].tolist())

    return train_indices, threshold_indices
