"""Checkpoint discovery, loading and saving for detectors trained on calibration data.

The on-disk format is the one `RNDBase` has always written, kept unchanged so that
existing `rnd_models/**/*.ckpt` files stay loadable:

    {
        "state_dict": <torch state dict>,
        "cfg": {"hparams": {...}, "input_dict": {...}},   # "input_dict" optional
        "kwargs": {...},
    }

A checkpoint is identified by its hyperparameters, not its filename: a directory may
hold several `.ckpt` files and the first one whose stored `hparams` match the requested
ones is used. That is what lets a re-run skip training -- and, just as importantly, what
makes it retrain when a hyperparameter changes instead of silently reusing stale weights.
So whatever influences the trained weights must be part of `desired_hparams`.

`RNDBase` deliberately keeps its own copy of this logic rather than delegating here.
Its matcher has two quirks we do not want to propagate -- a `desired_hparams` key that
appears in neither `hparams` nor `input_dict` inherits the previous key's verdict
instead of counting as a mismatch, and an empty `desired_hparams` raises rather than
accepting any checkpoint -- and rewriting it would put the existing ~24 GB of
`rnd_models/**/*.ckpt` and RND-OE's reproducibility at risk for no gain. The on-disk
format is shared; the matching rules are not.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import torch

from shared_utils.utility_functions import compare_dicts


def _hparams_match(cfg: dict, desired_hparams: dict) -> bool:
    """Whether a checkpoint's stored cfg matches every requested hyperparameter.

    Each requested key is looked up in the checkpoint's "hparams" and, failing that,
    its "input_dict". A key that appears in neither counts as a mismatch: a checkpoint
    that never recorded a hyperparameter cannot be shown to have been trained with the
    requested value.
    """
    stored_hparams = cfg.get("hparams", {}) or {}
    stored_input_dict = cfg.get("input_dict", {}) or {}
    for key, value in desired_hparams.items():
        if key in stored_hparams:
            if not compare_dicts(value, stored_hparams[key]):
                return False
        elif key in stored_input_dict:
            if not compare_dicts(value, stored_input_dict[key]):
                return False
        else:
            return False
    return True


def find_checkpoint(
    checkpoint_dir: str,
    desired_hparams: dict = {},
    checkpoint_name: Optional[str] = None,
) -> Optional[tuple[dict, dict]]:
    """Return `(cfg, checkpoint)` of the first matching checkpoint, or None.

    Args:
        checkpoint_dir: Directory scanned for `.ckpt` files (missing dir -> None).
        desired_hparams: Hyperparameters the checkpoint must have been trained with.
            Empty means "accept the first readable checkpoint".
        checkpoint_name: Optional substring filter on the filename.
    """
    if not os.path.isdir(checkpoint_dir):
        return None

    filenames = sorted(name for name in os.listdir(checkpoint_dir) if name.endswith(".ckpt"))
    if checkpoint_name is not None:
        filenames = [name for name in filenames if checkpoint_name in name]

    for filename in filenames:
        checkpoint = torch.load(os.path.join(checkpoint_dir, filename), weights_only=False)
        cfg = checkpoint.get("cfg", None)
        if not isinstance(cfg, dict) or "hparams" not in cfg:
            continue
        if not desired_hparams or _hparams_match(cfg, desired_hparams):
            return cfg, checkpoint
    return None


def save_checkpoint(
    checkpoint_dir: str,
    state_dict: dict,
    model_cfg: dict,
    checkpoint_name: str,
    overwrite: bool = False,
    **kwargs: Any,
) -> Optional[str]:
    """Write a checkpoint in the format documented above. Returns the path, or None if
    the file already existed and `overwrite` is False."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    if not checkpoint_name.endswith(".ckpt"):
        checkpoint_name += ".ckpt"
    path = os.path.join(checkpoint_dir, checkpoint_name)
    if os.path.exists(path) and not overwrite:
        return None
    torch.save({"state_dict": state_dict, "cfg": model_cfg, "kwargs": kwargs}, path)
    return path
