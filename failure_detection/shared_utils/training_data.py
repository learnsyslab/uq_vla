"""Where a learned failure detector (RND-*, logpZO) gets its training data from.

Two sources, selected by the `training_data` key of a method's eval config (defined once in
`configs/eval/base.yaml`, overridden per experiment through `pipeline.py`):

* "demos"       -- the policy's own training demonstrations, every frame embedded offline with the
                   policy's observation encoder (active_learning/scripts/failure_detection/run_{pusht,libero_plus}.sh)
                   and stored at `<task_data_path>/<demo_embeddings_file>`.
                   This is the FAIL-Detect setting for both logpZO and RND. Nothing is fit on
                   calibration rollouts, so the whole calibration set is available for thresholds.
* "calibration" -- FIPER's calibration-only protocol: fit on the training half of the calibration
                   rollouts (shared_utils.calibration_split), threshold on the other half. One model
                   per task.
* "calibration_pooled" -- same data, one model per *policy*: fit once on the union of the training
                   halves of every task the pipeline evaluates (pipeline.py builds all task datasets
                   first and passes them as `pooled_datasets`), stored under `pooled_models_dir` and
                   reused by every task; each task still thresholds on its own held-out half.

Both logpZO and the RND trainer/evaluator go through these helpers so the semantics, the file
format and the checkpoint-identity keys are defined exactly once.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

import torch
from omegaconf import DictConfig, open_dict

# "demos_pooled" (multi-task policies such as SmolVLA on LIBERO): one demo-trained model per *policy*, fit on
# the policy's whole multi-task training set. The demo-embedding file lives one level above the task
# dirs (<suite>/<demo_embeddings_file>, e.g. data/libero_10/demo_embeddings/...) and the model in
# <suite>/pooled_models/ (never cleared: like `demos`, the fit does not depend on the calibration set).
# Embeddings whose width differs per task (SmolVLA KV cache) are stored pre-padded by the extractor;
# `demo_aligner` pads each task's rollouts to that width with the same rule (`pooled_alignment` block).
# Thresholds stay per task on the task's full calibration set.
TRAINING_DATA_CHOICES = ("demos", "demos_pooled", "calibration", "calibration_pooled")
# Experiment-level keys that pipeline.py forwards into every method's eval config.
METHOD_CFG_OVERRIDE_KEYS = ("training_data", "demo_embeddings_file", "pooled_alignment")


def resolve_training_data(cfg) -> str:
    value = str(cfg.get("training_data", "calibration"))
    if value not in TRAINING_DATA_CHOICES:
        raise ValueError(f"training_data must be one of {TRAINING_DATA_CHOICES}, got {value!r}")
    return value


def trains_on_calibration_data(cfg) -> bool:
    """Only a detector fit on calibration rollouts has in-sample calibration scores to hold out."""
    return resolve_training_data(cfg) in ("calibration", "calibration_pooled")


def is_pooled(cfg) -> bool:
    """training_data=calibration_pooled: the model is fit on the union of the pooled tasks' calibration halves."""
    return resolve_training_data(cfg) == "calibration_pooled"


def is_demo_trained(cfg) -> bool:
    """Fit on the policy's demonstrations (per task or per policy) rather than on calibration rollouts."""
    return resolve_training_data(cfg) in ("demos", "demos_pooled")


def is_demo_pooled(cfg) -> bool:
    return resolve_training_data(cfg) == "demos_pooled"


def uses_pooled_models_dir(cfg) -> bool:
    """One model per policy, stored under pooled_models_dir instead of <task>/…_models."""
    return resolve_training_data(cfg) in ("calibration_pooled", "demos_pooled")


def pooled_models_dir(kwargs: dict, task_data_path: str) -> str:
    """Where a pooled model lives (pipeline.py passes `pooled_models_dir`; falls back to the suite dir)."""
    d = kwargs.get("pooled_models_dir")
    return str(d) if d else os.path.join(os.path.dirname(task_data_path), "pooled_models")


class KVCacheAligner:
    """Pads SmolVLA-style key/value-cache embeddings to a common token count.

    The flat embedding is `cat(keys, values)`, each `(kv_heads, tokens, head_dim)` head-major, with
    the prefix ordered [image tokens][instruction tokens][state tokens]. Only the instruction length
    differs between tasks, so the aligner inserts all-zero tokens right after the instruction block
    until every embedding has `target_tokens` tokens: image blocks, the start of the instruction and
    the state token then sit at identical offsets for every task (mode `pad_language`).
    """

    def __init__(self, kv_heads: int, head_dim: int, image_tokens: int, state_tokens: int,
                 target_tokens: int, mode: str = "pad_language"):
        if mode != "pad_language":
            raise NotImplementedError(f"pooled_alignment.mode={mode!r}; only 'pad_language' is implemented")
        self.kv_heads, self.head_dim = int(kv_heads), int(head_dim)
        self.image_tokens, self.state_tokens = int(image_tokens), int(state_tokens)
        self.target_tokens, self.mode = int(target_tokens), mode
        self.per_token = 2 * self.kv_heads * self.head_dim
        self.target_dim = self.target_tokens * self.per_token

    def tokens_of(self, width: int) -> int:
        if width % self.per_token:
            raise ValueError(f"embedding width {width} is not a multiple of {self.per_token} "
                             f"(2 x {self.kv_heads} heads x {self.head_dim} dims)")
        return width // self.per_token

    def identity(self) -> dict:
        return {"alignment_mode": self.mode, "alignment_target_tokens": self.target_tokens,
                "alignment_layout": [self.kv_heads, self.head_dim, self.image_tokens, self.state_tokens]}

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        squeeze = x.ndim == 1
        if squeeze:
            x = x.unsqueeze(0)
        n, width = x.shape[0], x.shape[-1]
        tokens = self.tokens_of(width)
        pad = self.target_tokens - tokens
        if pad < 0:
            raise ValueError(f"embedding has {tokens} tokens, more than the alignment target {self.target_tokens}")
        if pad == 0:
            return x.squeeze(0) if squeeze else x
        half = width // 2
        cut = self.image_tokens + (tokens - self.image_tokens - self.state_tokens)  # end of the instruction block
        out = []
        for part in (x[..., :half], x[..., half:]):
            part = part.reshape(n, self.kv_heads, tokens, self.head_dim)
            zeros = part.new_zeros((n, self.kv_heads, pad, self.head_dim))
            part = torch.cat([part[:, :, :cut], zeros, part[:, :, cut:]], dim=2)
            out.append(part.reshape(n, -1))
        y = torch.cat(out, dim=-1)
        return y.squeeze(0) if squeeze else y


def pooled_aligner(cfg, pooled_datasets: dict):
    """The embedding aligner for training_data=calibration_pooled, or None when the experiment
    config has no `pooled_alignment` block (embedding widths must then already agree)."""
    spec = cfg.get("pooled_alignment", None)
    if not spec:
        return None
    spec = dict(spec)
    per_token = 2 * int(spec["kv_heads"]) * int(spec["head_dim"])
    widths = [int(ds.get_tensor_shape("obs_embeddings")[-1]) for ds in pooled_datasets.values()]
    target = max(widths) // per_token
    return KVCacheAligner(spec["kv_heads"], spec["head_dim"], spec["image_tokens"], spec.get("state_tokens", 1),
                          target_tokens=target, mode=spec.get("mode", "pad_language"))


def pooled_training_tensors(pooled_datasets: dict, cfg, required_tensors, optional_tensors=None,
                            required_actions=None, optional_actions=None, history=1,
                            normalize_tensors=None, tag: str = "pooled") -> tuple[dict, dict]:
    """Union of the calibration *training* halves of every pooled task.

    `pooled_datasets` maps task label -> ProcessedRolloutDataset (pipeline.py). Each task is split
    with the same episode-level rule its own thresholding uses (get_calibration_split_indices), so
    no episode is ever both trained on and thresholded on. Returns `(tensors_by_key, identity)`;
    the identity (which tasks, how many training episodes per task, the split) goes into the
    checkpoint hparams so a pooled model is never confused with a per-task one.
    """
    from shared_utils.calibration_split import get_calibration_split, get_calibration_split_indices

    if not pooled_datasets:
        raise ValueError("training_data=calibration_pooled but no pooled_datasets were passed "
                         "(pipeline.py builds them when the experiment sets training_data=calibration_pooled).")
    aligner = pooled_aligner(cfg, pooled_datasets)
    tensors_by_key: dict[str, list[torch.Tensor]] = {}
    episodes_per_task: dict[str, int] = {}
    for label in sorted(pooled_datasets):
        dataset = pooled_datasets[label]
        train_indices, _ = get_calibration_split_indices(dataset, cfg)
        n = 0
        # Action kwargs only when the caller has them (RND): passing None makes datasets with an
        # action mapping look up an action called "None".
        action_kwargs = {k: v for k, v in (("required_actions", required_actions),
                                           ("optional_actions", optional_actions)) if v is not None}
        for episode_idx, episode_data in enumerate(
            dataset.iterate_episodes(
                subset="calibration",
                required_tensors=list(required_tensors),
                optional_tensors=list(optional_tensors or []),
                history=history,
                normalize_tensors=dict(normalize_tensors or {}),
                **action_kwargs,
            )
        ):
            if episode_idx not in train_indices:
                continue
            n += 1
            for key, tensor in episode_data.items():
                if key == "obs_embeddings" and aligner is not None:
                    tensor = aligner(tensor)
                tensors_by_key.setdefault(key, []).append(tensor)
        episodes_per_task[label] = n
    if not tensors_by_key:
        raise ValueError("pooled calibration training split produced no training episodes.")
    out = {key: torch.cat(tensors, dim=0) for key, tensors in tensors_by_key.items()}
    train_fraction, split_seed = get_calibration_split(cfg)
    identity = {
        "pooled_tasks": sorted(pooled_datasets),
        "pooled_train_episodes_per_task": [episodes_per_task[k] for k in sorted(pooled_datasets)],
        "calibration_train_fraction": float(train_fraction),
        "calibration_split_seed": int(split_seed),
    }
    if aligner is not None:
        identity.update(aligner.identity())
    n_rows = next(iter(out.values())).shape[0]
    print(f"[{tag}] pooled training set: {sum(episodes_per_task.values())} calibration episodes from "
          f"{len(pooled_datasets)} tasks -> {n_rows} samples ({', '.join(f'{k}:{v}' for k, v in episodes_per_task.items())})")
    return out, identity


def apply_method_cfg_overrides(
    cfg: DictConfig, overrides: dict[str, Any] | None, method_name: str | None = None
) -> DictConfig:
    """Set experiment-level keys on a per-method eval config.

    `load_config("eval", <method>)` recomposes the *default* config, so settings from the chosen
    experiment file never reach the method configs on their own; pipeline.py reads them up front
    and passes them here. Interpolations such as `${eval.training_data}` inside `hparams.model`
    see the new value because the subconfig is still attached to its root.

    Two kinds of entries: flat keys (`training_data`, ...) are set on every method's config; the
    special key `method_overrides` holds a `{method_name: {key: value}}` mapping whose entry for
    `method_name` is set on that method's config only (e.g. `bayesian: {num_scorers: 3}` for a
    scorer that stored three ensemble members). Values of None are ignored.
    """
    if not overrides:
        return cfg
    per_method = overrides.get("method_overrides") or {}
    with open_dict(cfg):
        for key, value in overrides.items():
            if key != "method_overrides" and value is not None:
                cfg[key] = value
        if method_name is not None and method_name in per_method:
            for key, value in per_method[method_name].items():
                if value is not None:
                    cfg[key] = value
    return cfg


def demo_embeddings_path(task_data_path: str, cfg) -> str:
    """demos: <task>/<file> (one demo set per task). demos_pooled: <suite>/<file>, one level up, shared
    by every task of the policy (for LIBERO the suite dir is the symlinked scoring run dir, so it is
    per policy as required)."""
    root = os.path.dirname(os.path.normpath(task_data_path)) if is_demo_pooled(cfg) else task_data_path
    return os.path.join(root, str(cfg.demo_embeddings_file))


@lru_cache(maxsize=8)
def _load(path: str) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def load_demo_embeddings(task_data_path: str, cfg) -> dict:
    path = demo_embeddings_path(task_data_path, cfg)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"training_data=demos but no demo embeddings at {path}. Produce them with "
            "active_learning/scripts/failure_detection/run_{pusht,libero_plus}.sh, or set training_data=calibration."
        )
    return _load(os.path.abspath(path))


def demo_training_identity(meta: dict) -> dict:
    """Checkpoint-identity keys for a demo-trained model: which episodes, how many frames, which
    encoder. A model fit on a different demo set or embedded by a different policy is a different
    model and must not be silently reused."""
    identity = {
        "num_demo_episodes": len(meta["episodes"]),
        "num_demo_frames": int(meta["embeddings"].shape[0]),
        "demo_episodes_sha1": str(meta["episodes_sha1"]),
        "demo_policy_path": str(meta["policy_path"]),
    }
    # Payloads of extract_demo_embeddings.py --frame_stride / --pad_kv_tokens (demos_pooled) carry how the
    # frames were subsampled and padded; both change the fitted weights. Older payloads lack the keys, so
    # the identity of every existing per-task demo checkpoint is unchanged.
    if meta.get("frame_stride", 1) not in (None, 1):
        identity["demo_frame_stride"] = int(meta["frame_stride"])
    if meta.get("alignment"):
        identity["demo_alignment"] = {k: (list(v) if isinstance(v, (list, tuple)) else v)
                                      for k, v in dict(meta["alignment"]).items()}
    return identity


def demo_aligner(cfg, meta: dict):
    """Aligner that pads a task's rollout embeddings to the width of a *pooled* demo-embedding file
    (training_data=demos_pooled), or None when the experiment has no `pooled_alignment` block (then
    every task's rollouts must already have the demo width). The extractor pads the demonstrations
    with the same rule (instruction block zero-padded to a common token count), so the layout block
    of the experiment config must agree with what the payload records."""
    spec = cfg.get("pooled_alignment", None)
    if not spec:
        return None
    spec = dict(spec)
    per_token = 2 * int(spec["kv_heads"]) * int(spec["head_dim"])
    width = int(meta["embedding_dim"])
    if width % per_token:
        raise ValueError(f"pooled demo embeddings are {width}-dim, not a multiple of {per_token} floats per token")
    aligner = KVCacheAligner(spec["kv_heads"], spec["head_dim"], spec["image_tokens"], spec.get("state_tokens", 1),
                             target_tokens=width // per_token, mode=spec.get("mode", "pad_language"))
    recorded = meta.get("alignment")
    if recorded:
        mine = {"kv_heads": aligner.kv_heads, "head_dim": aligner.head_dim, "image_tokens": aligner.image_tokens,
                "state_tokens": aligner.state_tokens, "target_tokens": aligner.target_tokens, "mode": aligner.mode}
        mismatch = {k: (recorded.get(k), v) for k, v in mine.items() if k in recorded and recorded[k] != v}
        if mismatch:
            raise ValueError(f"pooled_alignment disagrees with the demo payload's padding {mismatch} "
                             f"(payload -> config); the rollouts would be padded differently from the demos")
    return aligner


def demo_training_tensor(task_data_path: str, cfg, expected_dim: int, device, tag: str) -> torch.Tensor:
    """The (N, D) demo-embedding matrix, checked against the recorded rollouts' embedding width."""
    meta = load_demo_embeddings(task_data_path, cfg)
    emb = meta["embeddings"]
    if emb.shape[1] != expected_dim:
        raise ValueError(
            f"demo embeddings are {emb.shape[1]}-dim but the recorded rollouts are {expected_dim}-dim "
            f"({demo_embeddings_path(task_data_path, cfg)})"
        )
    print(f"[{tag}] training on {emb.shape[0]} demo-frame embeddings from "
          f"{len(meta['episodes'])} episodes (sha1 {meta['episodes_sha1']})")
    return emb.to(device)
