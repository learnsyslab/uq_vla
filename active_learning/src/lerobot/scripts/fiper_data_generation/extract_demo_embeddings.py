"""Embed a policy's *training demonstrations* with its own observation encoder.

Produces the training set FAIL-Detect's logpZO is fit on (`save_data.py` +
`train_diffusion_unet_hybrid_workspace_get_data.py` in the reference repo): every frame of the
chosen demonstration episodes, run through the policy's preprocessor and observation encoder,
saved as the flat `global_cond` vector. The FIPER recorder stores the very same quantity for
rollout observations (`FlowMatchingAdapter.prepare_fiper_obs_embedding`), so a detector trained on
these embeddings scores rollouts in the same space.

The frame -> observation -> conditioning path is the one used for demo-frame uncertainty scoring
in `iterative_fine_tuning.selection` (verified there to reproduce the policy's own conditioning);
the policy is in eval mode, so images are center-cropped exactly as at rollout time.

    PYTHONPATH=src python src/lerobot/scripts/fiper_data_generation/extract_demo_embeddings.py \
        --policy_path outputs/pretrain/pusht_all_30k_bs128_rerun/seed_0/checkpoints/030000/pretrained_model \
        --repo_id lerobot/pusht --episodes all \
        --output outputs/fiper_rollout_scoring/pusht_all206_s01/demo_embeddings/member_00_global_cond.pt

    --episodes all | 0,1,2,... | @<json_path>:<key>   (json: {key: [episode ids]})
    Repeated ids are kept: an episode listed k times contributes its frames k times (a policy
    fine-tuned on a demo multiset, e.g. the AL replays, is mirrored by the same multiset).
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from lerobot.configs.default import DatasetConfig
from lerobot.datasets.factory import make_dataset
from lerobot.policies.factory import (
    get_policy_class,
    make_flow_matching_adapter_from_policy,
    make_pre_post_processors,
)
from iterative_fine_tuning.selection import _prepare_observation_batch_for_sampler


def parse_episodes(spec: str) -> list[int] | None:
    if spec == "all":
        return None
    if spec.startswith("@"):
        path, _, key = spec[1:].partition(":")
        subsets = json.loads(Path(path).read_text())
        if key not in subsets:
            raise KeyError(f"{key!r} not in {path}; available: {sorted(subsets)}")
        return sorted(int(e) for e in subsets[key])  # duplicates kept
    return sorted(int(e) for e in spec.replace(" ", "").split(",") if e)


def _pad_kv_tokens(x: torch.Tensor, kv_heads: int, head_dim: int, image_tokens: int, state_tokens: int,
                   target_tokens: int) -> torch.Tensor:
    """Zero-pad one flat KV-cache embedding ([keys][values], each (kv_heads, tokens, head_dim) head-major,
    prefix = [image tokens][instruction tokens][state tokens]) to `target_tokens` tokens by inserting
    all-zero tokens right after the instruction block. Mirrors fiper_vfd KVCacheAligner (mode
    pad_language) exactly, so demos and rollouts are padded identically."""
    per_token = 2 * kv_heads * head_dim
    width = x.numel()
    if width % per_token:
        raise ValueError(f"embedding width {width} is not a multiple of {per_token} floats per token")
    tokens = width // per_token
    pad = target_tokens - tokens
    if pad < 0:
        raise ValueError(f"frame has {tokens} tokens, more than target_tokens={target_tokens}")
    if pad == 0:
        return x
    half = width // 2
    cut = image_tokens + (tokens - image_tokens - state_tokens)  # end of the instruction block
    out = []
    for part in (x[:half], x[half:]):
        part = part.reshape(kv_heads, tokens, head_dim)
        zeros = part.new_zeros((kv_heads, pad, head_dim))
        out.append(torch.cat([part[:, :cut], zeros, part[:, cut:]], dim=1).reshape(-1))
    return torch.cat(out)


def _index_columns(dataset, episodes: list[int] | None):
    """(episode_index, frame_index, task_index or None) per row of `dataset.hf_dataset`, from the parquet
    files (column projection only, no image bytes), in the same order the fork's load_nested_dataset
    concatenates them (sorted `data/*/*.parquet`), restricted to `episodes` when given."""
    import numpy as np
    import pyarrow.parquet as pq
    paths = sorted((dataset.root / "data").glob("*/*.parquet"))
    cols = ["episode_index", "frame_index"]
    has_task = "task_index" in pq.read_schema(paths[0]).names
    if has_task:
        cols.append("task_index")
    tables = [pq.read_table(p, columns=cols) for p in paths]
    ep = np.concatenate([t.column("episode_index").to_numpy() for t in tables])
    fr = np.concatenate([t.column("frame_index").to_numpy() for t in tables])
    task = np.concatenate([t.column("task_index").to_numpy() for t in tables]) if has_task else None
    if episodes is not None:
        keep = np.isin(ep, np.asarray(sorted(set(episodes))))
        ep, fr = ep[keep], fr[keep]
        task = task[keep] if task is not None else None
    return ep, fr, task


def _stack_padded(per_frame: list[torch.Tensor], pad_spec: str | None, target_tokens: int | None):
    """Stack per-frame embeddings; pad them to a common token count first when their widths differ."""
    widths = Counter(int(t.numel()) for t in per_frame)
    if pad_spec is None:
        if len(widths) > 1:
            raise ValueError(f"frames have {len(widths)} different embedding widths {dict(widths)}; pass "
                             "--pad_kv_tokens kv_heads,head_dim,image_tokens,state_tokens to pad them")
        return torch.stack(per_frame), None
    kv_heads, head_dim, image_tokens, state_tokens = (int(v) for v in pad_spec.split(","))
    per_token = 2 * kv_heads * head_dim
    tokens_seen = sorted(w // per_token for w in widths)
    target = max(tokens_seen) if target_tokens is None else int(target_tokens)
    padded = torch.stack([_pad_kv_tokens(t, kv_heads, head_dim, image_tokens, state_tokens, target) for t in per_frame])
    alignment = {"kv_heads": kv_heads, "head_dim": head_dim, "image_tokens": image_tokens,
                 "state_tokens": state_tokens, "target_tokens": target, "mode": "pad_language",
                 "tokens_seen": tokens_seen, "frames_per_width": {int(w // per_token): int(c) for w, c in widths.items()}}
    print(f"padded {len(per_frame)} frames with token counts {dict(alignment['frames_per_width'])} to {target} tokens "
          f"({padded.shape[1]} floats)")
    return padded, alignment


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy_path", required=True)
    p.add_argument("--repo_id", default="lerobot/pusht")
    p.add_argument("--root", default=None, help="local dataset directory (offline); default: the HF cache")
    p.add_argument("--video_backend", default=None,
                   help="frame decoder: torchcodec (default when importable) or pyav. Use pyav where "
                        "torchcodec cannot find the FFmpeg shared libraries; both decode to identical frames.")
    p.add_argument("--episodes", default="all")
    p.add_argument("--output", required=True)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--state_dims", type=int, default=None,
                   help="keep only the first N entries of observation.state before preprocessing "
                        "(default: the full state).")
    p.add_argument("--frame_stride", type=int, default=1,
                   help="embed only every N-th frame of each episode (frame_index %% N == 0). The multi-task "
                        "LIBERO set has 273k frames; every 10th gives ~27k, the reference's data regime.")
    p.add_argument("--pad_kv_tokens", default=None,
                   help="'kv_heads,head_dim,image_tokens,state_tokens' (SmolVLA LIBERO: 5,64,32,1). SmolVLA's "
                        "FIPER embedding is the flattened KV cache, 2 x kv_heads x head_dim floats per prefix "
                        "token, and the instruction length differs per task, so frames of different tasks have "
                        "different widths. With this option every frame is zero-padded right after its "
                        "instruction block to a common token count (fiper_vfd KVCacheAligner 'pad_language' "
                        "rule, which fiper_vfd then applies to the rollouts), so one model can be fit on all "
                        "tasks (training_data=demos_pooled). Required whenever widths differ.")
    p.add_argument("--target_tokens", type=int, default=None,
                   help="pad to this many tokens instead of the longest frame seen (must not be smaller).")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    episodes = parse_episodes(args.episodes)
    unique_episodes = None if episodes is None else sorted(set(episodes))
    multiplicity = Counter(episodes or [])

    # The policy class follows the checkpoint (flow_matching for Push-T, smolvla for LIBERO).
    policy_type = json.load(open(Path(args.policy_path) / "config.json"))["type"]
    policy_cls = get_policy_class(policy_type)
    policy = policy_cls.from_pretrained(pretrained_name_or_path=args.policy_path)
    policy.eval()
    policy.to(args.device)
    policy.eval()  # center crop, as at rollout time
    # The saved pipeline pins the training device; run it wherever we are (CPU smoke tests included).
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=policy.config, pretrained_path=args.policy_path,
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )
    adapter = make_flow_matching_adapter_from_policy(policy)

    dataset_kwargs = {} if args.video_backend is None else {"video_backend": args.video_backend}
    dataset = make_dataset(
        dataset_cfg=DatasetConfig(repo_id=args.repo_id, root=args.root, episodes=unique_episodes, **dataset_kwargs),
        policy_cfg=policy.config,
    )
    n = len(dataset)
    print(f"dataset {args.repo_id}: {n} frames over {dataset.num_episodes} episodes "
          f"({'all' if episodes is None else f'{len(unique_episodes)} unique of {len(episodes)}'} requested)")
    stride = max(1, int(args.frame_stride))
    # Per-row episode / frame / task indices, read straight from the parquet files with column projection.
    # NOT `dataset.hf_dataset["frame_index"]`: on the 273k-row LIBERO table (1,150 memory-mapped parquet
    # blocks with PNG columns) that whole-column access ran > 80 min at 100 % CPU (py-spy: table.select
    # rebuilding every block), while the projected parquet read takes seconds. Row order of the HF table =
    # the sorted file order load_nested_dataset uses (episode filter applied per row); verified below
    # against dataset[i] on a sample of rows.
    ep_col, fr_col, task_col = _index_columns(dataset, unique_episodes)
    if len(ep_col) != n:
        raise RuntimeError(f"parquet index columns have {len(ep_col)} rows but the dataset has {n}")
    for i in sorted(set([0, n - 1] + [int(x) for x in torch.randint(0, n, (12,))])):
        row = dataset.hf_dataset[i]
        if int(row["episode_index"]) != int(ep_col[i]) or int(row["frame_index"]) != int(fr_col[i]):
            raise RuntimeError(f"row {i}: parquet columns (ep {ep_col[i]}, frame {fr_col[i]}) disagree with the "
                               f"dataset (ep {int(row['episode_index'])}, frame {int(row['frame_index'])})")
    indices = [i for i in range(n) if fr_col[i] % stride == 0] if stride > 1 else list(range(n))
    print(f"embedding {len(indices)} of {n} frames (frame_stride {stride})")
    # Batches never straddle tasks: SmolVLA pads the instruction to the longest in the batch
    # (pad_language_to=longest), so a mixed batch would give shorter instructions extra pad tokens in the
    # KV cache and an embedding that depends on the batch, not the task. Within one task the instruction
    # is identical, so grouping by task_index keeps every batch pad-free -- exactly the recorder's batch-1 case.
    task_of = task_col if task_col is not None else np.zeros(n, dtype=np.int64)
    groups: dict[int, list[int]] = {}
    for i in indices:
        groups.setdefault(int(task_of[i]), []).append(i)
    batches = [g[s:s + args.batch_size] for g in groups.values() for s in range(0, len(g), args.batch_size)]
    print(f"{len(groups)} task group(s), {len(batches)} batches of <= {args.batch_size}")

    def reduce_state(frame: dict) -> dict:
        if args.state_dims is not None and "observation.state" in frame:
            frame["observation.state"] = frame["observation.state"][..., : args.state_dims]
        return frame

    # Two embedding paths, both identical to what the FIPER recorder stores for a rollout step:
    #  * flow-matching policies (Push-T): the flat `global_cond`, batched;
    #  * VLA policies (SmolVLA, X-VLA): `adapter.prepare_fiper_obs_embedding(conditioning, batch_index=b)`
    #    reads element b out of one batched prefix pass (~20x faster than a pass per frame).
    probe = adapter.prepare_conditioning(
        _prepare_observation_batch_for_sampler([preprocessor(reduce_state(dataset[0]))], args.device),
        num_action_samples=1,
    )
    batched_global_cond = "global_cond" in probe
    source = ("preprocessor -> FlowMatchingAdapter.prepare_conditioning(global_cond)" if batched_global_cond
              else "preprocessor -> adapter.prepare_conditioning -> adapter.prepare_fiper_obs_embedding (batched)")
    print(f"embedding path: {source}")

    embeddings, per_frame, episode_index, frame_index = [], [], [], []
    t0 = time.time()
    with torch.no_grad():
        for batch_indices in tqdm(batches, desc="embedding demo frames", unit="batch"):
            frames = [reduce_state(dataset[i]) for i in batch_indices]
            processed = [preprocessor(f) for f in frames]
            observation = _prepare_observation_batch_for_sampler(processed, args.device)
            conditioning = adapter.prepare_conditioning(observation, num_action_samples=1)
            if batched_global_cond:
                embeddings.append(conditioning["global_cond"].float().cpu())
            else:
                # One prefix pass for the whole batch, then read each element's embedding out of it.
                # Kept per frame: with a multi-task dataset the widths differ per task (see --pad_kv_tokens).
                per_frame.extend(
                    torch.as_tensor(adapter.prepare_fiper_obs_embedding(conditioning, batch_index=b)).float()
                    for b in range(len(processed))
                )
            episode_index.extend(int(f["episode_index"]) for f in frames)
            frame_index.extend(int(f["frame_index"]) for f in frames)

    alignment = None
    if batched_global_cond:
        embeddings = torch.cat(embeddings, dim=0)
    else:
        embeddings, alignment = _stack_padded(per_frame, args.pad_kv_tokens, args.target_tokens)
    episode_index = torch.tensor(episode_index, dtype=torch.long)
    frame_index = torch.tensor(frame_index, dtype=torch.long)
    if episodes is not None and len(episodes) > len(unique_episodes):
        # Repeat each frame as often as its episode appears in the requested multiset.
        repeats = torch.tensor([multiplicity[int(e)] for e in episode_index], dtype=torch.long)
        embeddings = embeddings.repeat_interleave(repeats, dim=0)
        episode_index = episode_index.repeat_interleave(repeats)
        frame_index = frame_index.repeat_interleave(repeats)
    # Multiset (sorted, repeats kept) so the sha1 identity distinguishes repeat structures;
    # identical to the old sorted-unique list when nothing repeats.
    episode_ids = sorted(episodes) if episodes is not None else sorted(set(episode_index.tolist()))
    payload = {
        "embeddings": embeddings,
        "episode_index": episode_index,
        "frame_index": frame_index,
        "episodes": episode_ids,
        "unique_episodes": sorted(set(episode_ids)),
        "episodes_sha1": hashlib.sha1(",".join(map(str, episode_ids)).encode()).hexdigest()[:12],
        "repo_id": args.repo_id,
        "policy_path": str(args.policy_path),
        "policy_type": policy.config.type,
        "policy_mode": "eval (center crop)",
        "embedding_dim": int(embeddings.shape[1]),
        "source": f"extract_demo_embeddings.py: {source}",
        "state_dims": args.state_dims,
        "video_backend": dataset.video_backend,
        # Both enter fiper_vfd's checkpoint identity (demo_training_identity) when set / non-trivial.
        "frame_stride": stride,
        "num_frames_total": n,
        "alignment": alignment,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    print(f"saved {tuple(embeddings.shape)} embeddings from {len(episode_ids)} episodes "
          f"({len(set(episode_ids))} unique) "
          f"(sha1 {payload['episodes_sha1']}) to {out} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
