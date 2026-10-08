#!/usr/bin/env python
"""Diversity ranking over the Push-T episodes (SigLIP embeddings + k-center-greedy).

The Push-T counterpart of ``k_greedy.py`` (which is LIBERO-only: it filters by libero
task and reads ``observation.images.image``). Push-T is a single unlabelled task with
one camera, so this embeds the first frame of each of the 206 episodes and ranks them
by k-center-greedy coverage.

The ranking is a property of the DATA, not of any policy, so one file is shared by
every seed group. It is written in the JSON shape that
``iterative_fine_tuning.selection._run_predefined_ranking_selection`` expects: a list
of objects with at least ``episode_id``, consumed ``episodes_per_round`` at a time.

    python scripts/active_learning/k_greedy_pusht.py --device cpu
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import pairwise_distances
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

from lerobot.configs.default import DatasetConfig
from lerobot.datasets.factory import make_dataset
from lerobot.policies.factory import get_policy_class

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

TASK_GROUP, TASK_ID = "pusht", 0


def load_vision_encoder(device):
    name = "google/siglip-so400m-patch14-384"
    logger.info(f"Loading vision encoder: {name}")
    model = AutoModel.from_pretrained(name).vision_model.to(device).eval()
    return model, AutoProcessor.from_pretrained(name)


def embed(model, processor, image_tensor, device):
    if image_tensor.max() <= 1.0:
        image_tensor = (image_tensor * 255).type(torch.uint8)
    inputs = processor(images=image_tensor.cpu(), return_tensors="pt", do_rescale=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs)
    emb = out.pooler_output if hasattr(out, "pooler_output") else out.last_hidden_state.mean(dim=1)
    return emb.float().cpu().numpy().flatten()


def k_center_greedy(embeddings, seed):
    """Rank every episode by greedy max-min coverage. Returns (order, scores)."""
    n = embeddings.shape[0]
    rng = np.random.default_rng(seed)
    first = int(rng.integers(n))
    order, scores = [first], [0.0]
    min_dists = pairwise_distances(embeddings, embeddings[[first]]).min(axis=1)
    for _ in tqdm(range(n - 1), desc="k-center-greedy"):
        nxt = int(np.argmax(min_dists))
        order.append(nxt)
        scores.append(float(min_dists[nxt]))
        d = pairwise_distances(embeddings, embeddings[nxt].reshape(1, -1)).flatten()
        min_dists = np.minimum(min_dists, d)
    return order, scores


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output_path", default="outputs/pusht_diversity_ranking.json")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0,
                   help="seeds the random first pick of k-center-greedy (kept fixed so the "
                        "shared ranking is reproducible)")
    p.add_argument("--image_key", default="observation.image")
    args = p.parse_args()

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.info("CUDA unavailable, falling back to CPU")
        device = "cpu"

    model, processor = load_vision_encoder(device)

    policy_cfg = get_policy_class("flow_matching").config_class()
    dataset_cfg = DatasetConfig(repo_id="lerobot/pusht", pusht_tasks={TASK_GROUP: [TASK_ID]})
    dataset = make_dataset(dataset_cfg=dataset_cfg, policy_cfg=policy_cfg)

    meta = dataset.meta.episodes
    episode_ids = [int(e) for e in meta["episode_index"]]
    starts = [int(i) for i in meta["dataset_from_index"]]
    logger.info(f"Embedding the first frame of {len(episode_ids)} Push-T episodes")

    embeddings, frames = [], []
    for ep_id, start in tqdm(list(zip(episode_ids, starts)), desc="embedding"):
        sample = dataset[start]
        embeddings.append(embed(model, processor, sample[args.image_key], device))
        frames.append(start)
    embeddings = np.stack(embeddings)

    order, scores = k_center_greedy(embeddings, args.seed)

    ranking = [
        {
            "episode_id": episode_ids[pos],
            "task_group": TASK_GROUP,
            "task_id": TASK_ID,
            "instruction": "",
            "frame_index": frames[pos],
            "rank": rank,
            "coverage_score": score,
        }
        for rank, (pos, score) in enumerate(zip(order, scores))
    ]

    out = Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(ranking, indent=2))
    logger.info(f"Wrote {len(ranking)} ranked episodes -> {out}")
    logger.info(f"First 10 episode ids: {[r['episode_id'] for r in ranking[:10]]}")


if __name__ == "__main__":
    main()
