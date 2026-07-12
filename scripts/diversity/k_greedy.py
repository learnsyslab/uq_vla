#!/usr/bin/env python
import argparse
import csv
import logging
import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import pairwise_distances
from transformers import AutoModel, AutoProcessor

# LeRobot dependencies
from lerobot.configs.default import DatasetConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.utils import filter_libero_episodes
from lerobot.envs.libero import get_task_instruction
from lerobot.policies.factory import get_policy_class

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_vision_encoder(device: str = "cuda"):
    model_name = "google/siglip-so400m-patch14-384"
    logger.info(f"Loading vision encoder: {model_name}")
    vision_model = AutoModel.from_pretrained(model_name).vision_model
    vision_model.to(device)
    vision_model.eval()
    processor = AutoProcessor.from_pretrained(model_name)
    return vision_model, processor

def get_visual_embedding(model, processor, image_tensor, device="cuda"):
    if image_tensor.max() <= 1.0:
        image_tensor = (image_tensor * 255).type(torch.uint8)
    
    inputs = processor(images=image_tensor.cpu(), return_tensors="pt", do_rescale=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs.last_hidden_state.mean(dim=1)
            
    return embedding.cpu().numpy().flatten()

def k_center_greedy(embeddings: np.ndarray):
    n_samples = embeddings.shape[0]
    
    global_center = np.mean(embeddings, axis=0, keepdims=True)
    
    first_idx = np.random.randint(n_samples)
    selected_indices = [first_idx]
    
    selection_scores = [0.0]  
    center_scores = [pairwise_distances(embeddings[[first_idx]], global_center)[0, 0]]

    dists = pairwise_distances(embeddings, embeddings[selected_indices]) 
    min_dists = np.min(dists, axis=1)

    logger.info(f"Running K-Center Greedy selection for {n_samples} episodes...")
    for _ in tqdm(range(n_samples - 1), desc="Ranking"):
        new_idx = np.argmax(min_dists)
        score_u = min_dists[new_idx] 
        
        dist_from_center = pairwise_distances(embeddings[[new_idx]], global_center)[0, 0]
        
        selected_indices.append(new_idx)
        selection_scores.append(score_u)
        center_scores.append(dist_from_center)
        
        new_dist_vec = pairwise_distances(embeddings, embeddings[new_idx].reshape(1, -1)).flatten()
        min_dists = np.minimum(min_dists, new_dist_vec)
        
    return selected_indices, selection_scores, center_scores

def main():
    default_model = "outputs/train/smolvla/libero_base_2/checkpoints/last/pretrained_model"

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=default_model)
    parser.add_argument("--output_path", type=str, default="outputs/active_learning/libero10_diversity_kcenter.csv")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        logger.error(f"Model path not found: {args.model_path}")
        return

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Models & Config
    vision_model, processor = load_vision_encoder(args.device)
    
    logger.info(f"Loading policy config from local: {args.model_path}")
    policy_cls = get_policy_class("smolvla")
    policy_cfg = policy_cls.from_pretrained(args.model_path).config

    all_embeddings, all_metadata = [], []
    task_group, num_tasks = "libero_10", 10 

    # 2. Data Collection
    for task_id in range(num_tasks):
        logger.info(f"Processing Task {task_id}")
        dataset_cfg = DatasetConfig(repo_id="HuggingFaceVLA/libero", libero_tasks={task_group: [task_id]})
        dataset = make_dataset(dataset_cfg=dataset_cfg, policy_cfg=policy_cfg)
        
        episode_ids = filter_libero_episodes(dataset=dataset, tasks_to_use={task_group: [task_id]})
        episodes_meta = dataset.meta.episodes
        all_ep_indices = list(episodes_meta["episode_index"])
        ep_id_to_pos = {ep_id: pos for pos, ep_id in enumerate(all_ep_indices)}

        for ep_id in tqdm(episode_ids, desc=f"Task {task_id}", leave=False):
            pos = ep_id_to_pos[ep_id]
            frame_idx = int(episodes_meta["dataset_from_index"][pos])
            
            sample = dataset[frame_idx]
            image_tensor = sample["observation.images.image"]
            emb = get_visual_embedding(vision_model, processor, image_tensor, device=args.device)
            
            all_embeddings.append(emb)
            all_metadata.append({
                "task_id": task_id, 
                "episode_id": ep_id, 
                "instr": get_task_instruction(task_group, task_id)
            })

    # 3. K-Center Rank with Scores
    X = np.stack(all_embeddings)
    rank_indices, selection_scores, center_scores = k_center_greedy(X)

    # 4. Save
    header = ["rank", "task_id", "episode_id", "task_instruction", "selection_distance", "center_distance"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        
        for rank, (idx, sel_dist, cent_dist) in enumerate(zip(rank_indices, selection_scores, center_scores), 1):
            m = all_metadata[idx]
            writer.writerow({
                "rank": rank, 
                "task_id": m["task_id"], 
                "episode_id": m["episode_id"], 
                "task_instruction": m["instr"],
                "selection_distance": f"{sel_dist:.6f}",
                "center_distance": f"{cent_dist:.6f}"
            })

    logger.info(f"Successfully saved results to {output_path}")

if __name__ == "__main__":
    main()