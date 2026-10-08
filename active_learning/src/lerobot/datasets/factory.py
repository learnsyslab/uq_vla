#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import random
from pprint import pformat

import torch

from lerobot.configs.default import DatasetConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
    MultiLeRobotDataset,
)
from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset
from lerobot.datasets.transforms import ImageTransforms
from lerobot.datasets.utils import patch_dataset_episode_boundaries
from lerobot.utils.constants import ACTION, OBS_PREFIX, REWARD

IMAGENET_STATS = {
    "mean": [[[0.485]], [[0.456]], [[0.406]]],  # (c,1,1)
    "std": [[[0.229]], [[0.224]], [[0.225]]],  # (c,1,1)
}


def resolve_delta_timestamps(
    cfg: PreTrainedConfig, ds_meta: LeRobotDatasetMetadata
) -> dict[str, list] | None:
    """Resolves delta_timestamps by reading from the 'delta_indices' properties of the PreTrainedConfig.

    Args:
        cfg (PreTrainedConfig): The PreTrainedConfig to read delta_indices from.
        ds_meta (LeRobotDatasetMetadata): The dataset from which features and fps are used to build
            delta_timestamps against.

    Returns:
        dict[str, list] | None: A dictionary of delta_timestamps, e.g.:
            {
                "observation.state": [-0.04, -0.02, 0]
                "observation.action": [-0.02, 0, 0.02]
            }
            returns `None` if the resulting dict is empty.
    """
    delta_timestamps = {}
    for key in ds_meta.features:
        if key == REWARD and cfg.reward_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.reward_delta_indices]
        if key == ACTION and cfg.action_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.action_delta_indices]
        if key.startswith(OBS_PREFIX) and cfg.observation_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.observation_delta_indices]

    if len(delta_timestamps) == 0:
        delta_timestamps = None

    return delta_timestamps


def make_dataset(
    dataset_cfg: DatasetConfig,
    policy_cfg: PreTrainedConfig,
    num_workers: int = 16,
) -> LeRobotDataset | MultiLeRobotDataset:
    """
    Create a LeRobot dataset instance with proper delta timestamp resolution and optional image transforms.

    Args:
        dataset_cfg: Dataset configuration.
        policy_cfg: Policy configuration used to resolve delta timestamps.
        num_workers: Number of workers for streaming datasets.

    Returns:
        LeRobotDataset | MultiLeRobotDataset: A dataset object ready for training or evaluation.
    """
    image_transforms = (
        ImageTransforms(dataset_cfg.image_transforms) if dataset_cfg.image_transforms.enable else None
    )

    if isinstance(dataset_cfg.repo_id, str):
        ds_meta = LeRobotDatasetMetadata(
            dataset_cfg.repo_id, root=dataset_cfg.root, revision=dataset_cfg.revision
        )
        delta_timestamps = resolve_delta_timestamps(policy_cfg, ds_meta)
        if not dataset_cfg.streaming:
            dataset = LeRobotDataset(
                dataset_cfg.repo_id,
                root=dataset_cfg.root,
                episodes=dataset_cfg.episodes,
                delta_timestamps=delta_timestamps,
                image_transforms=image_transforms,
                revision=dataset_cfg.revision,
                video_backend=dataset_cfg.video_backend,
            )
        else:
            dataset = StreamingLeRobotDataset(
                dataset_cfg.repo_id,
                root=dataset_cfg.root,
                episodes=dataset_cfg.episodes,
                delta_timestamps=delta_timestamps,
                image_transforms=image_transforms,
                revision=dataset_cfg.revision,
                max_num_shards=num_workers,
            )
    else:
        raise NotImplementedError("The MultiLeRobotDataset isn't supported for now.")
        dataset = MultiLeRobotDataset(
            dataset_cfg.repo_id,
            # TODO(aliberts): add proper support for multi dataset
            # delta_timestamps=delta_timestamps,
            image_transforms=image_transforms,
            video_backend=dataset_cfg.video_backend,
        )
        logging.info(
            "Multiple datasets were provided. Applied the following index mapping to the provided datasets: "
            f"{pformat(dataset.repo_id_to_index, indent=2)}"
        )

    if dataset_cfg.use_imagenet_stats:
        for key in dataset.meta.camera_keys:
            for stats_type, stats in IMAGENET_STATS.items():
                dataset.meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)

    dataset = patch_dataset_episode_boundaries(dataset=dataset)

    return dataset


def make_train_val_split(
    episode_ids: list[int], val_ratio: float, seed: int | None,
) -> tuple[list[int], list[int]]:
    """Split episode IDs into train and validation sets using the given ratio."""
    if seed is not None:
        rng = random.Random(seed)
        rng.shuffle(episode_ids)
    else:
        random.shuffle(episode_ids)

    num_val_episodes = max(1, int(len(episode_ids) * val_ratio))
    val_episode_ids = set(episode_ids[:num_val_episodes])
    train_episode_ids = set(episode_ids[num_val_episodes:])

    return train_episode_ids, val_episode_ids
