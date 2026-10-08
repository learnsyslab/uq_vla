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
from collections.abc import Iterator

import torch


class EpisodeAwareSampler:
    def __init__(
        self,
        dataset_from_indices: list[int],
        dataset_to_indices: list[int],
        episode_indices_to_use: list | None = None,
        episode_indices: list | None = None,
        drop_n_first_frames: int = 0,
        drop_n_last_frames: int = 0,
        shuffle: bool = False,
        fraction: float | None = None,
    ):
        """Sampler that optionally incorporates episode boundary information.

        Args:
            dataset_from_indices: List of indices containing the start of each episode in the dataset.
            dataset_to_indices: List of indices containing the end of each episode in the dataset.
            episode_indices_to_use: List of episode indices to use. If None, all episodes are used.
            episode_indices: The `episode_index` of each row of the boundary lists above. Pass this
                whenever only a subset of episodes is loaded, since the original episode index is then
                no longer the row position and `episode_indices_to_use` would otherwise be matched
                against positions. Defaults to the row positions, which is correct for full-dataset
                loads and for subsets that are contiguous from episode 0.
            drop_n_first_frames: Number of frames to drop from the start of each episode.
            drop_n_last_frames: Number of frames to drop from the end of each episode.
            shuffle: Whether to shuffle the indices.
            fraction: Fraction of the total filtered frames to keep.
        """
        episode_ids = (
            [int(episode_id) for episode_id in episode_indices]
            if episode_indices is not None
            else list(range(len(dataset_from_indices)))
        )
        keep = None if episode_indices_to_use is None else {int(e) for e in episode_indices_to_use}

        indices = []
        for episode_idx, start_index, end_index in zip(
            episode_ids, dataset_from_indices, dataset_to_indices, strict=True
        ):
            if keep is None or episode_idx in keep:
                indices.extend(range(start_index + drop_n_first_frames, end_index - drop_n_last_frames))

        if fraction is not None:
            indices = [indices[i] for i in torch.randperm(len(indices))]
            indices = indices[:int(fraction * len(indices))]

        self.indices = sorted(indices)
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[int]:
        if self.shuffle:
            for i in torch.randperm(len(self.indices)):
                yield self.indices[i]
        else:
            for i in self.indices:
                yield i

    def __len__(self) -> int:
        return len(self.indices)
