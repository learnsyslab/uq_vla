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

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from lerobot.datasets.transforms import ImageTransformsConfig
from lerobot.datasets.video_utils import get_safe_default_codec


@dataclass
class DatasetConfig:
    # You may provide a list of datasets here. `train.py` creates them all and concatenates them. Note: only data
    # keys common between the datasets are kept. Each dataset gets and additional transform that inserts the
    # "dataset_index" into the returned item. The index mapping is made according to the order in which the
    # datasets are provided.
    repo_id: str
    # Root directory where the dataset will be stored (e.g. 'dataset/path').
    root: str | None = None
    episodes: list[int] | None = None
    image_transforms: ImageTransformsConfig = field(default_factory=ImageTransformsConfig)
    revision: str | None = None
    use_imagenet_stats: bool = True
    video_backend: str = field(default_factory=get_safe_default_codec)
    streaming: bool = False
    libero_tasks: dict[str, list[int]] | None = None
    pusht_tasks: dict[str, list[int]] | None = None


@dataclass
class WandBConfig:
    enable: bool = True
    # Set to true to disable saving an artifact despite training.save_checkpoint=True
    disable_artifact: bool = True
    project: str = "lerobot"
    entity: str | None = None
    notes: str | None = None
    run_id: str | None = None
    mode: str | None = None  # Allowed values: 'online', 'offline' 'disabled'. Defaults to 'online'


@dataclass
class EvalConfig:
    n_episodes: int = 50
    # `batch_size` specifies the number of environments to use in a gym.vector.VectorEnv.
    batch_size: int = 50
    # `use_async_envs` specifies whether to use asynchronous environments (multiprocessing).
    use_async_envs: bool = False

    def __post_init__(self) -> None:
        if self.batch_size > self.n_episodes:
            raise ValueError(
                "The eval batch size is greater than the number of eval episodes "
                f"({self.batch_size} > {self.n_episodes}). As a result, {self.batch_size} "
                f"eval environments will be instantiated, but only {self.n_episodes} will be used. "
                "This might significantly slow down evaluation. To fix this, you should update your command "
                f"to increase the number of episodes to match the batch size (e.g. `eval.n_episodes={self.batch_size}`), "
                f"or lower the batch size (e.g. `eval.batch_size={self.n_episodes}`)."
            )

@dataclass
class EvalUncertEstConfig:
    n_episodes: int = 20

    # Which environment domains to evaluate. Allowed values: "id" (in-distribution), "ood" (out-of-distribution).
    domains: tuple[str] = ("id", "ood")

    # If True, merge success and failure episodes into a single bucket when plotting uncertainty scores
    collapse_success_failure: bool = False

    # Which uncertainty estimation methods to evaluate
    uncert_est_methods: list[str] = field(
        default_factory=lambda: [
            "composed_sequence",
            "composed_cross_bayesian",
            "cross_bayesian",
        ]
    )

    # Paths to ID and OoD failure seeds to balance success and failure cases
    id_failure_seeds_path: Path | None = None
    ood_failure_seeds_path: Path | None = None

    def validate(self):
        allowed_domains = {"id", "ood"}
        if len(self.domains) == 0:
            raise ValueError("At least one eval domain must be specified.")
        for d in self.domains:
            if d not in allowed_domains:
                raise ValueError(
                    f"Unknown eval domain '{d}'. Allowed: {sorted(allowed_domains)}."
                )

        allowed_methods = {
            "composed_sequence",
            "composed_cross_bayesian",
            "cross_bayesian",
        }
        # check every method the user passed
        for m in self.uncert_est_methods:
            if m not in allowed_methods:
                raise ValueError(
                    f"Unknown uncertainty-estimation method '{m}'. "
                    f"Allowed: {sorted(allowed_methods)}"
                )

@dataclass
class VisConfig:
    """Options that control what we draw and how we save/show it."""
    # Number of total rollouts to visualize
    num_rollouts: int = 10

    # If set, start drawing only after this environment step
    start_step: Optional[int] = None

    # Hard cap on how many env steps to visualize
    max_steps: Optional[int] = None

    # Names of the action dimensions to visualize
    action_dim_names: Optional[list[str]] = None
