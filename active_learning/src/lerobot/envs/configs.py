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

import abc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import draccus
import gymnasium as gym

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.envs.wrappers import PerturbationWrapper
from lerobot.utils.constants import (
    ACTION,
    LIBERO_KEY_EEF_MAT,
    LIBERO_KEY_EEF_POS,
    LIBERO_KEY_EEF_QUAT,
    LIBERO_KEY_GRIPPER_QPOS,
    LIBERO_KEY_GRIPPER_QVEL,
    LIBERO_KEY_JOINTS_POS,
    LIBERO_KEY_JOINTS_VEL,
    LIBERO_KEY_PIXELS_AGENTVIEW,
    LIBERO_KEY_PIXELS_EYE_IN_HAND,
    OBS_ENV_STATE,
    OBS_IMAGE,
    OBS_IMAGES,
    OBS_STATE,
)


@dataclass
class OODConfig(abc.ABC):
    enable: bool = False

    def tweak_gym_kwargs(self, kwargs: dict) -> dict:
        """Adjust kwargs for gym.make()."""
        return self._tweak_gym_kwargs_impl(kwargs) if self.enable else kwargs

    def _tweak_gym_kwargs_impl(self, kwargs: dict) -> dict:
        """Will be overwritten by subclasses."""
        return kwargs

    def wrap(self, env: gym.Env) -> gym.Env:
        """Return wrapped env."""
        return self._wrap_impl(env) if self.enable else env

    def _wrap_impl(self, env: gym.Env) -> gym.Env:
        """Will be overwritten by subclasses."""
        return env


@dataclass
class ImagePatchOODConfig(OODConfig):
    static: bool = False
    min_frac: float = 0.1
    max_frac: float = 0.2
    allowed_area: tuple[float, float] | None = None
    patch_color: tuple[int, int, int] = (0, 0, 0)

    def _tweak_gym_kwargs_impl(self, kwargs: dict) -> dict:
        return kwargs

    def _wrap_impl(self, env: gym.Env) -> gym.Env:
        return PerturbationWrapper(
            env,
            static=self.static,
            min_patch_frac=self.min_frac,
            max_patch_frac=self.max_frac,
            allowed_area=self.allowed_area,
            patch_color=self.patch_color,
        )


@dataclass
class BddlSwapOODConfig(OODConfig):
    root_dir: str = "src/lerobot/envs/libero_ood_bddl_files"
    disable_init_states: bool = True

    def _tweak_gym_kwargs_impl(self, kwargs: dict) -> dict:
        if self.disable_init_states:
            kwargs["init_states"] = False
        return kwargs

    def resolve_bddl_path(self, task_suite: Any, task_id: int) -> str | None:
        if not self.enable or not self.root_dir:
            return None

        task = task_suite.get_task(task_id)

        ood_bddl_path = Path(self.root_dir) / task.problem_folder / task.bddl_file

        if not ood_bddl_path.exists():
            raise FileNotFoundError(
                f"Could not find OOD BDDL at {ood_bddl_path}."
            )
        return ood_bddl_path


@dataclass
class EnvConfig(draccus.ChoiceRegistry, abc.ABC):
    task: str | None = None
    fps: int = 30
    features: dict[str, PolicyFeature] = field(default_factory=dict)
    features_map: dict[str, str] = field(default_factory=dict)
    max_parallel_tasks: int = 1
    disable_env_checker: bool = True
    ood: OODConfig = field(default_factory=OODConfig)

    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)

    @property
    def package_name(self) -> str:
        """Package name to import if environment not found in gym registry"""
        return f"gym_{self.type}"

    @property
    def gym_id(self) -> str:
        """ID string used in gym.make() to instantiate the environment"""
        return f"{self.package_name}/{self.task}"

    @property
    @abc.abstractmethod
    def gym_kwargs(self) -> dict:
        raise NotImplementedError()


@EnvConfig.register_subclass("pusht")
@dataclass
class PushtEnv(EnvConfig):
    task: str | None = "PushT-v0"
    fps: int = 10
    episode_length: int = 300
    obs_type: str = "pixels_agent_pos"
    render_mode: str = "rgb_array"
    visualization_width: int = 384
    visualization_height: int = 384
    observation_height: int = 384
    observation_width: int = 384
    features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,)),
            "agent_pos": PolicyFeature(type=FeatureType.STATE, shape=(2,)),
        }
    )
    features_map: dict[str, str] = field(
        default_factory=lambda: {
            ACTION: ACTION,
            "agent_pos": OBS_STATE,
            "environment_state": OBS_ENV_STATE,
            "pixels": OBS_IMAGE,
        }
    )

    # Out-of-distribution configuration
    ood: ImagePatchOODConfig = field(default_factory=ImagePatchOODConfig)

    def __post_init__(self):
        if self.obs_type == "pixels_agent_pos":
            self.features["pixels"] = PolicyFeature(
                type=FeatureType.VISUAL, shape=(self.observation_height, self.observation_width, 3)
            )
        elif self.obs_type == "environment_state_agent_pos":
            self.features["environment_state"] = PolicyFeature(type=FeatureType.ENV, shape=(16,))

    @property
    def gym_kwargs(self) -> dict:
        return {
            "obs_type": self.obs_type,
            "render_mode": self.render_mode,
            "visualization_width": self.visualization_width,
            "visualization_height": self.visualization_height,
            "max_episode_steps": self.episode_length,
        }


@EnvConfig.register_subclass("libero")
@dataclass
class LiberoEnv(EnvConfig):
    task: str = "libero_10"  # can also choose libero_spatial, libero_object, etc.
    task_ids: list[int] | None = None
    fps: int = 30
    episode_length: int | None = None
    obs_type: str = "pixels_agent_pos"
    render_mode: str = "rgb_array"
    camera_name: str = "agentview_image,robot0_eye_in_hand_image"
    init_states: bool = True
    camera_name_mapping: dict[str, str] | None = None
    observation_height: int = 360
    observation_width: int = 360
    features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
        }
    )
    features_map: dict[str, str] = field(
        default_factory=lambda: {
            ACTION: ACTION,
            LIBERO_KEY_EEF_POS: f"{OBS_STATE}.eef_pos",
            LIBERO_KEY_EEF_QUAT: f"{OBS_STATE}.eef_quat",
            LIBERO_KEY_EEF_MAT: f"{OBS_STATE}.eef_mat",
            LIBERO_KEY_GRIPPER_QPOS: f"{OBS_STATE}.gripper_qpos",
            LIBERO_KEY_GRIPPER_QVEL: f"{OBS_STATE}.gripper_qvel",
            LIBERO_KEY_JOINTS_POS: f"{OBS_STATE}.joint_pos",
            LIBERO_KEY_JOINTS_VEL: f"{OBS_STATE}.joint_vel",
            LIBERO_KEY_PIXELS_AGENTVIEW: f"{OBS_IMAGES}.image",
            LIBERO_KEY_PIXELS_EYE_IN_HAND: f"{OBS_IMAGES}.image2",
        }
    )
    control_mode: str = "relative"  # or "absolute"

    # Out-of-distribution configuration
    ood: BddlSwapOODConfig = field(default_factory=BddlSwapOODConfig)

    def __post_init__(self):
        if self.obs_type == "pixels":
            self.features[LIBERO_KEY_PIXELS_AGENTVIEW] = PolicyFeature(
                type=FeatureType.VISUAL, shape=(self.observation_height, self.observation_width, 3)
            )
            self.features[LIBERO_KEY_PIXELS_EYE_IN_HAND] = PolicyFeature(
                type=FeatureType.VISUAL, shape=(self.observation_height, self.observation_width, 3)
            )
        elif self.obs_type == "pixels_agent_pos":
            self.features[LIBERO_KEY_PIXELS_AGENTVIEW] = PolicyFeature(
                type=FeatureType.VISUAL, shape=(self.observation_height, self.observation_width, 3)
            )
            self.features[LIBERO_KEY_PIXELS_EYE_IN_HAND] = PolicyFeature(
                type=FeatureType.VISUAL, shape=(self.observation_height, self.observation_width, 3)
            )
            self.features[LIBERO_KEY_EEF_POS] = PolicyFeature(
                type=FeatureType.STATE,
                shape=(3,),
            )
            self.features[LIBERO_KEY_EEF_QUAT] = PolicyFeature(
                type=FeatureType.STATE,
                shape=(4,),
            )
            self.features[LIBERO_KEY_EEF_MAT] = PolicyFeature(
                type=FeatureType.STATE,
                shape=(3, 3),
            )
            self.features[LIBERO_KEY_GRIPPER_QPOS] = PolicyFeature(
                type=FeatureType.STATE,
                shape=(2,),
            )
            self.features[LIBERO_KEY_GRIPPER_QVEL] = PolicyFeature(
                type=FeatureType.STATE,
                shape=(2,),
            )
            self.features[LIBERO_KEY_JOINTS_POS] = PolicyFeature(
                type=FeatureType.STATE,
                shape=(7,),
            )
            self.features[LIBERO_KEY_JOINTS_VEL] = PolicyFeature(
                type=FeatureType.STATE,
                shape=(7,),
            )
        else:
            raise ValueError(f"Unsupported obs_type: {self.obs_type}")

    @property
    def gym_kwargs(self) -> dict:
        return {
            "obs_type": self.obs_type,
            "render_mode": self.render_mode,
            "task_ids": self.task_ids,
        }
