import random
import warnings
from typing import Any, Dict, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
from gymnasium.utils import seeding


class PerturbationWrapper(gym.Wrapper):
    """
    Inserts one black rectangle of configurable size into the environment.
    """
    def __init__(
        self,
        env: gym.Env,
        static: bool,
        min_patch_frac: float,
        max_patch_frac: float,
        allowed_area: Optional[Tuple[float, float]],
        patch_color: Union[Tuple[int, int, int], np.ndarray],
        seed: Optional[int] = None,
    ):
        """
        Args:
            env: The environment to wrap.
            static: If True, sample the patch region once and keep it fixed. If False,
                re-sample a new patch region on every frame.
            min_patch_frac: Minimum height/width fraction of the patch relative to image size.
            max_patch_frac: Maximum height/width fraction of the patch relative to image size.
            allowed_area: (width_frac, height_frac) of a central region
                within which patches must lie. If None, the whole image is allowed.
            patch_color: RGB color of the patch.
            seed: Optional seed for deterministic patch placement.
        """
        super().__init__(env)
        if not (0.0 <= min_patch_frac <= max_patch_frac <= 1.0):
            raise ValueError(
                f"Invalid patch fractions: ({min_patch_frac}, {max_patch_frac}). "
                "Values must be in [0.0, 1.0] and min <= max."
            )

        if allowed_area is not None:
            allowed_width, allowed_height = allowed_area
            if not (0.0 < allowed_width <= 1.0 and 0.0 < allowed_height <= 1.0):
                raise ValueError("allowed_area values must be in (0, 1].")
            self.allowed_area = (allowed_height, allowed_width)  # store as (height, width)
        else:
            self.allowed_area = None

        self.static = static
        self.min_patch_frac = min_patch_frac
        self.max_patch_frac = max_patch_frac

        # Normalised rectangle coordinates: (top_frac, left_frac, height_frac, width_frac) from [0, 1]
        self.patch_frac: Optional[Tuple[float, float, float, float]] = None

        # Color for the patch, stored as an array matching image dtype
        self.patch_color = np.array(patch_color, dtype=np.uint8)

        self.seed(seed)

    def seed(self, seed: Optional[int] = None):
        """
        Create (or re-create) the RNG used by this wrapper and return the seed actually used.
        """
        self.np_random, seed = seeding.np_random(seed)
        random.seed(seed)
        return [seed]

    def _reset_patch_fraction(self):
        """
        Randomly select a rectangular region within the image to blackout and store its
        normalized coordinates and size in `self.patch_frac`.
        """
        if self.allowed_area is not None:
            area_h, area_w = self.allowed_area
            max_h_for_area = min(self.max_patch_frac, area_h)
            max_w_for_area = min(self.max_patch_frac, area_w)
            if self.min_patch_frac > max_h_for_area or self.min_patch_frac > max_w_for_area:
                raise ValueError("allowed_area too small for the chosen patch fractions.")
            patch_height_frac = self.np_random.uniform(self.min_patch_frac, max_h_for_area)
            patch_width_frac  = self.np_random.uniform(self.min_patch_frac, max_w_for_area)
            # central region top-left corner
            center_top  = (1.0 - area_h) / 2.0
            center_left = (1.0 - area_w) / 2.0
            top_low  = center_top
            left_low = center_left
            top_high  = center_top + area_h - patch_height_frac
            left_high = center_left + area_w - patch_width_frac
        else:
            patch_height_frac = self.np_random.uniform(self.min_patch_frac, self.max_patch_frac)
            patch_width_frac  = self.np_random.uniform(self.min_patch_frac, self.max_patch_frac)
            top_low  = 0.0
            left_low = 0.0
            top_high  = 1.0 - patch_height_frac
            left_high = 1.0 - patch_width_frac

        top_frac  = self.np_random.uniform(top_low,  top_high)
        left_frac = self.np_random.uniform(left_low, left_high) 

        self.patch_frac = (top_frac, left_frac, patch_height_frac, patch_width_frac)

    def _apply_patch(self, img: np.ndarray) -> np.ndarray:
        """
        Apply a black rectangle to the input image using a fixed patch region.

        Args:
            img: The input image to apply the patch to.

        Returns:
            The patched image.
        """
        height, width = img.shape[:2]
        top_frac, left_frac, patch_height_frac, patch_width_frac = self.patch_frac

        patch_top = int(round(top_frac * height))
        patch_left = int(round(left_frac * width))
        patch_height = int(round(patch_height_frac * height))
        patch_width = int(round(patch_width_frac * width))

        img = img.copy()
        img[patch_top : patch_top + patch_height, patch_left : patch_left + patch_width] = self.patch_color
        return img

    def _patch_obs(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply the fixed black patch to each image in the observation.

        Args:
            observation (Dict[str, Any]): A dictionary containing environment observations.

        Returns:
            Dict[str, Any]: The observation dictionary with patched pixel data.
        """
        if "pixels" in observation:
            if isinstance(observation["pixels"], dict):  # Multi-camera
                for camera, img in observation["pixels"].items():
                    observation["pixels"][camera] = self._apply_patch(img.copy())
            else:  # Single-camera
                observation["pixels"] = self._apply_patch(observation["pixels"].copy())

        return observation

    # Gym API overrides
    def reset(self, seed: Optional[int] = None, **kwargs):
        # Call env.reset(...), then patch the returned obs
        if seed is not None:
            self.seed(seed)

        if self.patch_frac is None or not self.static:
            # Initialize the patch region using the first image
            self._reset_patch_fraction()

        obs, info = super().reset(seed=seed, **kwargs)
        return self._patch_obs(obs), info

    def step(self, action):
        # Call env.step(...), then patch the returned obs
        if not self.static:
            self._reset_patch_fraction()

        obs, rew, term, trunc, info = super().step(action)
        return self._patch_obs(obs), rew, term, trunc, info

    def render(self, *args, **kwargs):
        frame = self.unwrapped.render(*args, **kwargs)
        if frame is None:
            warnings.warn(
                "PerturbationWrapper: render() produced no image. "
                "Perturbation will not be visualized.",
                UserWarning,
                stacklevel=2,
            )
        else:
            frame = self._apply_patch(frame.copy())

        return frame

    def get_obs(self):
        # Call env.get_obs(), then patch the returned obs
        obs = self.env.get_obs()
        return self._patch_obs(obs)
