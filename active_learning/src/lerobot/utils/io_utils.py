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
import json
import warnings
from pathlib import Path
from typing import Dict, List, TypeVar, Union

import imageio
import numpy as np

JsonLike = str | int | float | bool | None | list["JsonLike"] | dict[str, "JsonLike"] | tuple["JsonLike", ...]
T = TypeVar("T", bound=JsonLike)


def save_episode_video(
    ep_frames: Union[List[np.ndarray], Dict[str, List[np.ndarray]]],
    out_root: Path,
    episode_idx: int,
    fps: int,
) -> None:
    """
    Save episode frames as an .mp4 video. Supports single-camera (list of frames)
    or multi-camera (dict of frame lists).
    """
    ep_str = f"rollout_ep{episode_idx:03d}.mp4"

    if isinstance(ep_frames, list):
        out_root.mkdir(parents=True, exist_ok=True)
        write_video(
            str(out_root / ep_str),
            np.stack(ep_frames, axis=0),           # (T, H, W, C)
            fps=fps,
        )

    elif isinstance(ep_frames, dict):
        for cam, frames in ep_frames.items():
            cam_dir = out_root / cam
            cam_dir.mkdir(parents=True, exist_ok=True)
            write_video(
                str(cam_dir / ep_str),
                np.stack(frames, axis=0),
                fps=fps,
            )

    else:
        raise TypeError(f"ep_frames must be list or dict, got {type(ep_frames)}")


def write_video(video_path, stacked_frames, fps):
    # Filter out DeprecationWarnings raised from pkg_resources
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", "pkg_resources is deprecated as an API", category=DeprecationWarning
        )
        imageio.mimsave(video_path, stacked_frames, fps=fps)


def deserialize_json_into_object(fpath: Path, obj: T) -> T:
    """
    Loads the JSON data from `fpath` and recursively fills `obj` with the
    corresponding values (strictly matching structure and types).
    Tuples in `obj` are expected to be lists in the JSON data, which will be
    converted back into tuples.
    """
    with open(fpath, encoding="utf-8") as f:
        data = json.load(f)

    def _deserialize(target, source):
        """
        Recursively overwrite the structure in `target` with data from `source`,
        performing strict checks on structure and type.
        Returns the updated version of `target` (especially important for tuples).
        """

        # If the target is a dictionary, source must be a dictionary as well.
        if isinstance(target, dict):
            if not isinstance(source, dict):
                raise TypeError(f"Type mismatch: expected dict, got {type(source)}")

            # Check that they have exactly the same set of keys.
            if target.keys() != source.keys():
                raise ValueError(
                    f"Dictionary keys do not match.\nExpected: {target.keys()}, got: {source.keys()}"
                )

            # Recursively update each key.
            for k in target:
                target[k] = _deserialize(target[k], source[k])

            return target

        # If the target is a list, source must be a list as well.
        elif isinstance(target, list):
            if not isinstance(source, list):
                raise TypeError(f"Type mismatch: expected list, got {type(source)}")

            # Check length
            if len(target) != len(source):
                raise ValueError(f"List length mismatch: expected {len(target)}, got {len(source)}")

            # Recursively update each element.
            for i in range(len(target)):
                target[i] = _deserialize(target[i], source[i])

            return target

        # If the target is a tuple, the source must be a list in JSON,
        # which we'll convert back to a tuple.
        elif isinstance(target, tuple):
            if not isinstance(source, list):
                raise TypeError(f"Type mismatch: expected list (for tuple), got {type(source)}")

            if len(target) != len(source):
                raise ValueError(f"Tuple length mismatch: expected {len(target)}, got {len(source)}")

            # Convert each element, forming a new tuple.
            converted_items = []
            for t_item, s_item in zip(target, source, strict=False):
                converted_items.append(_deserialize(t_item, s_item))

            # Return a brand new tuple (tuples are immutable in Python).
            return tuple(converted_items)

        # Otherwise, we're dealing with a "primitive" (int, float, str, bool, None).
        else:
            # Check the exact type.  If these must match 1:1, do:
            if type(target) is not type(source):
                raise TypeError(f"Type mismatch: expected {type(target)}, got {type(source)}")
            return source

    # Perform the in-place/recursive deserialization
    updated_obj = _deserialize(obj, data)
    return updated_obj


def get_task_group_dir(
    out_root: Path,
    task_group: str,
):
    """
    Return the output directory for a given task group.
    """
    if "libero" in task_group:
        return out_root / task_group
    return out_root


def get_task_dir(
    out_root: Path,
    task_group: str,
    task_id: int,
) -> Path:
    """
    Return the output directory for a given task.
    """
    task_group_dir = get_task_group_dir(out_root, task_group)
    if "libero" in task_group:
        return task_group_dir / f"task{task_id:02d}"
    return task_group_dir
