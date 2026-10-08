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
#
# Ported from upstream lerobot (policies/fastwam/processor_fastwam.py). The upstream version is built
# from `make_default_policy_processor_steps` / `make_policy_processor_pipelines`, which this fork does
# not have; the pipelines are assembled explicitly here in the same style as processor_xvla.py. The
# step order and semantics are unchanged: rename -> batch -> device -> normalize on the way in, and
# unnormalize -> optional action toggle -> cpu on the way out.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.policies.fastwam.configuration_fastwam import FastWAMConfig
from lerobot.processor import (
    ActionProcessorStep,
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.utils.constants import (
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)


@dataclass
@ProcessorStepRegistry.register(name="fastwam_action_toggle_processor")
class FastWAMActionToggleProcessorStep(ActionProcessorStep):
    """Apply FastWAM LIBERO toggle semantics to configured action dimensions."""

    toggle_dimensions: list[int]

    def action(self, action: PolicyAction) -> PolicyAction:
        if not self.toggle_dimensions:
            return action
        processed_action = action.clone()
        action_dim = int(processed_action.shape[-1])
        for dim in self.toggle_dimensions:
            resolved_dim = dim if dim >= 0 else action_dim + dim
            if resolved_dim < 0 or resolved_dim >= action_dim:
                raise ValueError(
                    f"FastWAM action toggle dimension {dim} is out of bounds for action dim {action_dim}."
                )
            value = processed_action[..., resolved_dim]
            value = value * 2.0 - 1.0
            processed_action[..., resolved_dim] = torch.sign(-value)
        return processed_action

    def get_config(self) -> dict[str, Any]:
        return {"toggle_dimensions": self.toggle_dimensions}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


def make_fastwam_pre_post_processors(
    config: FastWAMConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Create LeRobot pre- and post-processing pipelines for FastWAM.

    No visual normalization: VISUAL is IDENTITY in the config's normalization_mapping, so images pass
    through in [0, 1] and the model maps them to the Wan VAE's [-1, 1] at the encode boundary. A real
    dataset's per-channel image std is the tiny frame-to-frame brightness variance, which would push
    images far outside [-1, 1] and saturate them. STATE/ACTION still normalize with dataset stats.

    No resize step: the model is the single authority on input resolution and resizes each camera to
    its target inside `_stack_video_from_images` / `_prepare_infer_image` on every path.
    """
    features = {**config.input_features, **config.output_features}
    input_steps = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        DeviceProcessorStep(device=config.device),
        NormalizerProcessorStep(
            features=features, norm_map=config.normalization_mapping, stats=dataset_stats
        ),
    ]
    output_steps: list[Any] = [
        UnnormalizerProcessorStep(
            features=config.output_features,
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
    ]
    if config.toggle_action_dimensions:
        output_steps.append(
            FastWAMActionToggleProcessorStep(toggle_dimensions=list(config.toggle_action_dimensions))
        )
    output_steps.append(DeviceProcessorStep(device="cpu"))

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
