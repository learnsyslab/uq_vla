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

from __future__ import annotations  # noqa: I001

import importlib
import logging
from typing import Any, TypedDict

import torch
from torch import nn
from typing_extensions import Unpack

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.envs.configs import EnvConfig
from lerobot.envs.utils import env_to_policy_features
from lerobot.policies.flow_matching.configuration_flow_matching import FlowMatchingConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.fastwam.configuration_fastwam import FastWAMConfig
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.common.flow_matching.adapter import BaseFlowMatchingAdapter
from lerobot.policies.utils import validate_visual_features_consistency
from lerobot.policies.xvla.configuration_xvla import XVLAConfig
from lerobot.processor import PolicyAction, PolicyProcessorPipeline
from lerobot.processor.converters import (
    batch_to_transition,
    policy_action_to_transition,
    transition_to_batch,
    transition_to_policy_action,
)
from lerobot.uncertainty.uncertainty_samplers.configuration_uncertainty_sampler import (
    ScoringMetricConfig,
    UncertaintySamplerConfig,
)
from lerobot.uncertainty.uncertainty_samplers.uncertainty_sampler import (
    UncertaintySampler,
)
from lerobot.uncertainty.uncertainty_scoring.scorer_artifacts import (
    ScorerArtifacts,
)
from lerobot.uncertainty.uncertainty_scoring.scoring_metrics import UncertaintyMetric
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME


def get_policy_class(name: str) -> type[PreTrainedPolicy]:
    """
    Retrieves a policy class by its registered name.

    This function uses dynamic imports to avoid loading all policy classes into memory
    at once, improving startup time and reducing dependencies.

    Args:
        name: The name of the policy. Supported names are "flow_matching", "smolvla", "xvla", "fastwam".

    Returns:
        The policy class corresponding to the given name.

    Raises:
        NotImplementedError: If the policy name is not recognized.
    """
    if name == "flow_matching":
        from lerobot.policies.flow_matching.modelling_flow_matching import FlowMatchingPolicy

        return FlowMatchingPolicy
    elif name == "smolvla":
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        return SmolVLAPolicy
    elif name == "xvla":
        from lerobot.policies.xvla.modeling_xvla import XVLAPolicy

        return XVLAPolicy
    elif name == "fastwam":
        from lerobot.policies.fastwam.modeling_fastwam import FastWAMPolicy

        return FastWAMPolicy
    else:
        try:
            return _get_policy_cls_from_policy_name(name=name)
        except Exception as e:
            raise ValueError(f"Policy type '{name}' is not available.") from e


def make_policy_config(policy_type: str, **kwargs) -> PreTrainedConfig:
    """
    Instantiates a policy configuration object based on the policy type.

    This factory function simplifies the creation of policy configuration objects by
    mapping a string identifier to the corresponding config class.

    Args:
        policy_type: The type of the policy. Supported types are "flow_matching", "smolvla", "xvla", "fastwam".
        **kwargs: Keyword arguments to be passed to the configuration class constructor.

    Returns:
        An instance of a `PreTrainedConfig` subclass.

    Raises:
        ValueError: If the `policy_type` is not recognized.
    """
    if policy_type == "flow_matching":
        return FlowMatchingConfig(**kwargs)
    elif policy_type == "smolvla":
        return SmolVLAConfig(**kwargs)
    elif policy_type == "xvla":
        return XVLAConfig(**kwargs)
    elif policy_type == "fastwam":
        return FastWAMConfig(**kwargs)
    else:
        try:
            config_cls = PreTrainedConfig.get_choice_class(policy_type)
            return config_cls(**kwargs)
        except Exception as e:
            raise ValueError(f"Policy type '{policy_type}' is not available.") from e


class ProcessorConfigKwargs(TypedDict, total=False):
    """
    A TypedDict defining the keyword arguments for processor configuration.

    This provides type hints for the optional arguments passed to `make_pre_post_processors`,
    improving code clarity and enabling static analysis.

    Attributes:
        preprocessor_config_filename: The filename for the preprocessor configuration.
        postprocessor_config_filename: The filename for the postprocessor configuration.
        preprocessor_overrides: A dictionary of overrides for the preprocessor configuration.
        postprocessor_overrides: A dictionary of overrides for the postprocessor configuration.
        dataset_stats: Dataset statistics for normalization.
    """

    preprocessor_config_filename: str | None
    postprocessor_config_filename: str | None
    preprocessor_overrides: dict[str, Any] | None
    postprocessor_overrides: dict[str, Any] | None
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None


def make_pre_post_processors(
    policy_cfg: PreTrainedConfig,
    pretrained_path: str | None = None,
    **kwargs: Unpack[ProcessorConfigKwargs],
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    Create or load pre- and post-processor pipelines for a given policy.

    This function acts as a factory. It can either load existing processor pipelines
    from a pretrained path or create new ones from scratch based on the policy
    configuration. Each policy type has a dedicated factory function for its
    processors (e.g., `make_smolvla_pre_post_processors`).

    Args:
        policy_cfg: The configuration of the policy for which to create processors.
        pretrained_path: An optional path to load pretrained processor pipelines from.
            If provided, pipelines are loaded from this path.
        **kwargs: Keyword arguments for processor configuration, as defined in
            `ProcessorConfigKwargs`.

    Returns:
        A tuple containing the input (pre-processor) and output (post-processor) pipelines.

    Raises:
        NotImplementedError: If a processor factory is not implemented for the given
            policy configuration type.
    """
    if pretrained_path:
        return (
            PolicyProcessorPipeline.from_pretrained(
                pretrained_model_name_or_path=pretrained_path,
                config_filename=kwargs.get(
                    "preprocessor_config_filename", f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json"
                ),
                overrides=kwargs.get("preprocessor_overrides", {}),
                to_transition=batch_to_transition,
                to_output=transition_to_batch,
            ),
            PolicyProcessorPipeline.from_pretrained(
                pretrained_model_name_or_path=pretrained_path,
                config_filename=kwargs.get(
                    "postprocessor_config_filename", f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json"
                ),
                overrides=kwargs.get("postprocessor_overrides", {}),
                to_transition=policy_action_to_transition,
                to_output=transition_to_policy_action,
            ),
        )

    # Create a new processor based on policy type
    if isinstance(policy_cfg, FlowMatchingConfig):
        from lerobot.policies.flow_matching.processor_flow_matching import (
            make_flow_matching_pre_post_processors,
        )

        processors = make_flow_matching_pre_post_processors(
            config=policy_cfg,
            dataset_stats=kwargs.get("dataset_stats"),
        )

    elif isinstance(policy_cfg, SmolVLAConfig):
        from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors

        processors = make_smolvla_pre_post_processors(
            config=policy_cfg,
            dataset_stats=kwargs.get("dataset_stats"),
        )

    elif isinstance(policy_cfg, XVLAConfig):
        from lerobot.policies.xvla.processor_xvla import (
            make_xvla_pre_post_processors,
        )

        processors = make_xvla_pre_post_processors(
            config=policy_cfg,
            dataset_stats=kwargs.get("dataset_stats"),
        )
    elif isinstance(policy_cfg, FastWAMConfig):
        from lerobot.policies.fastwam.processor_fastwam import (
            make_fastwam_pre_post_processors,
        )

        processors = make_fastwam_pre_post_processors(
            config=policy_cfg,
            dataset_stats=kwargs.get("dataset_stats"),
        )

    else:
        try:
            processors = _make_processors_from_policy_config(
                config=policy_cfg,
                dataset_stats=kwargs.get("dataset_stats"),
            )
        except Exception as e:
            raise ValueError(f"Processor for policy type '{policy_cfg.type}' is not implemented.") from e

    return processors


def make_policy(
    cfg: PreTrainedConfig,
    ds_meta: LeRobotDatasetMetadata | None = None,
    env_cfg: EnvConfig | None = None,
    rename_map: dict[str, str] | None = None,
) -> PreTrainedPolicy:
    """
    Instantiate a policy model.

    This factory function handles the logic of creating a policy, which requires
    determining the input and output feature shapes. These shapes can be derived
    either from a `LeRobotDatasetMetadata` object or an `EnvConfig` object. The function
    can either initialize a new policy from scratch or load a pretrained one.

    Args:
        cfg: The configuration for the policy to be created. If `cfg.pretrained_path` is
             set, the policy will be loaded with weights from that path.
        ds_meta: Dataset metadata used to infer feature shapes and types. Also provides
                 statistics for normalization layers.
        env_cfg: Environment configuration used to infer feature shapes and types.
                 One of `ds_meta` or `env_cfg` must be provided.
        rename_map: Optional mapping of dataset or environment feature keys to match
                 expected policy feature names (e.g., `"left"` → `"camera1"`).

    Returns:
        An instantiated and device-placed policy model.

    Raises:
        ValueError: If both or neither of `ds_meta` and `env_cfg` are provided.
        NotImplementedError: If attempting to use an unsupported policy-backend
                             combination (e.g., VQBeT with 'mps').
    """
    if bool(ds_meta) == bool(env_cfg):
        raise ValueError("Either one of a dataset metadata or a sim env must be provided.")

    # NOTE: Currently, if you try to run vqbet with mps backend, you'll get this error.
    # TODO(aliberts, rcadene): Implement a check_backend_compatibility in policies?
    # NotImplementedError: The operator 'aten::unique_dim' is not currently implemented for the MPS device. If
    # you want this op to be added in priority during the prototype phase of this feature, please comment on
    # https://github.com/pytorch/pytorch/issues/77764. As a temporary fix, you can set the environment
    # variable `PYTORCH_ENABLE_MPS_FALLBACK=1` to use the CPU as a fallback for this op. WARNING: this will be
    # slower than running natively on MPS.
    policy_cls = get_policy_class(cfg.type)

    kwargs = {}
    if ds_meta is not None:
        features = dataset_to_policy_features(ds_meta.features)
        # Policies that derive their input layout from the dataset's real camera keys (FastWAM: one
        # visual feature per camera at image_size[1] // n_cams width) install it here, before the
        # visual-feature consistency check below.
        if hasattr(cfg, "set_dataset_feature_metadata"):
            cfg.set_dataset_feature_metadata(ds_meta.features)
    else:
        if not cfg.pretrained_path:
            logging.warning(
                "You are instantiating a policy from scratch and its features are parsed from an environment "
                "rather than a dataset. Normalization modules inside the policy will have infinite values "
                "by default without stats from a dataset."
            )
        if env_cfg is None:
            raise ValueError("env_cfg cannot be None when ds_meta is not provided")
        features = env_to_policy_features(env_cfg)

    cfg.output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    if not cfg.input_features:
        cfg.input_features = {key: ft for key, ft in features.items() if key not in cfg.output_features}
    kwargs["config"] = cfg

    if cfg.pretrained_path:
        # Load a pretrained policy and override the config if needed (for example, if there are inference-time
        # hyperparameters that we want to vary).
        kwargs["pretrained_name_or_path"] = cfg.pretrained_path
        policy = policy_cls.from_pretrained(**kwargs)
    else:
        # Make a fresh policy.
        policy = policy_cls(**kwargs)

    policy.to(cfg.device)
    assert isinstance(policy, torch.nn.Module)

    # policy = torch.compile(policy, mode="reduce-overhead")

    if not rename_map:
        validate_visual_features_consistency(cfg, features)
        # TODO: (jadechoghari) - add a check_state(cfg, features) and check_action(cfg, features)

    return policy


def make_uncertainty_sampler(
    uncertainty_sampler_config: UncertaintySamplerConfig,
    policy_config: PreTrainedConfig,
    model: nn.Module,
    scorer_artifacts: ScorerArtifacts,
) -> UncertaintySampler:
    # Initialize the uncertainty adapter
    uncertainty_adapter = make_flow_matching_adapter(
        model=model,
        policy_config=policy_config
    )

    if uncertainty_sampler_config.type == "composed_cross_bayesian":
        from lerobot.uncertainty.uncertainty_samplers.composed_cross_bayesian_sampler import (
            ComposedCrossBayesianSampler,
        )

        if uncertainty_sampler_config.composed_cross_bayesian_sampler.scorer_type == "ensemble" and not scorer_artifacts.ensemble_models:
            raise ValueError(
                "Composed Cross-Bayesian uncertainty sampler with scorer_type=ensemble requires an ensemble model."
            )
        if uncertainty_sampler_config.composed_cross_bayesian_sampler.scorer_type == "laplace" and scorer_artifacts.laplace_posterior is None:
            raise ValueError(
                "Composed Cross-Bayesian uncertainty sampler with scorer_type=laplace requires Laplace posterior "
                "to draw a scorer model from."
            )

        return ComposedCrossBayesianSampler(
            config=uncertainty_sampler_config.composed_cross_bayesian_sampler,
            sampler_model=uncertainty_adapter,
            scorer_artifacts=scorer_artifacts,
        )
    if uncertainty_sampler_config.type == "composed_sequence":
        from lerobot.uncertainty.uncertainty_samplers.composed_seq_sampler import (
            ComposedSequenceSampler,
        )

        return ComposedSequenceSampler(
            config=uncertainty_sampler_config.composed_sequence_sampler,
            model=uncertainty_adapter,
        )
    elif uncertainty_sampler_config.type == "cross_bayesian":
        from lerobot.uncertainty.uncertainty_samplers.cross_bayesian_sampler import (
            CrossBayesianSampler,
        )

        if uncertainty_sampler_config.cross_bayesian_sampler.scorer_type == "ensemble" and not scorer_artifacts.ensemble_models:
            raise ValueError(
                "Cross-Bayesian uncertainty sampler with scorer_type=ensemble requires an ensemble model."
            )
        if uncertainty_sampler_config.cross_bayesian_sampler.scorer_type == "laplace" and scorer_artifacts.laplace_posterior is None:
            raise ValueError(
                "Bayesian uncertainty sampler with scorer_type=laplace requires Laplace posterior to draw a scorer model from."
            )

        return CrossBayesianSampler(
            config=uncertainty_sampler_config.cross_bayesian_sampler,
            sampler_model=uncertainty_adapter,
            scorer_artifacts=scorer_artifacts,
        )
    elif uncertainty_sampler_config.type == "entropy":
        from lerobot.uncertainty.uncertainty_samplers.entropy_sampler import (
            EntropySampler,
        )

        return EntropySampler(
            config=uncertainty_sampler_config.entropy_sampler,
            model=uncertainty_adapter,
        )
    elif uncertainty_sampler_config.type == "ace":
        from lerobot.uncertainty.uncertainty_samplers.entropy_sampler import (
            ACESampler,
        )

        return ACESampler(
            config=uncertainty_sampler_config.ace_sampler,
            model=uncertainty_adapter,
        )
    elif uncertainty_sampler_config.type == "decu":
        from lerobot.uncertainty.uncertainty_samplers.decu_sampler import (
            DECUSampler,
        )

        if not scorer_artifacts.ensemble_models:
            raise ValueError(
                "DECU uncertainty sampler requires at least 2 ensemble models in scorer_artifacts."
            )

        return DECUSampler(
            config=uncertainty_sampler_config.decu_sampler,
            sampler_model=uncertainty_adapter,
            scorer_artifacts=scorer_artifacts,
        )
    else:
        raise ValueError(f"Unknown uncertainty sampler {uncertainty_sampler_config.type}.")


def make_uncertainty_scoring_metric(
    config: ScoringMetricConfig,
    uncertainty_sampler: UncertaintySampler | None = None,
) -> UncertaintyMetric:
    if config.metric_type == "inter_vel_diff":
        from lerobot.uncertainty.uncertainty_scoring.scoring_metrics import InterVelDiff

        return InterVelDiff(
            config=config,
            uncertainty_sampler=uncertainty_sampler
        )
    elif config.metric_type == "inter_vel_diff_2way":
        from lerobot.uncertainty.uncertainty_scoring.scoring_metrics import InterVelDiff2Way

        return InterVelDiff2Way(
            config=config,
            uncertainty_sampler=uncertainty_sampler
        )
    elif config.metric_type == "action_l2":
        from lerobot.uncertainty.uncertainty_scoring.scoring_metrics import ActionL2Distance

        return ActionL2Distance()
    elif config.metric_type == "terminal_variance":
        from lerobot.uncertainty.uncertainty_scoring.scoring_metrics import TerminalVariance

        return TerminalVariance()
    elif config.metric_type == "ensemble_terminal_variance":
        from lerobot.uncertainty.uncertainty_scoring.scoring_metrics import EnsembleTerminalVariance

        return EnsembleTerminalVariance()
    elif config.metric_type == "likelihood":
        from lerobot.uncertainty.uncertainty_scoring.scoring_metrics import Likelihood

        return Likelihood(
            config=config,
            uncertainty_sampler=uncertainty_sampler,
        )
    elif config.metric_type == "mode_distance":
        from lerobot.uncertainty.uncertainty_scoring.scoring_metrics import ModeDistance

        return ModeDistance(config=config)
    elif config.metric_type == "terminal_vel_norm":
        from lerobot.uncertainty.uncertainty_scoring.scoring_metrics import TerminalVelNorm

        return TerminalVelNorm(config=config)
    elif config.metric_type == "decu":
        from lerobot.uncertainty.uncertainty_scoring.scoring_metrics import DECUMetric

        return DECUMetric(config=config)
    else:
        raise ValueError(f"Unknown scoring metric type: {config.metric_type}.")


def make_flow_matching_adapter(model: nn.Module, policy_config: PreTrainedConfig) -> BaseFlowMatchingAdapter:
    if policy_config.type == "flow_matching":
        from lerobot.policies.flow_matching.flow_matching_adapter import FlowMatchingAdapter

        return FlowMatchingAdapter(config=policy_config, model=model)
    elif policy_config.type == "smolvla":
        from lerobot.policies.smolvla.smolvla_adapter import SmolVLAAdapter

        return SmolVLAAdapter(config=policy_config, model=model)
    elif policy_config.type == "xvla":
        from lerobot.policies.xvla.xvla_adapter import XVLAAdapter

        return XVLAAdapter(config=policy_config, model=model)
    elif policy_config.type == "fastwam":
        from lerobot.policies.fastwam.fastwam_adapter import FastWAMAdapter

        return FastWAMAdapter(config=policy_config, model=model)
    else:
        raise ValueError(f"No flow matching adapter available for policy type '{policy_config.type}'.")


def make_flow_matching_adapter_from_policy(policy: PreTrainedPolicy) -> BaseFlowMatchingAdapter:
    if policy.name == "flow_matching":
        from lerobot.policies.flow_matching.flow_matching_adapter import FlowMatchingAdapter

        return FlowMatchingAdapter(config=policy.config, model=policy.flow_matching)
    elif policy.name == "smolvla":
        from lerobot.policies.smolvla.smolvla_adapter import SmolVLAAdapter

        return SmolVLAAdapter(config=policy.config, model=policy.model)
    elif policy.name == "xvla":
        from lerobot.policies.xvla.xvla_adapter import XVLAAdapter

        return XVLAAdapter(config=policy.config, model=policy.model)
    elif policy.name == "fastwam":
        from lerobot.policies.fastwam.fastwam_adapter import FastWAMAdapter

        return FastWAMAdapter(config=policy.config, model=policy.model)
    else:
        raise ValueError(
            f"Cannot build flow matching adapter for policy type {policy.name}. "
            "This factory only knows how to extract the underlying flow matching model from "
            "a FlowMatchingPolicy via policy.flow_matching and SmolVLAPolicy via policy.model."
        )


def _get_policy_cls_from_policy_name(name: str) -> type[PreTrainedConfig]:
    """Get policy class from its registered name using dynamic imports.

    This is used as a helper function to import policies from 3rd party lerobot plugins.

    Args:
        name: The name of the policy.
    Returns:
        The policy class corresponding to the given name.
    """
    if name not in PreTrainedConfig.get_known_choices():
        raise ValueError(
            f"Unknown policy name '{name}'. Available policies: {PreTrainedConfig.get_known_choices()}"
        )

    config_cls = PreTrainedConfig.get_choice_class(name)
    config_cls_name = config_cls.__name__

    model_name = config_cls_name.removesuffix("Config")  # e.g., DiffusionConfig -> Diffusion
    if model_name == config_cls_name:
        raise ValueError(
            f"The config class name '{config_cls_name}' does not follow the expected naming convention."
            f"Make sure it ends with 'Config'!"
        )
    cls_name = model_name + "Policy"  # e.g., DiffusionConfig -> DiffusionPolicy
    module_path = config_cls.__module__.replace(
        "configuration_", "modeling_"
    )  # e.g., configuration_diffusion -> modeling_diffusion

    module = importlib.import_module(module_path)
    policy_cls = getattr(module, cls_name)
    return policy_cls


def _make_processors_from_policy_config(
    config: PreTrainedConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[Any, Any]:
    """Create pre- and post-processors from a policy configuration using dynamic imports.

    This is used as a helper function to import processor factories from 3rd party lerobot plugins.

    Args:
        config: The policy configuration object.
        dataset_stats: Dataset statistics for normalization.
    Returns:
        A tuple containing the input (pre-processor) and output (post-processor) pipelines.
    """

    policy_type = config.type
    function_name = f"make_{policy_type}_pre_post_processors"
    module_path = config.__class__.__module__.replace(
        "configuration_", "processor_"
    )  # e.g., configuration_diffusion -> processor_diffusion
    logging.debug(
        f"Instantiating pre/post processors using function '{function_name}' from module '{module_path}'"
    )
    module = importlib.import_module(module_path)
    function = getattr(module, function_name)
    return function(config, dataset_stats=dataset_stats)
