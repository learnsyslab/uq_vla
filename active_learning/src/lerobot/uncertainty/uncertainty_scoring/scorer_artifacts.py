from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from laplace import Laplace

from lerobot.configs.default import DatasetConfig
from lerobot.fiper_data_generator.configuration_fiper_rollout_scorer import (
    FiperRolloutScorerConfig,
)
from lerobot.policies.common.flow_matching.adapter import BaseFlowMatchingAdapter
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.processor import PolicyProcessorPipeline

from ..uncertainty_samplers.configuration_uncertainty_sampler import UncertaintySamplerConfig
from .ensemble_utils.factory import build_ensemble_models
from .laplace_utils.posterior_builder import get_laplace_posterior


@dataclass
class ScorerArtifacts:
    """
    Artifacts required by the uncertainty scorer.

    Attributes:
        ensemble_models: Model adapters used when scorer_type='ensemble'.
        laplace_posterior: Laplace posterior used when scorer_type='laplace'.
    """
    ensemble_models: list[BaseFlowMatchingAdapter] | None = None
    laplace_posterior: Laplace | None = None

def build_scorer_artifacts_for_uncertainty_sampler(
    uncertainty_sampler_cfg: UncertaintySamplerConfig,
    policy: PreTrainedPolicy,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    dataset_cfg: DatasetConfig | None = None,
) -> ScorerArtifacts:
    """
    Build scorer artifacts (ensemble model or Laplace posterior) from the active uncertainty sampler config.
    """
    active_config = uncertainty_sampler_cfg.active_config
    scorer_type = getattr(active_config, "scorer_type", None)
    if scorer_type is None:
        return ScorerArtifacts()
    if scorer_type == "ensemble":
        ensemble_models = build_ensemble_models(
            ensemble_model_paths=active_config.ensemble_model_paths,
            policy_cfg=policy.config,
        )
        return ScorerArtifacts(ensemble_models=ensemble_models)
    if scorer_type == "laplace":
        laplace_posterior = get_laplace_posterior(
            policy=policy,
            preprocessor=preprocessor,
            laplace_config=active_config.laplace_config,
            dataset_cfg=dataset_cfg,
        )
        return ScorerArtifacts(laplace_posterior=laplace_posterior)
    raise ValueError(f"Unknown scorer_type: {scorer_type!r}")

def build_scorer_artifacts_for_fiper_scorer(
    fiper_data_scorer_cfg: FiperRolloutScorerConfig,
    policy: PreTrainedPolicy,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    dataset_cfg: DatasetConfig | None = None,
) -> ScorerArtifacts:
    """
    Build both ensemble model and Laplace posterior artifacts for the FIPER data recorder.
    """
    if (
        fiper_data_scorer_cfg.is_method_enabled("bayesian_ensemble")
        or fiper_data_scorer_cfg.is_method_enabled("composed_bayesian_ensemble")
    ):
        ensemble_models = build_ensemble_models(
            ensemble_model_paths=fiper_data_scorer_cfg.ensemble_model_paths,
            policy_cfg=policy.config,
        )
    else:
        ensemble_models = None

    if (
        fiper_data_scorer_cfg.is_method_enabled("bayesian_laplace")
        or fiper_data_scorer_cfg.is_method_enabled("composed_bayesian_laplace")
    ):
        laplace_posterior = get_laplace_posterior(
            policy=policy,
            preprocessor=preprocessor,
            laplace_config=fiper_data_scorer_cfg.laplace_config,
            dataset_cfg=dataset_cfg,
        )
    else:
        laplace_posterior = None

    return ScorerArtifacts(ensemble_models=ensemble_models, laplace_posterior=laplace_posterior)
