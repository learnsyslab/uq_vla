from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

from lerobot.policies.common.flow_matching.adapter import BaseFlowMatchingAdapter
from lerobot.policies.factory import make_uncertainty_scoring_metric

from ..uncertainty_scoring.laplace_utils.posterior_builder import sample_from_posterior
from ..uncertainty_scoring.scorer_artifacts import ScorerArtifacts
from .configuration_uncertainty_sampler import (
    CrossBayesianSamplerConfig,
)
from .uncertainty_sampler import UncertaintySampler


class CrossBayesianSampler(UncertaintySampler):
    """
    Samples action sequences from a "sampler" flow-matching model and evaluates their
    uncertainty under a separate "scorer" flow-matching model. The "scorer" model can be
    either an independently trained ensemble model or a Laplace posterior draw.
    Uncertainty can be measured using several different metrics.
    """
    def __init__(
        self,
        config: CrossBayesianSamplerConfig,
        sampler_model: BaseFlowMatchingAdapter,
        scorer_artifacts: ScorerArtifacts,
    ):
        """
        Initializes the Initializes the cross bayesian sampler.

        Args:
            config: Sampler-specific settings.
            sampler_model: The flow matching model using for sampling actions.
            scorer_artifacts: Artifacts required by the scorer. Provide exactly one matching the configured scorer type.
                - ensemble_adapter: Adapter that wraps auxiliary flow matching model used for scoring when the scorer
                    type is "ensemble".
                - laplace_posterior: A Laplace approximation posterior used for scoring when the scorer type is "laplace".
        """
        extra_sampling_times = config.scoring_metric.velocity_eval_times if (config.scoring_metric.metric_type in ("vfd_oneway", "vfd")) else None

        super().__init__(
            model=sampler_model,
            num_action_samples=config.num_action_samples,
            extra_sampling_times=extra_sampling_times,
        )
        self.method_name = "cross_bayesian"

        # Initialize scoring metric
        self.scoring_metric = make_uncertainty_scoring_metric(
            config=config.scoring_metric,
            uncertainty_sampler=self,
        )

        self.ensemble_models = scorer_artifacts.ensemble_models
        self.laplace_posterior = scorer_artifacts.laplace_posterior
        self.num_laplace_samples = config.laplace_config.num_samples
        if config.scorer_type == "ensemble" and not self.ensemble_models:
            raise ValueError("At least one ensemble model is required for scorer_type='ensemble'.")
        elif config.scorer_type == "laplace" and not self.laplace_posterior:
            raise ValueError("Laplace posterior is required for scorer_type='laplace'.")
        elif config.scorer_type not in {"ensemble", "laplace"}:
            raise ValueError(f"Unknown scorer_type: {config.scorer_type!r}")
        self.scorer_models: List[BaseFlowMatchingAdapter] = []

        # Sampler-specific settings
        self.config = config

    def _batched_uncertainty(
        self,
        observation: Dict[str, Tensor],
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[Tensor, Tensor, int]:
        """
        Generates action sequences using a sampler flow matching model, then scores these
        samples under a Laplace-sampled or an ensemble model, and finally averages these scores
        to obtain an epistemic uncertainty meaure.

        Args:
            observation: Info about the environment used to create the conditioning for
                the flow matching model.
            generator: PyTorch random number generator.

        Returns:
            - Action sequences drawn from the sampler model.
              Shape: [num_action_samples, horizon, action_dim].
            - Uncertainty score where a higher value means more uncertain.
        """
        # Build the velocity function conditioned on the current observation
        batch_size = self._observation_batch_size(observation)
        conditioning = self.model.prepare_conditioning(observation, self.num_action_samples)
        velocity_fn = self.model.make_velocity_fn(conditioning=conditioning)

        # Sample noise priors
        noise_sample = self.model.sample_prior(
            num_samples=batch_size * self.num_action_samples,
            generator=generator,
        )

        # Solve ODE forward from noise to sample action sequences
        ode_states = self.sampling_ode_solver.sample(
            x_0=noise_sample,
            velocity_fn=velocity_fn,
            method=self.ode_solver_config["solver_method"],
            atol=self.ode_solver_config["atol"],
            rtol=self.ode_solver_config["rtol"],
            time_grid=self.sampling_time_grid,
            return_intermediate_states=True,
        )

        if self.config.scorer_type == "laplace":
            # Draw flow matching model from the Laplace posterior
            self.scorer_models = sample_from_posterior(
                laplace_posterior=self.laplace_posterior,
                uncertainty_adapter=self.model,
                num_samples=self.num_laplace_samples,
                generator=generator,
            )
        else:
            self.scorer_models = self.ensemble_models

        # Ensemble terminal variance: run all M members from the same noise_sample
        # and compute log-variance across members at the terminal state.
        if self.scoring_metric.name == "ensemble_terminal_variance":
            all_terminals = [ode_states[-1]]
            for scorer in self.scorer_models:
                sc_cond = scorer.prepare_conditioning(observation, self.num_action_samples)
                sc_vf = scorer.make_velocity_fn(conditioning=sc_cond)
                sc_states = self.sampling_ode_solver.sample(
                    x_0=noise_sample,
                    velocity_fn=sc_vf,
                    method=self.ode_solver_config["solver_method"],
                    atol=self.ode_solver_config["atol"],
                    rtol=self.ode_solver_config["rtol"],
                    time_grid=self.sampling_time_grid,
                    return_intermediate_states=True,
                )
                all_terminals.append(sc_states[-1])
            stacked = torch.stack(all_terminals, dim=0)  # (M, B*N, H, A)
            per_sample_scores = self.scoring_metric(ensemble_actions=stacked)  # (B*N,)
            uncertainty = per_sample_scores.reshape(batch_size, self.num_action_samples).mean(dim=1)
            action_candidates = ode_states[-1].reshape(
                batch_size, self.num_action_samples, self.horizon, self.action_dim
            )
            return action_candidates, uncertainty, batch_size

        # Compute uncertainty scores from each scorer model
        scorer_uncertainty_means: List[Tensor] = []
        # For vfd: store (ode_states, velocity_fn) per scorer for cross-scorer pairs.
        scorer_states_and_vfs: List[Tuple[Tensor, object]] = []
        for scorer in self.scorer_models:
            # Build the conditioned velocity function of the scorer
            scorer_conditioning = scorer.prepare_conditioning(observation, self.num_action_samples)
            scorer_velocity_fn = scorer.make_velocity_fn(conditioning=scorer_conditioning)

            # Compute uncertainty based on selected metric
            if self.scoring_metric.name in ("terminal_vel_norm", "mode_distance", "likelihood"):
                uncertainty_scores = self.scoring_metric(
                    action_sequences=ode_states[-1],
                    velocity_fn=scorer_velocity_fn,
                )
            elif self.scoring_metric.name == "vfd_oneway":
                uncertainty_scores = self.scoring_metric(
                    ref_ode_states=ode_states,
                    ref_velocity_fn=velocity_fn,
                    cmp_ode_states=ode_states,
                    cmp_velocity_fn=scorer_velocity_fn,
                )
            elif self.scoring_metric.name == "vfd":
                cmp_noise = self.model.sample_prior(
                    num_samples=batch_size * self.num_action_samples,
                    generator=generator,
                )
                cmp_ode_states = self.sampling_ode_solver.sample(
                    x_0=cmp_noise,
                    velocity_fn=scorer_velocity_fn,
                    method=self.ode_solver_config["solver_method"],
                    atol=self.ode_solver_config["atol"],
                    rtol=self.ode_solver_config["rtol"],
                    time_grid=self.sampling_time_grid,
                    return_intermediate_states=True,
                )
                uncertainty_scores = self.scoring_metric(
                    ref_ode_states=ode_states,
                    ref_velocity_fn=velocity_fn,
                    cmp_ode_states=cmp_ode_states,
                    cmp_velocity_fn=scorer_velocity_fn,
                )
                scorer_states_and_vfs.append((cmp_ode_states, scorer_velocity_fn))
            elif self.scoring_metric.name == "action_l2":
                # Sample independent noise for the scorer model
                cmp_noise = self.model.sample_prior(
                    num_samples=batch_size * self.num_action_samples,
                    generator=generator,
                )
                cmp_ode_states = self.sampling_ode_solver.sample(
                    x_0=cmp_noise,
                    velocity_fn=scorer_velocity_fn,
                    method=self.ode_solver_config["solver_method"],
                    atol=self.ode_solver_config["atol"],
                    rtol=self.ode_solver_config["rtol"],
                    time_grid=self.sampling_time_grid,
                    return_intermediate_states=True,
                )
                uncertainty_scores = self.scoring_metric(
                    ref_actions=ode_states[-1],
                    cmp_actions=cmp_ode_states[-1],
                    batch_size=batch_size,
                )
            elif self.scoring_metric.name == "terminal_variance":
                # Draw independent action sequences from the scorer model and compute
                # their empirical variance across the sample dimension
                cmp_noise = self.model.sample_prior(
                    num_samples=batch_size * self.num_action_samples,
                    generator=generator,
                )
                cmp_ode_states = self.sampling_ode_solver.sample(
                    x_0=cmp_noise,
                    velocity_fn=scorer_velocity_fn,
                    method=self.ode_solver_config["solver_method"],
                    atol=self.ode_solver_config["atol"],
                    rtol=self.ode_solver_config["rtol"],
                    time_grid=self.sampling_time_grid,
                    return_intermediate_states=True,
                )
                uncertainty_scores = self.scoring_metric(
                    cmp_actions=cmp_ode_states[-1],
                    batch_size=batch_size,
                )
            else:
                raise ValueError(f"Unknown uncertainty metric: {self.scoring_metric.name}.")

            # Average uncertainty scores of a single scorer over all action samples
            scorer_uncertainty_means.append(
                uncertainty_scores.reshape(batch_size, self.num_action_samples).mean(dim=1)
            )

        # For vfd: add all C(M, 2) scorer-scorer pairs. Each scorer's ODE states
        # were already computed during the loop above, so this only incurs velocity evaluations.
        for i, (states_i, vf_i) in enumerate(scorer_states_and_vfs):
            for j, (states_j, vf_j) in enumerate(scorer_states_and_vfs):
                if j <= i:
                    continue
                cross_scores = self.scoring_metric(
                    ref_ode_states=states_i,
                    ref_velocity_fn=vf_i,
                    cmp_ode_states=states_j,
                    cmp_velocity_fn=vf_j,
                )
                scorer_uncertainty_means.append(
                    cross_scores.reshape(batch_size, self.num_action_samples).mean(dim=1)
                )

        # Average uncertainty scores and store for logging
        uncertainty = torch.stack(scorer_uncertainty_means, dim=0).mean(dim=0)
        action_candidates = ode_states[-1].reshape(
            batch_size,
            self.num_action_samples,
            self.horizon,
            self.action_dim,
        )
        return action_candidates, uncertainty, batch_size

    def conditional_sample_with_uncertainty_batch(
        self,
        observation: Dict[str, Tensor],
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[Tensor, Tensor]:
        action_candidates, uncertainty, batch_size = self._batched_uncertainty(
            observation=observation,
            generator=generator,
        )

        self.uncertainty = float(uncertainty.mean().item())
        self.action_candidates = action_candidates[0] if batch_size == 1 else action_candidates
        actions, _ = self.rand_pick_action_batch(action_candidates=action_candidates)
        return actions.to(device="cpu", dtype=torch.float32), uncertainty.to(device="cpu", dtype=torch.float32)

    def conditional_sample_with_uncertainty(
        self,
        observation: Dict[str, Tensor],
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[Tensor, float]:
        batch_size = self._observation_batch_size(observation)
        if batch_size != 1:
            raise ValueError(
                "conditional_sample_with_uncertainty expects a single observation. "
                f"Received batch_size={batch_size}; use conditional_sample_with_uncertainty_batch instead."
            )

        action_candidates, uncertainty, _ = self._batched_uncertainty(
            observation=observation,
            generator=generator,
        )
        single_action_candidates = action_candidates[0]
        self.action_candidates = single_action_candidates
        self.uncertainty = float(uncertainty[0].item())
        actions, _ = self.rand_pick_action(action_candidates=single_action_candidates)

        return actions.to(device="cpu", dtype=torch.float32), self.uncertainty

    def reset(self):
        """
        Reset internal state to prepare for a new rollout.
        """
        pass
