from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

from lerobot.policies.common.flow_matching.adapter import BaseFlowMatchingAdapter
from lerobot.policies.factory import make_uncertainty_scoring_metric

from ..uncertainty_scoring.scorer_artifacts import ScorerArtifacts
from .configuration_uncertainty_sampler import DECUSamplerConfig
from .uncertainty_sampler import UncertaintySampler


class DECUSampler(UncertaintySampler):
    """
    DECU (Diffusion Ensembles for Capturing Uncertainty) sampler.

    Generates action sequences by integrating the flow-matching ODE in two phases:

      1. Shared trunk: integrate from t=0 (noise prior) up to t=b (branching time)
         with a single shared ensemble member. The intermediate state x_b is the
         common branching point.
      2. Scoring: evaluate the DECU PaiDE score using ALL ensemble members'
         velocities at (x_b, b) to obtain the epistemic uncertainty.
      3. Completion: continue the ODE integration with the same shared trunk model
         from t=b to t=1 to produce the final action candidates.
    """

    def __init__(
        self,
        config: DECUSamplerConfig,
        sampler_model: BaseFlowMatchingAdapter,
        scorer_artifacts: ScorerArtifacts,
    ):
        """
        Args:
            config: DECU sampler-specific settings.
            sampler_model: Adapter wrapping the primary flow matching model. Used as
                the trunk if `trunk_model_index` is None or out-of-range.
            scorer_artifacts: Provides `ensemble_models`, the list of M ensemble
                adapters used for the PaiDE score.
        """
        # Make sure the branching time is part of the sampling time grid so we can
        # cleanly split the integration into pre- and post-branching phases.
        branching_time = config.scoring_metric.branching_time
        super().__init__(
            model=sampler_model,
            num_action_samples=config.num_action_samples,
            extra_sampling_times=[branching_time],
        )
        self.method_name = "decu"

        if not scorer_artifacts.ensemble_models or len(scorer_artifacts.ensemble_models) < 2:
            raise ValueError(
                "DECUSampler requires at least 2 ensemble models in scorer_artifacts.ensemble_models."
            )
        self.ensemble_models: List[BaseFlowMatchingAdapter] = scorer_artifacts.ensemble_models

        self.scoring_metric = make_uncertainty_scoring_metric(
            config=config.scoring_metric,
            uncertainty_sampler=self,
        )

        self.config = config
        self.branching_time = branching_time

        # Pre-split the sampling time grid into the pre- and post-branching parts.
        # The branching time is guaranteed to be present thanks to extra_sampling_times.
        grid = self.sampling_time_grid
        b_tensor = torch.tensor(branching_time, device=grid.device, dtype=grid.dtype)
        pre_mask = grid <= b_tensor + 1e-6
        post_mask = grid >= b_tensor - 1e-6
        self.pre_branch_time_grid = grid[pre_mask]
        self.post_branch_time_grid = grid[post_mask]

    def _select_trunk_model(self, generator: Optional[torch.Generator]) -> BaseFlowMatchingAdapter:
        """Pick the ensemble member used as the shared trunk."""
        idx = self.config.trunk_model_index
        if idx is None:
            rand_idx = torch.randint(
                low=0,
                high=len(self.ensemble_models),
                size=(1,),
                generator=generator,
                device="cpu",
            ).item()
            return self.ensemble_models[rand_idx]
        if not (0 <= idx < len(self.ensemble_models)):
            raise ValueError(
                f"trunk_model_index={idx} is out of range for {len(self.ensemble_models)} ensemble models."
            )
        return self.ensemble_models[idx]

    def _batched_uncertainty(
        self,
        observation: Dict[str, Tensor],
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[Tensor, Tensor, int]:
        """
        Run the three-phase DECU procedure and return action candidates with their
        per-batch DECU uncertainty scores.
        """
        batch_size = self._observation_batch_size(observation)

        # Pick the trunk and prepare its conditioned velocity function once.
        trunk_model = self._select_trunk_model(generator)
        trunk_conditioning = trunk_model.prepare_conditioning(observation, self.num_action_samples)
        trunk_velocity_fn = trunk_model.make_velocity_fn(conditioning=trunk_conditioning)

        # Phase 1: integrate the shared trunk from t=0 up to the branching time.
        noise_sample = trunk_model.sample_prior(
            num_samples=batch_size * self.num_action_samples,
            generator=generator,
        )
        pre_states = self.sampling_ode_solver.sample(
            x_0=noise_sample,
            velocity_fn=trunk_velocity_fn,
            method=self.ode_solver_config["solver_method"],
            atol=self.ode_solver_config["atol"],
            rtol=self.ode_solver_config["rtol"],
            time_grid=self.pre_branch_time_grid,
            return_intermediate_states=True,
            allow_partial_time_grid=True,
        )
        x_b = pre_states[-1]

        # Phase 2: evaluate every ensemble member's velocity at (x_b, b) and apply PaiDE.
        ensemble_velocity_fns = []
        for member in self.ensemble_models:
            member_cond = member.prepare_conditioning(observation, self.num_action_samples)
            ensemble_velocity_fns.append(member.make_velocity_fn(conditioning=member_cond))

        time_b = torch.tensor(self.branching_time, device=x_b.device, dtype=x_b.dtype)
        per_sample_scores = self.scoring_metric(
            x_b=x_b,
            time_b=time_b,
            velocity_fns=ensemble_velocity_fns,
        )
        # Average across the num_action_samples axis to get a per-observation score.
        uncertainty = per_sample_scores.reshape(batch_size, self.num_action_samples).mean(dim=1)

        # Phase 3: continue the trunk integration from x_b at t=b through t=1.
        post_states = self.sampling_ode_solver.sample(
            x_0=x_b,
            velocity_fn=trunk_velocity_fn,
            method=self.ode_solver_config["solver_method"],
            atol=self.ode_solver_config["atol"],
            rtol=self.ode_solver_config["rtol"],
            time_grid=self.post_branch_time_grid,
            return_intermediate_states=True,
            allow_partial_time_grid=True,
        )
        final_actions = post_states[-1]

        action_candidates = final_actions.reshape(
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
        """Reset internal state to prepare for a new rollout."""
        pass
