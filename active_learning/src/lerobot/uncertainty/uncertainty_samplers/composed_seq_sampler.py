from typing import Callable, Optional, Tuple

import torch
from torch import Tensor

from lerobot.policies.common.flow_matching.adapter import BaseFlowMatchingAdapter
from lerobot.policies.factory import make_uncertainty_scoring_metric

from .configuration_uncertainty_sampler import (
    ComposedSequenceSamplerConfig,
)
from .uncertainty_sampler import UncertaintySampler
from .utils import compose_ode_states, select_and_expand_ode_states


class ComposedSequenceSampler(UncertaintySampler):
    """
    Samples action sequences, composes them with a previously executed sequence segment,
    and evaluates their uncertainty under the flow matching model.

    The key idea is that if the composed sequences have a low uncertainty, then the model
    successfully anticipated what is likely to happen next. This implies a good internal model
    of the environment.
    """
    def __init__(
        self,
        config: ComposedSequenceSamplerConfig,
        model: BaseFlowMatchingAdapter,
    ):
        """
        Initializes the composed sequence sampler.

        Args:
            config: Sampler-specific settings.
        """
        extra_sampling_times = config.scoring_metric.velocity_eval_times if (config.scoring_metric.metric_type == "inter_vel_diff") else None

        super().__init__(
            model=model,
            num_action_samples=config.num_action_samples,
            extra_sampling_times=extra_sampling_times,
        )
        self.method_name = "composed_sequence"

        # Initialize scoring metric
        self.scoring_metric = make_uncertainty_scoring_metric(
            config=config.scoring_metric,
            uncertainty_sampler=self,
        )

        if self.scoring_metric.name == "inter_vel_diff":
            raise ValueError(
                "Composed sequence sampler is not compatible with intermediate velocity difference score."
            )

        # Sampler-specific settings
        self.config = config

        # Store the velocity function and ODE states from the previous action sequence generation
        self.prev_velocity_fn: Callable[[Tensor, Tensor], Tensor] | None = None
        self.prev_ode_states: Tensor | None = None

        # Index of the selected action sequence from the previous actions batch
        self.prev_selected_action_idx: int | None = None

    def conditional_sample_with_uncertainty(
        self,
        observation: dict[str, Tensor],
        generator: Optional[torch.Generator] = None
    ) -> Tuple[Tensor, float]:
        """
        Samples num_action_samples many new action sequences and computes uncertainty
        scores by composing them with a previous action sequence and evaluating them under
        the flow model.

        Args:
            observation: Info about the environment used to create the conditioning for
                the flow matching model.
            generator: PyTorch random number generator.

        Returns:
            - Action sequence samples. Shape: [num_action_samples, horizon, action_dim].
            - Uncertainty score where a higher value means more uncertain.
        """
        # Build the velocity function conditioned on the current observation
        conditioning = self.model.prepare_conditioning(observation, self.num_action_samples)
        velocity_fn = self.model.make_velocity_fn(conditioning=conditioning)

        # Sample noise priors
        new_noise_sample = self.model.sample_prior(
            num_samples=self.num_action_samples,
            generator=generator,
        )

        # Solve ODE forward from noise to sample action sequences
        new_ode_states = self.sampling_ode_solver.sample(
            x_0=new_noise_sample,
            velocity_fn=velocity_fn,
            method=self.ode_solver_config["solver_method"],
            atol=self.ode_solver_config["atol"],
            rtol=self.ode_solver_config["rtol"],
            time_grid=self.sampling_time_grid,
            return_intermediate_states=True,
        )

        # If no previous trajectory is stored, return placeholder uncertainties
        if self.prev_selected_action_idx is None:
            self.uncertainty = float('-inf')
        else:
            # Compose full ODE states from stored previous and new action generation
            composed_ode_states = compose_ode_states(
                prev_ode_states=self.prev_ode_states[
                    :, self.prev_selected_action_idx:self.prev_selected_action_idx+1, :, :
                ],
                new_ode_states=new_ode_states,
                horizon=self.horizon,
                n_action_steps=self.n_action_steps,
                n_obs_steps=self.n_obs_steps
            )

            # Compute uncertainty based on selected metric
            if self.scoring_metric.name in ("terminal_vel_norm", "mode_distance", "likelihood"):
                uncertainty_scores = self.scoring_metric(
                    action_sequences=composed_ode_states[-1],
                    velocity_fn=self.prev_velocity_fn,
                )
            else:
                raise ValueError(f"Invalid uncertainty metric: {self.scoring_metric.name}.")

            # Average uncertainty scores and store for logging
            self.uncertainty = uncertainty_scores.mean().item()

        # Store velocity function and ODE states from the previous action sampling step
        self.prev_velocity_fn = velocity_fn
        self.prev_ode_states = new_ode_states

        # Pick one action sequence at random
        self.action_candidates = new_ode_states[-1]
        actions, self.prev_selected_action_idx = self.rand_pick_action(action_candidates=self.action_candidates)

        return actions.to(device="cpu", dtype=torch.float32), self.uncertainty

    def reset(self):
        """
        Reset internal state to prepare for a new rollout.
        """
        # Clear stored velocity function, ODE states and selected action sequence from previous step
        self.prev_velocity_fn = None
        self.prev_ode_states = None
        self.prev_selected_action_idx = None
