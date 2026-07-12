import pickle
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.distributions import Independent, Normal
from tqdm.auto import tqdm

from lerobot.policies.common.flow_matching.adapter import BaseFlowMatchingAdapter
from lerobot.policies.common.flow_matching.conditional_probability_path import (
    OTCondProbPath,
    VPDiffusionCondProbPath,
)
from lerobot.policies.common.flow_matching.ode_solver import (
    ODESolver,
    make_lik_estimation_time_grid,
    select_ode_states,
)
from lerobot.uncertainty.uncertainty_samplers.utils import compose_ode_states, select_and_expand_ode_states
from lerobot.uncertainty.uncertainty_scoring.laplace_utils.posterior_builder import (
    sample_from_posterior,
)
from lerobot.uncertainty.uncertainty_scoring.scorer_artifacts import ScorerArtifacts

from .configuration_fiper_rollout_scorer import FiperRolloutScorerConfig


class FiperRolloutScorer:
    def __init__(
        self,
        config: FiperRolloutScorerConfig,
        flow_matching_adapter: BaseFlowMatchingAdapter,
        scorer_artifacts: ScorerArtifacts,
    ):
        self.config = config
        self.flow_matching_adapter = flow_matching_adapter
        self.ode_solver = ODESolver()

        self.horizon = flow_matching_adapter.horizon
        self.n_action_steps = flow_matching_adapter.n_action_steps
        self.n_obs_steps = flow_matching_adapter.n_obs_steps
        self.action_dim = flow_matching_adapter.action_dim
        self.device = flow_matching_adapter.device
        self.dtype = flow_matching_adapter.dtype

        # Extract scorer artifacts
        self.ensemble_adapters = scorer_artifacts.ensemble_models
        self.laplace_posterior = scorer_artifacts.laplace_posterior
        self.num_laplace_samples = config.laplace_config.num_samples
        if (
            self.config.is_method_enabled("bayesian_ensemble")
            or self.config.is_method_enabled("composed_bayesian_ensemble")
        ) and self.ensemble_adapters is None:
            raise ValueError("At least one ensemble model is required for the 'bayesian_ensemble' method.")
        if (
            self.config.is_method_enabled("bayesian_laplace")
            or self.config.is_method_enabled("composed_bayesian_laplace")
        ) and self.laplace_posterior is None:
            raise ValueError("Laplace posterior is required for the 'bayesian_laplace' method.")

        # Noise distribution is an isotropic Gaussian
        self.gaussian_log_density = Independent(
            Normal(
                loc = torch.zeros(self.horizon, self.action_dim, device=self.device, dtype=self.dtype),
                scale = torch.ones(self.horizon, self.action_dim, device=self.device, dtype=self.dtype),
            ),
            reinterpreted_batch_ndims=2
        ).log_prob

        # Build time grid for likelihood estimation based on ODE solver method
        self.lik_estimation_time_grid = make_lik_estimation_time_grid(
            ode_solver_method=self.config.likelihood_ode_solver_cfg.method,
            device=self.device,
            dtype=self.dtype,
        )

        # Select conditional probability path for computing the intermediate velocity difference scaling factors
        self.cond_vf_config = flow_matching_adapter.cond_vf_config
        if self.cond_vf_config["type"] == "vp":
            self.cond_prob_path = VPDiffusionCondProbPath(
                beta_min=self.cond_vf_config["beta_min"],
                beta_max=self.cond_vf_config["beta_max"],
            )
        elif self.cond_vf_config["type"] == "ot":
            self.cond_prob_path = OTCondProbPath(self.cond_vf_config["sigma_min"])
        else:
            raise ValueError(
                f"Unknown conditional vector field type {self.cond_vf_config['type']}."
            )

        # Store the velocity function and ODE states from the previous action sequence generation
        self.prev_sampler_velocity_fn: Callable[[Tensor, Tensor], Tensor] | None = None
        self.prev_ensemble_velocity_fns: list[Callable[[Tensor, Tensor], Tensor]] | None = None
        self.prev_laplace_velocity_fns: list[Callable[[Tensor, Tensor], Tensor]] | None = None
        self.prev_ode_states: Tensor | None = None
        self.prev_velocities: Tensor | None = None
        self.prev_selected_action_idx: int | None = None
        self.prev_action_sample: Tensor | None = None

        # Store data from action generation steps across rollout
        self.fiper_data: list[dict[str, Any]] = []

    def eval_terminal_velocities(
        self,
        action_samples: Tensor,
        velocity_fns: list[Callable[[Tensor, Tensor], Tensor]],
    ) -> Tensor:
        """
        Evaluate terminal velocities for one or many velocity functions.
        """
        velocities_over_time = []
        for time in self.config.terminal_vel_eval_times:
            time_tensor = torch.tensor(time, device=self.device, dtype=self.dtype)
            velocities_at_t = torch.stack(
                [vf(x_t=action_samples, t=time_tensor) for vf in velocity_fns],
                dim=0
            )
            velocities_over_time.append(velocities_at_t)

        terminal_velocities = torch.stack(velocities_over_time, dim=0)

        return terminal_velocities

    def make_nan_terminal_velocities(
        self,
        reference_samples: Tensor,
        num_models: int | None = None,
    ) -> Tensor:
        """
        Create NaN-filled placeholder terminal velocities matching single/multi layout.
        """
        if num_models is None:
            out_shape = (len(self.config.terminal_vel_eval_times), *reference_samples.shape)
        else:
            out_shape = (len(self.config.terminal_vel_eval_times), num_models, *reference_samples.shape)

        return torch.full(out_shape, float("nan"), device=self.device, dtype=self.dtype)

    def compute_log_likelihood(
        self,
        action_samples: Tensor,
        velocity_fn: Callable[[Tensor, Tensor], Tensor],
        generator: torch.Generator | None = None
    ) -> Tensor:
        """Compute log-likelihood of sampled action sequences."""
        _, log_probs = self.ode_solver.sample_with_log_likelihood(
            x_init=action_samples,
            time_grid=self.lik_estimation_time_grid,
            velocity_fn=velocity_fn,
            log_p_0=self.gaussian_log_density,
            method=self.config.likelihood_ode_solver_cfg.method,
            atol=self.config.likelihood_ode_solver_cfg.atol,
            rtol=self.config.likelihood_ode_solver_cfg.rtol,
            exact_divergence=self.config.likelihood_ode_solver_cfg.exact_divergence,
            generator=generator,
        )

        return log_probs

    def compute_log_likelihoods(
        self,
        action_samples: Tensor,
        velocity_fns: list[Callable[[Tensor, Tensor], Tensor]],
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Compute log-likelihoods for a list of velocity functions."""
        return torch.stack(
            [
                self.compute_log_likelihood(
                    action_samples=action_samples,
                    velocity_fn=vel_fn,
                    generator=generator,
                )
                for vel_fn in velocity_fns
            ],
            dim=0
        )

    def make_nan_log_likelihoods(
        self,
        num_sequences: int,
        num_models: int | None = None,
    ) -> Tensor:
        """Create NaN-filled placeholders for log-likelihoods."""
        shape = (num_sequences,) if num_models is None else (num_models, num_sequences)

        return torch.full(shape, float("nan"), device=self.device, dtype=self.dtype)


    def eval_statewise_velocities(
        self,
        selected_states: Tensor,
        eval_times: list[float] | Tensor,
        velocity_fns: list[Callable[[Tensor, Tensor], Tensor]],
    ) -> Tensor:
        """Evaluate velocities for already-selected ODE states at their matching eval times."""
        vels_over_time: list[Tensor] = []
        for ode_state, time in zip(selected_states, eval_times, strict=True):
            time_tensor = torch.as_tensor(time, device=self.device, dtype=self.dtype)
            vels_at_t = torch.stack([vf(x_t=ode_state, t=time_tensor) for vf in velocity_fns], dim=0)
            vels_over_time.append(vels_at_t)

        return torch.stack(vels_over_time, dim=0)


    def make_nan_statewise_velocities(
        self,
        reference_selected_states: Tensor,
        num_models: int | None = None,
    ) -> Tensor:
        """NaN placeholders matching eval_statewise_velocities output shapes."""
        if num_models is None:
            shape = reference_selected_states.shape
        else:
            shape = (reference_selected_states.shape[0], num_models, *reference_selected_states.shape[1:])

        return torch.full(shape, float("nan"), device=self.device, dtype=self.dtype)

    def _integrated_velocity_difference(
        self,
        ref_velocities: Tensor,
        cmp_velocities: Tensor,
        eval_times: Tensor,
        vel_diff_scaling: Tensor,
    ) -> Tensor:
        """Integrate squared velocity disagreement and average over all non-time axes."""
        dt = torch.empty_like(eval_times)
        dt[:-1] = eval_times[1:] - eval_times[:-1]
        dt[-1] = 1.0 - eval_times[-1]

        vel_diffs = (ref_velocities - cmp_velocities).square().sum(dim=(-2, -1))
        weights = (vel_diff_scaling * dt).view(-1, *([1] * (vel_diffs.ndim - 1)))
        return (weights * vel_diffs).sum(dim=0).mean()

    def compute_ensemble_vfd_oneway_scores(
        self,
        sampler_velocities: Tensor,
        ensemble_velocities: Tensor,
        eval_times: Tensor,
        vel_diff_scaling: Tensor,
    ) -> Tensor:
        """One-way sampler-path velocity disagreement used by the existing FIPER score."""
        return self._integrated_velocity_difference(
            ref_velocities=sampler_velocities.unsqueeze(1),
            cmp_velocities=ensemble_velocities,
            eval_times=eval_times,
            vel_diff_scaling=vel_diff_scaling,
        )

    def eval_ensemble_velocities_on_sampler_states(
        self,
        selected_states: Tensor,
        sampler_velocities: Tensor,
        eval_times: list[float] | Tensor,
        ensemble_velocity_fns: list[Callable[[Tensor, Tensor], Tensor]],
    ) -> Tensor:
        """Evaluate ensemble velocities, reusing sampler velocities for a duplicated sampler member."""
        sampler_member_idx = self.config.sampler_ensemble_member_index
        if sampler_member_idx is None:
            return self.eval_statewise_velocities(
                selected_states=selected_states,
                eval_times=eval_times,
                velocity_fns=ensemble_velocity_fns,
            )

        if not 0 <= sampler_member_idx < len(ensemble_velocity_fns):
            raise ValueError(
                f"sampler_ensemble_member_index={sampler_member_idx} is outside "
                f"the ensemble range [0, {len(ensemble_velocity_fns) - 1}]."
            )

        ensemble_velocities = torch.empty(
            (sampler_velocities.shape[0], len(ensemble_velocity_fns), *sampler_velocities.shape[1:]),
            device=self.device,
            dtype=self.dtype,
        )
        ensemble_velocities[:, sampler_member_idx] = sampler_velocities

        other_indices = [
            idx for idx in range(len(ensemble_velocity_fns))
            if idx != sampler_member_idx
        ]
        if other_indices:
            other_velocities = self.eval_statewise_velocities(
                selected_states=selected_states,
                eval_times=eval_times,
                velocity_fns=[ensemble_velocity_fns[idx] for idx in other_indices],
            )
            for dst_idx, src_idx in enumerate(other_indices):
                ensemble_velocities[:, src_idx] = other_velocities[:, dst_idx]

        return ensemble_velocities

    def compute_ensemble_vfd_scores(
        self,
        sampling_time_grid: Tensor,
        initial_noise: Tensor,
        selected_sampler_states: Tensor,
        selected_sampler_velocities: Tensor,
        selected_ensemble_velocities_on_sampler_states: Tensor,
        eval_times: Tensor,
        vel_diff_scaling: Tensor,
        ensemble_velocity_fns: list[Callable[[Tensor, Tensor], Tensor]],
        sampler_velocity_fn: Callable[[Tensor, Tensor], Tensor],
    ) -> Tensor:
        """
        Compute two-way VFD from the recorded sampler trajectory plus scorer trajectories.

        The score averages sampler-vs-scorer disagreement on the sampler path, on each
        scorer's own path, and scorer-vs-scorer disagreement on scorer paths.
        """
        terms = [
            self.compute_ensemble_vfd_oneway_scores(
                sampler_velocities=selected_sampler_velocities,
                ensemble_velocities=selected_ensemble_velocities_on_sampler_states,
                eval_times=eval_times,
                vel_diff_scaling=vel_diff_scaling,
            )
        ]

        for scorer_idx, scorer_velocity_fn in enumerate(ensemble_velocity_fns):
            if scorer_idx == self.config.sampler_ensemble_member_index:
                selected_scorer_states = selected_sampler_states
                selected_scorer_velocities = selected_sampler_velocities
                terms.append(torch.zeros((), device=self.device, dtype=self.dtype))
            else:
                scorer_states, scorer_velocities = self.ode_solver.sample(
                    x_0=initial_noise,
                    velocity_fn=scorer_velocity_fn,
                    method="euler",
                    atol=None,
                    rtol=None,
                    time_grid=sampling_time_grid,
                    return_intermediate_states=True,
                    return_intermediate_vels=True,
                )
                selected_scorer_states, _ = select_ode_states(
                    time_grid=sampling_time_grid,
                    ode_states=scorer_states,
                    requested_times=eval_times,
                )
                selected_scorer_velocities, _ = select_ode_states(
                    time_grid=sampling_time_grid,
                    ode_states=scorer_velocities,
                    requested_times=eval_times,
                )

                sampler_velocities_on_scorer_states = self.eval_statewise_velocities(
                    selected_states=selected_scorer_states,
                    eval_times=eval_times,
                    velocity_fns=[sampler_velocity_fn],
                ).squeeze(1)
                terms.append(
                    self._integrated_velocity_difference(
                        ref_velocities=sampler_velocities_on_scorer_states,
                        cmp_velocities=selected_scorer_velocities,
                        eval_times=eval_times,
                        vel_diff_scaling=vel_diff_scaling,
                    )
                )

            if len(ensemble_velocity_fns) > 1:
                other_indices = [
                    other_idx for other_idx in range(len(ensemble_velocity_fns))
                    if other_idx != scorer_idx
                ]
                if scorer_idx == self.config.sampler_ensemble_member_index:
                    other_velocities_on_scorer_states = selected_ensemble_velocities_on_sampler_states[
                        :, other_indices
                    ]
                elif other_indices == [self.config.sampler_ensemble_member_index]:
                    other_velocities_on_scorer_states = sampler_velocities_on_scorer_states.unsqueeze(1)
                else:
                    other_velocity_fns = [ensemble_velocity_fns[other_idx] for other_idx in other_indices]
                    other_velocities_on_scorer_states = self.eval_statewise_velocities(
                        selected_states=selected_scorer_states,
                        eval_times=eval_times,
                        velocity_fns=other_velocity_fns,
                    )
                terms.append(
                    self._integrated_velocity_difference(
                        ref_velocities=selected_scorer_velocities.unsqueeze(1),
                        cmp_velocities=other_velocities_on_scorer_states,
                        eval_times=eval_times,
                        vel_diff_scaling=vel_diff_scaling,
                    )
                )

        return torch.stack(terms).mean()


    def score_step_data(self, rollout_step_data: dict[str, Any], generator: torch.Generator | None = None) -> dict[str, Any]:
        """Scores a single step of rollout data according to the configured scoring methods.

        Args:
            rollout_step_data: A dictionary containing the rollout step data to be scored.

        Returns:
            A dictionary containing the computed scores for the step data.
        """
        fiper_step_data: dict[str, Any] = {}
        fiper_step_data["obs_embedding"] = rollout_step_data["obs_embedding"]

        observation = {
            k: v.to(self.device, self.dtype) if torch.is_floating_point(v) else v.to(self.device)
            for k, v in rollout_step_data["observation"].items()
        }
        ode_states = rollout_step_data["ode_states"].to(self.device, self.dtype)
        velocities = rollout_step_data["velocities"].to(self.device, self.dtype)
        action_candidates = ode_states[-1]
        num_uncertainty_sequences = ode_states.shape[1]

        action_pred = action_candidates
        if action_pred.shape[-1] == 2:
            zero_padding = torch.zeros(
                (*action_pred.shape[:-1], 1),
                device=action_pred.device,
                dtype=action_pred.dtype,
            )
            action_pred = torch.cat([action_pred, zero_padding], dim=-1)

        fiper_step_data["action_pred"] = action_pred.detach().cpu().numpy()

        laplace_velocity_fns: list[Callable[[Tensor, Tensor], Tensor]] | None = None
        if self.config.is_method_enabled("bayesian_laplace") or self.config.is_method_enabled("composed_bayesian_laplace"):
            # Draw flow matching model from the Laplace posterior
            laplace_adapters = sample_from_posterior(
                laplace_posterior=self.laplace_posterior,
                uncertainty_adapter=self.flow_matching_adapter,
                num_samples=self.num_laplace_samples,
                generator=generator
            )

            laplace_velocity_fns: list[Callable[[Tensor, Tensor], Tensor]] = []
            for laplace_adapter in laplace_adapters:
                laplace_conditioning = laplace_adapter.prepare_conditioning(observation, num_uncertainty_sequences)
                laplace_velocity_fns.append(laplace_adapter.make_velocity_fn(conditioning=laplace_conditioning))

        ensemble_velocity_fns: list[Callable[[Tensor, Tensor], Tensor]] | None = None
        if self.config.is_method_enabled("bayesian_ensemble") or self.config.is_method_enabled("composed_bayesian_ensemble"):
            # Build the velocity functions for the ensemble members conditioned on the current observation
            ensemble_velocity_fns: list[Callable[[Tensor, Tensor], Tensor]] = []
            for ensemble_adapter in self.ensemble_adapters:
                ensemble_conditioning = ensemble_adapter.prepare_conditioning(observation, num_uncertainty_sequences)
                ensemble_velocity_fns.append(ensemble_adapter.make_velocity_fn(conditioning=ensemble_conditioning))

        vfd_enabled = self.config.should_compute("bayesian_ensemble", "vfd")

        composed_methods_enabled = (
            self.config.is_method_enabled("composed")
            or self.config.is_method_enabled("composed_bayesian_laplace")
            or self.config.is_method_enabled("composed_bayesian_ensemble")
        )
        sampler_velocity_fn: Callable[[Tensor, Tensor], Tensor] | None = None
        if composed_methods_enabled or vfd_enabled:
            # Build the velocity function for sampler and scorer model conditioned on the current observation
            sampler_conditioning = self.flow_matching_adapter.prepare_conditioning(observation, num_uncertainty_sequences)
            sampler_velocity_fn = self.flow_matching_adapter.make_velocity_fn(conditioning=sampler_conditioning)

        if self.prev_action_sample is not None and composed_methods_enabled:
            composed_action_samples = compose_ode_states(
                prev_ode_states=self.prev_action_sample.unsqueeze(0),
                new_ode_states=action_candidates,
                horizon=self.horizon,
                n_action_steps=self.n_action_steps,
                n_obs_steps=self.n_obs_steps
            )
            if "ode_states_for_composed_vfd_oneway" in rollout_step_data:
                composed_ode_states = compose_ode_states(
                    prev_ode_states=self.prev_ode_states[
                        :, self.prev_selected_action_idx:self.prev_selected_action_idx+1, :, :
                    ],
                    new_ode_states=rollout_step_data["ode_states_for_composed_vfd_oneway"].to(self.device, self.dtype),
                    horizon=self.horizon,
                    n_action_steps=self.n_action_steps,
                    n_obs_steps=self.n_obs_steps
                )

        if any(self.config.should_compute(m, "mode_distance") for m in self.config.scores_by_method):
            fiper_step_data["terminal_eval_times"] = np.asarray(self.config.terminal_vel_eval_times)

        # --------------------------------------------------------------
        # MODE DISTANCE COMPUTATION
        # --------------------------------------------------------------

        if self.config.should_compute("bayesian_laplace", "mode_distance"):
            laplace_terminal_velocities = self.eval_terminal_velocities(
                action_samples=action_candidates,
                velocity_fns=laplace_velocity_fns,
            )
            fiper_step_data["laplace_terminal_velocities"] = laplace_terminal_velocities.detach().cpu().numpy()

        if self.config.should_compute("bayesian_ensemble", "mode_distance"):
            ensemble_terminal_velocities = self.eval_terminal_velocities(
                action_samples=action_candidates,
                velocity_fns=ensemble_velocity_fns,
            )
            fiper_step_data["ensemble_terminal_velocities"] = ensemble_terminal_velocities.detach().cpu().numpy()

        if self.config.should_compute("composed", "mode_distance"):
            if self.prev_action_sample is not None:
                composed_terminal_velocities = self.eval_terminal_velocities(
                    action_samples=composed_action_samples,
                    velocity_fns=[self.prev_sampler_velocity_fn],
                ).squeeze(1)
            else:
                composed_terminal_velocities = self.make_nan_terminal_velocities(
                    reference_samples=action_candidates
                )
            fiper_step_data["composed_terminal_velocities"] = composed_terminal_velocities.detach().cpu().numpy()


        if self.config.should_compute("composed_bayesian_laplace", "mode_distance"):
            if self.prev_action_sample is not None:
                composed_laplace_terminal_velocities = self.eval_terminal_velocities(
                    action_samples=composed_action_samples,
                    velocity_fns=self.prev_laplace_velocity_fns,
                )
            else:
                composed_laplace_terminal_velocities = self.make_nan_terminal_velocities(
                    reference_samples=action_candidates,
                    num_models=len(laplace_velocity_fns)
                )
            fiper_step_data["composed_laplace_terminal_velocities"] = composed_laplace_terminal_velocities.detach().cpu().numpy()

        if self.config.should_compute("composed_bayesian_ensemble", "mode_distance"):
            if self.prev_action_sample is not None:
                composed_ensemble_terminal_velocities = self.eval_terminal_velocities(
                    action_samples=composed_action_samples,
                    velocity_fns=self.prev_ensemble_velocity_fns,
                )
            else:
                composed_ensemble_terminal_velocities = self.make_nan_terminal_velocities(
                    reference_samples=action_candidates,
                    num_models=len(ensemble_velocity_fns)
                )
            fiper_step_data["composed_ensemble_terminal_velocities"] = composed_ensemble_terminal_velocities.detach().cpu().numpy()

        # --------------------------------------------------------------
        # LIKELIHOOD COMPUTATION
        # --------------------------------------------------------------

        if self.config.should_compute("bayesian_laplace", "likelihood"):
            laplace_log_likelihood = self.compute_log_likelihoods(
                action_samples=action_candidates,
                velocity_fns=laplace_velocity_fns,
                generator=generator,
            )
            fiper_step_data["laplace_log_likelihood"] = laplace_log_likelihood.detach().cpu().numpy()

        if self.config.should_compute("bayesian_ensemble", "likelihood"):
            ensemble_log_likelihood = self.compute_log_likelihoods(
                action_samples=action_candidates,
                velocity_fns=ensemble_velocity_fns,
                generator=generator,
            )
            fiper_step_data["ensemble_log_likelihood"] = ensemble_log_likelihood.detach().cpu().numpy()

        if self.config.should_compute("composed", "likelihood"):
            if self.prev_action_sample is not None:
                composed_log_likelihood = self.compute_log_likelihood(
                    action_samples=composed_action_samples,
                    velocity_fn=self.prev_sampler_velocity_fn,
                    generator=generator,
                )
            else:
                composed_log_likelihood = self.make_nan_log_likelihoods(num_sequences=num_uncertainty_sequences)
            fiper_step_data["composed_log_likelihood"] = composed_log_likelihood.detach().cpu().numpy()

        if self.config.should_compute("composed_bayesian_laplace", "likelihood"):
            if self.prev_action_sample is not None:
                composed_laplace_log_likelihood = self.compute_log_likelihoods(
                    action_samples=composed_action_samples,
                    velocity_fns=self.prev_laplace_velocity_fns,
                    generator=generator,
                )
            else:
                composed_laplace_log_likelihood = self.make_nan_log_likelihoods(
                    num_sequences=num_uncertainty_sequences,
                    num_models=len(laplace_velocity_fns),
                )
            fiper_step_data["composed_laplace_log_likelihood"] = composed_laplace_log_likelihood.detach().cpu().numpy()

        if self.config.should_compute("composed_bayesian_ensemble", "likelihood"):
            if self.prev_action_sample is not None:
                composed_ensemble_log_likelihood = self.compute_log_likelihoods(
                    action_samples=composed_action_samples,
                    velocity_fns=self.prev_ensemble_velocity_fns,
                    generator=generator,
                )
            else:
                composed_ensemble_log_likelihood = self.make_nan_log_likelihoods(
                    num_sequences=num_uncertainty_sequences,
                    num_models=len(ensemble_velocity_fns),
                )
            fiper_step_data["composed_ensemble_log_likelihood"] = composed_ensemble_log_likelihood.detach().cpu().numpy()

        # --------------------------------------------------------------
        # INTERMEDIATE VELOCITY DIFFERENCE COMPUTATION
        # --------------------------------------------------------------

        if any(
            self.config.should_compute(m, "vfd_oneway") or self.config.should_compute(m, "vfd")
            for m in self.config.scores_by_method
        ):
            # Select the ODE states that correspond to the ODE evaluation times
            sampling_time_grid = rollout_step_data["sampling_time_grid"].to(self.device, self.dtype)
            ode_eval_times = rollout_step_data["ode_eval_times"]
            fiper_step_data["ode_eval_times"] = ode_eval_times
            selected_ode_states, _ = select_ode_states(
                time_grid=sampling_time_grid,
                ode_states=ode_states,
                requested_times=torch.tensor(ode_eval_times, device=self.device, dtype=self.dtype)
            )
            selected_velocities, _ = select_ode_states(
                time_grid=sampling_time_grid,
                ode_states=velocities,
                requested_times=torch.tensor(ode_eval_times, device=self.device, dtype=self.dtype)
            )
            fiper_step_data["velocities"] = selected_velocities.detach().cpu().numpy()

            vel_diff_scaling_factors: list[float] = []
            for time in ode_eval_times:
                vel_diff_scaling_factors.append(self.cond_prob_path.get_vel_diff_scaling_factor(t=time))
            fiper_step_data["vel_diff_scaling"] = np.asarray(vel_diff_scaling_factors)
            eval_times_tensor = torch.tensor(ode_eval_times, device=self.device, dtype=self.dtype)
            vel_diff_scaling_tensor = torch.tensor(vel_diff_scaling_factors, device=self.device, dtype=self.dtype)

        if self.config.should_compute("bayesian_laplace", "vfd_oneway"):
            laplace_vels = self.eval_statewise_velocities(
                selected_states=selected_ode_states,
                eval_times=ode_eval_times,
                velocity_fns=laplace_velocity_fns,
            )
            fiper_step_data["laplace_velocities"] = laplace_vels.detach().cpu().numpy()

        if self.config.should_compute("bayesian_ensemble", "vfd_oneway"):
            ensemble_vels = self.eval_ensemble_velocities_on_sampler_states(
                selected_states=selected_ode_states,
                sampler_velocities=selected_velocities,
                eval_times=ode_eval_times,
                ensemble_velocity_fns=ensemble_velocity_fns,
            )
            fiper_step_data["ensemble_velocities"] = ensemble_vels.detach().cpu().numpy()
            fiper_step_data["ensemble_vfd_oneway_scores"] = np.asarray(
                self.compute_ensemble_vfd_oneway_scores(
                    sampler_velocities=selected_velocities,
                    ensemble_velocities=ensemble_vels,
                    eval_times=eval_times_tensor,
                    vel_diff_scaling=vel_diff_scaling_tensor,
                ).detach().cpu().item(),
                dtype=np.float32,
            )

        if self.config.should_compute("bayesian_ensemble", "vfd"):
            if "ensemble_vels" not in locals():
                ensemble_vels = self.eval_ensemble_velocities_on_sampler_states(
                    selected_states=selected_ode_states,
                    sampler_velocities=selected_velocities,
                    eval_times=ode_eval_times,
                    ensemble_velocity_fns=ensemble_velocity_fns,
                )
                fiper_step_data["ensemble_velocities"] = ensemble_vels.detach().cpu().numpy()

            fiper_step_data["ensemble_vfd_scores"] = np.asarray(
                self.compute_ensemble_vfd_scores(
                    sampling_time_grid=sampling_time_grid,
                    initial_noise=ode_states[0],
                    selected_sampler_states=selected_ode_states,
                    selected_sampler_velocities=selected_velocities,
                    selected_ensemble_velocities_on_sampler_states=ensemble_vels,
                    eval_times=eval_times_tensor,
                    vel_diff_scaling=vel_diff_scaling_tensor,
                    ensemble_velocity_fns=ensemble_velocity_fns,
                    sampler_velocity_fn=sampler_velocity_fn,
                ).detach().cpu().item(),
                dtype=np.float32,
            )

        if (
            self.config.should_compute("composed", "vfd_oneway")
            or self.config.should_compute("composed_bayesian_laplace", "vfd_oneway")
            or self.config.should_compute("composed_bayesian_ensemble", "vfd_oneway")
        ):
            if self.prev_action_sample is not None:
                selected_prev_velocities, _ = select_ode_states(
                    time_grid=sampling_time_grid,
                    ode_states=select_and_expand_ode_states(self.prev_velocities, self.prev_selected_action_idx),
                    requested_times=torch.tensor(ode_eval_times, device=self.device, dtype=self.dtype)
                )
                fiper_step_data["prev_velocities"] = selected_prev_velocities.detach().cpu().numpy()
                selected_composed_ode_states, _ = select_ode_states(
                    time_grid=sampling_time_grid,
                    ode_states=composed_ode_states,
                    requested_times=torch.tensor(ode_eval_times, device=self.device, dtype=self.dtype)
                )
            else:
                fiper_step_data["prev_velocities"] = (
                    torch.full_like(selected_velocities, float("nan"), device=self.device, dtype=self.dtype).detach().cpu().numpy()
                )

        if self.config.should_compute("composed", "vfd_oneway"):
            if self.prev_action_sample is not None and selected_composed_ode_states is not None:
                composed_vels = self.eval_statewise_velocities(
                    selected_states=selected_composed_ode_states,
                    eval_times=ode_eval_times,
                    velocity_fns=[self.prev_sampler_velocity_fn],
                ).squeeze(1)
            else:
                composed_vels = self.make_nan_statewise_velocities(reference_selected_states=selected_ode_states)
            fiper_step_data["composed_velocities"] = composed_vels.detach().cpu().numpy()

        if self.config.should_compute("composed_bayesian_laplace", "vfd_oneway"):
            if self.prev_action_sample is not None and selected_composed_ode_states is not None:
                composed_laplace_vels = self.eval_statewise_velocities(
                    selected_states=selected_composed_ode_states,
                    eval_times=ode_eval_times,
                    velocity_fns=self.prev_laplace_velocity_fns,
                )
            else:
                composed_laplace_vels = self.make_nan_statewise_velocities(
                    reference_selected_states=selected_ode_states,
                    num_models=len(laplace_velocity_fns),
                )
            fiper_step_data["composed_laplace_velocities"] = composed_laplace_vels.detach().cpu().numpy()


        if self.config.should_compute("composed_bayesian_ensemble", "vfd_oneway"):
            if self.prev_action_sample is not None and selected_composed_ode_states is not None:
                composed_ensemble_vels = self.eval_statewise_velocities(
                    selected_states=selected_composed_ode_states,
                    eval_times=ode_eval_times,
                    velocity_fns=self.prev_ensemble_velocity_fns,
                )
            else:
                composed_ensemble_vels = self.make_nan_statewise_velocities(
                    reference_selected_states=selected_ode_states,
                    num_models=len(ensemble_velocity_fns),
                )
            fiper_step_data["composed_ensemble_velocities"] = composed_ensemble_vels.detach().cpu().numpy()

        # Store velocity functions, ODE states and selected action index from the previous sampling step
        self.prev_sampler_velocity_fn = sampler_velocity_fn
        self.prev_ode_states = ode_states
        self.prev_velocities = velocities
        self.prev_selected_action_idx = rollout_step_data["action_selection_idx"]
        self.prev_action_sample = rollout_step_data["action_sample"].to(self.device, self.dtype)
        self.prev_laplace_velocity_fns = laplace_velocity_fns
        self.prev_ensemble_velocity_fns = ensemble_velocity_fns

        return fiper_step_data

    def score_rollout_data(
        self,
        rollout_data: list[dict[str, Any]],
        generator: torch.Generator | None = None
    ) -> list[dict[str, Any]]:
        """Scores all steps in the given rollout data."""
        self.fiper_data = []

        for step_data in tqdm(
            rollout_data,
            total=len(rollout_data),
            desc="Scoring rollout steps",
            leave=False,
        ):
            fiper_step_data = self.score_step_data(step_data, generator=generator)
            self.fiper_data.append(fiper_step_data)

        return self.fiper_data

    def reset(self):
        """
        Reset internal state to prepare for a new rollout.
        """
        # Clear stored velocity functions, ODE states and selected action sequence from previous step
        self.prev_sampler_velocity_fn: Callable[[Tensor, Tensor], Tensor] | None = None
        self.prev_ode_states: Tensor | None = None
        self.prev_velocities: Tensor | None = None
        self.prev_selected_action_idx: int | None = None
        self.prev_action_sample: Tensor | None = None
        self.prev_laplace_velocity_fns: list[Callable[[Tensor, Tensor], Tensor]] | None = None
        self.prev_ensemble_velocity_fns: list[Callable[[Tensor, Tensor], Tensor]] | None = None

        # Clear recorded rollout data
        self.fiper_data.clear()

    def save_data(
        self,
        output_dir: str | Path,
        episode_metadata: dict[str, Any],
    ) -> None:
        """Saves the recorded FIPER data to a specified path."""
        episode_idx = episode_metadata.get("episode")
        if episode_idx is None:
            raise ValueError("episode_metadata must contain an 'episode' key.")

        data = {
            "metadata": episode_metadata,
            "rollout": self.fiper_data,
            "config": asdict(self.config),
        }

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        success_flag = "s" if episode_metadata["successful"] else "f"

        filename = f"episode_{success_flag}_{episode_idx:04d}"
        task = episode_metadata["task"]
        task_id = episode_metadata.get("task_id")
        if "libero" in task and task_id is not None:
            filename += f"_task{task_id:02d}"
        output_path = output_dir / (filename + ".pkl")

        with output_path.open("wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        self.reset()

        print(f"Saved FIPER data for episode {episode_idx} to {output_path}.")
