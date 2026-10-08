import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from lerobot.policies.common.flow_matching.adapter import BaseFlowMatchingAdapter
from lerobot.policies.common.flow_matching.ode_solver import (
    ADAPTIVE_SOLVERS,
    FIXED_STEP_SOLVERS,
    ODESolver,
    make_sampling_time_grid,
)
from lerobot.uncertainty.uncertainty_samplers.utils import splice_noise_with_prev

from .configuration_fiper_rollout_recorder import FiperRolloutRecorderConfig


class FiperRolloutRecorder:
    def __init__(
        self,
        config: FiperRolloutRecorderConfig,
        flow_matching_adapter: BaseFlowMatchingAdapter,
    ):
        self.config = config
        self.flow_matching_adapter = flow_matching_adapter
        self.ode_solver = ODESolver()

        self.horizon = flow_matching_adapter.horizon
        self.n_action_steps = flow_matching_adapter.n_action_steps
        self.n_obs_steps = flow_matching_adapter.n_obs_steps
        self.env_action_dim = flow_matching_adapter.env_action_dim
        self.device = flow_matching_adapter.device
        self.dtype = flow_matching_adapter.dtype

        # Build time grid for sampling according to ODE solver method and scoring metric
        extra_times = []
        if self.config.ode_eval_times is not None:
            extra_times = [t for t in self.config.ode_eval_times if 0.0 < t < 1.0]
        self.ode_solver_config = flow_matching_adapter.ode_solver_config
        if self.ode_solver_config["solver_method"] in FIXED_STEP_SOLVERS:
            self.sampling_time_grid = make_sampling_time_grid(
                step_size=self.ode_solver_config["step_size"],
                extra_times=extra_times,
                device=self.device,
                dtype=self.dtype
            )
        elif self.ode_solver_config["solver_method"] in ADAPTIVE_SOLVERS:
            self.sampling_time_grid = torch.tensor(
                [0.0, *extra_times, 1.0],
                device=self.device, dtype=self.dtype
            )
        else:
            raise ValueError(f"Unknown ODE solver method: {self.ode_solver_config['solver_method']}.")

        # Store noise sample from the previous action sequence iteration
        self.prev_noise_sample: Tensor | None = None

        # Store data from action generation steps across rollout
        self.rollout_data: list[dict[str, Any]] = []

    def conditional_sample_with_recording(
        self,
        observation: dict[str, Tensor],
        generator: torch.Generator | None = None
    ) -> Tensor:
        """
        Sample an action sequence conditioned on an observation and record rollout data.

        Args:
            observation: Info about the environment used to create the conditioning for
                the flow matching model.
            generator: PyTorch random number generator.

        Returns:
            - Action sequence drawn from the flow matching model.
              Shape: (horizon, action_dim).
        """
        step_data: dict[str, Any] = {}

        # Store the observation
        # Tensors are stored detached on the CPU; the language conditioning is kept as well, because the
        # scorer re-runs the ensemble members on this observation and every VLA needs it. SmolVLA and X-VLA
        # carry it as tensors (OBS_LANGUAGE_TOKENS), but FastWAM takes a plain `task` / `prompt` STRING,
        # which a tensor-only filter drops -- and the scorer then fails with "Either `prompt` or both
        # `context/context_mask` must be provided".
        step_data["observation"] = {
            k: v.detach().cpu()
            for k, v in observation.items()
            if torch.is_tensor(v)
        }
        step_data["observation"].update(
            {k: v for k, v in observation.items() if k in ("task", "prompt") and isinstance(v, (str, list, tuple))}
        )

        conditioning = self.flow_matching_adapter.prepare_conditioning(observation, self.config.num_uncertainty_sequences)
        velocity_fn = self.flow_matching_adapter.make_velocity_fn(conditioning=conditioning)
        step_data["obs_embedding"] = self.flow_matching_adapter.prepare_fiper_obs_embedding(conditioning=conditioning)

        # Sample noise priors
        noise_sample = self.flow_matching_adapter.sample_prior(
            num_samples=self.config.num_uncertainty_sequences,
            generator=generator,
        )

        # Solve ODE forward from noise to sample action sequences
        ode_states, velocities = self.ode_solver.sample(
            x_0=noise_sample,
            velocity_fn=velocity_fn,
            method=self.ode_solver_config["solver_method"],
            atol=self.ode_solver_config["atol"],
            rtol=self.ode_solver_config["rtol"],
            time_grid=self.sampling_time_grid,
            return_intermediate_states=True,
            return_intermediate_vels=True
        )
        step_data["sampling_time_grid"] = self.sampling_time_grid.detach().cpu()
        step_data["ode_eval_times"] = np.asarray(self.config.ode_eval_times)
        # ODE states are model inputs and stay in the full model action space;
        # velocities are score inputs and keep only the real (environment) dims.
        step_data["ode_states"] = ode_states.detach().cpu()
        step_data["velocities"] = velocities[..., :self.env_action_dim].detach().cpu()

        if self.prev_noise_sample is not None and self.config.record_composed_inter_vel_diff:
            # Reuse overlapping segment of noise from the previously selected trajectory
            # so that the newly sampled noise remains consistent with already executed actions
            composed_noise_sample = splice_noise_with_prev(
                new_noise_sample=noise_sample,
                prev_noise_sample=self.prev_noise_sample,
                horizon=self.horizon,
                n_action_steps=self.n_action_steps,
                n_obs_steps=self.n_obs_steps,
            )
            ode_states_for_composed_inter_vel_diff, _ = self.ode_solver.sample(
                x_0=composed_noise_sample,
                velocity_fn=velocity_fn,
                method=self.ode_solver_config["solver_method"],
                atol=self.ode_solver_config["atol"],
                rtol=self.ode_solver_config["rtol"],
                time_grid=self.sampling_time_grid,
                return_intermediate_states=True,
                return_intermediate_vels=True
            )
            step_data["ode_states_for_composed_inter_vel_diff"] = ode_states_for_composed_inter_vel_diff.detach().cpu()

        # Pick one action sequence at random to return
        action_candidates = ode_states[-1]  # (num_uncertainty_sequences, horizon, action_dim)
        action_selection_idx = torch.randint(
            low=0,
            high=self.config.num_uncertainty_sequences,
            size=(1,),
            generator=generator,
            device=self.device
        ).item()
        step_data["action_selection_idx"] = action_selection_idx
        action_sample = action_candidates[action_selection_idx : action_selection_idx+1]  # (1, horizon, action_dim)
        step_data["action_sample"] = action_sample[..., :self.env_action_dim].detach().cpu()

        self.prev_noise_sample = noise_sample[action_selection_idx]

        # Store data from this action generation step
        self.rollout_data.append(step_data)

        return action_sample

    def reset(self):
        """
        Reset internal state to prepare for a new rollout.
        """
        self.prev_noise_sample = None

        # Clear recorded rollout data
        self.rollout_data.clear()

    def save_data(
        self,
        output_dir: str | Path,
        episode_metadata: dict[str, Any],
    ) -> None:
        """
        Save the recorded data (episode metadata + rollout data) as a .pkl file.
        The filename is constructed using the episode index from episode_metadata.
        """
        episode_idx = episode_metadata.get("episode")
        if episode_idx is None:
            raise ValueError("episode_metadata must contain an 'episode' key.")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        success_flag = "s" if episode_metadata["successful"] else "f"

        filename = f"episode_{success_flag}_{episode_idx:04d}"
        task = episode_metadata["task"]
        task_id = episode_metadata.get("task_id")
        if "libero" in task and task_id is not None:
            filename += f"_task{task_id:02d}"
        output_path = output_dir / (filename + ".pkl")

        data = {
            "metadata": episode_metadata,
            "rollout": self.rollout_data,
        }

        with output_path.open("wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        self.reset()

        print(f"Saved FIPER data for episode {episode_idx} to {output_path}.")
