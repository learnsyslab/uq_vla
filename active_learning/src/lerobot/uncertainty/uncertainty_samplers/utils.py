import torch
from torch import Tensor


def splice_noise_with_prev(
    new_noise_sample: Tensor,
    prev_noise_sample: Tensor,
    horizon: int,
    n_action_steps: int,
    n_obs_steps: int,
) -> Tensor:
    """
    Splice newly sampled noise with the overlapping segment from the
    previously selected trajectory to maintain temporal consistency.

    Args:
        new_noise_sample: Freshly sampled noise for this step.
            Shape: (num_action_samples, horizon, action_dim)
        prev_noise_sample: Noise from the previously selected action sequence.
            Shape: (horizon, action_dim).
        horizon: Total number of timesteps in the planning horizon.
        n_action_steps: Number of action steps to execute before replanning.
        n_obs_steps: Number of observation steps used for conditioning.

    Returns:
        Updated noise tensor where the overlapping part has been replaced
        with the corresponding segment from the previous trajectory, ensuring
        consistency with already executed actions.
    """
    # Indices marking the portion of the trajectory that will actually be executed
    exec_start_idx = n_obs_steps - 1
    exec_end_idx = exec_start_idx + n_action_steps

    # Compute the cutoff index: how far the new noise sample overlaps with the old one
    new_noise_overlap_end = exec_start_idx + (horizon - exec_end_idx)

    batch_size = new_noise_sample.shape[0]
    prev_noise_sample_duplicated = prev_noise_sample.expand(batch_size, -1, -1)

    new_noise_sample[:, exec_start_idx:new_noise_overlap_end, :] = \
        prev_noise_sample_duplicated[:, exec_end_idx:, :]

    return new_noise_sample

def compose_ode_states(
    prev_ode_states: Tensor,
    new_ode_states: Tensor,
    horizon: int,
    n_action_steps: int,
    n_obs_steps: int,
) -> Tensor:
    """
    Splice ODE states by keeping the executed prefix from the previous rollout and appending the freshly
    sampled suffix from the new action generation. Inputs can be full ODE integration states with time
    dimension or final sampled action sequences only.

    Args:
        prev_action_seq: ODE states collected during the previous action generation step.
            Shape: (timesteps, 1, horizon, action_dim) or (1, horizon, action_dim) for final action sequences.
        new_action_seq: Newly generated action sequence or ODE states.
            Shape: (timesteps, batch_size, horizon, action_dim) or (batch_size, horizon, action_dim) for
            final action sequences.
        horizon: Total number of timesteps in the planning horizon.
        n_action_steps: Number of action steps to execute before replanning.
        n_obs_steps: Number of observation steps used for conditioning.

    Returns:
        The composed ODE states. Shape: (timesteps, batch_size, horizon, action_dim) or
            (batch_size, horizon, action_dim) for final action sequences.
    """
    def add_time_dimension(ode_states: Tensor) -> tuple[Tensor, bool]:
        if ode_states.ndim == 3:   # (batch_size, horizon, action_dim)
            return ode_states.unsqueeze(0), False  # (1, batch_size, horizon, action_dim)
        if ode_states.ndim == 4:   # (time_step, batch_size, horizon, action_dim)
            return ode_states, True
        raise ValueError(f"Expected 3D or 4D tensor, got shape {tuple(ode_states.shape)}")

    prev_ode_states, prev_had_time_dim = add_time_dimension(prev_ode_states)
    new_ode_states, new_had_time_dim = add_time_dimension(new_ode_states)

    if prev_ode_states.shape[1] != 1:
        raise ValueError(
            "Selected ODE states from previous action generation are expected to have batch size "
            f"of one but got batch_size={prev_ode_states.shape[1]}."
        )

    if [prev_ode_states.size(i) for i in (0, 2, 3)] != [new_ode_states.size(i) for i in (0, 2, 3)]:
        raise ValueError(
            "ODE states to compose are expected to have the same time dimension, horizon, "
            f"and action dimension. Got shapes {prev_ode_states.shape} and {new_ode_states.shape}."
        )

    # Indices marking the portion of the trajectory that will actually be executed
    exec_start_idx = n_obs_steps - 1
    exec_end_idx = exec_start_idx + n_action_steps

    # Compute the cutoff index: how far the new sequence overlaps with the old one
    new_action_seq_end = exec_start_idx + (horizon - exec_end_idx)

    # Repeat prefix from previous ODE states to match batch dimension
    batch_size = new_ode_states.shape[1]
    prev_ode_states_duplicated = prev_ode_states.expand(
        -1, batch_size, -1, -1
    )

    # Compose from stored prefix and newly generated ODE states
    composed_ode_states = torch.cat([
        prev_ode_states_duplicated[:, :, :exec_end_idx, :],
        new_ode_states[:, :, exec_start_idx:new_action_seq_end, :]
    ], dim=2)

    if not (prev_had_time_dim and new_had_time_dim):
        return composed_ode_states.squeeze(0)
    else:
        return composed_ode_states

def select_and_expand_ode_states(
    ode_states: torch.Tensor,
    traj_idx: int,
) -> torch.Tensor:
    """
    Select a single trajectory from the ODE states (by index) and
    broadcast it across the batch dimension.

    Args:
        ode_states:  ODE states. Shape: (timesteps, batch_size, horizon, action_dim).
        traj_idx: Index of the trajectory to select from the batch dimension.

    Returns:
        Selected trajectory duplicated across all batch entries.
        Shape: (timesteps, batch_size, horizon, action_dim),
    """
    num_repeats = ode_states.shape[1]
    return ode_states[:, traj_idx:traj_idx+1, ...].expand(-1, num_repeats, -1, -1)
