import torch
from torch import Tensor
from typing import Any, Dict

from ..utils import add_multi_scorer_axis, build_eval_time_mask


def likelihood(log_likelihood: Tensor) -> float:
    """
    Compute the uncertainty score based on the negative log-likelihood for a batch of action sequences.

    Args:
        log_likelihood: Log-likelihoods for a batch of action sequences.

    Returns:
        Average negative log-likelihood score.
    """
    return float((-log_likelihood).mean().item())

def mode_distance(terminal_vels: Tensor, eval_times: Tensor, config: Dict[str, Any]) -> float:
    """
    Compute the proxy "distance-from-mode" score computed by averaging (1 - t) * ‖v(x; t)‖
    of the velocity at the terminal action sequences x across the specified evaluation times.

    Args:
        terminal_vels: Velocities at terminal action sequence.
            Shape: (ode_timesteps, batch_size, horizon, action_dim) or (ode_timesteps, num_scorer, batch_size, horizon, action_dim).
        eval_times: Times at which velocity at terminal action sequence has been evaluated.
            Shape: (ode_timesteps,)
        config: Configuration of a failure prediction method expected to contain "mode_distance_eval_times".

    Returns:
        Average distance-from-mode score.
    """
    # Select the termninal velocities that correspond to the configured velocity evaluation times
    eval_times_mask = build_eval_time_mask(
        cfg_times=torch.tensor(
            config.mode_distance_eval_times,
            dtype=eval_times.dtype,
            device=eval_times.device
        ),
        recorded_times=eval_times
    )
    selected_times = eval_times[eval_times_mask]
    selected_vels = terminal_vels[eval_times_mask]

    if selected_times.numel() == 0:
        raise ValueError(
            f"Could not find any evaluation times {config.mode_distance_eval_times} in the "
            f"recorded times {eval_times.tolist()}."
        )

    selected_vels = add_multi_scorer_axis(selected_vels, num_scorers=config.get("num_scorers", 1),
                                          scorer_indices=config.get("scorer_indices", None)) # Shape: (num_eval_times, num_scorer, batch_size, horizon, action_dim)

    velocity_norms = torch.norm(selected_vels, dim=(3, 4)) # Shape: (num_eval_times, num_scorer, batch_size)
    
    # Scale by (1 - time) as a simple proxy for “distance from the mode”
    # (i.e. how far a particle would still travel under constant velocity)
    distances = (1.0 - selected_times).view(-1, 1, 1) * velocity_norms
    return float(distances.mean().item())

def inter_vel_diff(
    ref_velocities: Tensor,
    cmp_velocities: Tensor,
    ode_eval_times: Tensor,
    vel_diff_scaling_factors: Tensor,
    config: Dict[str, Any]
) -> float:
    """
    Compute the average velocity discrepancy between two trajectories at intermediate ODE states.

    For each evaluation time t, this metric computes the L2 difference between the velocity 
    predicted by the reference model at x_ref(t) and the comparison model at x_cmp(t):

        ||v_ref(x_ref(t), t) - v_cmp(x_cmp(t), t)||

    The discrepancies are integrated over time and averaged across the batch of sampled actions.

    Args:
        ref_velocities: Velocities at ODE integration states of the reference trajectory.
            Shape: (ode_timesteps, batch_size, horizon, action_dim) or (ode_timesteps, num_scorer, batch_size, horizon, action_dim).
        cmp_velocities: Velocities at ODE integration states of the comparison trajectory.
            Shape: (ode_timesteps, batch_size, horizon, action_dim) or (ode_timesteps, num_scorer, batch_size, horizon, action_dim).
        ode_eval_times: Times at which velocities were recorded during ODE integration.
            Shape: (ode_timesteps,)
        vel_diff_scaling_factors: Scaling factors applied to velocity differences at each 
            evaluation time. Shape: (ode_timesteps,)
        config: Configuration of a failure prediction method expected to contain "inter_vel_diff_eval_times".

    Returns:
        Average intermediate velocity difference score.
    """
    # Select the ODE states that correspond to the velocity evaluation times
    eval_times_mask = build_eval_time_mask(
        cfg_times=torch.tensor(
            config.inter_vel_diff_eval_times,
            dtype=ode_eval_times.dtype,
            device=ode_eval_times.device
        ),
        recorded_times=ode_eval_times
    )

    selected_ode_eval_times = ode_eval_times[eval_times_mask] # Shape: (num_eval_times,)
    selected_ref_vels = ref_velocities[eval_times_mask]
    selected_cmp_vels = cmp_velocities[eval_times_mask]
    selected_ref_vels = add_multi_scorer_axis(selected_ref_vels, num_scorers=config.get("num_scorers", 1),
                                              scorer_indices=config.get("scorer_indices", None)) # Shape: (num_eval_times, 1, batch_size, horizon, action_dim)
    selected_cmp_vels = add_multi_scorer_axis(selected_cmp_vels, num_scorers=config.get("num_scorers", 1),
                                              scorer_indices=config.get("scorer_indices", None)) # Shape: (num_eval_times, num_scorer, batch_size, horizon, action_dim)
    selected_vel_diff_scaling = vel_diff_scaling_factors[eval_times_mask] # Shape: (num_eval_times,)
    # Optionally score with only the first `num_action_samples` of the recorded action samples (the
    # batch axis), e.g. 8 of the 128 recorded on Push-T. Samples are
    # i.i.d. draws from the same noise distribution, so the first B of them are an unbiased B-sample batch.
    num_samples = config.get("num_action_samples", None)
    if num_samples is not None:
        num_samples = int(num_samples)
        if not 0 < num_samples <= selected_ref_vels.shape[2]:
            raise ValueError(f"num_action_samples={num_samples} but only {selected_ref_vels.shape[2]} action samples were recorded")
        selected_ref_vels = selected_ref_vels[:, :, :num_samples]
        selected_cmp_vels = selected_cmp_vels[:, :, :num_samples]

    if selected_ode_eval_times.numel() == 0:
        raise ValueError(
            f"Could not find any evaluation times {config.inter_vel_diff_eval_times} in the "
            f"recorded times {ode_eval_times.tolist()}."
        )
    
    # Determine dt: difference to next time or to 1.0 for last step
    dt = torch.empty_like(selected_ode_eval_times)
    dt[:-1] = selected_ode_eval_times[1:] - selected_ode_eval_times[:-1]
    dt[-1]  = 1.0 - selected_ode_eval_times[-1]

    # L2 norm across horizon and action dims gives magnitude of velocity differences 
    vel_diffs = (selected_ref_vels - selected_cmp_vels).square().sum(dim=(3, 4)) # Shape: (num_eval_times, num_scorer, batch_size)

    inter_vel_diff_score = ((selected_vel_diff_scaling * dt).view(-1, 1, 1) * vel_diffs).sum(dim=0).mean()
    return float(inter_vel_diff_score.item())