import itertools
import math
import warnings
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor
from torchdiffeq import odeint

FIXED_STEP_SOLVERS = {
    "euler",
    "midpoint",
    "rk4",
    "explicit_adams",
    "implicit_adams",
}
ADAPTIVE_SOLVERS = {
    "dopri8",
    "dopri5",
    "bosh3",
    "fehlberg2",
    "adaptive_heun",
}

class ODESolver():
    """
    Wrapper class for solving flow-matching-based ODEs using a velocity function.

    Supports sampling from a learned distribution and computing log-likelihoods
    of a target sample via numerical ODE integration with torchdiffeq.
    """
    def sample(
        self,
        x_0: Tensor,
        velocity_fn: Callable[[Tensor, Tensor], Tensor],
        method: str,
        atol: float,
        rtol: float,
        step_size: Optional[float] = None,
        time_grid: Tensor = torch.tensor([0.0, 1.0]),
        return_intermediate_states: bool = False,
        return_intermediate_vels: bool = False,
        enable_grad: bool = False,
        allow_partial_time_grid: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """
        Solve the flow matching ODE with the conditioned velocity model.

        Args:
            x_0: Initial sample from source distribution. Shape: [batch_size, ...].
            velocity_fn: Velocity function defining the right-hand side of the flow matching ODE
                d/dt φ_t(x) = v_t(φ_t(x), conditoning).
                    Args:
                        t: Current scalar time of the ODE integration.
                        x_t: Current state x_t along the flow trajectory.

                    Returns:
                        Velocity v_t(φ_t(x), conditoning) with the same shape as `x`.
            method: An ODE solver method supported by torchdiffeq. For a complete list, see torchdiffeq.
                Some examples are:
                - Fixed-step examples: "euler", "midpoint", "rk4", ...
                - Adaptive examples: "dopri5", "bosh3", ...
            step_size: Size of an integration step only for fixed-step solvers. Ignored
                for adaptive solvers.
            atol, rtol: Absolute/relative error tolerances for accepting an adaptive solver step.
                Ignored for fixed-step solvers.
            time_grid: Times at which ODE is evaluated. Integration runs from time_grid[0] to time_grid[-1].
            return_intermediate_states: If True then return intermediate evaluation points according to time_grid.
            return_intermediate_vels: If True then return velocities at intermediate evaluation points acoording
                to time_grid.
            enable_grad: If True then compute gradients during sampling.
            allow_partial_time_grid: If True, allow monotone integration over a
                subinterval of [0, 1], e.g. DECU's 0 -> b and b -> 1 phases.

        Returns:
            - The solution of the ODE at time 1.0 when return_intermediate_states = False, otherwise all
            evaluation points specified in time_grid.
            - If return_intermediate_vels = True, additionally the velocities at the intermediate evaluation
            points specified in time_grid.
        """
        # Ensure all tensors are on the same device
        time_grid = time_grid.to(x_0.device)

        # Validate input shapes and solver parameters and build keyword arguments for odeint method
        ode_kwargs = self._validate_and_configure_solver(
            time_grid=time_grid,
            method=method,
            step_size=step_size,
            atol=atol,
            rtol=rtol,
            allow_partial_time_grid=allow_partial_time_grid,
        )

        with torch.set_grad_enabled(enable_grad):
            if method == "euler":
                trajetory, velocities = euler_integrate(
                    velocity_fn=velocity_fn,
                    x_0=x_0,
                    time_grid=time_grid,
                    step_size=step_size,
                )
            else:
                # Approximate ODE solution with numerical ODE solver
                trajetory = odeint(
                    velocity_fn,
                    x_0,
                    time_grid,
                    method=method,
                    **ode_kwargs,
                )
                if return_intermediate_vels:
                    velocities = []
                    for t, x_t in zip(time_grid, trajetory, strict=False):
                        velocities.append(velocity_fn(t=t, x_t=x_t))
                    velocities = torch.stack(velocities, dim=0)

        outputs: list[Tensor] = []

        if return_intermediate_states:
            outputs.append(trajetory)
        else:
            outputs.append(trajetory[-1])

        if return_intermediate_vels:
            outputs.append(velocities)

        return outputs[0] if len(outputs) == 1 else tuple(outputs)

    def sample_with_log_likelihood(
        self,
        x_init: Tensor,
        time_grid: Tensor,
        velocity_fn: Callable[[Tensor, Tensor], Tensor],
        log_p_0: Callable[[Tensor], Tensor],
        method: str,
        atol: float | None,
        rtol: float | None,
        step_size: float | None = None,
        return_intermediate_states: bool = False,
        exact_divergence: bool = True,
        num_hutchinson_samples: int | None = 3,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor] | tuple[Sequence[Tensor], Tensor]:
        """
        Integrate the combined flow matching ODE in either direction and return both the terminal
        sample and the log-likelihood log(p_1(x_1)) of the target sample x_1.

        This method supports both:
        - Forward integration from x_0 ~ p_0 using an increasing `time_grid` (0.0 to 1.0),
        which produces a target sample x_1 and computes log(p_1(x_1)).
        - Reverse integration from x_1 ~ p_1 using a decreasing `time_grid` (1.0 to 0.0),
        which reconstructs the corresponding x_0 ~ p_0 and computes log(p_1(x_1)).

        The integration direction is automatically inferred from the ordering of `time_grid`.

        Args:
            x_init: Initial state. Shape: [batch_size, ...].
                - If time_grid[0] = 0.0, sample from the source distribution, i.e. `x_init` ~ p_0.
                - If time_grid[0] = 1.0, sample from the target distribution, i.e. `x_init` ~ p_1.
            time_grid: Times at which ODE is evluated. Integration runs from time_grid[0] to time_grid[-1].
                Use [0.0, ..., 1.0] for forward integration (starting from a source sample),
                or [1.0, ..., 0.0] for reverse integration (starting from a target sample).
            velocity_fn: Velocity function defining the right-hand side of the flow matching ODE
                d/dt φ_t(x) = v_t(φ_t(x), conditoning).
                    Args:
                        t: Current scalar time of the ODE integration.
                        x_t: Current state x_t along the flow trajectory.

                    Returns:
                        Velocity v_t(φ_t(x), conditoning) with the same shape as `x`.
            log_p_0: Log-probability function of the source distribution p_0.
            method: An ODE solver method supported by torchdiffeq. For a complete list, see torchdiffeq.
                Some examples are:
                - Fixed-step examples: "euler", "midpoint", "rk4", ...
                - Adaptive examples: "dopri5", "bosh3", ...
            step_size: Size of an integration step only for fixed-step solvers. Ignored
                for adaptive solvers.
            atol, rtol: Absolute/relative error tolerances for accepting an adaptive solver step.
                Ignored for fixed-step solvers.
            return_intermediate_states: If True then return intermediate evaluation points according to time_grid.
            exact_divergence: Whether to compute the exact divergence or estimate it using the Hutchinson
                trace estimator.
            num_hutchinson_samples: Number of Hutchinson samples to use when estimating the divergence. Higher
                values reduce variance but increase computation. Ignored if `exact_divergence=True`.
            generator : Pass a pre-seeded generator for reproducible results.

        Returns:
            - Either the terminal state (x_0 in forward and x_1 in backward direction) when
              `return_intermediate_states` = False or all evaluation points x_t specified in time_grid
              when `return_intermediate_states` = True
            - The estimated log-likelihood log(p_1(x_1)).
        """
        if not exact_divergence and num_hutchinson_samples is None:
            raise ValueError("`num_hutchinson_samples` must be specified when `exact_divergence` is False.")

        # Ensure all tensors are on the same device
        time_grid = time_grid.to(x_init.device)

        # Validate input shapes and solver parameters and build keyword arguments for odeint method
        ode_kwargs = self._validate_and_configure_solver(
            time_grid=time_grid,
            method=method,
            step_size=step_size,
            atol=atol,
            rtol=rtol,
        )

        # Sample `num_hutchinson_samples` fixed Rademacher noise vector for which E[zz^T] = I
        # for the Hutchinson divergence estimator
        if not exact_divergence:
            z_samples = torch.randint(
                0, 2,
                (num_hutchinson_samples, *x_init.shape),
                device=x_init.device,
                dtype=x_init.dtype,
                generator=generator,
            ) * 2 - 1

        def combined_dynamics(t: Tensor, states: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
            """
            Helper function defining the right-hand side of the combined ODE to compute the
            flow trajectory as well as the log-likelihood of a target sample x_1 with
                d/dt φ_t(x) = v_t(φ_t(x), global_cond)
            and
                d/dt f(t) = div(v_t(φ_t(x), global_cond)) (exact divergence computation) or
                d/dt f(t) = z^T D_{x_t} v_t(x_t, global_cond) z (Hutchinson divergence estimator)
            For the log-likelihood computation the ODE is solved in forward or reverse direction.
            `global_cond` is captured from the outer scope.

            Args:
                t: Current scalar time of the ODE integration.
                states: Current state x_t along the flow trajectory (`states[0]`) with shape like `x_1`
                    and integrated scalar divergence f(t) (`states[1]`).

            Returns:
                Velocity d/dt φ_t(x) with the same shape as `states[0]` and scalar divergence term d/dt f(t).
            """
            # Current state φ_t(x) along the flow trajectory
            x_t = states[0]
            with torch.set_grad_enabled(True):
                x_t.requires_grad_()
                # Compute velocity v_t(φ_t(x), global_cond)
                v_t = velocity_fn(t=t, x_t=x_t)

                # Compute or estimate divergence div(v_t(φ_t(x), global_cond))
                if exact_divergence:
                    # Compute exact divergence
                    # div(v_t(φ_t(x), global_cond)) = tr(D_{x_t} v_t(x_t, global_cond))
                    div = torch.zeros(x_t.shape[0], device=x_t.device)
                    feature_dims = v_t.shape[1:]
                    for idx in itertools.product(*(range(d) for d in feature_dims)):
                        # Add batch dimension to index
                        idx = [slice(None)] + list(idx)
                        div += torch.autograd.grad(
                            outputs=v_t[idx],
                            inputs=x_t,
                            grad_outputs=torch.ones_like(v_t[idx]).detach(),
                            retain_graph=True,
                        )[0][idx]
                else:
                    # Compute Hutchinson divergence estimator z^T D_x(v_t(x_t)) z by averaging
                    # over `num_hutchinson_samples` samples z
                    div = torch.zeros(x_t.shape[0], device=x_t.device)
                    for z in z_samples:
                        # Dot product v_t · z for each batch element
                        z = z.to(dtype=v_t.dtype)
                        v_t_dot_z = torch.einsum(
                            "ij,ij->i", v_t.flatten(start_dim=1), z.flatten(start_dim=1)
                        )
                        # Gradient of v_t · z w.r.t. x_t
                        grad_v_t_dot_z = torch.autograd.grad(
                            outputs=v_t_dot_z,
                            inputs=x_t,
                            grad_outputs=torch.ones_like(v_t_dot_z).detach(),
                            retain_graph=True,
                        )[0]
                        # Single-probe divergence estimate: zᵀ ∇_x (v_t · z)
                        div += torch.einsum(
                            "ij,ij->i",
                            grad_v_t_dot_z.flatten(start_dim=1),
                            z.flatten(start_dim=1),
                        )

                    div *= (1.0 / num_hutchinson_samples)
            return v_t, div

        # Set initial state of the reverse-time combined ODE for initial noise sample
        # and log-likelihood computation
        initial_state = (x_init, torch.zeros(x_init.shape[0], device=x_init.device))

        # Solve the combined flow matching ODE to obtain a terminal state (x_1 for
        # forward direction and x_0 for reverse direction) and log-probability difference
        # of log(p_1(x_1)) and log(p_0(x_0))
        trajectory, log_prob_diff = odeint(
            combined_dynamics,
            initial_state,
            time_grid,
            method=method,
            **ode_kwargs,
        )

        if time_grid[-1] == 0:
            # Extract initial noise sample from reverse flow trajectory
            x_0 = trajectory[-1]

            # Compute log-probability of target sample using log-probability of initial noise
            # sample and change-of-variables correction
            log_p_1_x_1 = log_p_0(x_0) + log_prob_diff[-1]
        elif time_grid[-1] == 1:
            # Compute log-probability of target sample using log-probability of initial noise
            # sample and change-of-variables correction
            log_p_1_x_1 = log_p_0(x_init) - log_prob_diff[-1]

        return (trajectory, log_p_1_x_1) if return_intermediate_states else (trajectory[-1], log_p_1_x_1)

    def _validate_and_configure_solver(
        self,
        time_grid: Tensor,
        method: str,
        step_size: Optional[float],
        atol: Optional[float],
        rtol: Optional[float],
        allow_partial_time_grid: bool = False,
    ) ->  Dict[str, Any]:
        """
        Validate input shapes and solver parameters, and construct appropriate keyword arguments for `torchdiffeq.odeint`.
        """
        # Ensure time grid is one-dimensional and strictly monotone
        if time_grid.ndim != 1:
            raise ValueError("`time_grid` must be one-dimensional.")
        time_deltas = time_grid[1:] - time_grid[:-1]
        if not (torch.all(time_deltas > 0) or torch.all(time_deltas < 0)):
            raise ValueError("`time_grid` must be strictly monotone (all increasing or all decreasing).")

        if allow_partial_time_grid:
            if torch.any(time_grid < 0.0) or torch.any(time_grid > 1.0):
                raise ValueError(
                    f"`time_grid` entries must stay within [0.0, 1.0] when "
                    f"`allow_partial_time_grid=True`. Got {time_grid}."
                )
        elif not (
            (time_grid[0] == 0.0 and time_grid[-1] == 1.0) or
            (time_grid[0] == 1.0 and time_grid[-1] == 0.0)
        ):
            raise ValueError(f"Time grid must go from 0.0 to 1.0 or from 1.0 to 0.0. Got {time_grid}.")

        # Check ODE solver parameters
        if method in FIXED_STEP_SOLVERS:
            # Positive step‑size required
            if atol is not None or rtol is not None:
                warnings.warn(
                    f"`atol`/`rtol` is ignored by fixed-step solver '{method}'.", stacklevel=2
                )
            if step_size is not None:
                if step_size <= 0.0:
                    raise ValueError(
                        f"`step_size` must be a positive float for fixed-step solver '{method}'."
                    )
                return {"options": {"step_size": float(step_size)}}
            else:
                return {}

        if method in ADAPTIVE_SOLVERS:
            if atol is None or rtol is None or atol <= 0.0 or rtol <= 0.0:
                raise ValueError(
                    f"`atol` and `rtol` must be positive for adaptive solver '{method}'."
                )
            if step_size is not None:
                raise warnings.warn(
                    f"`step_size` is ignored by adaptive solver '{method}'.", stacklevel=2
                )
            return {"atol": float(atol), "rtol": float(rtol)}

        raise ValueError(
            f"Unknown solver '{method}'. Choose one of "
            f"{sorted(FIXED_STEP_SOLVERS | ADAPTIVE_SOLVERS)}."
        )

def euler_integrate(
    x_0: Tensor,
    velocity_fn: Callable[[Tensor, Tensor], Tensor],
    time_grid: Tensor,
    step_size: Optional[float] = None
) -> Tuple[Tensor, Tensor]:
    """
    Explicit Euler integrator over a time grid.

    Args:
        x_0: Initial sample from source distribution. Shape: [batch_size, ...].
        velocity_fn: Velocity function defining the right-hand side of the flow matching ODE
            d/dt φ_t(x) = v_t(φ_t(x), conditoning).
                Args:
                    t: Current scalar time of the ODE integration.
                    x_t: Current state x_t along the flow trajectory.

                Returns:
                    Velocity v_t(φ_t(x), conditoning) with the same shape as `x`.
        time_grid: Times at which ODE is evaluated. Integration runs from time_grid[0] to time_grid[-1].
        step_size: Take fixed-magnitude substeps of |step_size| starting from t_i until reaching t_{i+1}.
            If None, take a single Euler step from t_i to t_{i+1}.

    Returns:
        - The ODE states evaluated at each time in `time_grid`.
        - The velocities v(t_i, x_i) at these ODE states and evaluation times.
    """
    states: List[Tensor] = []
    vels: List[Tensor] = []

    x_t = x_0
    for i in range(len(time_grid) - 1):
        t = time_grid[i]
        t_next = time_grid[i + 1]

        # Record at current time t
        states.append(x_t)
        v_t = velocity_fn(x_t=x_t, t=t)
        vels.append(v_t)

        if step_size is None:
            # Single Euler step from t to t_next
            dt = t_next - t
            x_t = x_t + dt * v_t
        else:
            # Multiple substeps of signed size 'direction * step'
            t_cur = t
            direction = torch.sign(t_next - t)
            while True:
                remaining_diff = t_next - t_cur
                # If within one step, take the final shortened step
                if torch.abs(remaining_diff) <= step_size:
                    dt = remaining_diff
                    x_t = x_t + dt * v_t
                    break
                else:
                    # Take a full substep
                    dt = direction * step_size
                    x_t = x_t + dt * v_t
                    t_cur = t_cur + dt
                    # Update velocity for the next substep
                    v_t = velocity_fn(t=t_cur, x_t=x_t)

    # Append final state at t_N and its velocity for completeness
    states.append(x_t)
    vels.append(velocity_fn(t=time_grid[-1], x_t=x_t))

    trajectory = torch.stack(states, dim=0)
    velocities = torch.stack(vels, dim=0)
    return trajectory, velocities


def make_sampling_time_grid(
    step_size: float,
    device: torch.device,
    dtype: torch.dtype,
    extra_times: Optional[Union[Tensor, Sequence]] = None,
) -> Tensor:
    """
    Build a time grid from 0.0 to 1.0 with fixed step_size, plus extra points.

    Args:
        step_size: Spacing between regular points.
        device: The device on which to create the time grid.
        dtype: The dtype of the resulting tensor
        extra_times: Additional timepoints to include.

    Returns:
        A time grid of unique, sorted times in [0.0, 1.0.]
    """
    if not (0 < step_size <= 1.0):
        raise ValueError("step_size must be > 0 and <= 1.")

    # How many full steps of step_size fit into [0,1]
    n = math.floor(1.0 / step_size)

    # Regular grid from 0.0 to (n * step_size)
    time_grid = torch.linspace(
        0.0,
        n * step_size,
        steps=n + 1,
        device=device,
        dtype=dtype,
    )

    # Ensure time grid ends with 1.0
    if time_grid[-1] < 1.0:
        time_grid = torch.cat([
            time_grid, torch.tensor([1.0], device=device, dtype=dtype)
        ])

    # Merge step size time grid with extra times and sort
    if extra_times is not None:
        time_grid = torch.cat([
            time_grid, torch.tensor(extra_times, device=device, dtype=dtype).clamp(0.0, 1.0)
        ])

    # Remove near-duplicates and sort
    time_grid, _ = torch.sort(time_grid)
    keep = torch.ones_like(time_grid, dtype=torch.bool)
    keep[1:] = torch.diff(time_grid) > 1e-4
    time_grid = time_grid[keep]

    if time_grid[0].item() != 0.0 or time_grid[-1].item() != 1.0:
        raise RuntimeError("Sampling time grid must start at 0.0 and end at 1.0.")

    return time_grid


def make_lik_estimation_time_grid(ode_solver_method: str, device: torch.device, dtype: torch.dtype) -> Tensor:
    """
    Build time grid to estimate likelihood according to ODE solver method.

    For a fixed step solver the time grid consists of a fine segment of 10 points evenly
    spaced from 1.0 up to 0.93 and a coarse segment of 10 points evenly spaced from 0.9 up to 0.0.

    Returns:
        A 1D time grid.
    """
    if ode_solver_method in FIXED_STEP_SOLVERS:
        fine = torch.linspace(1.0, 0.93, steps=10, device=device, dtype=dtype)
        coarse = torch.linspace(0.9, 0.0,  steps=10, device=device, dtype=dtype)
        return torch.cat((fine, coarse))
    elif ode_solver_method in ADAPTIVE_SOLVERS:
        lik_estimation_time_grid = torch.tensor([1.0, 0.0], device=device, dtype=dtype)
    else:
        raise ValueError(
            f"Unknown ODE solver method {ode_solver_method}. "
            f"Expected one of {sorted(FIXED_STEP_SOLVERS | ADAPTIVE_SOLVERS)}."
        )

    return lik_estimation_time_grid


def select_ode_states(
    time_grid: Tensor, ode_states: Tensor, requested_times: Tensor
) -> tuple[Tensor, Tensor]:
    """
    Extract the ODE states (and their timestamps) that correspond to a set of
    requested times.

    Args:
        time_grid: A tensor of time points at which the ODE was evaluated.
        ode_states: A tensor of ODE states corresponding to the time points in time_grid.
        requested_times: A tensor of times for which to extract the ODE states.

    Returns:
        A tuple containing:
            - A tensor of ODE states corresponding to the requested times.
            - A tensor of the corresponding time points from `time_grid`.
    """
    if time_grid.size(0) != ode_states.size(0):
        raise ValueError(
            f"`time_grid` and `ode_states` must have the same length; "
            f"got {time_grid.size(0)} and {ode_states.size(0)}."
        )
    # Map each requested time to its index in the time grid
    matched_indices = []
    for req_t in requested_times:
        # Locate entries equal to requested time (within tolerance)
        time_mask = torch.isclose(time_grid, req_t, atol=1e-5, rtol=0)
        match_count = int(time_mask.sum().item())
        if match_count == 0:
            raise ValueError(f"Requested time {req_t.item()} not found in time_grid")
        if match_count > 1:
            raise ValueError(
                f"Requested time {req_t.item()} matched {match_count} entries in time_grid; "
                "expected exactly one."
            )

        # Grab index of match
        index = time_mask.nonzero(as_tuple=True)[0].item()
        matched_indices.append(index)

    # Select only the ODE states and time points that correspond to those indices
    selected_ode_states = ode_states[matched_indices]
    selected_grid_times = time_grid[matched_indices]

    return selected_ode_states, selected_grid_times
