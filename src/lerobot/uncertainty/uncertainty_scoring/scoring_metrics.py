from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

import numpy as np
import torch # type: ignore
from torch import Tensor # type: ignore
from torch.distributions import Independent, Normal # type: ignore

from lerobot.policies.common.flow_matching.conditional_probability_path import (
    OTCondProbPath,
    VPDiffusionCondProbPath,
)
from lerobot.policies.common.flow_matching.ode_solver import (
    ODESolver,
    make_lik_estimation_time_grid,
    select_ode_states,
)

from ..uncertainty_samplers.configuration_uncertainty_sampler import ScoringMetricConfig
from ..uncertainty_samplers.uncertainty_sampler import UncertaintySampler


class UncertaintyMetric(ABC):  # noqa: B024
    """Abstract base class for uncertainty metrics."""
    name: str = "base"
    type: str = "base"


class TerminalStateMetric(UncertaintyMetric, ABC):
    """
    Abstract base class for uncertainty metrics that operate on terminal action sequences,
    i.e., the final outputs of the flow matching ODE integration.
    Subclasses define how the velocity model is applied to score these sequences and produce
    an uncertainty score.
    """
    type: str = "terminal"

    @abstractmethod
    def __call__(
        self,
        velocity_fn: Callable[[Tensor, Tensor], Tensor],
        action_sequences: Tensor,
        **kwargs: Any,
    ) -> Tensor:
        """
        Compute an uncertainty score for a batch of action sequences using a scorer flow matching model.
        Args:
            velocity_fn: Velocity function defining the right-hand side of the flow matching ODE
                d/dt φ_t(x) = v_t(φ_t(x), conditoning).
            action_sequence: Final action sequences to score. Shape: (batch_size,
                horizon, action_dim).

        Returns:
            Uncertainty scores per action sequence where larger values indicate higher uncertainty.
            Shape: (batch_size,).
        """
        raise NotImplementedError


class TerminalVelNorm(TerminalStateMetric):
    """
    Average L2 norm of scorer velocities evaluated on the terminal action sequence
    across specified evaluation times.
    """
    name: str = "terminal_vel_norm"

    def __init__(self, config: ScoringMetricConfig):
        """
        Args:
            config: Scoring metric settings.
        """
        self.velocity_eval_times = config.velocity_eval_times

    def __call__(
        self,
        velocity_fn: Callable[[Tensor, Tensor], Tensor],
        action_sequences: Tensor,
        **_: Any
    ) -> Tensor:
        """
        Evaluate the velocity only at the terminal action sequences for multiple evaluation
        times and return the mean L2 norm as the uncertainty score.
        """
        # Evaluate velocity on the final sampled sequence
        terminal_vel_norms: list[float] = []
        for time in self.velocity_eval_times:
            velocity = velocity_fn(
                x_t=action_sequences, t=torch.tensor(time, device=action_sequences.device)
            )
            terminal_vel_norms.append(torch.norm(velocity, dim=(1, 2)))

        # Use average velocity norm as uncertainty score
        return torch.stack(terminal_vel_norms, dim=0).mean(dim=0)


class ModeDistance(TerminalStateMetric):
    """
    Estimates "distance to the next mode" by averaging (1 - t) · ‖v(x; t)‖ over specified
    evaluation times.
    """
    name: str = "mode_distance"

    def __init__(self, config: ScoringMetricConfig):
        """
        Args:
            config: Scoring metric settings.
            device: The PyTorch device on which to store and perform tensor operations.
        """
        self.velocity_eval_times = config.velocity_eval_times

    def __call__(
        self,
        velocity_fn: Callable[[Tensor, Tensor], Tensor],
        action_sequences: Tensor,
        **_: Any,
    ) -> Tensor:
        """
        Compute the proxy "distance-from-mode" score computed by averaging (1 - t) * ‖v(x; t)‖
        of the velocity at the terminal action sequence x across the specified evaluation times.
        """
        distances: list[Tensor] = []
        # Loop over each time in [0, 1) at which we want to probe the velocity field
        for time in self.velocity_eval_times:
            # Query the velocity field at the terminal action sequence and this time
            velocity = velocity_fn(
                x_t=action_sequences, t=torch.tensor(time, device=action_sequences.device)
            )
            velocity_norm = torch.norm(velocity, dim=(1, 2))
            # Scale by (1 - time) as a simple proxy for “distance from the mode”
            # (i.e. how far a particle would still travel under constant velocity)
            distance = (1 - time) * velocity_norm
            distances.append(distance)

        return torch.stack(distances, dim=0).mean(dim=0)


class Likelihood(TerminalStateMetric):
    """
    Uncertainty metric that scores an action sequence by its negative log-likelihood under a scorer model.
    Uses an ODE solver to estimate likelihood along a reverse-time trajectory.
    """
    name: str = "likelihood"

    def __init__(self, config: ScoringMetricConfig, uncertainty_sampler: UncertaintySampler):
        """
        Args:
            config: Scoring metric settings.
        """
        self.device = uncertainty_sampler.device
        self.dtype = uncertainty_sampler.dtype
        # Noise distribution is an isotropic Gaussian
        horizon = uncertainty_sampler.horizon
        action_dim = uncertainty_sampler.action_dim
        self.gaussian_log_density = Independent(
            Normal(
                loc = torch.zeros(horizon, action_dim, device=self.device, dtype=self.dtype),
                scale = torch.ones(horizon, action_dim, device=self.device, dtype=self.dtype),
            ),
            reinterpreted_batch_ndims=2
        ).log_prob

        # ODE solver settings for likelihood estimation
        self.lik_ode_solver_cfg = config.likelihood_ode_solver_config

        # Build time grid for likelihood estimation based on ODE solver method
        self.lik_estimation_time_grid = make_lik_estimation_time_grid(
            ode_solver_method=self.lik_ode_solver_cfg.method,
            device=self.device,
            dtype=self.dtype,
        )

        self.ode_solver = ODESolver()

    def __call__(
        self,
        velocity_fn: Callable[[Tensor, Tensor], Tensor],
        action_sequences: Tensor,
        generator: torch.Generator | None = None,
        **_: Any,
    ) -> Tensor:
        """
        Run a reverse-time ODE under the scorer model to compute the log-likelihood of the
        action sequence; the score is the negative log-likelihood.
        """
        # Compute log-likelihood of sampled action sequences in scorer model
        _, log_probs = self.ode_solver.sample_with_log_likelihood(
            x_init=action_sequences,
            time_grid=self.lik_estimation_time_grid,
            velocity_fn=velocity_fn,
            log_p_0=self.gaussian_log_density,
            method=self.lik_ode_solver_cfg.method,
            atol=self.lik_ode_solver_cfg.atol,
            rtol=self.lik_ode_solver_cfg.rtol,
            exact_divergence=self.lik_ode_solver_cfg.exact_divergence,
            generator=generator,
        )

        # Use negative log-likelihood as uncertainty score
        return -log_probs


class DECUMetric(UncertaintyMetric):
    """
    Diffusion Ensembles for Capturing Uncertainty (DECU) scoring metric.

    Estimates epistemic uncertainty using the Pairwise-Distance Estimator (PaiDE)
    on velocities predicted by an ensemble of flow-matching models at a single
    branching point x_b on the ODE trajectory.

    Given M ensemble members with velocities v_i = v_{theta_i}(x_b, t_b), let
        D_{i,j} = || v_i - v_j ||_2^2     (reduced over horizon and action dims)
    The mutual-information approximation is
        Score = - (1/M) * sum_i log( (1/M) * sum_j exp( -D_{i,j} ) ).
    """
    name: str = "decu"
    type: str = "branching"

    def __init__(self, config: ScoringMetricConfig):
        """
        Args:
            config: Scoring metric settings. Uses `branching_time` to record the
                ODE time at which the velocities are evaluated (informational only;
                the actual evaluation time is passed in via __call__).
        """
        self.branching_time = config.branching_time

    def __call__(
        self,
        x_b: Tensor,
        time_b: Tensor | float,
        velocity_fns: list[Callable[[Tensor, Tensor], Tensor]],
        **_: Any,
    ) -> Tensor:
        """
        Compute DECU uncertainty for a batch of branching states.

        Args:
            x_b: Intermediate ODE state at the branching time.
                Shape: (batch_size, horizon, action_dim).
            time_b: Scalar branching time at which to evaluate the velocities.
            velocity_fns: List of M velocity functions, one per ensemble member.

        Returns:
            DECU uncertainty score per batch element. Shape: (batch_size,).
        """
        if len(velocity_fns) < 2:
            raise ValueError(
                f"DECUMetric requires at least 2 ensemble velocity functions; got {len(velocity_fns)}."
            )

        # Convert time to a tensor on the right device if needed
        if not isinstance(time_b, Tensor):
            time_b = torch.tensor(time_b, device=x_b.device, dtype=x_b.dtype)

        # Evaluate all ensemble velocities at (x_b, time_b)
        # Each entry has shape (batch_size, horizon, action_dim)
        velocities: list[Tensor] = [vf(x_t=x_b, t=time_b) for vf in velocity_fns]
        # Stack into (M, batch_size, H, A) and flatten action/horizon dims
        v_stack = torch.stack(velocities, dim=0)
        M, B = v_stack.shape[0], v_stack.shape[1]
        v_flat = v_stack.reshape(M, B, -1)  # (M, B, H*A)

        # Pairwise squared L2 distances over flattened velocity dim
        # D[b, i, j] = || v_i(b) - v_j(b) ||_2^2
        v_bm = v_flat.permute(1, 0, 2)  # (B, M, H*A)
        diff = v_bm.unsqueeze(2) - v_bm.unsqueeze(1)  # (B, M, M, H*A)
        D = diff.pow(2).sum(dim=-1)  # (B, M, M)

        # Log-Sum-Exp form of the PaiDE mutual-information approximation:
        #   inner_i = log( (1/M) * sum_j exp(-D_{i,j}) )
        #          = logsumexp(-D_{i,j}, j) - log(M)
        log_M = torch.log(torch.tensor(M, device=D.device, dtype=D.dtype))
        inner = torch.logsumexp(-D, dim=2) - log_M  # (B, M)
        # Score = - (1/M) * sum_i inner_i
        score = -inner.mean(dim=1)  # (B,)
        return score


class VfdOneway(UncertaintyMetric):
    """
    Uncertainty metric based on intermediate velocity discrepancies.

    Compares the velocities predicted by two flow matching models along their
    respective ODE trajectories. Large values indicate stronger disagreement
    between the reference and comparison models.
    """
    name: str = "vfd_oneway"
    type: str = "trajectory"

    def __init__(self, config: ScoringMetricConfig, uncertainty_sampler: UncertaintySampler):
        """
        Args:
            config: Scoring metric settings.
        """
        self.velocity_eval_times = config.velocity_eval_times
        self.device = uncertainty_sampler.device
        self.dtype = uncertainty_sampler.dtype
        self.ode_solver = uncertainty_sampler.sampling_ode_solver
        self.sampling_time_grid = uncertainty_sampler.sampling_time_grid
        self.cond_vf_type = uncertainty_sampler.cond_vf_config["type"]
        if self.cond_vf_type == "vp":
            self.cond_prob_path = VPDiffusionCondProbPath(
                beta_min=uncertainty_sampler.cond_vf_config["beta_min"],
                beta_max=uncertainty_sampler.cond_vf_config["beta_max"],
            )
        elif self.cond_vf_type == "ot":
            self.cond_prob_path = OTCondProbPath(uncertainty_sampler.cond_vf_config["sigma_min"])
        else:
            raise ValueError(
                f"Unknown conditional vector field type {self.cond_vf_type}."
            )

    def __call__(
        self,
        ref_ode_states: Tensor,
        ref_velocity_fn: Callable[[Tensor, Tensor], Tensor],
        cmp_ode_states: Tensor,
        cmp_velocity_fn: Callable[[Tensor, Tensor], Tensor],
    ) -> Tensor:
        """
        Compute the average velocity discrepancy between two trajectories at intermediate ODE states.

        For each evaluation time t, this metric computes the L2 difference between the velocity
        predicted by the reference model at x_ref(t) and the comparison model at x_cmp(t):

            ||v_ref(x_ref(t), t) - v_cmp(x_cmp(t), t)||^2

        The discrepancies are integrated over time.

        Args:
            ref_ode_states: ODE integration states of the reference trajectory.
                Shape: (timesteps, batch_size, horizon, action_dim).
            ref_velocity_fn: Conditional velocity function associated with the reference trajectory.
            ref_global_cond: Conditioning vector for the reference model.
                Shape: (batch_size, cond_dim).
            cmp_ode_states: ODE integration states of the comparison trajectory.
                Shape: (timesteps, batch_size, horizon, action_dim).
            cmp_velocity_fn: Conditional velocity function associated with the comparison trajectory.
            cmp_global_cond: Conditioning vector for the comparison model.
                Shape: (batch_size, cond_dim).

        Returns:
            Uncertainty scores per trajectory sample, where larger values indicate stronger
            disagreement between reference and comparison velocities. Shape: (batch_size,).
        """
        # Select the ODE states that correspond to the velocity evaluation times
        selected_ref_ode_states, selected_ref_grid_times = select_ode_states(
            time_grid=self.sampling_time_grid,
            ode_states=ref_ode_states,
            requested_times=torch.tensor(self.velocity_eval_times, device=self.device, dtype=self.dtype)
        )
        selected_cmp_ode_states, selected_cmp_grid_times = select_ode_states(
            time_grid=self.sampling_time_grid,
            ode_states=cmp_ode_states,
            requested_times=torch.tensor(self.velocity_eval_times, device=self.device, dtype=self.dtype)
        )
        if not torch.equal(selected_ref_grid_times, selected_cmp_grid_times):
            raise ValueError(
                f"Mismatch in evaluation times: reference times {selected_ref_grid_times.tolist()} "
                f"vs comparison times {selected_cmp_grid_times.tolist()}."
            )

        # Evaluate velocity difference between reference and comparison trajectory at each intermediate time point
        batch_size = ref_ode_states.shape[1]
        vfd_oneway_score: Tensor = torch.zeros(batch_size, device=self.device, dtype=self.dtype)
        for idx, (time, ref_inter_state, cmp_inter_state) in enumerate(zip(
            selected_ref_grid_times, selected_ref_ode_states, selected_cmp_ode_states, strict=False
        )):
            # Determine dt: difference to next time or to 1.0 for last step
            dt = selected_ref_grid_times[idx + 1] - time if idx < len(selected_ref_grid_times) - 1 else 1.0 - time

            ref_velocity = ref_velocity_fn(x_t=ref_inter_state, t=time)
            cmp_velocity = cmp_velocity_fn(x_t=cmp_inter_state, t=time)
            # L2 norm across horizon and action dims gives magnitude of velocity difference
            velocity_difference = torch.norm(ref_velocity - cmp_velocity, dim=(1, 2)) ** 2

            # Scale velocity difference by factor that depends on conditional vector field type
            vfd_oneway_score += (
                self.cond_prob_path.get_vel_diff_scaling_factor(time)
                * velocity_difference
                * dt
            )

        return vfd_oneway_score


class Vfd(VfdOneway):
    """
    Two-way variant of VfdOneway.

    Averages the velocity disagreement evaluated at the reference model's ODE trajectory
    and at the comparison model's ODE trajectory.
    """
    name: str = "vfd"

    def __call__(
        self,
        ref_ode_states: Tensor,
        ref_velocity_fn: Callable[[Tensor, Tensor], Tensor],
        cmp_ode_states: Tensor,
        cmp_velocity_fn: Callable[[Tensor, Tensor], Tensor],
    ) -> Tensor:
        score_at_ref = super().__call__(ref_ode_states, ref_velocity_fn, ref_ode_states, cmp_velocity_fn)
        score_at_cmp = super().__call__(cmp_ode_states, ref_velocity_fn, cmp_ode_states, cmp_velocity_fn)
        return (score_at_ref + score_at_cmp) / 2


class ActionL2Distance(UncertaintyMetric):
    """
    Uncertainty metric based on the mean pairwise L2 distance between terminal action chunks
    sampled by two ensemble models from independent noises.

    Each model draws N action sequences from its own noise; for each reference sequence we
    average its L2 distance to all N scorer sequences.  This gives one score per reference
    sample, compatible with the [batch_size * N] output shape expected by the sampler.
    """
    name: str = "action_l2"
    type: str = "trajectory"

    def __call__(
        self,
        ref_actions: Tensor,
        cmp_actions: Tensor,
        batch_size: int,
    ) -> Tensor:
        """
        Args:
            ref_actions: Terminal actions from the reference (sampler) model.
                Shape: (batch_size * N, horizon, action_dim).
            cmp_actions: Terminal actions from the scorer model, from independent noise.
                Shape: (batch_size * N, horizon, action_dim).
            batch_size: Number of environment observations in the batch.

        Returns:
            Mean pairwise L2 distance per reference sample. Shape: (batch_size * N,).
        """
        total, H, A = ref_actions.shape
        N = total // batch_size
        ref = ref_actions.reshape(batch_size, N, H * A)
        cmp = cmp_actions.reshape(batch_size, N, H * A)
        # [batch_size, N_ref, N_cmp] pairwise distances
        dists = torch.cdist(ref, cmp, p=2)
        # mean over scorer samples → [batch_size, N], then flatten
        return dists.mean(dim=2).reshape(batch_size * N)


class EnsembleTerminalVariance(UncertaintyMetric):
    """
    Epistemic uncertainty via the log-variance of terminal action sequences across
    M ensemble members, all integrated from the same shared noise z.

    For M members and N noise samples per observation:
      - Stack terminal actions → (M, B*N, H, A)
      - Unbiased variance over M: var(dim=0) → (B*N, H, A)
      - Score = Σ_{h,a} log(Var_m(X_{b,m,h,a}) + ε) → (B*N,)

    Unlike TerminalVariance (aleatoric: spread of one model's own samples),
    this measures epistemic uncertainty: how much the M members disagree at
    the terminal action given the same starting noise.
    """
    name: str = "ensemble_terminal_variance"
    type: str = "trajectory"

    def __call__(self, ensemble_actions: Tensor, **_: Any) -> Tensor:
        """
        Args:
            ensemble_actions: Terminal action sequences from M members, all integrated
                from the same shared noise. Shape: (M, B*N, horizon, action_dim).

        Returns:
            Sum of log per-element variances across (horizon, action_dim).
            Shape: (B*N,).
        """
        var = ensemble_actions.var(dim=0, unbiased=True)  # (B*N, H, A)
        return torch.log(var + 1e-8).sum(dim=(1, 2))  # (B*N,)


class TerminalVariance(UncertaintyMetric):
    """
    Uncertainty metric based on the empirical variance of raw terminal action sequences
    sampled independently from a scorer model.

    For N independently drawn action sequences per batch element, the score is the sum
    of per-element sample variances across horizon and action dimensions:

        u_b = sum_{h=1}^{H} sum_{a=1}^{A} Var_n(X_{b,n,h,a})

    where Var is the unbiased sample variance across the N draw dimension.

    This is proportional to the log-determinant of a diagonal Gaussian fitted to the
    samples, making it a tractable proxy for Gaussian entropy.
    """
    name: str = "terminal_variance"
    type: str = "trajectory"

    def __call__(
        self,
        cmp_actions: Tensor,
        batch_size: int,
    ) -> Tensor:
        """
        Args:
            cmp_actions: Terminal action sequences from the scorer model, drawn from
                independent noise samples. Shape: (batch_size * N, horizon, action_dim).
            batch_size: Number of environment observations in the batch.

        Returns:
            Sum of per-element unbiased sample variances across N draws, broadcast back to
            per-sample shape so the caller can average across N consistently with other
            metrics. Shape: (batch_size * N,).
        """
        total, H, A = cmp_actions.shape
        N = total // batch_size
        x = cmp_actions.reshape(batch_size, N, H, A)
        # Unbiased variance across the sample dimension, summed over (H, A): (batch_size,)
        variance_scores = torch.log(x.var(dim=1, unbiased=True) + 1e-8).sum(dim=(1, 2))
            # Broadcast to (batch_size * N,) to match the shape expected by the sampler loop
        return variance_scores.repeat_interleave(N)


class ACEEntropy(UncertaintyMetric):
    """
    Grid-based Shannon entropy over N sampled action chunks, averaged over timesteps.

    Follows the ACE method: for each timestep, bin the N sampled actions into a
    per-dimension grid whose cell size is scaled from the observed range, then compute
    Shannon entropy over the bin counts.  No second model is required.

    entropy_space controls what is binned:
      - "position": integrate delta-position actions (dims 0-2) from the current EEF
                    position forward in time, then compute entropy over the resulting
                    absolute position trajectory.  Requires current_pos to be passed
                    to __call__.  Matches the original ACE paper.
      - "velocity": compute entropy directly over the raw action deltas (dims 0-2).

    cellsize_factor controls the grid resolution: cell_size = range * cellsize_factor
    per dimension, calibrated from the full set of samples across all timesteps.
    """
    name: str = "ace"
    type: str = "sample_entropy"

    def __init__(
        self,
        cellsize_factor: float = 0.03,
        entropy_space: str = "position",
    ):
        if entropy_space not in ("position", "velocity"):
            raise ValueError(f"entropy_space must be 'position' or 'velocity', got {entropy_space!r}")
        self.cellsize_factor = cellsize_factor
        self.entropy_space = entropy_space

    def __call__(self, action_samples: Tensor, current_pos: Optional[np.ndarray] = None) -> float:
        """
        Args:
            action_samples: shape [N, horizon, action_dim]
            current_pos: EEF position at the current timestep, shape [3].
                Required when entropy_space="position"; ignored otherwise.

        Returns:
            Mean Shannon entropy (bits) across timesteps.
        """
        from scipy.stats import entropy as scipy_entropy

        deltas = action_samples.cpu().float().numpy()[:, :, :3]  # [N, horizon, 3]

        if self.entropy_space == "position":
            if current_pos is None:
                raise ValueError("current_pos is required when entropy_space='position'")
            # Cumulative sum of deltas from current position → absolute positions
            # samples[:, t, :] = current_pos + sum(deltas[:, 0:t+1, :], axis=1)
            samples = current_pos[np.newaxis, np.newaxis, :] + np.cumsum(deltas, axis=1)  # [N, horizon, 3]
        else:
            samples = deltas  # [N, horizon, 3]

        N, horizon, D = samples.shape

        # Calibrate cell sizes from the full distribution of samples
        flat = samples.reshape(-1, D)
        ranges = flat.max(axis=0) - flat.min(axis=0)
        max_range = float(ranges.max()) or 1e-8
        cell_size = np.where(ranges == 0, max_range, ranges) * self.cellsize_factor

        entropy_values = []
        for t in range(horizon):
            pts = samples[:, t, :]  # [N, D]
            indices = np.stack(
                [
                    np.digitize(pts[:, d], np.arange(pts[:, d].min(), pts[:, d].max() + cell_size[d], cell_size[d])) - 1
                    for d in range(D)
                ],
                axis=1,
            )
            grid_sizes = [max(int(np.ceil((pts[:, d].max() - pts[:, d].min()) / cell_size[d])), 1) for d in range(D)]
            strides = np.ones(D, dtype=np.int64)
            for d in range(D - 2, -1, -1):
                strides[d] = strides[d + 1] * grid_sizes[d + 1]
            flat_idx = (indices * strides).sum(axis=1)
            counts = np.bincount(flat_idx, minlength=1)
            entropy_values.append(float(scipy_entropy(counts, base=2)))

        return float(np.mean(entropy_values))
