from dataclasses import dataclass, field


@dataclass
class LikelihoodODESolverConfig:
    """Configuration for the ODE solver used in likelihood estimation."""
    method: str = "euler"
    atol: float | None = None
    rtol: float | None = None
    exact_divergence: bool = False

@dataclass
class LaplaceConfig:
    """Configuration parameters for the Laplace approximation."""
    scopes: list[str] | None = None
    calib_fraction: float = 1.0
    batch_size: int = 1
    num_samples: int = 5

@dataclass
class FiperRolloutScorerConfig:
    # Which scores to compute for each method
    scores_by_method: dict[str, list[str]] = field(
        default_factory=lambda: {
            "bayesian_laplace": ["mode_distance", "likelihood", "inter_vel_diff"],
            "bayesian_ensemble": ["mode_distance", "likelihood", "inter_vel_diff"],
            "composed": ["mode_distance", "likelihood", "inter_vel_diff"],
            "composed_bayesian_laplace": ["mode_distance", "likelihood", "inter_vel_diff"],
            "composed_bayesian_ensemble": ["mode_distance", "likelihood", "inter_vel_diff"],
        }
    )

    # Parameters for the ensemble model
    ensemble_model_paths: list[str] | None = None
    # If the rollout/sampler policy is also present in the ensemble, this index can
    # be skipped during two-way scorer trajectory integration and treated exactly.
    sampler_ensemble_member_index: int | None = None

    # Parameters for the Laplace approximation
    laplace_config: LaplaceConfig = field(default_factory=LaplaceConfig)

    # Times at which to evaluate the velocity of the sampled action sequence under a scorer model
    terminal_vel_eval_times: list[float] = field(
        default_factory=lambda: [round(i * 0.1, 1) for i in range(10)]
    )

    # Configuration for the ODE solver used in likelihood estimation
    likelihood_ode_solver_cfg: LikelihoodODESolverConfig = field(
        default_factory=LikelihoodODESolverConfig
    )

    def should_compute(self, method: str, score: str) -> bool:
        """Convenience helper to check if a given score should be computed."""
        return score in self.scores_by_method.get(method, [])

    def is_method_enabled(self, method: str) -> bool:
        """Return True if this method should be computed at all."""
        scores = self.scores_by_method.get(method, [])
        return len(scores) > 0
