from dataclasses import dataclass, field


def process_ode_eval_times(
    ode_eval_times: list[float],
) -> list[float]:
    """
    Clean and normalize ODE evaluation times.

    - Removes any values outside (0, 1).
    - Sorts the list in ascending order and removes duplicates.
    """
    # Remove points outside the range of (0,1)
    processed_ode_eval_times: list[float] = []
    for t in ode_eval_times:
        if round(t, 3) < 0.0 or round(t, 3) > 1.0:
            continue
        processed_ode_eval_times.append(round(t, 3))

    # Sort and remove duplicates
    processed_ode_eval_times = sorted(set(processed_ode_eval_times))

    return processed_ode_eval_times

@dataclass
class FiperRolloutRecorderConfig:
    # Number of action sequences sampled per observation for uncertainty estimation
    num_uncertainty_sequences: int = 256

    # Times at which to record ODE states and velocities during the flow integration
    ode_eval_times: list[float] = field(
        default_factory=lambda: [round(i * 0.05, 2) for i in range(20)] + [1.0]
    )

    record_composed_inter_vel_diff: bool = True

    def __post_init__(self) -> None:
        self.ode_eval_times = process_ode_eval_times(self.ode_eval_times)