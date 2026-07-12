import numpy as np
import torch

from iterative_fine_tuning.selection import _prepare_observation_batch_for_sampler
from lerobot.policies.common.flow_matching.adapter import BaseFlowMatchingAdapter


class _DummyConfig:
    n_action_steps = 1
    n_obs_steps = 1


class _DummyAdapter(BaseFlowMatchingAdapter):
    @property
    def horizon(self) -> int:
        return 2

    @property
    def action_dim(self) -> int:
        return 3

    @property
    def ode_solver_config(self) -> dict[str, str | float | None]:
        return {
            "solver_method": "euler",
            "step_size": 0.1,
            "atol": None,
            "rtol": None,
        }

    @property
    def cond_vf_config(self) -> dict[str, str | float]:
        return {"type": "ot", "sigma_min": 0.0}

    def prepare_conditioning(self, observation: dict[str, torch.Tensor], num_action_samples: int):
        return self.expand_observation(observation, num_action_samples)

    def make_velocity_fn(self, conditioning):
        def _velocity_fn(x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            del t
            return torch.zeros_like(x_t)

        return _velocity_fn

    def prepare_fiper_obs_embedding(self, conditioning):
        del conditioning
        return np.zeros(1, dtype=np.float32)


def test_expand_observation_supports_batched_candidates():
    adapter = _DummyAdapter(model=torch.nn.Linear(1, 1), config=_DummyConfig())
    observation = {
        "observation.state": torch.tensor(
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ]
        )
    }

    expanded = adapter.expand_observation(observation, num_action_samples=3)

    assert expanded["observation.state"].shape == (6, 2)
    assert torch.equal(
        expanded["observation.state"],
        torch.tensor(
            [
                [1.0, 2.0],
                [1.0, 2.0],
                [1.0, 2.0],
                [3.0, 4.0],
                [3.0, 4.0],
                [3.0, 4.0],
            ]
        ),
    )


def test_prepare_observation_batch_for_sampler_concatenates_candidates():
    processed_samples = [
        {"observation.state": torch.tensor([1.0, 2.0])},
        {"observation.state": torch.tensor([3.0, 4.0])},
    ]

    observation_batch = _prepare_observation_batch_for_sampler(processed_samples, device="cpu")

    assert observation_batch["observation.state"].shape == (2, 2)
    assert torch.equal(
        observation_batch["observation.state"],
        torch.tensor(
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ]
        ),
    )
