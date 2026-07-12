from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from torch import Tensor

from lerobot.policies.flow_matching.configuration_flow_matching import FlowMatchingConfig
from lerobot.policies.flow_matching.modelling_flow_matching import FlowMatchingModel

from ..common.flow_matching.adapter import BaseFlowMatchingAdapter


class FlowMatchingAdapter(BaseFlowMatchingAdapter):
    def __init__(self, config: FlowMatchingConfig, model: FlowMatchingModel):
        super().__init__(model=model, config=config)

    @property
    def horizon(self) -> int:
        return self.config.horizon

    @property
    def action_dim(self) -> int:
        return self.config.action_feature.shape[0]

    @property
    def ode_solver_config(self) -> dict[str, Any]:
        return {
            "solver_method": self.config.ode_solver_method,
            "step_size": self.config.ode_step_size,
            "atol": self.config.atol,
            "rtol": self.config.rtol,
        }

    @property
    def cond_vf_config(self) -> dict[str, Any]:
        return {
            "type": self.config.cond_vf_type,
            "sigma_min": self.config.sigma_min,
            "beta_min": self.config.beta_min,
            "beta_max": self.config.beta_max,
        }

    @torch.no_grad()
    def prepare_conditioning(self, observation: dict[str, Tensor], num_action_samples: int) -> dict[str, Tensor]:
        observation = self.expand_observation(observation=observation, num_action_samples=num_action_samples)
        global_cond = self.model.prepare_global_conditioning(observation)

        return {
            "global_cond": global_cond,
        }

    def make_velocity_fn(self, conditioning: dict[str, Tensor]) -> Callable[[Tensor, Tensor], Tensor]:
        def v_t(t: Tensor, x_t: Tensor) -> Tensor:
            batch_size = x_t.shape[0]
            return self.model.unet(
                x_t, t.expand(batch_size), global_cond=conditioning["global_cond"]
            )
        return v_t

    def prepare_fiper_obs_embedding(self, conditioning: dict[str, Tensor]) -> np.ndarray:
        return conditioning["global_cond"][0].detach().cpu().numpy()
