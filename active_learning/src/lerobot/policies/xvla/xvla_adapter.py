from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from torch import Tensor

from lerobot.policies.xvla.configuration_xvla import XVLAConfig
from lerobot.policies.xvla.modeling_xvla import XVLAModel
from lerobot.utils.constants import OBS_LANGUAGE_TOKENS

from ..common.flow_matching.adapter import BaseFlowMatchingAdapter


class XVLAAdapter(BaseFlowMatchingAdapter):
    """Flow matching adapter for the X-VLA vision-language-action policy.

    X-VLA's transformer predicts the clean action x_0_hat with a reversed time
    convention (model time 0 = data, 1 = noise) instead of a velocity field. The
    adapter exposes the equivalent velocity v(t, x_t) = (x_0_hat - x_t) / (1 - t)
    in the shared convention (t=0 noise, t=1 data) used by the ODE solver.
    """

    def __init__(self, config: XVLAConfig, model: XVLAModel):
        super().__init__(model=model, config=config)

    @property
    def horizon(self) -> int:
        return self.config.chunk_size

    @property
    def action_dim(self) -> int:
        return self.model.dim_action

    @property
    def env_action_dim(self) -> int:
        return self.config.action_feature.shape[0]

    @property
    def dtype(self) -> torch.dtype:
        return self.model._get_target_dtype()

    @property
    def ode_solver_config(self) -> dict[str, Any]:
        return {
            "solver_method": "euler",
            "step_size": 0.1,
            "atol": None,
            "rtol": None,
        }

    @property
    def cond_vf_config(self) -> dict[str, Any]:
        return {
            "type": "ot",
            "sigma_min": 0,
            "beta_min": None,
            "beta_max": None,
        }

    @torch.no_grad()
    def prepare_conditioning(self, observation: dict[str, Tensor], num_action_samples: int) -> dict[str, Tensor]:
        """Encode the observation through the X-VLA VLM backbone once for reuse
        across all denoising steps.

        Returns:
            A dict with keys:

            - "domain_id": Domain identifier tensor used by the action transformer.
            - "proprio": Proprioceptive state tensor.
            - "vlm_encoding": Dict of VLM feature tensors ("vlm_features", "aux_visual_inputs").
        """
        input_ids_raw = observation[OBS_LANGUAGE_TOKENS]
        batch_size = int(input_ids_raw.shape[0])

        # domain_id does not use the "observation" prefix, so it is dropped by
        # expand_observation; expand it explicitly alongside the other tensors.
        domain_id = self.model.get_domain_id(observation, batch_size, input_ids_raw.device)
        domain_id = (
            domain_id.unsqueeze(1)
            .expand(batch_size, num_action_samples)
            .reshape(batch_size * num_action_samples)
        )

        observation = self.expand_observation(observation=observation, num_action_samples=num_action_samples)

        target_dtype = self.model._get_target_dtype()
        input_ids = observation[OBS_LANGUAGE_TOKENS]
        images, image_mask = self.model.prepare_images(observation)
        proprio = self.model.prepare_state(observation, input_ids.shape[0], images.device)

        images = images.to(dtype=target_dtype)
        proprio = proprio.to(dtype=target_dtype)

        vlm_encoding = self.model.forward_vlm(input_ids, images, image_mask)

        return {
            "domain_id": domain_id,
            "proprio": proprio,
            "vlm_encoding": vlm_encoding,
        }

    def make_velocity_fn(self, conditioning: dict[str, Tensor]) -> Callable[[Tensor, Tensor], Tensor]:
        def v_t(t: Tensor, x_t: Tensor) -> Tensor:
            t_batched = t.expand(x_t.shape[0])
            # X-VLA denoises with reversed timestep model_t = 1 - t
            model_t = 1 - t_batched

            proprio_m, x_t_m = self.model.action_space.preprocess(
                conditioning["proprio"], x_t
            )

            # Model predicts the clean action x_0_hat; derive velocity from it
            x0_hat = self.model.transformer(
                domain_id=conditioning["domain_id"],
                action_with_noise=x_t_m,
                proprio=proprio_m,
                t=model_t,
                **conditioning["vlm_encoding"],
            )
            return (x0_hat - x_t_m) / (1 - t_batched).view(x_t.shape[0], 1, 1)
        return v_t

    def prepare_fiper_obs_embedding(self, conditioning: dict[str, Tensor], batch_index: int = 0) -> np.ndarray:
        """Extract a flat observation embedding used by the FIPER nearest-neighbour lookup.

        Concatenates the domain ID, proprioceptive state, a strided sample of the VLM token
        features, and a strided sample of the auxiliary visual tokens from real camera views
        (excluding empty/padding cameras).

        `batch_index` selects the element of the conditioning batch. Rollout recording builds the
        conditioning from a single observation, so the default 0 is what the recorder and the scorer
        store; embedding a *batch* of demonstration frames (extract_demo_embeddings.py) runs one prefix
        pass and reads each element in turn, which is far cheaper than a pass per frame.
        """
        flat_domain_id = conditioning["domain_id"][batch_index].flatten()
        flat_proprio = conditioning["proprio"][batch_index].flatten()

        flat_vlm_tokens = conditioning["vlm_encoding"]["vlm_features"][batch_index].flatten()[::5]

        aux_visual_tokens = conditioning["vlm_encoding"]["aux_visual_inputs"][batch_index]
        num_aux_views = self.config.num_image_views - 1
        num_real_aux_views = num_aux_views - self.config.empty_cameras
        num_aux_visual_tokens_per_view = aux_visual_tokens.shape[0] // num_aux_views
        aux_visual_tokens = aux_visual_tokens.view(num_aux_views, num_aux_visual_tokens_per_view, -1)
        flat_real_aux_view_tokens = aux_visual_tokens[:num_real_aux_views].flatten()[::5]

        fiper_obs_embedding = torch.cat(
            [flat_domain_id, flat_proprio, flat_vlm_tokens, flat_real_aux_view_tokens]
        )

        return fiper_obs_embedding.detach().to(torch.float32).cpu().numpy()
