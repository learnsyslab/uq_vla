from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from torch import Tensor

from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import PI05Pytorch, make_att_2d_masks, resize_with_pad_torch
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

from ..common.flow_matching.adapter import BaseFlowMatchingAdapter


class Pi05Adapter(BaseFlowMatchingAdapter):
    def __init__(self, config: PI05Config, model: PI05Pytorch):
        super().__init__(model=model, config=config)

    @property
    def horizon(self) -> int:
        return self.config.chunk_size

    @property
    def action_dim(self) -> int:
        return self.config.max_action_dim

    @property
    def dtype(self) -> torch.dtype:
        return torch.float32

    @property
    def ode_solver_config(self) -> dict[str, Any]:
        return {
            "solver_method": "euler",
            "step_size": 1.0 / self.config.num_inference_steps,
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

    def _preprocess_images(self, observation: dict[str, Tensor]) -> tuple[list[Tensor], list[Tensor]]:
        """Preprocess images for embed_prefix (mirrors PI05Policy._preprocess_images)."""
        images = []
        img_masks = []
        device = self.device

        present_img_keys = [k for k in self.config.image_features if k in observation]
        missing_img_keys = [k for k in self.config.image_features if k not in observation]

        if not present_img_keys:
            raise ValueError(
                f"No image features found in observation. Expected one of: {list(self.config.image_features)}"
            )

        img = mask = None
        for key in present_img_keys:
            img = observation[key].to(device=device, dtype=torch.float32)
            is_channels_first = img.shape[1] == 3
            if is_channels_first:
                img = img.permute(0, 2, 3, 1)
            if img.shape[1:3] != self.config.image_resolution:
                img = resize_with_pad_torch(img, *self.config.image_resolution)
            img = img * 2.0 - 1.0
            if is_channels_first:
                img = img.permute(0, 3, 1, 2)
            images.append(img)
            mask = torch.ones(img.shape[0], dtype=torch.bool, device=device)
            img_masks.append(mask)

        for _ in missing_img_keys:
            images.append(torch.ones_like(img) * -1)
            img_masks.append(torch.zeros_like(mask))

        return images, img_masks

    @torch.no_grad()
    def prepare_conditioning(self, observation: dict[str, Tensor], num_action_samples: int) -> dict[str, Tensor]:
        observation = self.expand_observation(observation=observation, num_action_samples=num_action_samples)

        images, img_masks = self._preprocess_images(observation)
        tokens = observation[OBS_LANGUAGE_TOKENS]
        masks = observation[OBS_LANGUAGE_ATTENTION_MASK]

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.model.embed_prefix(
            images=images, img_masks=img_masks, tokens=tokens, masks=masks
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = self.model._prepare_attention_masks_4d(prefix_att_2d_masks)

        self.model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"

        _, past_key_values = self.model.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        return {
            "prefix_pad_masks": prefix_pad_masks,
            "past_key_values": past_key_values,
        }

    def make_velocity_fn(self, conditioning: dict[str, Tensor]) -> Callable[[Tensor, Tensor], Tensor]:
        def v_t(t: Tensor, x_t: Tensor) -> Tensor:
            # Pi0.5 denoise_step expects timestep where 1=noise, 0=action.
            # The ODE solver integrates in the opposite direction (0→1), so invert t and negate v.
            s = 1 - t
            v_s = self.model.denoise_step(
                prefix_pad_masks=conditioning["prefix_pad_masks"],
                past_key_values=conditioning["past_key_values"],
                x_t=x_t,
                timestep=s.expand(x_t.shape[0]),
            )
            return -v_s

        return v_t

    def prepare_fiper_obs_embedding(self, conditioning: dict[str, Tensor]) -> np.ndarray:
        past_kv = conditioning["past_key_values"]
        mid = max(0, (len(past_kv) // 2) - 1)
        kv_layer = past_kv[mid]
        keys, values = kv_layer[0], kv_layer[1]
        return (
            torch.cat([keys.reshape(-1), values.reshape(-1)]).detach().to(torch.float32).cpu().numpy()
        )
