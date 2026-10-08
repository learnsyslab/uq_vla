from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from torch import Tensor

from lerobot.policies.fastwam.configuration_fastwam import FastWAMConfig
from lerobot.policies.fastwam.modeling_fastwam import _input_image_from_batch, _prompt_from_batch, _proprio_from_batch
from lerobot.policies.fastwam.wan.modular import FastWAM

from ..common.flow_matching.adapter import BaseFlowMatchingAdapter


def _expand_samples(x: Tensor, num_action_samples: int) -> Tensor:
    """[B, ...] -> [B * N, ...] with the sample index fastest, matching `expand_observation`."""
    batch_size = int(x.shape[0])
    return (
        x.unsqueeze(1)
        .expand(batch_size, num_action_samples, *x.shape[1:])
        .reshape(batch_size * num_action_samples, *x.shape[1:])
    )


class FastWAMAdapter(BaseFlowMatchingAdapter):
    """Flow matching adapter for the action branch of FastWAM.

    FastWAM samples actions exactly the way the paper describes inference: the video expert runs
    once on the first-frame VAE latents at video timestep 0 and its per-layer K/V are cached; the
    action DiT is then denoised alone, attending to that fixed cache (`FastWAM.infer_action`). The
    adapter reproduces this split -- `prepare_conditioning` is the prefill, the velocity function is
    one cached action-denoiser call -- so no video tokens are recomputed per ODE step.

    Time convention. `WanContinuousFlowMatchScheduler` noises with sigma running from 1 (noise) to 0
    (data), x_sigma = (1 - sigma) x_0 + sigma eps, and the network predicts eps - x_0 = dx/dsigma. In
    the shared convention of the ODE solver (t = 0 noise, t = 1 data) this is sigma = 1 - t and
    v(t, x) = -pred(x, timestep = (1 - t) * num_train_timesteps).
    """

    def __init__(self, config: FastWAMConfig, model: FastWAM):
        super().__init__(model=model, config=config)

    @property
    def horizon(self) -> int:
        return int(self.config.action_horizon)

    @property
    def action_dim(self) -> int:
        return int(self.model.action_expert.action_dim)

    @property
    def env_action_dim(self) -> int:
        feature = self.config.action_feature if self.config.output_features else None
        if feature is None:
            return self.action_dim
        return min(int(feature.shape[0]), self.action_dim)

    @property
    def device(self) -> torch.device:
        return self.model.device

    @property
    def dtype(self) -> torch.dtype:
        return self.model.torch_dtype

    @property
    def ode_solver_config(self) -> dict[str, Any]:
        # Same fixed-step Euler grid as the X-VLA adapter (10 steps; FastWAM's own
        # `num_inference_steps` default is also 10, on a shifted sigma grid).
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
    def prepare_conditioning(self, observation: dict[str, Tensor], num_action_samples: int) -> dict[str, Any]:
        """Prefill the video branch once per observation and expand the cache for the action samples.

        The observation is the preprocessed policy batch: `observation.images.*` in [0, 1] (first frame
        is used), optional `observation.state`, and either `task` (looked up in the precomputed prompt
        table or encoded by the UMT5 text encoder) or precomputed `context` / `context_mask`.

        Returns:
            A dict with the tensors `_predict_action_noise_with_cache` needs at batch B * N
            ("video_kv_cache", "context", "context_mask", "attention_mask", "video_seq_len") plus the
            unexpanded per-observation "first_frame_latents" and "proprio" used for FIPER embeddings.
        """
        model = self.model
        input_image = _input_image_from_batch(observation, self.config)  # [B, 3, H, W], resized
        batch_size = int(input_image.shape[0])
        _, _, height, width = input_image.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"FastWAM input image must have H, W multiples of 16, got ({height}, {width}).")
        input_image = input_image.to(device=model.device, dtype=model.torch_dtype)

        prompt = _prompt_from_batch(observation, self.config)
        context = observation.get("context")
        context_mask = observation.get("context_mask")
        if prompt is not None and (context is not None or context_mask is not None):
            # `_prepare_infer_context` treats them as mutually exclusive; precomputed context wins.
            prompt = None
        if isinstance(prompt, str):
            prompt = [prompt] * batch_size
        proprio = _proprio_from_batch(observation)
        if proprio is not None and model.proprio_encoder is not None:
            proprio = proprio.to(device=model.device, dtype=model.torch_dtype)
        else:
            proprio = None

        first_frame_latents = torch.cat(
            [model._encode_input_image_latents_tensor(input_image=input_image[i : i + 1]) for i in range(batch_size)],
            dim=0,
        )  # [B, C, 1, h, w]
        context, context_mask = model._prepare_infer_context(prompt, context, context_mask, proprio)
        context = context.to(device=model.device)
        context_mask = context_mask.to(device=model.device)

        timestep_video = torch.zeros((batch_size,), dtype=first_frame_latents.dtype, device=model.device)
        video_pre = model.video_expert.pre_dit(
            x=first_frame_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=bool(getattr(model.video_expert, "fuse_vae_embedding_in_latents", False)),
        )
        video_seq_len = int(video_pre["tokens"].shape[1])
        attention_mask = model._build_mot_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=self.horizon,
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )
        video_kv_cache = model.mot.prefill_video_cache(
            video_tokens=video_pre["tokens"],
            video_freqs=video_pre["freqs"],
            video_t_mod=video_pre["t_mod"],
            video_context_payload={"context": video_pre["context"], "mask": video_pre["context_mask"]},
            video_attention_mask=attention_mask[:video_seq_len, :video_seq_len],
        )

        return {
            "video_kv_cache": [
                {
                    "k": _expand_samples(layer["k"], num_action_samples),
                    "v": _expand_samples(layer["v"], num_action_samples),
                }
                for layer in video_kv_cache
            ],
            "context": _expand_samples(context, num_action_samples),
            "context_mask": _expand_samples(context_mask, num_action_samples),
            "attention_mask": attention_mask,
            "video_seq_len": video_seq_len,
            "first_frame_latents": first_frame_latents,
            "proprio": proprio,
            "num_action_samples": int(num_action_samples),
        }

    def make_velocity_fn(self, conditioning: dict[str, Any]) -> Callable[[Tensor, Tensor], Tensor]:
        model = self.model
        num_train_timesteps = float(model.infer_action_scheduler.num_train_timesteps)

        def v_t(t: Tensor, x_t: Tensor) -> Tensor:
            batch_size = int(x_t.shape[0])
            t_batched = torch.as_tensor(t, device=x_t.device, dtype=torch.float32).reshape(-1)
            if t_batched.numel() == 1:
                t_batched = t_batched.expand(batch_size)
            # shared t (0 = noise) -> scheduler sigma (1 = noise); `infer_action` feeds the timestep in
            # the latents' dtype, keep that so bf16 scoring matches bf16 inference.
            timestep_action = ((1.0 - t_batched) * num_train_timesteps).to(dtype=model.torch_dtype)

            pred = model._predict_action_noise_with_cache(
                latents_action=x_t.to(dtype=model.torch_dtype),
                timestep_action=timestep_action,
                context=conditioning["context"],
                context_mask=conditioning["context_mask"],
                video_kv_cache=conditioning["video_kv_cache"],
                attention_mask=conditioning["attention_mask"],
                video_seq_len=conditioning["video_seq_len"],
            )
            # The network predicts eps - x_0 = dx/dsigma; the shared-convention velocity is dx/dt = -that.
            return (-pred).to(dtype=x_t.dtype)

        return v_t

    def prepare_fiper_obs_embedding(self, conditioning: dict[str, Any], batch_index: int = 0) -> np.ndarray:
        """Flat observation embedding for the FIPER nearest-neighbour lookup.

        Concatenates the first-frame VAE latents, the proprioceptive state (if any) and a strided sample
        of the middle MoT layer's cached video keys and values -- the only video information the action
        branch ever sees. `batch_index` addresses the observation; the K/V cache is stored expanded, so
        its row is `batch_index * num_action_samples`.
        """
        parts = [conditioning["first_frame_latents"][batch_index].flatten()]
        if conditioning.get("proprio") is not None:
            parts.append(conditioning["proprio"][batch_index].flatten())
        cache = conditioning["video_kv_cache"]
        mid = cache[(len(cache) // 2) - 1] if len(cache) > 1 else cache[0]
        row = batch_index * int(conditioning["num_action_samples"])
        parts.append(mid["k"][row].flatten()[::8])
        parts.append(mid["v"][row].flatten()[::8])
        return torch.cat([p.to(torch.float32) for p in parts]).detach().cpu().numpy()
