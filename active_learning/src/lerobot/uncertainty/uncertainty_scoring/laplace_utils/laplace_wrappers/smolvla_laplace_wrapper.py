from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data.dataloader import default_collate

from lerobot.policies.common.aloha import (
    pi_aloha_decode_state,
    pi_aloha_encode_actions_inv,
)
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import VLAFlowMatching, make_att_2d_masks
from lerobot.processor import PolicyProcessorPipeline
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE
from lerobot.utils.utils import get_safe_torch_device

from .laplace_wrapper import LaplaceBatch, LaplaceWrapper


class SmolVLALaplaceWrapper(LaplaceWrapper):
    """Laplace wrapper for VLAFlowMatching models."""
    def __init__(self, config: SmolVLAConfig, model: VLAFlowMatching, scopes: list[str]):
        """
        Initialize a Laplace approximation wrapper around a VLAFlowMatching model.

        Args:
            config: SmolVLA policy configuration.
            model: The underlying VLAFlowMatching model to calibrate.
            scopes: Which submodules to expose to the Laplace posterior.
                Available scopes:
                    - "action_out_proj": Final linear projection from expert hidden size
                        to action dimension.
                    - "action_time_embed": The action-time fusion layer that mixes action tokens with
                        the timestep embedding.
                    - "expert_last": The final transformer block of the action expert.
                    - "state_proj": The projection layer from the state to the hidden size of the text model.
                    - "vlm_connector": VLM connector mapping vision tokens to text space.
                    - "vision_last": Last layer of the vision encoder.
                Defaults to ["expert_last", "action_out_proj"].
        """
        scopes = scopes or ["action_out_proj"]
        super().__init__(model=model, config=config, scopes=scopes)

        self.scope_abbr: dict[str, str] = {
            "action_out_proj": "aop",
            "action_time_embed": "ate",
            "expert_last": "el",
            "state_proj": "sp",
            "vlm_connector": "vc",
            "vision_last": "vl",
        }

    def build_collate_fn(
        self, preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]]
    ) -> Callable[[dict[str, Tensor]], tuple[LaplaceBatch, Tensor]]:
        """
        Factory that builds a dataloader collate function tailored for Laplace calibration of
        a VLAFlowMatching model.

        The returned method converts a list of raw samples coming from the training dataset into the
        pair (LaplaceBatch, target_velocity) to fit a Laplace posterior using laplace-torch.

        Args:
            preprocessor: Preprocessor to apply to raw dataset samples.
        """
        # Check device is available
        device = get_safe_torch_device(self.device)

        def collate_fn(batch: dict[str, Tensor]) -> tuple[LaplaceBatch, Tensor]:
            # Turn into batched dict of tensors
            batch = default_collate(batch)
            # Apply the dataset preprocessor
            batch = preprocessor(batch)

            if self.config.adapt_to_pi_aloha:
                batch[OBS_STATE] = pi_aloha_decode_state(batch[OBS_STATE])
                batch[ACTION] = pi_aloha_encode_actions_inv(batch[ACTION])

            # Extract actions as stored in the batch
            actions = batch[ACTION]

            # Sample random time and noise
            time = self.model.sample_time(actions.shape[0], device)
            noise = self.model.sample_noise(actions.shape, device)

            # Form noisy actions according to optimal transport path
            x_t = time[:, None, None] * noise + (1 - time[:, None, None]) * actions

            # Target velocity along the path is u_t = noise - actions
            u_t = noise - actions

            # Create observation only dictionary
            observation = {
                key: value.cpu()
                for key, value in batch.items()
                if key.startswith("observation.")
            }

            # Apply mask to target velocity so padded steps outside of the episode contribute zero loss
            actions_is_pad = batch.get("action_is_pad")
            if actions_is_pad is None:
                in_episode_mask = torch.ones(actions.shape[:2], dtype=torch.bool, device=device)
            else:
                in_episode_mask = ~actions_is_pad
            u_t = u_t * in_episode_mask.unsqueeze(-1)

            # Package inputs and targets exactly as laplace-torch expects
            input_batch = LaplaceBatch(
                interp_traj=x_t,
                time=time,
                observation=observation,
                in_episode_mask=in_episode_mask,
            )
            target_batch = u_t.flatten(start_dim=1)

            return input_batch, target_batch

        return collate_fn

    def forward(self, batch: LaplaceBatch) -> Tensor:
        """
        Compute the vector-field prediction v_t for an input (x_t, t) of noisy actions and time
        conditioned on some observation.

        This method nearly copies the VLAFlowMatching's forward method.
        """
        images, img_masks = self.model.prepare_images(batch.observation)
        state = self.model.prepare_state(batch.observation)
        lang_tokens = batch.observation[f"{OBS_LANGUAGE_TOKENS}"]
        lang_masks = batch.observation[f"{OBS_LANGUAGE_ATTENTION_MASK}"]

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.model.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.model.embed_suffix(
            noisy_actions=batch.interp_traj,
            timestep=batch.time,
        )

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        (_, suffix_out), _ = self.model.vlm_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            fill_kv_cache=False,
        )
        suffix_out = suffix_out[:, -self.config.chunk_size :]

        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.model.action_out_proj(suffix_out)

        # Remove padded action dims
        v_t = v_t[:, :, :self.config.action_feature.shape[0]]

        # Ensure padded steps outside of the episode contribute zero loss
        v_t = v_t * batch.in_episode_mask.unsqueeze(-1)

        return v_t.flatten(start_dim=1)

    def apply_laplace_scope(self):
        """
        Freeze the entire model and unfreeze only the submodules selected by the specified
        scopes to be fitted by the Laplace approximation.
        """
        # Select target modules for Laplace approximation
        target_modules: list[nn.Module] = []
        for scope in self.scopes:
            if scope == "action_out_proj":
                target_modules.append(self.model.action_out_proj)
            elif scope == "action_time_embed":
                target_modules.append(self.model.action_time_mlp_out)
            elif scope == "expert_last":
                action_expert = self.model.vlm_with_expert.lm_expert
                layer = action_expert.layers[-1]
                target_modules.extend([
                    layer.self_attn.o_proj,
                    layer.mlp.gate_proj,
                    layer.mlp.up_proj,
                    layer.mlp.down_proj
                ])
            elif scope == "state_proj":
                target_modules.append(self.model.state_proj)
            elif scope == "vlm_connector":
                connector = self.model.vlm_with_expert.get_vlm_model().connector
                target_modules.append(connector)
            elif scope == "vision_last":
                vision_model = self.model.vlm_with_expert.get_vlm_model().vision_model
                last_vision_layer = vision_model.encoder.layers[-1]
                # Attention projections
                target_modules.extend([
                    last_vision_layer.self_attn.q_proj,
                    last_vision_layer.self_attn.k_proj,
                    last_vision_layer.self_attn.v_proj,
                    last_vision_layer.self_attn.out_proj,
                ])

                # MLP linears within the vision last layer
                for m in last_vision_layer.mlp.modules():
                    if isinstance(m, nn.Linear):
                        target_modules.append(m)
            else:
                raise ValueError(
                    f"Unknown Laplace approximation target {scope}. "
                    "Choose from ['action_out_proj', 'action_time_embed', 'expert_last', "
                    "'state_proj', 'vlm_connector', 'vision_last']."
                )

        # Freeze all parameters
        for p in self.model.parameters():
            p.requires_grad_(False)

        # Unfreeze only the selected subnetwork for Laplace fitting
        for module in target_modules:
            for p in module.parameters():
                p.requires_grad_(True)
