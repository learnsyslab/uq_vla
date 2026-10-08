#!/usr/bin/env python

"""Flow Matching Policy as per "Flow Matching for Generative Modelling"."""
from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable, Sequence

import einops
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from torch import Tensor, nn
from tqdm import tqdm

from lerobot.fiper_data_generator.configuration_fiper_rollout_recorder import (
    FiperRolloutRecorderConfig,
)
from lerobot.policies.common.flow_matching.conditional_probability_path import make_cond_prob_path
from lerobot.policies.common.flow_matching.ode_solver import ODESolver
from lerobot.policies.flow_matching.configuration_flow_matching import FlowMatchingConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import (
    get_device_from_parameters,
    get_dtype_from_parameters,
    get_output_shape,
    populate_queues,
    replace_submodules,
)
from lerobot.uncertainty.uncertainty_samplers.configuration_uncertainty_sampler import (
    UncertaintySamplerConfig,
)
from lerobot.utils.constants import (
    ACTION,
    FINAL_FEATURE_MAP_MODULE,
    OBS_ENV_STATE,
    OBS_IMAGES,
    OBS_STATE,
)


class FlowMatchingPolicy(PreTrainedPolicy):
    """
    Flow Matching Policy as per "Flow Matching for Generative Modelling"
    (paper: https://arxiv.org/abs/2210.02747)
    """

    config_class = FlowMatchingConfig
    name = "flow_matching"

    def __init__(
        self,
        config: FlowMatchingConfig,
    ):
        """
        Args:
            config: Policy configuration class instance.
            dataset_stats: Dataset statistics to be used for normalization. If not passed here, it is expected
                that they will be passed with a call to `load_state_dict` before the policy is used.
        """
        super().__init__(config)
        config.validate_features()
        self.config = config

        # queues are populated during rollout of the policy, they contain the n latest observations and actions
        self._queues = None

        self.flow_matching = FlowMatchingModel(config)

        self.uncertainty_sampler = None
        self.fiper_rollout_recorder = None

        self.reset()

    def init_uncertainty_sampler(
        self,
        config: UncertaintySamplerConfig,
        scorer_artifacts,
    ):
        """
        Constructs the uncertainty sampler based on the config.
        """
        from lerobot.policies.factory import make_uncertainty_sampler

        self.uncertainty_sampler = make_uncertainty_sampler(
            uncertainty_sampler_config=config,
            policy_config=self.config,
            model=self.flow_matching,
            scorer_artifacts=scorer_artifacts,
        )

    def init_fiper_rollout_recorder(
        self,
        config: FiperRolloutRecorderConfig,
    ):
        """
        Constructs the FIPER data recorder based on the config.
        """
        from lerobot.fiper_data_generator.fiper_rollout_recorder import FiperRolloutRecorder
        from lerobot.policies.factory import make_flow_matching_adapter

        flow_matching_adapter = make_flow_matching_adapter(
            model=self.flow_matching,
            policy_config=self.config
        )

        self.fiper_rollout_recorder = FiperRolloutRecorder(
            config=config,
            flow_matching_adapter=flow_matching_adapter,
        )

    def get_optim_params(self) -> dict:
        return self.flow_matching.parameters()

    def reset(self):
        """Clear observation and action queues. Should be called on `env.reset()`"""
        self._queues = {
            "observation.state": deque(maxlen=self.config.n_obs_steps),
            "action": deque(maxlen=self.config.n_action_steps),
        }
        if self.config.image_features:
            self._queues["observation.images"] = deque(maxlen=self.config.n_obs_steps)
        if self.config.env_state_feature:
            self._queues["observation.environment_state"] = deque(maxlen=self.config.n_obs_steps)

        if self.uncertainty_sampler is not None:
            self.uncertainty_sampler.reset()

        if self.fiper_rollout_recorder is not None:
            self.fiper_rollout_recorder.reset()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        # stack n latest observations from the queue
        batch = {k: torch.stack(list(self._queues[k]), dim=1) for k in batch if k in self._queues}
        actions = self.generate_actions(batch)

        return actions

    def generate_actions(
        self,
        batch: dict[str, Tensor],
        generator: torch.Generator | Sequence[torch.Generator] | None = None,
    ) -> Tensor:
        """
        This function expects `batch` to have:
        {
            "observation.state": (B, n_obs_steps, state_dim)

            "observation.images": (B, n_obs_steps, num_cameras, C, H, W)
                AND/OR
            "observation.environment_state": (B, environment_dim)
        }
        """
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
        assert n_obs_steps == self.config.n_obs_steps

        # Run sampling.
        if not self.training and self.uncertainty_sampler is not None:
            if batch_size != 1:
                raise ValueError(
                    f"Sampling with uncertainty currently only supports batch size of 1, but got {batch_size}."
                )

            # Sample action sequence candidates and compute their uncertainty.
            actions, uncertainty = self.uncertainty_sampler.conditional_sample_with_uncertainty(
                observation=batch, generator=generator
            )
            tqdm.write(f"{self.uncertainty_sampler.method_name} uncertainty: {uncertainty:.4f}")
        elif not self.training and self.fiper_rollout_recorder is not None:
            if batch_size != 1:
                raise ValueError(
                    f"Recording FIPER data requires batch size of 1, but got {batch_size}."
                )

            # Sample actions and record sampling data.
            actions = self.fiper_rollout_recorder.conditional_sample_with_recording(
                observation=batch, generator=generator
            )
        else:
            # Encode image features and concatenate them all together along with the state vector.
            global_cond = self.flow_matching.prepare_global_conditioning(batch)  # (B, global_cond_dim)

            actions = self.flow_matching.conditional_sample(batch_size, global_cond=global_cond, generator=generator)

        # Extract `n_action_steps` steps worth of actions (from the current observation).
        start = n_obs_steps - 1
        end = start + self.config.n_action_steps
        actions = actions[:, start:end]

        return actions

    @torch.no_grad
    def select_action(
        self,
        batch: dict[str, Tensor],
        generator: torch.Generator | Sequence[torch.Generator] | None = None,
    ) -> Tensor:
        """Select a single action given environment observations.

        This method handles caching a history of observations and an action trajectory generated by the
        underlying flow matching model. Here's how it works:
          - `n_obs_steps` steps worth of observations are cached (for the first steps, the observation is
            copied `n_obs_steps` times to fill the cache).
          - The flow matching model generates `horizon` steps worth of actions.
          - `n_action_steps` worth of actions are actually kept for execution, starting from the current step.
        Schematically this looks like:
            ----------------------------------------------------------------------------------------------
            (legend: o = n_obs_steps, h = horizon, a = n_action_steps)
            |timestep            | n-o+1 | n-o+2 | ..... | n     | ..... | n+a-1 | n+a   | ..... | n-o+h |
            |observation is used | YES   | YES   | YES   | YES   | NO    | NO    | NO    | NO    | NO    |
            |action is generated | YES   | YES   | YES   | YES   | YES   | YES   | YES   | YES   | YES   |
            |action is used      | NO    | NO    | NO    | YES   | YES   | YES   | NO    | NO    | NO    |
            ----------------------------------------------------------------------------------------------
        Note that this means we require: `n_action_steps <= horizon - n_obs_steps + 1`. Also, note that
        "horizon" may not the best name to describe what the variable actually means, because this period is
        actually measured from the first observation which (if `n_obs_steps` > 1) happened in the past.
        """
        # NOTE: for offline evaluation, we have action in the batch, so we need to pop it out
        if ACTION in batch:
            batch.pop(ACTION)

        if self.config.image_features:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch[OBS_IMAGES] = torch.stack(
                [batch[key] for key in self.config.image_features], dim=-4
            )
        # NOTE: It's important that this happens after stacking the images into a single key.
        self._queues = populate_queues(self._queues, batch)

        if len(self._queues[ACTION]) == 0:
            # stack n latest observations from the queue
            batch = {k: torch.stack(list(self._queues[k]), dim=1) for k in batch if k in self._queues}
            actions = self.generate_actions(batch, generator)
            self._queues[ACTION].extend(actions.transpose(0, 1))

        action = self._queues[ACTION].popleft()
        return action

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, None]:
        """Run the batch through the model and compute the loss for training or validation."""
        if self.config.image_features:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original

            imgs = []
            for key in self.config.image_features:
                img = batch[key]

                if img.dim() == 4:
                    img = img.unsqueeze(1)
                elif img.dim() != 5:
                    raise ValueError(f"Unexpected image tensor shape {img.shape} for key {key}")

                imgs.append(img)

            batch[OBS_IMAGES] = torch.stack(imgs, dim=2)
        loss = self.flow_matching.compute_loss(batch)
        # no output_dict so returning None
        return loss, None


class FlowMatchingModel(nn.Module):
    def __init__(self, config: FlowMatchingConfig):
        super().__init__()
        self.config = config

        # Build observation encoders (depending on which observations are provided).
        global_cond_dim = self.config.robot_state_feature.shape[0]
        if self.config.image_features:
            num_images = len(self.config.image_features)
            if self.config.use_separate_rgb_encoder_per_camera:
                encoders = [FlowMatchingRgbEncoder(config) for _ in range(num_images)]
                self.rgb_encoder = nn.ModuleList(encoders)
                global_cond_dim += encoders[0].feature_dim * num_images
            else:
                self.rgb_encoder = FlowMatchingRgbEncoder(config)
                global_cond_dim += self.rgb_encoder.feature_dim * num_images
        if self.config.env_state_feature:
            global_cond_dim += self.config.env_state_feature.shape[0]

        self.unet = FlowMatchingConditionalUnet1d(config, global_cond_dim=global_cond_dim * config.n_obs_steps)
        self.cond_prob_path = make_cond_prob_path(
            cond_vf_type=self.config.cond_vf_type,
            sigma_min=self.config.sigma_min,
            beta_min=self.config.beta_min,
            beta_max=self.config.beta_max,
        )
        self.ode_solver = ODESolver()

    def forward(
        self, interpolated_trajectory: Tensor, timestep: Tensor, observation: dict[str, Tensor]
    ) -> Tensor:
        """
        Args:
            interpolated_trajectory: Sample from OT conditional probability path based on some
                initial noise sample and target actions. Shape: (batch_size, horizon, action_dim).
            timestep: Flow matching ODE timesteps. Shape: (batch_size,).
            observation: State of the agent stored under "observation.state" as well as
                RGB images of the environment stored under "observation.images" AND/OR state of the
                environment stored under "observation.environment_state".
        Returns:
            Predicted velocity of the flow matching model. Shape: (batch_size, horizong, action_dim).
        """
        # Input validation.
        assert "observation.state" in observation
        assert "observation.images" in observation or "observation.environment_state" in observation

        # Encode image features and concatenate them all together along with the state vector.
        global_cond = self.prepare_global_conditioning(observation)  # (B, global_cond_dim)

        # Run the model that predicts the velocity
        predicted_velocity = self.unet(interpolated_trajectory, timestep, global_cond)

        return predicted_velocity

    def make_velocity_fn(self, global_cond: Tensor) -> Callable[[Tensor, Tensor], Tensor]:
        def v_t(t: Tensor, x_t: Tensor) -> Tensor:
            return self.unet(x_t, t.expand(x_t.shape[0]), global_cond=global_cond)
        return v_t

    # ========= inference  ============
    def conditional_sample(
        self,
        batch_size: int,
        global_cond: Tensor,
        generator: torch.Generator | Sequence[torch.Generator] | None = None,
    ) -> Tensor:
        device = get_device_from_parameters(self)
        dtype = get_dtype_from_parameters(self)

        # Sample noise prior.
        if isinstance(generator, Sequence):
            if len(generator) != batch_size:
                raise ValueError(
                    f"Expected {batch_size} generators but got {len(generator)}."
                )
            noise_sample = torch.stack([
                torch.randn(
                    self.config.horizon,
                    self.config.action_feature.shape[0],
                    dtype=dtype,
                    device=device,
                    generator=g,
                )
                for g in generator
            ])
        else:
            noise_sample = torch.randn(
                batch_size,
                self.config.horizon,
                self.config.action_feature.shape[0],
                dtype=dtype,
                device=device,
                generator=generator,
            )

        # Use the velocity field model and an ODE solver to predict a sample from the target distribution.
        sample = self.ode_solver.sample(
            x_0=noise_sample,
            velocity_fn=self.make_velocity_fn(global_cond=global_cond),
            step_size=self.config.ode_step_size,
            method=self.config.ode_solver_method,
            atol=self.config.atol,
            rtol=self.config.rtol,
        )

        return sample

    def prepare_global_conditioning(self, batch: dict[str, Tensor]) -> Tensor:
        """Encode image features and concatenate them all together along with the state vector."""
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
        global_cond_feats = [batch[OBS_STATE]]
        # Extract image features.
        if self.config.image_features:
            if self.config.use_separate_rgb_encoder_per_camera:
                # Combine batch and sequence dims while rearranging to make the camera index dimension first.
                images_per_camera = einops.rearrange(batch["observation.images"], "b s n ... -> n (b s) ...")
                img_features_list = torch.cat(
                    [
                        encoder(images)
                        for encoder, images in zip(self.rgb_encoder, images_per_camera, strict=True)
                    ]
                )
                # Separate batch and sequence dims back out. The camera index dim gets absorbed into the
                # feature dim (effectively concatenating the camera features).
                img_features = einops.rearrange(
                    img_features_list, "(n b s) ... -> b s (n ...)", b=batch_size, s=n_obs_steps
                )
            else:
                # Combine batch, sequence, and "which camera" dims before passing to shared encoder.
                img_features = self.rgb_encoder(
                    einops.rearrange(batch["observation.images"], "b s n ... -> (b s n) ...")
                )
                # Separate batch dim and sequence dim back out. The camera index dim gets absorbed into the
                # feature dim (effectively concatenating the camera features).
                img_features = einops.rearrange(
                    img_features, "(b s n) ... -> b s (n ...)", b=batch_size, s=n_obs_steps
                )
            global_cond_feats.append(img_features)

        if self.config.env_state_feature:
            global_cond_feats.append(batch[OBS_ENV_STATE])

        # Concatenate features then flatten to (B, global_cond_dim).
        return torch.cat(global_cond_feats, dim=-1).flatten(start_dim=1)

    def compute_loss(self, batch: dict[str, Tensor]) -> Tensor:
        """
        This function expects `batch` to have (at least):
        {
            "observation.state": (B, n_obs_steps, state_dim)

            "observation.images": (B, n_obs_steps, num_cameras, C, H, W)
                AND/OR
            "observation.environment_state": (B, environment_dim)

            "action": (B, horizon, action_dim)
            "action_is_pad": (B, horizon)
        }
        """
        # Input validation.
        assert set(batch).issuperset({"observation.state", "action", "action_is_pad"})
        assert "observation.images" in batch or "observation.environment_state" in batch
        n_obs_steps = batch["observation.state"].shape[1]
        horizon = batch["action"].shape[1]
        assert horizon == self.config.horizon
        assert n_obs_steps == self.config.n_obs_steps

        device = get_device_from_parameters(self)

        # Encode image features and concatenate them all together along with the state vector.
        global_cond = self.prepare_global_conditioning(batch)  # (B, global_cond_dim)

        # Get ground-truth trajectory (x_1)
        trajectory = batch["action"]

        # Sample a random time for each item in the batch.
        times = torch.rand(
            size=(trajectory.shape[0],),
            device=device,
        ) * 0.999

        # Sample from conditional probability path and vector field.
        interpolated_trajectory = self.cond_prob_path.sample(
            x_1=trajectory,
            t=times[:, None, None],
        )
        target_velocity = self.cond_prob_path.velocity(
            x_t=interpolated_trajectory,
            x_1=trajectory,
            t=times[:, None, None],
        )

        # Run the model that predicts the velocity
        predicted_velocity = self.unet(interpolated_trajectory, times, global_cond=global_cond)

        # Compute the loss.
        loss = F.mse_loss(predicted_velocity, target_velocity, reduction="none")

        # Mask loss wherever the action is padded with copies (edges of the dataset trajectory).
        if self.config.do_mask_loss_for_padding:
            if "action_is_pad" not in batch:
                raise ValueError(
                    "You need to provide 'action_is_pad' in the batch when "
                    f"{self.config.do_mask_loss_for_padding}."
                )
            in_episode_bound = ~batch["action_is_pad"]
            loss = loss * in_episode_bound.unsqueeze(-1)

        return loss.mean()


class SpatialSoftmax(nn.Module):
    """
    Spatial Soft Argmax operation described in "Deep Spatial Autoencoders for Visuomotor Learning" by Finn et al.
    (https://arxiv.org/pdf/1509.06113). A minimal port of the robomimic implementation.

    At a high level, this takes 2D feature maps (from a convnet/ViT) and returns the "center of mass"
    of activations of each channel, i.e., keypoints in the image space for the policy to focus on.

    Example: take feature maps of size (512x10x12). We generate a grid of normalized coordinates (10x12x2):
    -----------------------------------------------------
    | (-1., -1.)   | (-0.82, -1.)   | ... | (1., -1.)   |
    | (-1., -0.78) | (-0.82, -0.78) | ... | (1., -0.78) |
    | ...          | ...            | ... | ...         |
    | (-1., 1.)    | (-0.82, 1.)    | ... | (1., 1.)    |
    -----------------------------------------------------
    This is achieved by applying channel-wise softmax over the activations (512x120) and computing the dot
    product with the coordinates (120x2) to get expected points of maximal activation (512x2).

    The example above results in 512 keypoints (corresponding to the 512 input channels). We can optionally
    provide num_kp != None to control the number of keypoints. This is achieved by a first applying a learnable
    linear mapping (in_channels, H, W) -> (num_kp, H, W).
    """

    def __init__(self, input_shape, num_kp=None):
        """
        Args:
            input_shape (list): (C, H, W) input feature map shape.
            num_kp (int): number of keypoints in output. If None, output will have the same number of channels as input.
        """
        super().__init__()

        assert len(input_shape) == 3
        self._in_c, self._in_h, self._in_w = input_shape

        if num_kp is not None:
            self.nets = torch.nn.Conv2d(self._in_c, num_kp, kernel_size=1)
            self._out_c = num_kp
        else:
            self.nets = None
            self._out_c = self._in_c

        # we could use torch.linspace directly but that seems to behave slightly differently than numpy
        # and causes a small degradation in pc_success of pre-trained models.
        pos_x, pos_y = np.meshgrid(np.linspace(-1.0, 1.0, self._in_w), np.linspace(-1.0, 1.0, self._in_h))
        pos_x = torch.from_numpy(pos_x.reshape(self._in_h * self._in_w, 1)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self._in_h * self._in_w, 1)).float()
        # register as buffer so it's moved to the correct device.
        self.register_buffer("pos_grid", torch.cat([pos_x, pos_y], dim=1))

    def forward(self, features: Tensor) -> Tensor:
        """
        Args:
            features: (B, C, H, W) input feature maps.
        Returns:
            (B, K, 2) image-space coordinates of keypoints.
        """
        if self.nets is not None:
            features = self.nets(features)

        # [B, K, H, W] -> [B * K, H * W] where K is number of keypoints
        features = features.reshape(-1, self._in_h * self._in_w)
        # 2d softmax normalization
        attention = F.softmax(features, dim=-1)
        # [B * K, H * W] x [H * W, 2] -> [B * K, 2] for spatial coordinate mean in x and y dimensions
        expected_xy = attention @ self.pos_grid
        # reshape to [B, K, 2]
        feature_keypoints = expected_xy.view(-1, self._out_c, 2)

        return feature_keypoints


class FlowMatchingRgbEncoder(nn.Module):
    """Encodes an RGB image into a 1D feature vector.

    Includes the ability to normalize and crop the image first.
    """
    def __init__(self, config: FlowMatchingConfig):
        super().__init__()
        # Set up optional preprocessing.
        if config.crop_shape is not None:
            self.do_crop = True
            # Always use center crop for eval
            self.center_crop = torchvision.transforms.CenterCrop(config.crop_shape)
            if config.crop_is_random:
                self.maybe_random_crop = torchvision.transforms.RandomCrop(config.crop_shape)
            else:
                self.maybe_random_crop = self.center_crop
        else:
            self.do_crop = False

        # Set up backbone.
        backbone_model = getattr(torchvision.models, config.vision_backbone)(
            weights=config.pretrained_backbone_weights
        )
        # Extract feature extractor.
        if config.vision_backbone not in FINAL_FEATURE_MAP_MODULE:
            raise ValueError(f"No feature-map module registered for backbone “{config.vision_backbone}”")
        final_feature_map_module = FINAL_FEATURE_MAP_MODULE[config.vision_backbone]
        self.backbone = nn.Sequential()
        final_feature_map_module_found = False
        for name, module in backbone_model.named_children():
            self.backbone.add_module(name, module)
            if name == final_feature_map_module:
                final_feature_map_module_found = True
                break
        if not final_feature_map_module_found:
            raise RuntimeError(f"Final feature-map module “{final_feature_map_module}” not found in {config.vision_backbone}")

        if config.use_group_norm:
            if config.pretrained_backbone_weights:
                raise ValueError(
                    "You can't replace BatchNorm in a pretrained model without ruining the weights!"
                )
            self.backbone = replace_submodules(
                root_module=self.backbone,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(num_groups=x.num_features // 16, num_channels=x.num_features),
            )

        # Set up pooling and final layers.
        # Use a dry run to get the feature map shape.
        # The dummy input should take the number of image channels from `config.image_features` and it should
        # use the height and width from `config.crop_shape` if it is provided, otherwise it should use the
        # height and width from `config.image_features`.

        # Note: we have a check in the config class to make sure all images have the same shape.
        images_shape = next(iter(config.image_features.values())).shape
        dummy_shape_h_w = config.crop_shape if config.crop_shape is not None else images_shape[1:]
        dummy_shape = (1, images_shape[0], *dummy_shape_h_w)
        feature_map_shape = get_output_shape(self.backbone, dummy_shape)[1:]

        self.pool = SpatialSoftmax(feature_map_shape, num_kp=config.spatial_softmax_num_keypoints)
        self.feature_dim = config.spatial_softmax_num_keypoints * 2
        self.out = nn.Linear(config.spatial_softmax_num_keypoints * 2, self.feature_dim)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, C, H, W) image tensor with pixel values in [0, 1].
        Returns:
            (B, D) image feature.
        """
        # Preprocess: maybe crop (if it was set up in the __init__).
        if self.do_crop:
            if self.training:
                x = self.maybe_random_crop(x)
            else:
                # Always use center crop for eval.
                x = self.center_crop(x)
        # Extract backbone feature.
        x = torch.flatten(self.pool(self.backbone(x)), start_dim=1)
        # Final linear layer with non-linearity.
        x = self.relu(self.out(x))
        return x


class FlowMatchingSinusoidalPosEmb(nn.Module):
    """1D sinusoidal positional embeddings as in Attention is All You Need."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class FlowMatchingConv1dBlock(nn.Module):
    """Conv1d --> GroupNorm --> Mish"""

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class FlowMatchingConditionalUnet1d(nn.Module):
    """A 1D convolutional UNet with FiLM modulation for conditioning."""

    def __init__(self, config: FlowMatchingConfig, global_cond_dim: int):
        super().__init__()

        self.config = config

        # Encoder for the flow matching timestep.
        self.flow_matching_step_encoder = nn.Sequential(
            FlowMatchingSinusoidalPosEmb(config.flow_matching_step_embed_dim),
            nn.Linear(config.flow_matching_step_embed_dim, config.flow_matching_step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(config.flow_matching_step_embed_dim * 4, config.flow_matching_step_embed_dim),
        )

        # The FiLM conditioning dimension.
        cond_dim = config.flow_matching_step_embed_dim + global_cond_dim

        # In channels / out channels for each downsampling block in the Unet's encoder. For the decoder, we
        # just reverse these.
        in_out = [(config.action_feature.shape[0], config.down_dims[0])] + list(
            zip(config.down_dims[:-1], config.down_dims[1:], strict=True)
        )

        # Unet encoder.
        common_res_block_kwargs = {
            "cond_dim": cond_dim,
            "kernel_size": config.kernel_size,
            "n_groups": config.n_groups,
            "use_film_scale_modulation": config.use_film_scale_modulation,
        }
        self.down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.down_modules.append(
                nn.ModuleList(
                    [
                        FlowMatchingConditionalResidualBlock1d(dim_in, dim_out, **common_res_block_kwargs),
                        FlowMatchingConditionalResidualBlock1d(dim_out, dim_out, **common_res_block_kwargs),
                        # Downsample as long as it is not the last block.
                        nn.Conv1d(dim_out, dim_out, 3, 2, 1) if not is_last else nn.Identity(),
                    ]
                )
            )

        # Processing in the middle of the auto-encoder.
        self.mid_modules = nn.ModuleList(
            [
                FlowMatchingConditionalResidualBlock1d(
                    config.down_dims[-1], config.down_dims[-1], **common_res_block_kwargs
                ),
                FlowMatchingConditionalResidualBlock1d(
                    config.down_dims[-1], config.down_dims[-1], **common_res_block_kwargs
                ),
            ]
        )

        # Unet decoder.
        self.up_modules = nn.ModuleList([])
        for ind, (dim_out, dim_in) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            self.up_modules.append(
                nn.ModuleList(
                    [
                        # dim_in * 2, because it takes the encoder's skip connection as well
                        FlowMatchingConditionalResidualBlock1d(dim_in * 2, dim_out, **common_res_block_kwargs),
                        FlowMatchingConditionalResidualBlock1d(dim_out, dim_out, **common_res_block_kwargs),
                        # Upsample as long as it is not the last block.
                        nn.ConvTranspose1d(dim_out, dim_out, 4, 2, 1) if not is_last else nn.Identity(),
                    ]
                )
            )

        self.final_conv = nn.Sequential(
            FlowMatchingConv1dBlock(config.down_dims[0], config.down_dims[0], kernel_size=config.kernel_size),
            nn.Conv1d(config.down_dims[0], config.action_feature.shape[0], 1),
        )

    def forward(self, x: Tensor, timestep: Tensor | int, global_cond=None) -> Tensor:
        """
        Args:
            x: (B, T, input_dim) tensor for input to the Unet.
            timestep: (B,) tensor of flow matching ODE timesteps.
            global_cond: (B, global_cond_dim)
            output: (B, T, input_dim)
        Returns:
            (B, T, input_dim) flow matching model prediction.
        """
        # For 1D convolutions we'll need feature dimension first.
        x = einops.rearrange(x, "b t d -> b d t")

        timesteps_embed = self.flow_matching_step_encoder(timestep)

        # If there is a global conditioning feature, concatenate it to the timestep embedding.
        if global_cond is not None:
            global_feature = torch.cat([timesteps_embed, global_cond], axis=-1)
        else:
            global_feature = timesteps_embed

        # Run encoder, keeping track of skip features to pass to the decoder.
        encoder_skip_features: list[Tensor] = []
        for resnet, resnet2, downsample in self.down_modules:
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            encoder_skip_features.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        # Run decoder, using the skip features from the encoder.
        for resnet, resnet2, upsample in self.up_modules:
            x = torch.cat((x, encoder_skip_features.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)

        x = einops.rearrange(x, "b d t -> b t d")
        return x


class FlowMatchingConditionalResidualBlock1d(nn.Module):
    """ResNet style 1D convolutional block with FiLM modulation for conditioning."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        kernel_size: int = 3,
        n_groups: int = 8,
        # Set to True to do scale modulation with FiLM as well as bias modulation (defaults to False meaning
        # FiLM just modulates bias).
        use_film_scale_modulation: bool = False,
    ):
        super().__init__()

        self.use_film_scale_modulation = use_film_scale_modulation
        self.out_channels = out_channels

        self.conv1 = FlowMatchingConv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups)

        # FiLM modulation (https://arxiv.org/abs/1709.07871) outputs per-channel bias and (maybe) scale.
        cond_channels = out_channels * 2 if use_film_scale_modulation else out_channels
        self.cond_encoder = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, cond_channels))

        self.conv2 = FlowMatchingConv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups)

        # A final convolution for dimension matching the residual (if needed).
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        """
        Args:
            x: (B, in_channels, T)
            cond: (B, cond_dim)
        Returns:
            (B, out_channels, T)
        """
        out = self.conv1(x)

        # Get condition embedding. Unsqueeze for broadcasting to `out`, resulting in (B, out_channels, 1).
        cond_embed = self.cond_encoder(cond).unsqueeze(-1)
        if self.use_film_scale_modulation:
            # Treat the embedding as a list of scales and biases.
            scale = cond_embed[:, : self.out_channels]
            bias = cond_embed[:, self.out_channels :]
            out = scale * out + bias
        else:
            # Treat the embedding as biases.
            out = out + cond_embed

        out = self.conv2(out)
        out = out + self.residual_conv(x)
        return out
