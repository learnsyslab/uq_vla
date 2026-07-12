from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data.dataloader import default_collate

from lerobot.policies.flow_matching.configuration_flow_matching import FlowMatchingConfig
from lerobot.policies.flow_matching.modelling_flow_matching import FlowMatchingModel
from lerobot.processor import PolicyProcessorPipeline
from lerobot.utils.constants import ACTION, OBS_IMAGES
from lerobot.utils.utils import get_safe_torch_device

from .laplace_wrapper import LaplaceBatch, LaplaceWrapper


class FlowMatchingLaplaceWrapper(LaplaceWrapper):
    """Laplace wrapper for FlowMatching models."""
    def __init__(self, config: FlowMatchingConfig, model: FlowMatchingModel, scopes: list[str]):
        """
        Initialize a Laplace approximation wrapper around a FlowMatching model.

        Args:
            config: FlowMatching policy configuration.
            model: The underlying FlowMatchingModel to calibrate.
            scopes: Which submodules to expose to the Laplace posterior.
                Available scopes:
                  - "velocity_last": The final layer of the velocity head.
                  - "rgb_last": The final projection layer of the RGB encoder.
                Defaults to ["velocity_last", "rgb_last"].
        """
        scopes = scopes or ["velocity_last", "rgb_last"]
        super().__init__(model=model, config=config, scopes=scopes)

        self.scope_abbr: dict[str, str] = {
            "velocity_last": "vel",
            "rgb_last": "rgb",
        }

        self.cond_prob_path = self.model.cond_prob_path

    def build_collate_fn(
        self, preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]]
    ) -> Callable[[dict[str, Tensor]], tuple[LaplaceBatch, Tensor]]:
        """
        Factory that builds a dataloader collate function tailored for Laplace calibration of
        a FlowMatchingModel.

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

            # Get ground-truth trajectory (x_1)
            trajectory = batch[ACTION]
            batch_size = trajectory.shape[0]

            # Sample a random time for each item in the batch
            times = torch.rand(
                size=(batch_size,),
                device=device,
            ) * 0.999

            # Sample from conditional probability path and vector field.
            interpolated_trajectory = self.cond_prob_path.sample(
                x_1=trajectory,
                t=times[:, None, None],
            )
            target_vel = self.cond_prob_path.velocity(
                x_t=interpolated_trajectory,
                x_1=trajectory,
                t=times[:, None, None],
            )

            # Create observation only dictionary
            observation = {
                key: value.cpu()
                for key, value in batch.items()
                if key.startswith("observation.")
            }

            # Apply mask to target velocity so padded steps outside of the episode contribute zero loss
            if self.config.do_mask_loss_for_padding:
                if "action_is_pad" not in batch:
                    raise ValueError(
                        "You need to provide 'action_is_pad' in the batch when "
                        f"{self.config.do_mask_loss_for_padding}."
                    )
                in_episode_mask = ~batch["action_is_pad"]
            else:
                in_episode_mask = torch.ones(trajectory.shape[:2], dtype=torch.bool, device=device)
            target_vel = target_vel * in_episode_mask.unsqueeze(-1)

            # Package inputs and targets exactly as laplace-torch expects
            input_batch = LaplaceBatch(
                interp_traj=interpolated_trajectory,
                time=times,
                observation=observation,
                in_episode_mask=in_episode_mask
            )
            target_batch = target_vel.flatten(start_dim=1)

            return input_batch, target_batch

        return collate_fn

    def forward(self, batch: LaplaceBatch) -> Tensor:
        """
        Compute the vector-field prediction v_t for an input (x_t, t) of noisy actions and time
        conditioned on some observation.
        """
        # Unpack FlowMatchingInput attributes and forward them to the velocity model
        pred_vel = self.model(batch.interp_traj, batch.time, batch.observation)

        # Ensure padded steps at the end of an episode contribute zero loss
        pred_vel = pred_vel * batch.in_episode_mask.unsqueeze(-1)

        return pred_vel.flatten(start_dim=1)

    def apply_laplace_scope(self):
        """
        Freeze the entire model and unfreeze only the submodules selected by the specified
        scopes to be fitted by the Laplace approximation.
        """
        # Select target modules for Laplace approximation
        target_modules: list[nn.Module] = []
        for scope in self.scopes:
            if scope == "velocity_last":
                if isinstance(self.model.unet.final_conv[1], nn.Conv1d):
                    self.model.unet.final_conv[1] = PointwiseConv1dToLinear(
                        self.model.unet.final_conv[1]
                    )
                elif not isinstance(self.model.unet.final_conv[1], PointwiseConv1dToLinear):
                    raise ValueError(
                        "Expected final layer of velocity model to be nn.Conv1d, "
                        f"got {type(self.model.unet.final_conv[1])}"
                    )
                target_modules.append(self.model.unet.final_conv[1].linear_layer)
            elif scope == "rgb_last":
                rgb_encoder = self.model.rgb_encoder

                if isinstance(rgb_encoder, nn.ModuleList):
                    for enc in rgb_encoder:
                        target_modules.append(enc.out)
                else:
                    target_modules.append(rgb_encoder.out)
            else:
                raise ValueError(
                    f"Unknown Laplace approximation target {scope}. "
                    "Choose from ['velocity_last', 'rgb_last']"
                )

        # Freeze all parameters
        for p in self.model.parameters():
            p.requires_grad_(False)

        # Unfreeze only the selected subnetwork for Laplace fitting
        for module in target_modules:
            for p in module.parameters():
                p.requires_grad_(True)


class PointwiseConv1dToLinear(nn.Module):
    """
    Adapter that replaces a pretrained 1x1 Conv1D layer with an equivalent nn.Linear,
    so that Laplace-torch can recognize it as a linear leaf module. The functional
    output is identical to the original convolution.
    """
    def __init__(self, conv_layer: nn.Conv1d):
        super().__init__()
        # Number of output channels for reshaping
        self.out_channels = conv_layer.out_channels
        # Create a Linear layer matching the conv_layer parameters
        self.linear_layer = nn.Linear(
            in_features=conv_layer.in_channels,
            out_features=conv_layer.out_channels,
            bias=(conv_layer.bias is not None),
            device=conv_layer.weight.device,
            dtype=conv_layer.weight.dtype,
        )
        # Copy pretrained conv weights and bias
        with torch.no_grad():
            # conv_layer.weight shape: (C_out, C_in, 1)
            self.linear_layer.weight.copy_(conv_layer.weight.squeeze(-1))
            if conv_layer.bias is not None:
                self.linear_layer.bias.copy_(conv_layer.bias)

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Forward pass that applies the equivalent linear transformation at each time step.

        Args:
            inputs: Inputs to the final 1x1 Conv1D layer. Shape (batch_size, C_in, horizon).

        Returns:
           Outputs of the final 1x1 Conv1D layer. Shape (batch_size, action_dim, horizon).
        """
        # Inputs Shape: (batch_size, C_in, horizon)
        batch_size, _, horizon = inputs.shape
        # Flatten horizon dimension into batch dimension: (bach_size*horizon, C_in)
        flat_inputs = inputs.permute(0, 2, 1).reshape(batch_size * horizon, -1)
        # Apply linear layer: (bach_size*horizon, action_dim)
        flat_outputs = self.linear_layer(flat_inputs)
        # Restore shape: (bach_size*horizon, action_dim, horizon)
        outputs = flat_outputs.view(batch_size, horizon, self.out_channels).permute(0, 2, 1)
        return outputs
