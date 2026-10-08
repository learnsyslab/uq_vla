from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torchvision
import numpy as np
import os
from shared_utils.data_compression import perform_pca_tensor
from shared_utils.utility_functions import compare_dicts

from rnd.rnd_to_subblocks import RND_TO_base
from rnd.rnd_to_subblocks import Predictor, Target


class RNDBase(nn.Module):
    def __init__(self, input_dict: Dict[str, any] = None, checkpoint_dir=None, desired_hparams: dict = {}, **kwargs):
        assert checkpoint_dir is not None or input_dict is not None, (
            "Either input_size or checkpoint_dir must be provided."
        )
        self.input_dict = input_dict
        if checkpoint_dir is not None:
            cfg, checkpoint = self._load_checkpoint(checkpoint_dir, desired_hparams=desired_hparams)
            input_dict = cfg.get("input_dict", None)
            assert input_dict is not None, "input_dict must be provided in the checkpoint."
            kwargs = cfg.get("kwargs", {})
        else:
            input_dict = self._compute_input_sizes(input_dict)
        super().__init__()

        input_dict = self._check_action_batch_handling(input_dict)
        # Set class attributes from input_dict
        for key, value in input_dict.items():
            setattr(self, key, value)
        self.kwargs = kwargs
        self.input_dict = input_dict

        # Init network layers

        self.target_network = self.get_target_network()
        self.predictor_network = self.get_predictor_network()
        self._get_additional_layers(input_dict)
        # Init rnd loss function
        if self.hyperparameters["rnd_loss"] == "mse":
            self.rnd_mse_loss = nn.MSELoss(reduction="none")
            self.rnd_loss_function = lambda target, prediction: self.rnd_mse_loss(target, prediction).mean(dim=-1)
        elif self.hyperparameters["rnd_loss"] == "l2":
            self.rnd_loss_function = nn.PairwiseDistance(p=2)
        # Init weights and set requires_grad to False for target network
        self.post_process(init=checkpoint_dir is None)

        # Load the model weights if checkpoint_dir is not None
        if checkpoint_dir is not None:
            # Load the model weights
            self.load_state_dict(checkpoint["state_dict"])

        self.input_dict = input_dict
        self._get_transform_loss_function()

    def _compute_input_sizes(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Update the input dict with model specific computations if needed"""
        return input_dict

    def _get_additional_layers(self, input_dict):
        pass

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        target = self.target_network(input)
        pred = self.predictor_network(input)
        loss = self.rnd_loss_function(pred, target)
        loss = self._transform_loss_function(loss)
        return loss

    def _get_transform_loss_function(self):
        if self.use_action_batch:
            if self.action_batch_style == "var":
                # Transform the loss to match the action batch size
                self._transform_loss_function = self._transform_loss_var
            else:
                # No transformation needed
                self._transform_loss_function = self._transform_loss_svd
        else:
            # No transformation needed
            self._transform_loss_function = self._pass_loss

    def _transform_loss_var(self, loss: torch.Tensor) -> torch.Tensor:
        loss = loss.view(-1, self.var_batch_size)
        # loss = loss_executed_action + mean loss of the action batch + std loss of the action batch (remove mean/var by setting factors to 0)
        loss = loss[:, 0] + loss.mean(dim=-1) * self.mean_factor + loss.std(dim=-1) * self.var_factor
        return loss

    def _transform_loss_svd(self, loss: torch.Tensor) -> torch.Tensor:
        return loss

    @staticmethod
    def _pass_loss(loss: torch.Tensor) -> torch.Tensor:
        return loss

    def _check_action_batch_handling(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check the action batch handling configuration and override the input_dict if necessary.
        Args:
            Input_dict (Dict[str, Any]): The input dictionary containing the configuration.
        Returns:
            Input_dict (Dict[str, Any]): The updated input dictionary after checking the action batch handling configuration.
        """
        self.model_uses_action_preds = True if self.__class__.__name__ in ["RND_TO", "RND_T", "RND_TO_L"] else False
        self.action_pred_shape = input_dict["action_pred_shape"]
        handling_dict = input_dict.get("action_batch_handling", None)
        if handling_dict is None or not self.model_uses_action_preds:
            self.use_action_batch = False
            return input_dict
        self.use_action_batch = handling_dict.get("use_action_batch", False) and self.model_uses_action_preds
        if self.use_action_batch:
            self.action_batch_style = handling_dict.get("action_batch_style", "svd")  # "svd" or "var"
            self.action_batch_size = self.action_pred_shape[0]

            if self.action_batch_style == "svd":
                svd_dim = self.action_pred_shape[-2] * self.action_pred_shape[-1]

                self.svd_components = min(handling_dict.get("svd_components", 3), svd_dim)
                self.svd_project = handling_dict.get("svd_project", False)
                self.svd_num_sigmas = min(handling_dict.get("svd_num_sigmas", 10), svd_dim)
                # Determine input size based on the action batch style
            elif self.action_batch_style == "var":
                self.var_batch_size = min(handling_dict.get("var_batch_size", 32), self.action_batch_size)
                self.var_factor = handling_dict.get("var_factor", 1)
                self.mean_factor = handling_dict.get("mean_factor", 1)
            elif self.action_batch_style == "pca":
                self.pca_components = min(handling_dict.get("pca_components", 5), self.action_batch_size)
                self.pca_part_trajectory = handling_dict.get("pca_part_trajectory", False)

                self.pca_pred_horizon = self.action_pred_shape[-2] if not self.pca_part_trajectory else 3
                self.pca_components = min(self.pca_components, self.action_pred_shape[-1] * self.pca_pred_horizon)
                # Determine input size based on the action batch style
            input_dict = self._update_input_sizes_from_action_batch_handling(input_dict)
        return input_dict

    def _update_input_sizes_from_action_batch_handling(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Only implemented in the action prediction using models"""
        return input_dict

    def _perform_pca(self, action_preds: torch.Tensor) -> torch.Tensor:
        """
        Perform PCA on a batch of action predictions and return the transformed tensor.
        Args:
            action_preds (torch.Tensor): Shape (batch_size, action_batch_size, prediction_horizon, action_dim).
        Returns:
            tensor (torch.Tensor): The transformed tensor after PCA of shape (pca_components, prediction_horizon * action_dim).
        """

        assert action_preds.ndim == 4, "action_preds must be a 4D tensor."
        batch_size = action_preds.shape[0]

        if self.pca_part_trajectory:
            pred_horizon = action_preds.shape[-2]
            indices = torch.tensor([0, pred_horizon // 2, pred_horizon - 1], device=action_preds.device)
            action_preds = torch.index_select(action_preds, dim=-2, index=indices)

        action_preds = action_preds.flatten(start_dim=-2, end_dim=-1)

        tensor = torch.empty(
            (batch_size, self.pca_components, action_preds.shape[-1]),
            device=action_preds.device,
        )

        for i in range(batch_size):
            tensor[i] = perform_pca_tensor(action_preds[i].T, self.pca_components).T

        return tensor

    def _perform_var(self, action_preds: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Perform variance calculation on a batch of action predictions and return the transformed tensor.
        Args:
            action_preds (torch.Tensor): Shape (batch_size, action_batch_size, prediction_horizon, action_dim).
        Returns:
            tensor (torch.Tensor): The transformed tensor after variance calculation of shape (var_batch_size, prediction_horizon * action_dim).
        """

        assert action_preds.ndim == 4, "action_preds must be a 4D tensor."

        action_preds = action_preds[:, : self.var_batch_size, ...].flatten(start_dim=0, end_dim=1)

        results = (action_preds,)
        for key in kwargs:
            if kwargs[key] is not None:
                results += (kwargs[key].repeat(self.var_batch_size, 1),)
        if len(results) == 1:
            results = results[0]
        return results

    def _perform_svd(self, action_preds: torch.Tensor) -> torch.Tensor:
        """
        Perform SVD on a batch of action predictions and return the transformed tensor.
        Args:
            action_preds (torch.Tensor): Shape (batch_size, action_batch_size, prediction_horizon, action_dim).
        Returns:
            tensor (torch.Tensor): The transformed tensor after SVD of shape (svd_components, prediction_horizon * action_dim).
        """

        assert action_preds.ndim == 4, "action_preds must be a 4D tensor."
        with_projected_executed_action = self.svd_project and action_preds.shape[1] > 1
        batch_size = action_preds.shape[0]

        action_preds = action_preds.flatten(start_dim=-2, end_dim=-1)

        action_batch_size = self.svd_components + 1 if with_projected_executed_action else self.svd_components

        tensor = torch.empty(
            (batch_size, action_batch_size, action_preds.shape[-1]),
            device=action_preds.device,
        )

        for i in range(batch_size):
            _, _, vh = torch.linalg.svd(action_preds[i], full_matrices=False)
            top_modes = vh[: self.svd_components, :]

            if with_projected_executed_action:
                projection_coefficients = action_preds[i, :1, ...].flatten(start_dim=-2) @ vh.T
                executed_action = projection_coefficients @ vh
                top_modes = torch.cat((executed_action, top_modes), dim=0)

            tensor[i] = top_modes

        return tensor

    def get_target_network(self):
        raise NotImplementedError("get_target_network() must be implemented in subclasses")

    def get_predictor_network(self):
        raise NotImplementedError("get_predictor_network() must be implemented in subclasses")

    def _load_checkpoint(
        self,
        checkpoint_dir: str,
        checkpoint_name: Optional[str] = None,
        desired_hparams: dict = {},
        **kwargs,
    ) -> Tuple:
        """
        Load the model checkpoint from the specified directory.
        Args:
            checkpoint_dir (str): Directory containing the checkpoint file.
            checkpoint_name (Optional[str]): Name of the checkpoint file.
            desired_hparams (Optional[Dict[str, Any]]): Hyperparameters for the model. Used to identify whether the model exists.
            kwargs: Additional arguments to identify the correct checkpoint.
        Returns:
            Dict[str, Any]: Dictionary containing the loaded checkpoint data.
        """
        if not os.path.exists(checkpoint_dir):
            raise FileNotFoundError(f"Checkpoint dir {checkpoint_dir} not found.")

        filenames = sorted([filename for filename in os.listdir(checkpoint_dir) if filename.endswith(".ckpt")])
        if checkpoint_name is not None:
            filenames = [filename for filename in filenames if checkpoint_name in filename]
        if not filenames:
            raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}.")
        found = False
        for filename in filenames:
            checkpoint = torch.load(os.path.join(checkpoint_dir, filename), weights_only=False)
            cfg = checkpoint["cfg"]
            if "hparams" not in cfg.keys() or "input_dict" not in cfg.keys():
                continue

            if not desired_hparams:
                Warning("No desired_hparams provided. Using the first checkpoint found in the directory.")
                break
            for key, value in desired_hparams.items():
                if key in cfg["hparams"].keys():
                    found = compare_dicts(value, cfg["hparams"][key])
                elif key in cfg["input_dict"].keys():
                    found = compare_dicts(value, cfg["input_dict"][key])

                if not found:
                    break

            if found:
                break
        if not found:
            raise FileNotFoundError(
                f"No checkpoint files found in {checkpoint_dir} with the desired hyperparameters: {desired_hparams}."
            )

        return cfg, checkpoint

    def _save_checkpoint(
        self,
        checkpoint_dir: str,
        model_cfg,
        checkpoint_name: Optional[str] = None,
        overwrite=False,
        state_dict=None,
        **kwargs,
    ) -> None:
        """
        Save the model configs and weights to a checkpoint file.
        Args:
            checkpoint_dir (str): Path to the checkpoint directory.
            checkpoint_name (str): Name of the checkpoint file.
        """
        os.makedirs(checkpoint_dir, exist_ok=True)

        if checkpoint_name is None:
            checkpoint_name = f"checkpoint_{self.__class__.__name__}.ckpt"
        else:
            if not checkpoint_name.endswith(".ckpt"):
                checkpoint_name += ".ckpt"

        if os.path.exists(os.path.join(checkpoint_dir, checkpoint_name)) and not overwrite:
            Warning(
                f"Checkpoint {checkpoint_name} already exists in {checkpoint_dir}. Use overwrite=True to overwrite it."
            )
            return

        # Save the model configs and weights
        checkpoint = {
            "state_dict": self.state_dict() if state_dict is None else state_dict,
            "cfg": model_cfg,
            "kwargs": kwargs,
        }
        torch.save(
            checkpoint,
            os.path.join(checkpoint_dir, checkpoint_name),
        )

    def _check_for_existance(self, checkpoint_dir: str, hparams: dict) -> bool:
        """
        Check if a checkpoint file with the same hyperparameters exists in the specified directory.
        Args:
            checkpoint_dir (str): Directory containing the checkpoint file.
            hparams (dict): Hyperparameters for the model. Used to identify the correct checkpoint.
        Returns:
            bool: True if the checkpoint file exists, False otherwise.
        """
        if not os.path.exists(checkpoint_dir):
            return False
        if not hasattr(self, "input_dict"):
            return False

        filenames = sorted([filename for filename in os.listdir(checkpoint_dir) if filename.endswith(".ckpt")])

        for filename in filenames:
            checkpoint = torch.load(os.path.join(checkpoint_dir, filename), weights_only=False)
            if "cfg" not in checkpoint.keys():
                continue
            if "hparams" not in checkpoint["cfg"].keys():  # or "input_dict" not in checkpoint["cfg"].keys():
                continue
            if compare_dicts(
                checkpoint["cfg"]["hparams"], hparams
            ):  # compare_dicts(checkpoint["cfg"]["input_dict"], self.input_dict) and
                return True
        return False

    def datasets_to_model_inputs(self, datasets: Dict[str, torch.tensor]) -> Union[torch.Tensor, tuple]:
        """Converts datasets to models inputs. Must be implemented in subclasses.
        Args:
            datasets (Dict[str, torch.tensor]): A dictionary containing the datasets.
        Returns:
            Union[torch.Tensor, tuple]: The model input tensor or a tuple of tensors.
        """
        raise NotImplementedError("datasets_to_model_inputs() must be implemented in subclasses")

    def post_process(self, init: bool = False):
        if init:
            for p in self.modules():
                if isinstance(p, nn.Conv1d):
                    nn.init.orthogonal_(p.weight, np.sqrt(2))
                    if p.bias is not None:
                        p.bias.data.zero_()

                if isinstance(p, nn.Linear):
                    nn.init.orthogonal_(p.weight, np.sqrt(2))
                    if p.bias is not None:
                        p.bias.data.zero_()

        for param in self.target_network.parameters():
            param.requires_grad = False


class RND_OE(RNDBase):
    def __init__(self, input_dict: Dict[str, Any] = None, checkpoint_dir=None, desired_hparams: dict = {}, **kwargs):
        """
        Args:
            input_dict:  Input dictionary containing the model configuration. Is unpacked in the base class constructor.
        """
        super().__init__(input_dict, checkpoint_dir, desired_hparams, **kwargs)

    def get_target_network(self):
        # Define the target network architecture
        net = nn.Sequential(
            nn.Linear(in_features=self.obs_embedding_dim, out_features=1024),
            nn.LeakyReLU(),
            nn.Linear(in_features=1024, out_features=2048),
            nn.LeakyReLU(),
            nn.Linear(in_features=2048, out_features=4096),
            nn.LeakyReLU(),
            nn.Linear(in_features=4096, out_features=self.output_size),
        )
        return net

    def get_predictor_network(self):
        # Define the prediction network architecture
        net = nn.Sequential(
            nn.Linear(in_features=self.obs_embedding_dim, out_features=1024),
            nn.LeakyReLU(),
            nn.Linear(in_features=1024, out_features=2048),
            nn.LeakyReLU(),
            nn.Linear(in_features=2048, out_features=4096),
            nn.LeakyReLU(),
            nn.Linear(in_features=4096, out_features=2048),
            nn.ReLU(),
            nn.Linear(in_features=2048, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=self.output_size),
        )
        return net

    def datasets_to_model_inputs(self, datasets):
        return {"input": datasets["obs_embeddings"]}


class RND_OI(RNDBase):
    def __init__(self, input_dict: Dict[str, Any] = None, checkpoint_dir=None, desired_hparams: dict = {}, **kwargs):
        super().__init__(input_dict, checkpoint_dir, desired_hparams, **kwargs)

    def _compute_input_sizes(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        input_dict["output_size"] = 256
        return input_dict

    def get_target_network(self):
        resnet_target_layers = self.get_resnet_layers("resnet18")
        net = nn.Sequential(
            *resnet_target_layers,  # Include all ResNet layers
            nn.Flatten(),  # -> (B, 512)
            nn.Linear(512, self.output_size),  # Additional fully connected layer (ResNet-18 outputs 1000 features)
        )
        return net

    def get_predictor_network(self):
        resnet_pred_layers = self.get_resnet_layers("resnet18")
        net = nn.Sequential(
            *resnet_pred_layers,  # Include all ResNet layers; Careful: One of the layers (flattening) is not included as it is only part of the forwart block in resnet implementation (see there for more details)
            nn.Flatten(),  # -> (B, 512)
            nn.Linear(512, 512),
            nn.ReLU(),  # also tried LeakyReLU()
            nn.Linear(512, self.output_size),
            nn.ReLU(),
            nn.Linear(self.output_size, self.output_size),  # Final layer
        )
        return net

    def get_resnet_layers(self, name: str, weights=None, **kwargs) -> nn.Module:
        # Use standard ResNet implementation from torchvision
        func = getattr(torchvision.models, name)
        resnet = func(weights=weights, **kwargs)

        # remove the final fully connected layer
        # # for resnet18, the output dim should be 512
        resnet.fc = torch.nn.Identity()
        # Replace the last fully connected layer (fc) with your own
        # resnet.fc = nn.Linear(resnet.fc.in_features, 512)  # First layer of your custom FC network
        layers = list(resnet.children())  # Remove the last layer
        return layers

    def datasets_to_model_inputs(self, datasets):
        return {"input": datasets["rgb_images"]}


class RND_TO_L(RNDBase):
    def __init__(self, input_dict: Dict[str, Any] = None, checkpoint_dir=None, desired_hparams: dict = {}, **kwargs):
        """
        Args:
            input_size: (Observation_embeddings + flattened trajectory + states )
        """
        super().__init__(input_dict, checkpoint_dir, desired_hparams, **kwargs)

    def _compute_input_sizes(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        input_dict["input_size"] = (
            input_dict["obs_embedding_dim"]
            + input_dict["action_pred_shape"][-2] * input_dict["action_pred_shape"][-1]
            + input_dict.get("state_dim", 0)
        )
        return input_dict

    def get_target_network(self):
        # Define the target network architecture
        net = nn.Sequential(
            nn.Linear(in_features=self.input_size, out_features=1024),
            nn.LeakyReLU(),
            nn.Linear(in_features=1024, out_features=2048),
            nn.LeakyReLU(),
            nn.Linear(in_features=2048, out_features=4096),
            nn.LeakyReLU(),
            nn.Linear(in_features=4096, out_features=self.output_size),
        )
        return net

    def get_predictor_network(self):
        # Define the prediction network architecture
        net = nn.Sequential(
            nn.Linear(in_features=self.input_size, out_features=1024),
            nn.LeakyReLU(),
            nn.Linear(in_features=1024, out_features=2048),
            nn.LeakyReLU(),
            nn.Linear(in_features=2048, out_features=4096),
            nn.LeakyReLU(),
            nn.Linear(in_features=4096, out_features=2048),
            nn.ReLU(),
            nn.Linear(in_features=2048, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=self.output_size),
        )
        return net

    def _update_input_sizes_from_action_batch_handling(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Only implemented in the action prediction using models"""
        if self.action_batch_style == "svd":
            # (batch_size, action_dim * prediction_horizon * svd_components)
            compressed_batch_size = self.svd_components + 1 if self.svd_project else self.svd_components
            input_dict["input_size"] = (
                input_dict["obs_embedding_dim"]
                + input_dict.get("state_dim", 0)
                + compressed_batch_size * self.action_pred_shape[-2] * self.action_pred_shape[-1]
            )
        elif self.action_batch_style == "var":
            # Input size stays the same, batch size increases
            pass
        elif self.action_batch_style == "pca":
            # (batch_size, action_dim * prediction_horizon * (pca_components + 1))
            input_dict["input_size"] = (
                input_dict["obs_embedding_dim"]
                + input_dict.get("state_dim", 0)
                + (self.pca_components) * self.pca_pred_horizon * self.action_pred_shape[-1]
            )
        return input_dict

    def datasets_to_model_inputs(self, datasets: Dict[str, torch.Tensor]) -> torch.Tensor:
        # (batch_size, policy_batch_size, prediction_horizon, action_dim)
        action_preds = datasets["action_preds"]
        obs_embeddings = datasets["obs_embeddings"]
        states = datasets.get("states", None)
        executed_action_batch = action_preds[:, 0, :, :].flatten(start_dim=-2)
        if not self.use_action_batch:
            # (batch_size, prediction_horizon * action_dim)
            action_preds = executed_action_batch
        else:
            if self.action_batch_style == "svd":
                # (batch_size, svd_components* prediction_horizon * action_dim)
                action_preds = self._perform_svd(action_preds).flatten(start_dim=1)
            elif self.action_batch_style == "var":
                # (batch_size * action_batch_size, prediction_horizon * action_dim)
                action_preds, obs_embeddings, states = self._perform_var(
                    action_preds, obs_embeddings=obs_embeddings, states=states
                )
                action_preds = action_preds.flatten(start_dim=1)
            elif self.action_batch_style == "pca":
                action_preds = self._perform_pca(action_preds).flatten(start_dim=1)

        if states is not None:
            model_input = torch.cat((obs_embeddings, action_preds, states), dim=-1)
        else:
            model_input = torch.cat((obs_embeddings, action_preds), dim=-1)

        return {"input": model_input}


class RND_T(RNDBase):
    def __init__(
        self,
        input_dict: Dict[str, Any] = None,
        checkpoint_dir=None,
        desired_hparams: dict = {},
        **kwargs,
    ):
        super().__init__(input_dict, checkpoint_dir, desired_hparams, **kwargs)

    def _compute_input_sizes(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        input_dict["input_size"] = torch.Size(
            [input_dict["action_pred_shape"][-1], input_dict["action_pred_shape"][-2]]
        )
        input_dict["output_size"] = 128
        return input_dict

    def get_target_network(self):
        # Define the target network architecture
        kernel_size = min(self.input_size[0], 4)
        net = nn.Sequential(
            nn.Conv1d(in_channels=self.input_size[0], out_channels=64, kernel_size=kernel_size, stride=2),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=min(kernel_size - 1, 2), stride=1),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=min(kernel_size - 2, 2), stride=1),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool1d(output_size=4),  # reduces length of last dimension to output_size, here 4 -> 4
            nn.Flatten(),
            nn.Linear(in_features=1024, out_features=self.output_size),  # in_features = 256 * 4
        )
        return net

    def get_predictor_network(self):
        # Define the prediction network architecture
        kernel_size = min(self.input_size[0], 4)
        net = nn.Sequential(
            nn.Conv1d(in_channels=self.input_size[0], out_channels=64, kernel_size=kernel_size, stride=2),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=min(kernel_size - 1, 2), stride=1),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=min(kernel_size - 2, 2), stride=1),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool1d(output_size=4),
            nn.Flatten(),
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=self.output_size),
            nn.ReLU(),
            nn.Linear(in_features=self.output_size, out_features=self.output_size),
        )
        return net

    def datasets_to_model_inputs(self, datasets: Dict[str, torch.Tensor]) -> torch.Tensor:
        action_preds = datasets["action_preds"]
        executed_action_batch = action_preds[:, 0, :, :]
        if not self.use_action_batch:
            # (batch_size, action_dim, prediction_horizon)
            action_preds = executed_action_batch.swapaxes(-2, -1)
            # action_preds = action_preds
        # No swap axis for action batch handling
        else:
            if self.action_batch_style == "svd":
                # (batch_size, svd_components, action_dim * prediction_horizon)
                action_preds = self._perform_svd(action_preds)

            elif self.action_batch_style == "var":
                # (batch_size * action_batch_size, prediction_horizon * action_dim)
                action_preds = self._perform_var(action_preds)
            elif self.action_batch_style == "pca":
                # (batch_size, pca_components, action_dim * prediction_horizon)
                action_preds = self._perform_pca(action_preds)

        return {"input": action_preds}

    def _update_input_sizes_from_action_batch_handling(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Only implemented in the action prediction using models"""
        # Input sizes stay the same (batch sizes increase)

        if self.action_batch_style == "svd":
            compressed_batch_size = self.svd_components + 1 if self.svd_project else self.svd_components
            input_dict["input_size"] = torch.Size(
                [compressed_batch_size, self.action_pred_shape[-1] * self.action_pred_shape[-2]]
            )
        elif self.action_batch_style == "var":
            # Size stays the same, batch size increases
            pass
        elif self.action_batch_style == "pca":
            input_dict["input_size"] = torch.Size(
                [self.pca_components, self.action_pred_shape[-1] * self.pca_pred_horizon]
            )
        return input_dict


class RND_TO(RNDBase):
    def __init__(
        self,
        input_dict: Dict[str, Any] = None,
        checkpoint_dir=None,
        desired_hparams: dict = {},
        **kwargs,
    ):
        """
        Args:
            input_size: (prediction_horizon, action_dim)
            condition_vector_size: embedding dim + state dim

        """

        super().__init__(
            input_dict=input_dict, checkpoint_dir=checkpoint_dir, desired_hparams=desired_hparams, **kwargs
        )

    def _compute_input_sizes(self, input_dict):
        input_dict["input_size"] = input_dict["action_pred_shape"][-2:]
        input_dict["condition_vector_size"] = input_dict["obs_embedding_dim"] + input_dict.get("state_dim", 0)
        return input_dict

    def get_target_network(self):
        # Define the target network architecture
        # RND_to_base only takes the action dim part of the input size
        net = Target(
            RND_TO_base(
                input_dim=self.input_size[-1],
                global_cond_dim=self.condition_vector_size,
                down_dims=self.hyperparameters.down_dims,
                kernel_size=self.hyperparameters.kernel_size,
                n_groups=self.hyperparameters.num_groups,
            ),
            in_features=self.input_size[-2]
            * self.hyperparameters["down_dims"][0],  # Necessary to adapt to action prediction horizons other than 16
        )
        return net

    def get_predictor_network(self):
        # RND_to_base only takes the action dim part of the input size
        net = Predictor(
            RND_TO_base(
                input_dim=self.input_size[-1],
                global_cond_dim=self.condition_vector_size,
                down_dims=self.hyperparameters.down_dims,
                kernel_size=self.hyperparameters.kernel_size,
                n_groups=self.hyperparameters.num_groups,
            ),
            in_features=self.input_size[-2]
            * self.hyperparameters["down_dims"][0],  # Necessary to adapt to action prediction horizons other than 16
        )
        return net

    def forward(self, action_preds: torch.Tensor, condition_vector: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.
        Args:
            input (torch.Tensor): Action_predictions (batch_size, prediction_horizon, action_dim).
            cond_vector (torch.Tensor): Obs_emeddings (batc_size, embedding_dim) optionally concatenated with states (batch_size, embedding_dim).
        """
        target = self.target_network(action_preds, condition_vector)
        pred = self.predictor_network(action_preds, condition_vector)
        loss = self.rnd_loss_function(pred, target)
        loss = self._transform_loss_function(loss)
        return loss

    def _update_input_sizes_from_action_batch_handling(self, input_dict):
        if self.action_batch_style == "svd":
            input_dict["action_batch_handling"]["svd_components"] = 7 if self.svd_project else 8
            self.svd_components = input_dict["action_batch_handling"]["svd_components"]
            # (svd_components + 1, action_dim * prediction_horizon)
            input_dict["input_size"] = torch.Size(
                [self.svd_components, self.action_pred_shape[-2] * self.action_pred_shape[-1]]
            )
        elif self.action_batch_style == "var":
            pass
        elif self.action_batch_style == "pca":
            input_dict["action_batch_handling"]["pca_components"] = 8
            self.pca_components = input_dict["action_batch_handling"]["pca_components"]
            # (pca_components + 1, action_dim * prediction_horizon)
            input_dict["input_size"] = torch.Size(
                [self.pca_components, self.pca_pred_horizon * self.action_pred_shape[-1]]
            )
        return input_dict

    def datasets_to_model_inputs(self, datasets: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        action_preds = datasets["action_preds"]
        executed_action_batch = action_preds[:, 0, :, :]
        obs_embeddings = datasets["obs_embeddings"]
        states = datasets.get("states", None)  # Optional dataset
        if not self.use_action_batch:
            # (batch_size, prediction_horizon, action_dim)
            action_preds = executed_action_batch
        else:
            if self.action_batch_style == "svd":
                action_preds = self._perform_svd(action_preds)
            elif self.action_batch_style == "var":
                # (batch_size * action_batch_size, prediction_horizon * action_dim)
                action_preds = self._perform_var(action_preds)
            elif self.action_batch_style == "pca":
                # (batch_size, action_batch_size, action_dim * prediction_horizon)
                action_preds = self._perform_pca(action_preds)

        if states is not None:
            condition_vector = torch.cat((obs_embeddings, states), dim=-1)
        else:
            condition_vector = obs_embeddings

        return {"action_preds": action_preds, "condition_vector": condition_vector}
