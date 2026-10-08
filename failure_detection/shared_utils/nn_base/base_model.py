import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torchvision
from typing import Any, Dict, List, Optional, Tuple, Union
from shared_utils.utility_functions import compare_dicts
import hydra


class BaseModel(nn.Module):
    def __init__(
        self, input_dict: Dict[str, Any] = None, checkpoint_dir: str = None, model_id_params: dict = {}, **kwargs
    ):
        assert checkpoint_dir is not None or input_dict is not None, (
            "Either input_size or checkpoint_dir must be provided."
        )
        if checkpoint_dir is not None:
            cfg, state_dict = self._load_checkpoint(
                checkpoint_dir, model_id_params, checkpoint_name=kwargs.get("checkpoint_name", None)
            )
            input_dict = cfg.get("input_dict", None)
            assert input_dict is not None, "input_dict must be provided in the checkpoint."
            kwargs = cfg.get("kwargs", {})
        else:
            input_dict = self._process_input_dict(input_dict)
        super(BaseModel, self).__init__()
        self.input_dict = input_dict
        # Set class attributes
        for key, value in input_dict.items():
            setattr(self, key, value)
        self.model_id_params = model_id_params
        self.kwargs = kwargs

        # Initialize the model
        self.layers_info = {}
        self._init_layers(input_dict, **kwargs)
        self._post_process_weights(init=checkpoint_dir is None)
        self._init_submodules(input_dict, **kwargs)

        # Load the state dict if provided
        if checkpoint_dir is not None and state_dict is not None:
            self.load_state_dict(state_dict, strict=False)

    def _process_input_dict(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        return input_dict

    def _init_layers(self, input_dict: Dict[str, Any], **kwargs):
        """
        Initialize the model layers based on the input dictionary.
        This method should be overridden in subclasses to define the model architecture.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def _init_submodules(self, input_dict: Dict[str, Any], **kwargs):
        """
        Initialize submodules based on the input dictionary.
        This method should be overridden in subclasses to define the model architecture.
        """
        for module_name, cfg in input_dict["submodules"].items():
            module_class: BaseModel = hydra.utils.get_class(cfg["target"])
            self.add_module(module_name, module_class(**cfg["inputs"]))
            if cfg["fixed"]:
                for param in getattr(self, module_name).parameters():
                    param.requires_grad = False

    def _post_process_weights(self, init: bool = False):
        """
        Post-process the weights of the model.
        Initialize weights of all modules with appropriate initialization values.
        """
        if init:
            for p in self.modules():
                if isinstance(p, nn.Conv1d):
                    nn.init.orthogonal_(p.weight, np.sqrt(2))
                    if p.bias is not None:
                        p.bias.data.zero_()
                elif isinstance(p, nn.Conv2d):
                    nn.init.kaiming_normal_(p.weight, mode="fan_out", nonlinearity="relu")
                    if p.bias is not None:
                        p.bias.data.zero_()
                elif isinstance(p, nn.Linear):
                    nn.init.xavier_uniform_(p.weight)
                    if p.bias is not None:
                        p.bias.data.zero_()
                elif isinstance(p, nn.BatchNorm1d) or isinstance(p, nn.BatchNorm2d):
                    p.weight.data.fill_(1)
                    if p.bias is not None:
                        p.bias.data.zero_()

    def _load_checkpoint(
        self,
        checkpoint_dir: str,
        model_id_params: dict = {},
        checkpoint_name: Optional[str] = None,
    ) -> Tuple:
        """
        Load the model checkpoint from the specified directory (and optionally checkpoint name).
        Args:
            checkpoint_dir (str): Directory containing the checkpoint file.
            checkpoint_name (Optional[str]): Name of the checkpoint file.
            model_id_params (Optional[Dict[str, Any]]): Parameters to identify the correct checkpoint.
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
            if not model_id_params:
                Warning("No model_id_params provided. Using the last checkpoint found in the directory.")
                found = True
                break

            if "model_id_params" not in cfg.keys():
                continue

            for key, value in model_id_params.items():
                found = compare_dicts(value, cfg["model_id_params"][key])
                if not found:
                    break

            if found:
                break

        if not found:
            raise FileNotFoundError(
                f"No checkpoint files found in {checkpoint_dir} with the desired model parameters: {model_id_params}."
            )

        return cfg, checkpoint["state_dict"]

    def _save_checkpoint(
        self,
        checkpoint_dir: str,
        checkpoint_name: Optional[str] = None,
        model_cfg: Optional[Dict[str, Any]] = None,
        overwrite=False,
        state_dict=None,
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
        }
        if model_cfg is not None:
            checkpoint["cfg"] = model_cfg
        else:
            checkpoint["cfg"] = {
                "model_id_params": self.model_id_params,
                "input_dict": self.input_dict,
                "kwargs": self.kwargs,
            }
        torch.save(
            checkpoint,
            os.path.join(checkpoint_dir, checkpoint_name),
        )

    def _check_for_existance(self, checkpoint_dir: str, model_id_params: dict) -> bool:
        """
        Check if a checkpoint file with the same hyperparameters exists in the specified directory.
        Args:
            checkpoint_dir (str): Directory containing the checkpoint file.
            model_id_params (dict): Parameters to identify the correct checkpoint.
        Returns:
            bool: True if the checkpoint file exists, False otherwise.
        """
        if not os.path.exists(checkpoint_dir):
            return False

        filenames = sorted([filename for filename in os.listdir(checkpoint_dir) if filename.endswith(".ckpt")])

        for filename in filenames:
            checkpoint = torch.load(os.path.join(checkpoint_dir, filename), weights_only=False)
            if "cfg" not in checkpoint.keys():
                continue
            if "model_id_params" not in checkpoint["cfg"].keys():
                continue
            if compare_dicts(checkpoint["cfg"]["model_id_params"], model_id_params):
                return True
        return False
