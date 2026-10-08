import os
import numpy as np
from typing import Optional, Union
from shared_utils.hydra_utils import load_config
from shared_utils.data_management import (
    load_raw_rollouts,
    _get_filenames,
)
from shared_utils.utility_functions import ensure_list, list_to_tensor_padded
from rollout_data.rollout_datasets import ProcessedRolloutDataset
import torch
import re
from omegaconf import DictConfig


class TaskManager:
    def __init__(self, cfg: DictConfig, task: str, base_config_path: str, task_data_path: str, **kwargs):
        """
        Manages task-specific data processing and rollouts.

        Responsibilities:
        - Load and process raw rollouts.
        - Generate datasets for training and evaluation.
        - Interface with task-specific environments.

        Args:
            cfg (DictConfig): Configuration for the task.
            task (str): Name of the task.
            base_config_path (str): Path to the configuration folder.
            task_data_path (str): Path to the task-specific data.
            kwargs: Additional arguments for processing rollouts.
                - required_tensors (list): List of required tensors to be included in the dataset.
                - optional_tensors (list): List of optional tensors to be included in the dataset.
                - device (str): Device to use for processing (e.g., "cpu", "cuda").
                - normalize_tensors (dict): Dictionary specifying which tensors to normalize and their normalization parameters.
                - calibration_subset_seed (int): If set, draw the `max_calibration_rollouts`
                  calibration rollouts at random with this seed instead of taking the first N,
                  so several runs over one recording differ only in which subset they calibrate on.
                - max_calibration_rollouts (int): If set, use only the first N calibration
                  rollouts on disk (filenames are sorted, so the subset is deterministic).
                  Lets a recording of many calibration episodes be evaluated at a smaller
                  calibration-set size without re-recording.
        """
        self.task = task
        self.base_config_path = base_config_path
        self.task_data_path = task_data_path
        # Task-specific config
        self.cfg = load_config("task", task, return_only_subdict=True, as_dict=True)
        self.calibration_rollout_dir = os.path.join(self.task_data_path, "rollouts", "calibration")
        self.test_rollout_dir = os.path.join(self.task_data_path, "rollouts", "test")
        self.required_tensors = kwargs.get("required_tensors", [])
        self.optional_tensors = list(kwargs.get("optional_tensors", []))
        # self.optional_tensors = list(set(kwargs.get("optional_tensors", [])+ ["states"]))
        self.device = kwargs.get("device", "cpu")
        self.normalize_tensors = cfg.eval.get("normalize_tensors", {})
        self.max_calibration_rollouts = kwargs.get("max_calibration_rollouts", None)
        # None: take the first N sorted filenames; an int: draw N at random with this seed.
        self.calibration_subset_seed = kwargs.get("calibration_subset_seed", None)

    def get_rollout_dataset(
        self,
        load_dataset_if_exists=True,
        check_for_new_rollouts=False,
    ):
        """Get the processed rollout dataset by either creating it from the raw rollouts or loading it.

        If the processed rollout dataset is not available or load_dataset_if_exists is False, it will process the raw rollouts and create a dataset.
        If check_for_new_rollouts is True, it will check for new rollouts in the rollout folders and append them to the dataset if possible.

        Args:
            load_dataset_if_exists (bool, optional): Whether to prioritize loading the dataset if it already exists. Defaults to True.
            check_for_new_rollouts (bool, optional): Whether to check for new rollouts in the data path and append them to the dataset if possible. Defaults to False.

        Returns:
            rollout_dataset (ProcessedRolloutDataset): A custom dataset with loaded and normalized data inlcuding global metadata.
        """
        # Initialize the dataset
        processed_rollout_dataset = ProcessedRolloutDataset(
            task_data_path=self.task_data_path,
            base_config_path=self.base_config_path,
            required_tensors=self.required_tensors,
            optional_tensors=self.optional_tensors,
            normalize_tensors=self.normalize_tensors,
        )
        # Check if the dataset already exists
        if load_dataset_if_exists:
            processed_rollout_dataset.load_dataset()
            if processed_rollout_dataset.dataset_loaded:
                # Check whether dataset should and can be extended with new rollouts
                if check_for_new_rollouts:
                    processed_rollout_dataset = self._init_or_update_dataset(processed_rollout_dataset, extend=True)
                return processed_rollout_dataset

        # Check if the row rollouts already exist
        processed_rollout_dataset = self._init_or_update_dataset(processed_rollout_dataset, extend=False)
        if processed_rollout_dataset.dataset_loaded:
            return processed_rollout_dataset

        raise NotImplementedError(
            "Generating new rollouts with the diffusion policy is not implemented yet. Make sure raw rollouts are available."
        )

    def _init_or_update_dataset(self, processed_rollout_dataset: ProcessedRolloutDataset, extend=False):
        """Check if (new) raw rollouts are available, process them, and append them to the dataset or init the dataset with them if all required tensors are given.

        Args:
            processed_rollout_dataset (ProcessedRolloutDataset): The dataset object. Defaults to ["actions", "states"].
            extend (bool, optional): Whether to extend the dataset with new rollouts or initialize it with them. Defaults to False.

        Returns:
            processed_rollout_dataset (ProcessedRolloutDataset): The updated/initialized dataset object.
        """
        # Check the filenames in the rollout directory
        calibration_filenames = _get_filenames(self.calibration_rollout_dir, keywords=["episode", "eps", "rollout"])
        test_filenames = _get_filenames(self.test_rollout_dir, keywords=["episode", "eps", "rollout"])

        # Optionally evaluate against a smaller calibration set than was recorded. Without a subset
        # seed the filenames are sorted, so the first N is a stable subset across runs and methods.
        # With `calibration_subset_seed` set, N of them are drawn at random with that seed instead —
        # the seeds of a multi-seed run over one recording then differ only in *which* calibration
        # rollouts they see, and the drawn subset is kept in filename
        # order so every downstream index still lines up with the metadata.
        if self.max_calibration_rollouts is not None and len(calibration_filenames) > self.max_calibration_rollouts:
            if self.calibration_subset_seed is None:
                calibration_filenames = calibration_filenames[: self.max_calibration_rollouts]
                how = "first"
            else:
                rng = np.random.default_rng(int(self.calibration_subset_seed))
                drawn = rng.permutation(len(calibration_filenames))[: self.max_calibration_rollouts]
                calibration_filenames = [calibration_filenames[i] for i in sorted(drawn)]
                how = f"seed {self.calibration_subset_seed}"
            print(
                f"[task_manager] Using {self.max_calibration_rollouts} of "
                f"{len(calibration_filenames) if self.calibration_subset_seed is None else 'all'} "
                f"calibration rollouts ({how})."
            )

        if len(calibration_filenames) < 5 or len(test_filenames) < 5:
            print(
                f"Only {len(calibration_filenames)} calibration rollouts and {len(test_filenames)} test rollouts found in the rollout directory. "
            )

        # Obtain the filenames of the new rollouts
        if extend:
            calibration_filenames = processed_rollout_dataset.compare_old_and_new_rollouts(
                calibration_filenames, "calibration"
            )
            test_filenames = processed_rollout_dataset.compare_old_and_new_rollouts(test_filenames, "test")

        # Obtain metadata, and processed rollouts
        new_data_dict = self._load_and_convert_raw_rollouts(calibration_filenames, test_filenames)
        if new_data_dict:
            if extend:
                # Append the new data to the existing dataset
                processed_rollout_dataset.add_data(
                    new_data_dict, extend=True, replace=False, overwrite_saved_dataset=True
                )
            else:
                # Initialize the dataset with the new data
                processed_rollout_dataset.init_dataset(**new_data_dict)

        return processed_rollout_dataset

    def _load_and_convert_raw_rollouts(
        self,
        calibration_rollout_filenames: Optional[list] = [],
        test_rollout_filenames: Optional[list] = [],
    ) -> dict:
        """Load and convert raw calibration and test rollouts with the specified filenames

        Args:
            calibration_rollout_filenames (list, optional): List of calibration rollout filenames. Defaults to None.
            test_rollout_filenames (list, optional): List of test rollout filenames. Defaults to None.

        Returns:
            data_dict: A dictionary containing the converted rollouts with metadata.
        """

        # Load the raw rollouts
        calibration_rollouts = load_raw_rollouts(
            load_dirs=self.calibration_rollout_dir,
            searched_filenames=calibration_rollout_filenames,
            required_keys=self.required_tensors,
        )
        test_rollouts = load_raw_rollouts(
            load_dirs=self.test_rollout_dir,
            searched_filenames=test_rollout_filenames,
            required_keys=self.required_tensors,
        )
        if len(calibration_rollouts) == 0 and len(test_rollouts) == 0:
            return {}

        # Obtain calibration and test rollout labels
        num_rollouts = len(calibration_rollouts) + len(test_rollouts)
        calibration_rollout_labels = np.zeros(num_rollouts, dtype=bool)
        calibration_rollout_labels[: len(calibration_rollouts)] = np.ones(len(calibration_rollouts), dtype=bool)
        test_rollout_labels = ~calibration_rollout_labels

        rollouts = calibration_rollouts + test_rollouts
        kwargs = {
            "calibration_rollout_labels": calibration_rollout_labels,
            "test_rollout_labels": test_rollout_labels,
            "test_rollout_filenames": test_rollout_filenames,
            "calibration_rollout_filenames": calibration_rollout_filenames,
        }

        # Process the raw rollouts
        data_dict = self._convert_raw_rollouts(rollouts, **kwargs)

        return data_dict

    def _convert_raw_rollouts(self, rollouts: list, **kwargs):
        """Process the raw rollouts.

        Contents:
        - Extract metadata from the rollouts.
        - Convert the rollouts to a standard format which includes global metadata.

        Args:
            rollouts (list): List of raw rollouts.
            **kwargs: Additional arguments for processing the rollouts.
        Returns:
            data_dict: A dictionary containing the converted rollouts with metadata.
        """
        global_metadata = self.cfg.get("metadata", {})
        required_metadata_keys = [
            "num_robots", "num_steps", "num_rollouts", "episode_start_indices", 
            "episode_end_indices", "successful_rollout_labels", "calibration_rollout_labels", 
            "test_rollout_labels", "id_rollout_labels", "ood_rollout_labels"
        ]
        tensor_to_rollout_key_mapping = {
            "obs_embeddings": "obs_embedding", "actions": "action", 
            "action_preds": "action_pred", "states": "agent_pos", "rgb_images": "rgb"
        }

        global_metadata["num_rollouts"] = len(rollouts)
        global_metadata["num_robots"] = global_metadata.get("num_robots", 1)
        data_dict = {key: [] for key in self.required_tensors + self.optional_tensors}
        for key in required_metadata_keys:
            if key not in global_metadata:
                global_metadata[key] = []

        global_metadata.update({
            "calibration_rollout_labels": kwargs.get("calibration_rollout_labels", []),
            "test_rollout_labels": kwargs.get("test_rollout_labels", []),
            "rollout_filenames": {
                "test": kwargs.get("test_rollout_filenames", []),
                "calibration": kwargs.get("calibration_rollout_filenames", [])
            },
            "episode_keys": []
        })

        filenames = global_metadata["rollout_filenames"]["calibration"] + global_metadata["rollout_filenames"]["test"]
        step_counter, episode_counter = 0, 0
        # Execution horizons recorded per episode by the recorder. Policy-specific
        # and may differ from the static task-config
        # default; collected here to override that default in the metadata below.
        recorded_exec_horizons: set = set()

        for rollout in rollouts:
            metadata = self._parse_filename_for_metadata(filenames[episode_counter])
            if isinstance(rollout, dict):
                metadata.update(rollout.get("metadata", {}))
                rollout = rollout["rollout"]
            elif isinstance(rollout, list) and rollout[0].get("metadata", False):
                metadata.update(rollout[0])
                rollout = rollout[1:]
            else:
                raise ValueError("Rollout must be either a dict or a list")

            recorded_exec_horizon = metadata.get("action_execution_horizon")
            if recorded_exec_horizon is not None:
                recorded_exec_horizons.add(int(recorded_exec_horizon))

            global_metadata["successful_rollout_labels"].append(metadata.get("successful", False))
            global_metadata["episode_keys"].append(rollout[0].keys())
            global_metadata["episode_start_indices"].append(step_counter)

            for step in rollout:
                for i in range(global_metadata["num_robots"]):
                    new_step = {key: step[key][i] if isinstance(step[key], list) else step[key] for key in step}
                    for tensor in self.required_tensors + self.optional_tensors:
                        if tensor_to_rollout_key_mapping.get(tensor, tensor) in new_step and new_step[tensor_to_rollout_key_mapping.get(tensor, tensor)] is not None:
                            data_dict[tensor].append(new_step[tensor_to_rollout_key_mapping.get(tensor, tensor)])
                        elif tensor in self.optional_tensors:
                            data_dict.pop(tensor, None)
                        else:
                            raise ValueError(
                                f"Tensor {tensor} mapping to {tensor_to_rollout_key_mapping[tensor]} not found in rollout step. Available keys: {step.keys()}"
                            )
                    step_counter += 1

            global_metadata["episode_end_indices"].append(step_counter)
            rollout_subtype = metadata.get("rollout_subtype", "id" if global_metadata["calibration_rollout_labels"][episode_counter] else "ood")
            rollout_subtype = "id" if rollout_subtype in ["ca", "id", "calibration", "na"] else "ood"
            global_metadata["ood_rollout_labels"].append(rollout_subtype == "ood")
            global_metadata["id_rollout_labels"].append(rollout_subtype == "id")
            episode_counter += 1

        global_metadata["rollouts_consistent"] = all(keys == global_metadata["episode_keys"][0] for keys in global_metadata["episode_keys"])
        global_metadata["episode_keys"] = list(map(str, global_metadata["episode_keys"][0])) if global_metadata["rollouts_consistent"] else [list(map(str, keys)) for keys in global_metadata["episode_keys"]]
        global_metadata["num_steps"] = step_counter

        # Prefer the execution horizon actually recorded with the rollouts over the
        # static task-config default. The recorder stores it per episode; it is
        # policy-specific while the config hardcodes one value.
        # Methods such as TC (STAC) align consecutive action chunks by this horizon, so
        # a wrong value can collapse the chunk overlap (backtrack_steps -> 0). Fall back
        # to the config value only when nothing was recorded.
        actions_meta = global_metadata.get("actions")
        if recorded_exec_horizons and actions_meta is not None:
            if len(recorded_exec_horizons) > 1:
                print(
                    "[task_manager] WARNING: inconsistent recorded action_execution_horizon "
                    f"values {sorted(recorded_exec_horizons)}; using {min(recorded_exec_horizons)}."
                )
            actions_meta["action_execution_horizon"] = min(recorded_exec_horizons)

        # Check of all required keys are present in the metadata and convert them to numpy arrays
        for key in required_metadata_keys:
            if key not in global_metadata:
                raise ValueError(f"Key {key} not found in metadata. Available keys: {global_metadata.keys()}")
            if isinstance(global_metadata[key], list) and "ids" or "indices" or "labels" in key:
                global_metadata[key] = np.array(global_metadata[key])

        global_metadata["episode_lengths"] = global_metadata["episode_end_indices"] - global_metadata["episode_start_indices"]
        global_metadata["failed_rollout_labels"] = ~global_metadata["successful_rollout_labels"]

        data_dict["metadata"] = global_metadata
        for key in list(data_dict.keys()):
            if key in self.required_tensors + self.optional_tensors:
                data_dict[key] = list_to_tensor_padded(data_dict[key])

        if "action_preds" in data_dict:
            data_dict = self._augment_actions(data_dict)

        return data_dict

    def _augment_actions(self, data_dict):
        """Augment the actions in the 'actions' and 'action_preds' tensors.

        Augmentations include:
        - Converting Velocity actions to position trajectories.
        - Converting angular velocity actions to rotation trajectories.
        """
        states_available = data_dict.get("states", None) is not None
        if states_available:
            state_mapping = data_dict["metadata"]["states"]["state_mapping"]
            states = data_dict["states"]
            if state_mapping["position"] is not None:
                initial_positions = states[..., state_mapping["position"]]
            else:
                initial_positions = torch.zeros_like(states[..., 0])
            if state_mapping["rotation"] is not None:
                initial_rotations = states[..., state_mapping["rotation"]]
            else:
                initial_rotations = torch.zeros_like(states[..., 0])

        action_mapping = data_dict["metadata"]["actions"]["action_mapping"]
        ts = data_dict["metadata"].get("env", {"ts", 0.1}).get("ts", 0.1)
        # Check which actions are available and which can be augmented
        available_actions = [key for key in action_mapping.keys() if action_mapping[key] is not None]
        if "velocity" in available_actions and "position" not in available_actions:
            # Convert velocity actions to position trajectories, starting from either zero or the starting position given by the states
            linear_velocity = data_dict["action_preds"][..., action_mapping["velocity"]]
            position_trajectories = torch.cumsum(linear_velocity * ts, dim=-2)
            if states_available:
                position_trajectories += initial_positions.unsqueeze(1).unsqueeze(2)
            action_mapping["position"] = list(
                np.arange(
                    data_dict["action_preds"].shape[-1],
                    data_dict["action_preds"].shape[-1] + position_trajectories.shape[-1],
                )
            )
            data_dict["action_preds"] = torch.cat([data_dict["action_preds"], position_trajectories], dim=-1)
        if "angular_velocity" in available_actions and "rotation" not in available_actions:
            # Convert angular velocity actions to rotation trajectories, starting from either zero or the starting rotation given by the states
            angular_velocity = data_dict["action_preds"][..., action_mapping["angular_velocity"]]
            rotation_trajectories = torch.cumsum(angular_velocity * ts, dim=-2)
            if states_available:
                rotation_trajectories += initial_rotations.unsqueeze(1).unsqueeze(2)
            action_mapping["rotation"] = np.arange(
                data_dict["action_preds"].shape[-1],
                data_dict["action_preds"].shape[-1] + rotation_trajectories.shape[-1],
            )
            data_dict["action_preds"] = torch.cat([data_dict["action_preds"], position_trajectories], dim=-1)
        # Update the action mapping in the metadata
        data_dict["metadata"]["actions"]["action_mapping"] = action_mapping
        return data_dict

    def _parse_filename_for_metadata(
        self, filename, success_keywords=["success", "_s_"], failure_keywords=["failure", "_f_"]
    ):
        """Parse the filename to extract metadata.

        Args:
            filename (dict): Fileinfo with filename and filepath.

        Returns:
            dict: A dictionary containing the extracted metadata.
        """
        # Get episode ID from the filename
        matches = re.findall(r"(?<!\d)\d{2,4}(?!\d)", filename)
        # episodes_id = filename.split("_")[-1].split(".")[0]
        episode_id = int(matches[0]) if matches else None
        # Get successful flag from the filename
        successful = True if any(success_keyword in filename for success_keyword in success_keywords) else False
        if not successful:
            # Check if the filename contains any failure keywords
            # If so, set the successful flag to False
            successful = False if any(failure_keyword in filename for failure_keyword in failure_keywords) else True
        # Check whether we can extract action execution horizon from the filename
        match = re.search(r"exec_horizon_(\d+)_", filename)
        if match:
            exec_horizon = int(match.group(1))  # Extract the number as an integer
        else:
            exec_horizon = None
        return {
            "episode_id": episode_id,
            "successful": successful,
            "exec_horizon": exec_horizon,
        }
