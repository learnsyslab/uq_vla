import torch
from shared_utils.normalizer import LinearNormalizer
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from shared_utils.data_management import load_data, save_data
from shared_utils.utility_functions import ensure_list
from typing import Union, Optional, Dict, Any
from omegaconf import DictConfig


class ProcessedRolloutDataset(Dataset):
    def __init__(
        self,
        task_data_path,
        base_config_path,
        required_tensors: list = ["action_preds", "rgb_images", "obs_embeddings"],
        **kwargs,
    ):
        """
        Dataset for processed rollouts.
        Each rollout data key, e.g., "action_preds", "rgb_images", etc., is stored separately as a entry in the "self.data" dictionary as a tensor of shape (N, {key.shape}), where N is the total number of steps across all rollouts.
        "self.data["metadata"]" contains the metadata dictionary with information about the dataset, including the episode start and end indices, the action and observation spaces, and the rollout types.

        Included functionality:
        - Initialize the dataset with metadata and tensors (`init_dataset`).
        - Create normalizers based on calibration rollouts (`normalize`, `_create_normalizer`).
        - Load and save the dataset to/from a specified directory (`load_dataset`, `_save_dataset`).
        - Retrieve the available tensors in the dataset (`get_available_tensors`).
        - Retrieve various information about the dataset (`get_metadata`, `print_dataset_summary`).
        - Iterate over specified episodes filtered by rollout types and subtypes (`iterate_episodes`).
        - Retrieve a (filtered) subset of the data (`get_subset`).

        Args:
            task_data_path (str): Path to the task specific data directory.
            base_config_path (str): Path to the base configuration directory.
            required_tensors (list): List of required tensors to be included in the dataset. Default is ["action_preds", "rgb_images", "obs_embeddings"].
            kwargs: Additional keyword arguments, currently supported are:
                - normalize_tensors (dict): Dictionary containing the base normalization parameters for each tensor. 
                  Example: {"actions": {"mode": "minmax", "limits": [0, 1]}}.
                - optional_tensors (list): List of optional tensors to include in the dataset. 
                - shuffle (bool): Whether to shuffle the dataset during initialization. Default is False.
                - calibration_set_size (float): Fraction of rollouts to use for calibration. Default is 0.2.
        Returns:
            rollout_dataset (ProcessedRolloutDataset): An empty dataset object that can be initialized with metadata and tensors.
        """
        self.data = {}
        self.task_data_path = task_data_path
        self.save_dir = os.path.join(self.task_data_path, "processed_rollouts")
        self.base_config_path = base_config_path
        self.dataset_loaded = False
        self.required_tensors: list = required_tensors
        self.kwargs = kwargs
        # For filtering tensors and providing dataset information
        self.allowed_subsets = ["all", "calibration", "test", "successful", "failed"]
        self.allowed_subsubsets = ["all", "id", "ood"]
        # Although some keys are redundant, they are included for consistency and future-proofing.
        self.required_metadata_keys = [
            "episode_start_indices",
            "episode_end_indices",
            "calibration_rollout_labels",
            "test_rollout_labels",
            "successful_rollout_labels",
            "failed_rollout_labels",
            "id_rollout_labels",
            "ood_rollout_labels",
            "num_steps",
            "num_rollouts",
            "episode_lengths",
        ]
        # Holds the normalizer objects for each tensor
        self.normalizer = {}
        self.normalize_tensors: dict = kwargs.get("normalize_tensors", {})
        self.optional_tensors: list = kwargs.get("optional_tensors", [])

    def get_available_tensors(self):
        """Returns the available tensors in the dataset."""
        self._assert_metadata()
        return self.data["metadata"]["available_tensors"]

    def compare_old_and_new_rollouts(self, filenames: list, rollout_type: str):
        """
        Compare the raw rollout filenames (saved in the metadata) with the new filenames.
        Returns the new filenames that are not in the old filenames.
        This is used to check whether new rollouts can be added to the dataset.

        Args:
            filenames (list): List of filenames to compare.
            rollout_type (str): Either "calibration" or "test".
        Returns:
            new_filenames (list): List of new filenames.
        """
        assert rollout_type in ["calibration", "test"], "rollout_type must be either 'calibration' or 'test'."
        metadata: dict = self.data.get("metadata", None)
        if metadata is None:
            return filenames
        if metadata.get("rollout_filenames", None) is None:
            return filenames
        if rollout_type not in metadata["rollout_filenames"]:
            return filenames
        filenames = ensure_list(filenames)
        old_filenames = metadata["rollout_filenames"][rollout_type]
        new_filenames = [filename for filename in filenames if filename not in old_filenames]
        return new_filenames

    def _dataset_exists(
        self,
        save_dir: Optional[str] = None,
        tensors_required: Optional[list] = None,
        check_for_normalizer: bool = False,
    ) -> bool:
        """
        Check if the dataset (mandatory elements and the specified optional ones) already exists.

        Args:
            save_dir (str, Optional): Directory to check for the dataset. If None, the default save directory is used.
            tensors_required (list): List of tensors to check for existence. Default is self.required_tensors.
        Returns:
            all_found (bool): True if all mandatory and optional tensors are found, False otherwise.
        """
        # Check which tensors are required
        if tensors_required is None:
            tensors_required = self.required_tensors
        else:
            tensors_required = ensure_list(tensors_required)
            # tensors_required = list(set(tensors_required + self.required_tensors))

        # Check if the save directory exists
        if save_dir is None:
            save_dir = self.save_dir
        if not os.path.exists(save_dir) or not os.path.isdir(save_dir):
            return False

        elements_to_check = ["metadata"] + tensors_required
        if check_for_normalizer:
            elements_to_check.append("normalizer")

        filenames = os.listdir(save_dir)
        all_found = True
        missing_files = []
        for element in elements_to_check:
            if not any([element in filename for filename in filenames]):
                all_found = False
                missing_files.append(element)

        if not all_found:
            print(f"Following files in {save_dir} are missing. Loading dataset not possible.{missing_files}")
        return all_found

    def _check_and_update_metadata(self, metadata: dict, update_metadata: bool = False):
        """
        Check if the metadata dictionary contains the required keys. Updates or replaces the current metadata.

        Args:
            metadata (dict): New metadata dictionary to check.
            update_metadata (bool): If True, the old metadata is updated. If False, the metadata is replaced.
        """
        assert isinstance(metadata, dict), "Metadata must be a dictionary."
        always_required_keys = [
            "episode_start_indices",
            "episode_end_indices",
            "calibration_rollout_labels",
            "test_rollout_labels",
            "successful_rollout_labels",
            "failed_rollout_labels",
        ]
        if update_metadata:
            self._assert_dataset_loaded()
            self._assert_metadata()
            required_keys = always_required_keys
            for key in required_keys:
                if key not in metadata.keys():
                    raise ValueError(f"Metadata must contain the key '{key}'.")
                else:
                    # Indices refer to steps, so we need to add the number of steps to the indices
                    if "_indices" in key:
                        metadata[key] += self.data["metadata"]["num_steps"]
                    # Rollout ids refer to rollouts, so we need to add the number of rollouts to the ids
                    if "_ids" in key:
                        metadata[key] += self.data["metadata"]["num_rollouts"]
                    # Append the new metadata to the existing metadata
                    self.data["metadata"][key] = np.concatenate((self.data["metadata"][key], metadata[key]), axis=0)
        else:
            self.data["metadata"] = {}
            # New metadata, so we need to check if num_robots and action space are in the metadata
            required_keys = always_required_keys + ["num_robots", "actions"]
            for key in required_keys:
                if key not in metadata.keys():
                    raise ValueError(f"Metadata must contain the key '{key}'.")
                else:
                    self.data["metadata"][key] = metadata[key]
        # These keys can be calculated from the metadata (dont change the order of the keys)
        calcable_keys = ["num_rollouts", "episode_lengths", "num_steps"]
        for key in calcable_keys:
            if key == "num_steps":
                self.data["metadata"][key] = sum(self.data["metadata"]["episode_lengths"])
            elif key == "num_rollouts":
                self.data["metadata"][key] = len(self.data["metadata"]["episode_start_indices"])
            elif key == "episode_lengths":
                self.data["metadata"][key] = (
                    self.data["metadata"]["episode_end_indices"] - self.data["metadata"]["episode_start_indices"]
                )
        # Check whether there are additional keys in the metadata and add them
        for key in metadata.keys():
            if key not in self.data["metadata"].keys():
                self.data["metadata"][key] = metadata[key]

        if "available_tensors" not in self.data["metadata"].keys():
            self.data["metadata"]["available_tensors"] = []
        return

    def init_dataset(self, metadata: dict, normalize_tensors: dict = None, **kwargs):
        """
        Initialize the dataset with metadata and the rollout data tensors included in kwargs.
        """
        # Update normalization dict if normalize is not None
        if normalize_tensors is not None:
            if isinstance(normalize_tensors, dict):
                for key in normalize_tensors.keys():
                    self.normalize_tensors[key] = normalize_tensors[key]
            elif isinstance(normalize_tensors, bool):
                for key in kwargs.keys():
                    self.normalize_tensors[key] = normalize_tensors

        # Check and init metadata
        self._check_and_update_metadata(metadata, update_metadata=False)

        for key, tensor in kwargs.items():
            if tensor is not None:
                assert isinstance(key, str), f"Key {key} must be a string."
                assert isinstance(tensor, torch.Tensor), f"Tensor {key} must be a torch tensor."
                tensor = self.normalize(tensor, key=key, only_create_normalizer=True)
                self.data[key] = tensor
                self.data["metadata"]["available_tensors"].append(key)

        # Splits the dataset into calibration and test sets if not already defined.
        if (
            self.data["metadata"].get("calibration_rollout_labels", None) is None
            or self.data["metadata"].get("test_rollout_labels", None) is None
        ):
            self._init_rollout_types(calibration_set_size=0.2, shuffle=False)

        # Save metadata, tensors, and normalizer
        # self._save_dataset(save_dir=self.save_dir)
        # Set the dataset_loaded flag to True
        self.dataset_loaded = True

    def _extend_tensor(self, tensor: torch.Tensor, key: str):
        """Extends an existing tensor in the dataset.

        Args:
            tensor (torch.Tensor): Tensor to be appended.
            key (str): Key to store the tensor in the dataset.
        """
        assert isinstance(tensor, torch.Tensor), f"Tensor {key} must be a torch tensor."
        
        
        # For rgb_images, the tensor is converted to (N, C, H, W) format
        if key == "rgb_images" and self.normalize_tensors.get(key, False):
            tensor = self._normalize_rgb_images(tensor, only_shape=True)

        # Check whether key is already in the dataset and the tensor has the same shape
        new_shape = tensor.shape[1:]
        self._assert_tensor(key, shape=new_shape)

        # Concatenate the new tensor with the existing one and create a new normalizer
        tensor = torch.cat((self.data[key], tensor), dim=0)
        self.data[key] = self.normalize(self.data[key], key=key, only_create_normalizer=True, overwrite_normalizer=True)

    def _add_tensor(self, tensor: torch.Tensor, key: str, replace: bool = True):
        """Adds a tensor to the dataset. 

        Args:
            tensor (torch.Tensor): Tensor to be added.
            key (str): Key to store the tensor in the dataset.
            replace (bool): If True, an existing tensor is replaced.
        """
        assert tensor is not None, f"Tensor {key} cannot be None."
        assert isinstance(tensor, torch.Tensor), f"Tensor {key} must be a torch tensor."
        self._assert_metadata()  # Ensure metadata integrity
        assert tensor.shape[0] == self.data["metadata"]["num_steps"], (
            f"Tensor {key} must have the same number of elements as the current dataset which is {self.data['metadata']['num_steps']}."
        )
        if not replace:
            assert key not in self.data.keys(), (
                f"Tensor {key} already exists in the dataset. Use replace=True to replace the tensor or call '_extend_tensor' to extend existing data."
            )

        tensor = self.normalize(tensor, key=key, only_create_normalizer=True, overwrite_normalizer=True)

        self.data[key] = tensor

    def add_data(
        self,
        data: dict,
        extend: bool = True,
        replace: bool = False,
        overwrite_saved_dataset: bool = False,
    ):
        """Appends data and corresponding metadata to the dataset.

        Args:
            data (dict): Dictionary containing key-value pairs of data/metadata to be added to the dataset.
            extend (bool): If True, the data is appended to the existing tensors. If False, a new tensor is added to the dataset.
            replace (bool): If True, the existing tensor is replaced. If False, the new tensor is appended to the dataset.
            overwrite_saved_dataset (bool): If True, the dataset is saved after adding the data.

        Options:
        - Either add a new tensor to the dataset:
            - Then the tensor must have the same number of elements as the current dataset
        - Or append new data to the existing tensors. Then a metadata dict must be included that has the required keys for the new data.
        """
        self._assert_dataset_loaded()
        assert isinstance(data, dict), "Data must be a dictionary."

        if extend:
            assert "metadata" in data.keys(), "If extend is True, a metadata dict must be included."
            self._check_and_update_metadata(data["metadata"], update_metadata=True)
            for key, tensor in data.items():
                if key == "metadata":
                    continue
                if tensor is not None:
                    self._extend_tensor(tensor, key=key)
        else:
            for key, tensor in data.keys():
                if key == "metadata":
                    continue
                self._add_tensor(tensor=tensor, key=key, replace=replace)
        # Save metadata, tensors, and normalizer
        if overwrite_saved_dataset:
            self._save_dataset()

    def normalize(
        self,
        tensor: torch.Tensor,
        key: str,
        only_if_normalizer_exists: bool = False,
        only_create_normalizer: bool = False,
        overwrite_normalizer: bool = False,
    ):
        """Interface for normalizing a tensor:
        - Converts the tensor to the correct shape
        - Either:
            - Normalizes the tensor using the normalizer stored in the dataset or by creating a new one.
            - Or, creates a new normalizer without normalizing the tensor.
        - Reconverts the tensor to the original shape
        - Returns the normalized tensor.

        Args:
            tensor (torch.Tensor): Tensor to be normalized.
            key (str): Key to store the tensor in the dataset.
            only_if_normalizer_exists (bool): If True, the tensor is not normalized if the normalizer does not exist.
            only_create_normalizer (bool): If True, the tensor is not normalized but the normalizer created.
        """
        assert isinstance(tensor, torch.Tensor), f"Tensor {key} must be a torch tensor."
        assert isinstance(key, str), f"Key {key} must be a string."

        # if self.normalize_tensors.get(key, False) is False and not only_create_normalizer:
        #     return tensor

        original_shape = tensor.shape

        if key == "rgb_images":
            return self._normalize_rgb_images(tensor, only_shape=only_create_normalizer)

        if only_create_normalizer:
            # The normalizer is fitted on the calibration set, so we need to filter the tensor
            start_indices, end_indices, _ = self._filter_start_end_episode_indices(
                subset="calibration", during_init=True
            )
            slices = self._get_slices_from_indices(start_indices, end_indices)
        else:
            slices = None
        if key == "obs_embeddings":
            tensor = tensor.view(tensor.shape[0], -1)
        elif key == "states":
            tensor = tensor.view(tensor.shape[0], -1)
        elif key == "actions" or key == "action_preds":
            tensor = tensor.view(-1, tensor.shape[-1])
        else:
            tensor = tensor.view(tensor.shape[0], -1)
            Warning(f"Tensor {key} is not supported for normalization.")

        # Normalize the tensor
        tensor = self._normalize_tensor(
            tensor,
            key=key,
            only_if_normalizer_exists=only_if_normalizer_exists,
            only_create_normalizer=only_create_normalizer,
            overwrite=overwrite_normalizer,
            slices=slices,
        )
        # Revert to the original shape
        tensor = tensor.view(original_shape)

        return tensor

    def _create_normalizer(self, normalizer_key: str, data: torch.Tensor, slices: Optional[Union[list, slice]] = None):
        if slices is not None:
            data = data[slices]
        normalizer_object = LinearNormalizer()
        normalizer_object.fit(
            data=data,
            last_n_dims=1,
            mode=self.normalize_tensors["mode"],
            range_eps=self.normalize_tensors["range_eps"],
            fit_offset=self.normalize_tensors["fit_offset"],
            output_min=self.normalize_tensors["limits"][0],
            output_max=self.normalize_tensors["limits"][1],
        )
        self.normalizer[normalizer_key] = normalizer_object
        return

    def _normalize_tensor(
        self,
        tensor: torch.Tensor,
        key: str,
        normalizer_key: str = None,
        only_if_normalizer_exists: bool = False,
        only_create_normalizer: bool = False,
        overwrite: bool = False,
        slices: Optional[Union[list, slice]] = None,
    ):
        """Normalizes a tensor using the normalizer stored in the dataset or by creating a new one.
        Args:
            tensor (torch.Tensor): Tensor to be normalized.
            key (str): Key to store the tensor in the dataset.
            normalizer_key (str): Key to store the normalizer in the dataset.
            only_if_normalizer_exists (bool): If True, the tensor is not normalized if the normalizer does not exist.
        """
        assert isinstance(tensor, torch.Tensor), f"Tensor {key} must be a torch tensor."
        assert isinstance(key, str), f"Key {key} must be a string."
        if normalizer_key is None:
            normalizer_key = key
        if not hasattr(self, "normalizer"):
            self.normalizer: dict = {}
        if only_create_normalizer:
            self._create_normalizer(normalizer_key=normalizer_key, data=tensor, slices=slices)
            return tensor
        if normalizer_key not in self.normalizer.keys() or overwrite:
            if only_if_normalizer_exists:
                Warning(f"Requested tensor {key} is not normalized because the normalizer does not exist.")
                return tensor
            self._create_normalizer(normalizer_key=normalizer_key, data=tensor, slices=slices)

        # if key == "obs_embeddings":
        #     tensor = self.normalizer[normalizer_key]["dpie"].normalize(tensor)
        if key != "rgb_images":
            tensor = self.normalizer[normalizer_key].normalize(tensor)

        return tensor

    def _normalize_rgb_images(self, rgb_images: torch.Tensor, only_shape: bool = False):
        """
        Processes RGB images to ensure they are in (N, C, H, W) format and normalized to [0, 1].

        Args:
            rgb_images (torch.Tensor): Tensor containing a single RGB image or a batch of RGB images.
                - For a single image: Shape should be (H, W, C) or (C, H, W).
                - For a batch of images: Shape should be (N, H, W, C) or (N, C, H, W).

        Returns:
            torch.Tensor: Processed tensor in (N, C, H, W) format with pixel values normalized to [0, 1].
        """
        # Ensure the input is a 4D tensor (batch of images)
        if rgb_images.ndim == 3:  # Single image
            rgb_images = rgb_images.unsqueeze(0)  # Add batch dimension

        # Check if the last dimension is the channel dimension (H, W, C) or (N, H, W, C)
        if rgb_images.shape[-1] == 3:  # Convert (N, H, W, C) to (N, C, H, W)
            rgb_images = rgb_images.permute(0, 3, 1, 2)

        # Normalize pixel values to [0, 1] if necessary
        if not only_shape and rgb_images.max() > 1.1:
            rgb_images = rgb_images / 255.0

        return rgb_images

    def load_dataset(
        self,
        load_dir: str = None,
        optional_tensors_required: list = [],
        weights_only: bool = True,
    ):
        """Loads a dataset from a directory, including the dataset metadata and the tensors specified in the required_tensors list."""
        self.dataset_loaded = True
        tensors_required = list(set(optional_tensors_required + self.required_tensors))

        if load_dir is None:
            load_dir = self.save_dir
        if not self._dataset_exists(save_dir=load_dir, tensors_required=tensors_required):
            self.dataset_loaded = False
            return

        # Load tensors and metadata
        data = {}
        data["metadata"] = load_data(load_dirs=load_dir, keywords="metadata", data_types="pkl")
        for tensor_keyword in tensors_required:
            tensor_keyword = (
                tensor_keyword + ".pt"
                if (not tensor_keyword.endswith(".pt") or not tensor_keyword.endswith("."))
                else tensor_keyword
            )
            tensor = load_data(
                load_dirs=load_dir,
                keywords=tensor_keyword,
                data_types="pt",
                weights_only=weights_only,
                error_if_not_found=True,
            )
            data[tensor_keyword] = tensor
        self.data = data
        self.normalizer = load_data(
            load_dirs=load_dir,
            keywords="normalizer",
            data_types="pkl",
            error_if_not_found=False,
        )
        if not self.normalizer:
            self.normalizer = {}

    def _save_dataset(self, save_dir: str = None):
        # Save normalizers
        if save_dir is None:
            save_dir = self.save_dir
        if self.normalizer:
            save_data(save_dir=save_dir, data=self.normalizer, filenames="normalizer", data_types="pkl", overwrite=True)
        # Save metadata
        save_data(
            save_dir=save_dir,
            data=self.data["metadata"],
            filenames="metadata",
            data_types="pkl",
            overwrite=True,
        )
        # Save tensors
        for tensor_keyword, tensor in self.data.items():
            if tensor_keyword == "metadata":
                continue
            save_data(
                save_dir=save_dir,
                data=tensor,
                filenames=tensor_keyword,
                data_types="pt",
                overwrite=True,
            )
        return

    def __getitem__(self, index) -> Union[dict, torch.Tensor]:
        """Functionality dependent on type of index:
        - If "str": Assumes index is a key and returns the complete dataset entry.
        - If "int": Returns a dictionary with the data for a specific step.
        - If "slice": Returns a dictionary with the data for a slice of steps."""
        if isinstance(index, str):
            self._assert_dataset_entries(dataset_entries=[index])
            return self.data[index]
        if isinstance(index, slice):
            return {key: tensor[index.start : index.stop] for key, tensor in self.data.items() if key != "metadata"}
        elif isinstance(index, int):
            # Return data for a specific step
            return {key: tensor[index] for key, tensor in self.data.items() if key != "metadata"}
        else:
            raise TypeError("Index must be an str, int, or slice.")

    def __len__(self) -> int:
        self._assert_metadata()
        return self.data["metadata"]["num_steps"]

    def num_rollouts(self) -> int:
        return self.data["metadata"]["num_rollouts"]

    def get_tensor_shape(self, tensor: str, **kwargs) -> torch.Size:
        """Returns the shape of a tensor. If the tensor is not in the dataset, returns torch.Size([0, 0, 0]).
        Optionally filters the actions and action_preds tensors to include only the specified actions (specified in kwargs)."""
        if tensor not in self.data:
            return torch.Size([0, 0, 0])
        self._assert_tensor(tensor)
        if kwargs.get("filter_actions", False) and tensor in ["actions", "action_preds"]:
            # Filter the actions and action_preds tensors to include only the specified actions
            action_indices = self._get_action_slices(**kwargs)
            tensor = self.data[tensor][..., action_indices]
        else:
            tensor = self.data[tensor]
        return tensor.shape

    def _filter_start_end_episode_indices(
        self, subset: str = "all", subsubset: str = "all", during_init=False
    ) -> tuple[np.ndarray, np.ndarray]:
        """Returns the episode start and end indices and successlabels given a subset and subsubset."""
        if not during_init:
            self._assert_dataset_loaded()
            self._assert_subset(subset=subset)
            self._assert_subsubset(subsubset=subsubset)
            self._assert_metadata()
        metadata = self.data["metadata"]
        start_indices = metadata["episode_start_indices"]
        end_indices = metadata["episode_end_indices"]

        mask = np.ones(len(start_indices), dtype=bool)
        mask = mask & metadata[subset + "_rollout_labels"] if subset != "all" else mask
        mask = mask & metadata[subsubset + "_rollout_labels"] if subsubset != "all" else mask

        start_indices = start_indices[mask]
        end_indices = end_indices[mask]
        success_labels = metadata["successful_rollout_labels"][mask]

        return start_indices, end_indices, success_labels

    def _get_slices_from_indices(self, start_indices: np.ndarray, end_indices: np.ndarray) -> np.ndarray:
        """Returns a numpy array with indices between numpy arrays of start and end indices."""
        if len(start_indices) != len(end_indices):
            raise ValueError("start_indices and end_indices must have the same length.")

        # Generate slices for each start and end index pair
        all_indices = np.concatenate([np.arange(start, end) for start, end in zip(start_indices, end_indices)])
        return all_indices

    def _check_normalize_tensors(self, normalize_tensors: Union[bool, dict]):
        """Checks if the normalize_tensors argument is valid. If it is a bool, it is converted to a dictionary.
        If it is a dictionary, it updates self.normalize_tensors with the entries."""
        # Default case
        if normalize_tensors is None or not normalize_tensors:
            return self.normalize_tensors.copy()
        # If normalize tensors is a bool use that values for all tensors
        elif isinstance(normalize_tensors, bool):
            normalize_tensors = {key: normalize_tensors for key in self.required_tensors + self.optional_tensors}
            for key in self.normalize_tensors.keys():
                if key not in normalize_tensors.keys() and key is not None:
                    normalize_tensors[key] = self.normalize_tensors.get(key, False)
        # If normalize tensors is a dict or dict config update the default values
        elif isinstance(normalize_tensors, dict) or isinstance(normalize_tensors, DictConfig):
            for key in list(set(self.required_tensors + self.optional_tensors + list(self.normalize_tensors.keys()))):
                if key not in normalize_tensors.keys() and key is not None:
                    normalize_tensors[key] = self.normalize_tensors.get(key, False)
        else:
            raise ValueError(f"Type of normalize_tensors argument {type(normalize_tensors)} is not supported.")
        return normalize_tensors

    def iterate_episodes(
        self,
        subset: str = "all",
        subsubset: Optional[str] = "all",
        required_tensors: Union[str, list] = "all",
        optional_tensors: Optional[list] = [],
        required_actions: Union[str, list] = "all",
        optional_actions: Optional[Union[str, list]] = [],
        with_success_labels: bool = False,
        normalize_tensors: Union[bool, dict] = None,
        history: int = 0,
        **kwargs: Optional[dict],
    ):
        """Iterate over episodes in the dataset. Returns the data for the specified subset of episodes and the specified tensors.

        Args:
            subset (str): The subset of episodes to iterate over. Options are "all", "calibration", "test", "successful", "failed".
            subsubset (str): The subsubset (rollout_subtypes) of episodes to iterate over. Options are "id" and "ood".
            required_tensors (list): List of tensors to include in the output. Default is "all", which includes all tensors.
            optional_tensors (list): List of optional tensors to include in the output. These tensors are included if they exist in the dataset.
            with_success_labels (bool): If True, the success labels are included in the output.
            normalize_tensors: If True, all tensors are returned in normalized form. If False, all tensors are returned in unnormalized form. If None, the tensors are returned according to self.normalize_tensors. If a list, the tensors are returned in the specified form.
        """
        required_tensors, required_actions, optional_tensors, optional_actions = ensure_list(
            required_tensors, required_actions, optional_tensors, optional_actions
        )
        normalize_tensors = self._check_normalize_tensors(normalize_tensors)

        self._assert_dataset_loaded()
        self._assert_subset(subset=subset)
        self._assert_subsubset(subsubset=subsubset)
        self._assert_dataset_entries(dataset_entries=required_tensors)
        start_indices, end_indices, success_labels = self._filter_start_end_episode_indices(
            subset=subset, subsubset=subsubset
        )
        episode_lengths = end_indices - start_indices
        action_indices = self._get_action_slices(required_actions=required_actions, optional_actions=optional_actions)
        for i in range(len(start_indices)):
            start = start_indices[i].item()
            end = end_indices[i].item()
            rollout_data = {
                key: tensor[start:end]
                for key, tensor in self.data.items()
                if (key in required_tensors or required_tensors == "all" or key in optional_tensors)
                and key != "metadata"
            }
            for key in rollout_data.keys():
                if normalize_tensors[key]:
                    rollout_data[key] = self.normalize(rollout_data[key], key=key)
                if "action" in key:
                    # Filter the actions and action_preds tensors to include only the specified actions
                    rollout_data[key] = rollout_data[key][..., action_indices]

                if history > 1:
                    rollout_data[key] = self._augment_tensor_with_history(
                        tensor=rollout_data[key],
                        history=history,
                        episode_lengths=[episode_lengths[i]],
                    )

            if with_success_labels:
                rollout_data["successful"] = success_labels[i].item()
            yield rollout_data

    def get_subset(
        self,
        subset: str = "all",
        subsubset: Optional[str] = "all",
        required_tensors: Union[str, list] = "all",
        optional_tensors: Optional[list] = [],
        required_actions: Union[str, list] = "all",
        optional_actions: Optional[Union[str, list]] = [],
        with_success_labels: bool = False,
        normalize_tensors: Union[bool, dict] = None,
        with_metadata: bool = False,
        history: int = 0,
        **kwargs: Optional[dict],
    ) -> Union[Dict[str, Any], list]:
        """Returns the data for the specified subset of episodes and the specified tensors.

        Args:
            subset (str,list): The subset of episodes to iterate over. Options are "all", "calibration", "test", "successful", "failed".
            required_tensors (list): List of tensors to include in the output. Default is "all", which includes all mandatory tensors.
            return_as_list (bool): If True, returns the data as a list. If False, returns the data as a dictionary.
            optional_tensors (list): List of optional tensors to include in the output. These tensors are included if they exist in the dataset.
            required_actions (list): List of actions to filter the 'actions' and 'action_preds' tensors. Default is "all", which includes all actions.
        """
        required_tensors, required_actions, optional_tensors, optional_actions = ensure_list(
            required_tensors, required_actions, optional_tensors, optional_actions
        )
        normalize_tensors = self._check_normalize_tensors(normalize_tensors)

        checked_tensors = [tensor for tensor in required_tensors if tensor not in optional_tensors]
        self._assert_subset(subset=subset)
        self._assert_subsubset(subsubset=subsubset)
        self._assert_dataset_entries(dataset_entries=checked_tensors)
        self._assert_dataset_loaded()
        start_indices, end_indices, success_labels = self._filter_start_end_episode_indices(
            subset=subset, subsubset=subsubset
        )
        slices = self._get_slices_from_indices(start_indices, end_indices)

        action_indices = self._get_action_slices(required_actions=required_actions, optional_actions=optional_actions)
        data = {}
        for key, tensor in self.data.items():
            if key == "metadata":
                if with_metadata:
                    Warning("The full metadata is included in the data dict, not just the subset.")
                    data[key] = tensor.copy()
                continue
            elif key in required_tensors or key in optional_tensors or required_tensors == "all":
                # tensor = tensor[slices].clone()
                tensor = tensor[slices]
                if normalize_tensors[key]:
                    tensor = self.normalize(tensor, key=key, only_if_normalizer_exists=True)
                if key == "actions" or key == "action_preds":
                    # Filter the actions and action_preds tensors to include only the specified actions
                    tensor = tensor[..., action_indices]
                # Replace NaN values with 0.0
                if torch.isnan(tensor).any():
                    print(f"Tensor {key} in dataset class contains NaN values. Replacing them with 0.0.")
                    tensor = torch.nan_to_num(tensor, nan=0.0)
                data[key] = tensor

            if history > 1:
                data[key] = self._augment_tensor_with_history(
                    tensor=data[key],
                    history=history,
                    episode_lengths=end_indices - start_indices,
                )

        if with_success_labels:
            data["successful"] = success_labels

        if kwargs.get("return_episode_lengths", False):
            data["episode_lengths"] = end_indices - start_indices
        return data

    def _augment_tensor_with_history(
        self,
        tensor: torch.Tensor,
        history: int,
        episode_lengths: np.ndarray = None,
    ) -> torch.Tensor:
        """Augments a tensor with history information. Translates a tensor of shape (N, ...) to (N, history, ...), where history is the number of steps to include in the history.
        The history is ordered from oldest to current step, i.e., the first step in the history is the oldest step and the last step in the history is the current step.

        Args:
            tensor (torch.Tensor): The tensor to augment.
            history (int): The number of steps to include in the history including the current step.
            episode_lengths (np.ndarray): The lengths of each episode in the tensor.
        Returns:
            torch.Tensor: The augmented tensor.
        """
        # Create a new tensor with the new history dimension as the second dimension
        augmented_tensor = torch.zeros(
            (tensor.shape[0], history, *tensor.shape[1:]),
            dtype=tensor.dtype,
            device=tensor.device,
        )

        if episode_lengths is None:
            episode_lengths = np.array([tensor.shape[0]])

        # Fill in the augmented tensor with the original tensor and its history
        episode_start = 0
        for episode_length in episode_lengths:
            episode_end = episode_start + episode_length
            for i in range(episode_start, episode_end):
                start = max(episode_start, i - history + 1)
                num_history = i - start + 1
                augmented_tensor[i, history - num_history :] = tensor[start : i + 1]
            episode_start = episode_end

        return augmented_tensor

    def _get_action_slices(
        self, required_actions: Union[str, list], optional_actions: Optional[Union[str, list]] = [], **kwargs
    ):
        """
        Args:
            required_actions (list): List of actions to include in the output. Default is "all", which includes all actions.
            optional_actions (list): List of optional actions to include in the output. These actions are included if they exist in the dataset.
        Returns:
            action_indices (np.ndarray): Array of action indices to include in the output.
        """

        if "actions" not in self.data.keys() and "action_preds" not in self.data.keys():
            return np.array([])

        required_actions, optional_actions = ensure_list(required_actions, optional_actions)
        if "all" in required_actions:
            return np.arange(self.data["metadata"]["actions"]["action_dim"])
        action_indices = []
        for action in required_actions + optional_actions:
            if (
                action in self.data["metadata"]["actions"]["action_mapping"]
                and self.data["metadata"]["actions"]["action_mapping"][action] is not None
            ):
                action_indices.extend(self.data["metadata"]["actions"]["action_mapping"][action])
            elif action in required_actions:
                available_actions = list(self.data['metadata']['actions']['action_mapping'].keys())
                raise ValueError(
                    f"Required action '{action}' not found in action mapping. "
                    f"Available actions are: {available_actions}."
                )
        return np.array(action_indices)

    def __iter__(self):
        """Iterates over all steps in the dataset and returns a dictionary with the current step for all tensors in the dataset."""
        for i in range(self.data["metadata"]["num_steps"]):
            yield {key: tensor[i] for key, tensor in self.data.items() if key != "metadata"}

    def get_metadata(self):
        """Get metadata from the dataset."""
        self._assert_metadata()
        return self.data["metadata"]

    def create_dataloader(self, keys, batch_size, shuffle=True):
        """Create a DataLoader for the specified keys in the dataset."""
        selected_tensors = [self.data[key] for key in keys if key in self.data.keys() and key != "metadata"]
        combined_dataset = torch.utils.data.TensorDataset(*selected_tensors)
        return DataLoader(combined_dataset, batch_size=batch_size, shuffle=shuffle)

    def get_rollout_subtypes(self, subset: str = "all", subsubset: str = "all", reduce: bool = True, count=False):
        """Get the rollout subtypes for the specified subset.

        Returns a numpy array of booleans indicating the rollout subtypes (subsubsets) for the specified subset.
        Args:
            subset (str, list): The subset(s) of episodes to return. Options are (individual or as list) "all", "calibration", "test", "successful", "failed".
            subsubset (str): The subsubset of episodes to filter. Options are "id" and "ood".
            reduce (bool): If False, returns the logical combination of subset and subsubset, i.e., an array of length "num_rollouts". If True, returns the subsubset filtered by the subset, i.e., an array with length sum(subset == True).
            count (bool): If True, returns the number of rollouts in the specified subset. If False, returns a boolean mask indicating the rollouts in the specified subset.
        """
        self._assert_dataset_loaded()
        mask = self.get_rollout_types(subset=subset, count=False)
        self._assert_subsubset(subsubset=subsubset)
        metadata = self.data["metadata"]

        if not reduce:
            mask = metadata[subsubset + "_rollout_labels"] & mask
        else:
            mask = metadata[subsubset + "_rollout_labels"][mask]

        if count:
            return np.sum(mask)
        return mask

    def get_rollout_types(self, subset: Union[str, list] = "all", count=False, reduce_mask_to: str = []):
        """Get the rollout types for the specified subset.

        Returns a numpy array of booleans indicating the rollout types for the specified subset.
        Args:
            subset (str, list): The subset(s) of episodes to return. Options are (individual or as list) "all", "calibration", "test", "successful", "failed".
            count (bool): If True, returns the number of rollouts in the specified subset. If False, returns a boolean mask indicating the rollouts in the specified subset.
            reduce_mask_to (str, list): If specified, the mask is reduced to the specified subset.
        """
        subset, reduce_mask_to = ensure_list(subset, reduce_mask_to)
        self._assert_dataset_loaded()
        for s in subset + reduce_mask_to:
            self._assert_subset(subset=s)
        metadata = self.data["metadata"]
        mask = np.ones(metadata["num_rollouts"], dtype=bool)
        for s in subset:
            mask = mask & metadata[s + "_rollout_labels"] if s != "all" else mask
        if reduce_mask_to:
            assert len(reduce_mask_to) == 1, "reduce_mask_to must be a str or a list of length 1."
            for s in reduce_mask_to:
                mask = mask[metadata[s + "_rollout_labels"]] if s != "all" else mask
        if count:
            return np.sum(mask)
        return mask

    def _init_rollout_types(self, calibration_set_size=0.2, shuffle=False):
        """Inits the calibration and test rollout indices. Optionally shuffles the rollout indices..
        Args:
            calibration_set_size (float): The size of the calibration set as a fraction of the total rollouts.
        """
        assert 0.01 < calibration_set_size < 0.5, "calibration_set_size must be between 0.01 and 0.5."
        self._assert_dataset_loaded()
        metadata = self.data["metadata"]
        num_rollouts = metadata["num_rollouts"]
        successful_rollout_labels = metadata["successful_rollout_labels"]
        failed_rollout_labels = metadata["failed_rollout_labels"]

        num_calibration_rollouts = min(np.sum(successful_rollout_labels) // 2, int(num_rollouts * calibration_set_size))
        num_test_rollouts = num_rollouts - num_calibration_rollouts
        failed_ratio = np.sum(failed_rollout_labels) / num_test_rollouts

        successful_indices = np.where(successful_rollout_labels)[0]

        if failed_ratio > 0.7 or failed_ratio < 0.3:
            Warning(
                f"Ratio of failed rollouts in the test dataset is {failed_ratio}. This might lead to biased results. Consider using a different dataset. Increasing/decreasing the calibration_set_size will increase/decrease the failed_ratio. (current calibration_set_size: {calibration_set_size})."
            )
        if shuffle:
            # Randomly select calibration rollouts from successful rollouts
            calibration_indices = np.random.choice(successful_indices, num_calibration_rollouts, replace=False)

        else:
            calibration_indices = successful_indices[:num_calibration_rollouts]

        calibration_rollout_labels = np.zeros(num_rollouts, dtype=bool)
        calibration_rollout_labels[calibration_indices] = True
        test_rollout_labels = ~calibration_rollout_labels

        self.data["metadata"]["calibration_rollout_labels"] = calibration_rollout_labels
        self.data["metadata"]["test_rollout_labels"] = test_rollout_labels

    def shuffle_rollout_types(self, calibration_set_size=0.2):
        """Shuffle the calibration and test rollouts.

        The calibration dataset is randomly selected from the successful rollouts, while the test dataset is selected from the remaining successful rollouts and failed rollouts.

        Args:
            calibration_set_size (float): The size of the calibration set as a fraction of the total rollouts.
        """
        self._init_rollout_types(calibration_set_size=calibration_set_size, shuffle=True)

    def _assert_subset(self, subset: str):
        """Check if the subset is valid."""
        assert subset in self.allowed_subsets, (
            f"Subset {subset} is not valid. Allowed subsets are: {self.allowed_subsets}"
        )

    def _assert_subsubset(self, subsubset: str):
        """Check if the subsubset is valid."""
        assert subsubset in self.allowed_subsubsets, (
            f"Subsubset {subsubset} is not valid. Allowed subsubsets are: {self.allowed_subsubsets}"
        )

    def _assert_dataset_loaded(self):
        """Check if the dataset is loaded."""
        assert self.dataset_loaded, "Dataset is not initialized yet. Either load or initialize the dataset."
        self._assert_dataset_entries(dataset_entries="all")

    def _assert_dataset_entries(self, dataset_entries: Union[str, list]):
        """Check if the tensor exists in the dataset."""
        dataset_entries = ensure_list(dataset_entries)
        assert isinstance(self.data, dict), "Dataset is not a dictionary."
        assert len(self.data) > 0, "Dataset is empty."

        for entry in dataset_entries:
            if entry == "all":
                self._assert_metadata()
                for tensor in self.required_tensors:
                    self._assert_tensor(tensor)
                return
            else:
                if entry == "metadata":
                    self._assert_metadata()
                else:
                    self._assert_tensor(entry)

    def _assert_metadata(self):
        """Check if the metadata exists in the dataset."""
        assert "metadata" in self.data, "Metadata does not exist in the dataset."
        assert self.data["metadata"] is not None, "Metadata is None."
        assert isinstance(self.data["metadata"], dict), "Metadata is not a dictionary."
        assert all(key in self.data["metadata"] for key in self.required_metadata_keys), (
            "Metadata does not contain the required keys."
        )

    def _assert_tensor(self, tensor: str, shape: tuple = None):
        """Check if the tensor exists in the dataset and has the expected shape."""
        assert tensor in self.data, f"Tensor {tensor} does not exist in the dataset."
        assert self.data[tensor] is not None, f"Tensor {tensor} is None."
        assert isinstance(self.data[tensor], torch.Tensor), f"Tensor {tensor} is not a torch tensor."
        if shape is not None:
            assert self.data[tensor].shape[1:] == shape, f"Tensor {tensor} does not have the expected shape {shape}."

    def print_dataset_summary(self):
        """Prints a summary of the dataset, including the number of rollouts, steps, and the available tensors."""
        print("Dataset summary:")
        print(f"Number of rollouts: {self.data['metadata']['num_rollouts']}")
        print(f"Num Calibration rollouts: {np.sum(self.data['metadata']['calibration_rollout_labels'])}")
        print(f"Num Test rollouts: {np.sum(self.data['metadata']['test_rollout_labels'])}")
        print(f"Total Number of steps: {self.data['metadata']['num_steps']}")
        print("Available tensors and shapes:")
        for tensor in self.data["metadata"]["available_tensors"]:
            print(f"{tensor}: {self.data[tensor].shape}")
        print("Normalizer Entries:")
        for normalizer in list(self.normalizer.keys()):
            print(f"{normalizer}")

    def get_dataset_statistics(self, print = False, return_df = True):
        """Returns the statistics of the dataset as a panda dataframe and/or prints it."""
        self._assert_metadata()

        metadata = self.data["metadata"]

        # Prepare the statistics dictionary
        stats = {
            "Number of rollouts": metadata["num_rollouts"],
            "Num Calibration rollouts": int(np.sum(metadata["calibration_rollout_labels"])),
            "Num Test rollouts": int(np.sum(metadata["test_rollout_labels"])),
            "Test OOD Rate": float(
                np.sum(metadata["ood_rollout_labels"] & metadata["test_rollout_labels"]) / np.sum(metadata["test_rollout_labels"])
            ),
            "Test ID Rate": float(
                np.sum(metadata["id_rollout_labels"] & metadata["test_rollout_labels"]) / np.sum(metadata["test_rollout_labels"])
            ),
            "Test Success Rate": float(
                np.sum(metadata["successful_rollout_labels"] & metadata["test_rollout_labels"]) / np.sum(metadata["test_rollout_labels"])
            ),
            "Success Rate ID": float(
                np.sum(metadata["successful_rollout_labels"] & metadata["id_rollout_labels"]) / np.sum(metadata["id_rollout_labels"])
            ),
            "Success Rate OOD": float(
                np.sum(metadata["successful_rollout_labels"] & metadata["ood_rollout_labels"]) / np.sum(metadata["ood_rollout_labels"])
            ),
            "Max Episode Length": int(max(metadata["episode_lengths"])),
            "Avg Episode Length": int(metadata["num_steps"] / metadata["num_rollouts"]),
        }
        for tensor in self.data.keys():
            if tensor != "metadata":
                stats[tensor] = str(list(self.data[tensor].shape[1:]))

        # # Print the statistics in YAML format
        # if print:
        #     print("Dataset Statistics:")
        #     print(OmegaConf.to_yaml(stats, resolve=True))
        if return_df:
            import pandas as pd
            stats_df = pd.DataFrame(stats, index=[0])
            return stats_df
