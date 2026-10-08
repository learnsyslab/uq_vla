"""This file contains shared utility functions for data management including saving/loading etc."""

import os
import pickle
import numpy as np
import pathlib
from typing import Optional, Union, Dict, Any, Tuple, Iterable, Set
import h5py
import torch
from shared_utils.utility_functions import ensure_list

supported_data_types = ["npy", "pkl", "h5py", "pt"]


class LazyTensorManager:
    """
    A class to manage lazy loading/saving of tensors.
    """

    def __init__(
        self,
        save_dir: str,
        key: str,
        batch_size: int = 1000,
        dtype=torch.float32,
        max_batches_in_cache: int = 1,
        max_bytes_per_batch: int = 2**30,
        element_size: int = None,
    ):
        """
        Args:
            save_dir (str): Directory where the tensors are saved.
            key (str): Key for the tensor.
            batch_size (int, optional): Size of each batch. Defaults to 1000.
            max_batches_in_cache (int, optional): Maximum number of batches to keep in cache. Defaults to 1.
            max_bytes (int, optional): Maximum size of a batch in bytes. Only applies when 'element_size' is given, then batch_size is calculated. Defaults to 2**30.
            element_size (int, optional): Size of each element in bytes, i.e., tensor.shape[1:].numel() * tensor.element_size().
        """
        if element_size is not None:
            batch_size = max_bytes_per_batch // element_size
        self.save_dir = save_dir
        self.key = key
        self.batch_size = batch_size
        self.max_batches_in_cache = max_batches_in_cache
        self.cache = {}  # Cache for loaded parts
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        if not os.path.exists(self._get_filepath(0)):
            self.num_batches = 0
        else:
            self.num_batches = len([f for f in os.listdir(save_dir) if f.startswith(f"{key}_part_")])

    def get_batch(self, batch_index):
        if batch_index in self.cache:
            return self.cache[batch_index]
        # Load batch from disk
        batch = torch.load(self._get_filepath(batch_index), weights_only=True)
        self.cache[batch_index] = batch
        # Evict old batches if cache size exceeds limit
        if len(self.cache) > self.max_batches_in_cache:
            self.cache.pop(next(iter(self.cache)))
        return batch

    def __getitem__(self, index):
        batch_index = index // self.batch_size
        batch_offset = index % self.batch_size
        batch = self.get_batch(batch_index)
        return batch[batch_offset]

    def __len__(self):
        return self.num_batches * self.batch_size

    def save_tensor_batch(self, batch_index, tensor, overwrite=True):
        os.makedirs(self.save_dir, exist_ok=True)
        filepath = self._get_filepath(batch_index)
        if os.path.exists(filepath) and not overwrite:
            raise FileExistsError(f"File {filepath} already exists. Use overwrite=True to replace it.")
        torch.save(tensor, filepath)

    def save_list_as_tensor(self, list: list[Union[np.ndarray, torch.Tensor]], save_dir):
        pass

    def save_full_tensor_in_batches(self, tensor, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        num_batches = (tensor.shape[0] + self.batch_size - 1) // self.batch_size
        for i in range(num_batches):
            batch = tensor[i * self.batch_size : (i + 1) * self.batch_size]
            torch.save(batch, self._get_filepath(i))

    def _get_filepath(self, batch_index):
        return os.path.join(self.save_dir, f"{self.key}_part_{batch_index}.pt")


def create_memmap_tensor(save_dir, filename, shape, dtype=torch.float32):
    """
    Create a memory-mapped tensor.

    Args:
        save_dir (str): Directory where the memory-mapped file will be created.
        filename (str): Name of the memory-mapped file.
        shape (tuple): Shape of the tensor.
        dtype (torch.dtype): Data type of the tensor (default: torch.float32).

    Returns:
        torch.Tensor: Memory-mapped tensor.
    """
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    if os.path.exists(filepath):
        os.remove(filepath)
    num_elements = torch.prod(torch.tensor(shape))
    storage = torch.FloatStorage.from_file(filepath, shared=True, size=num_elements)
    return torch.tensor(storage, dtype=dtype).reshape(shape)


def append_to_memmap_tensor(filepath, tensor: torch.Tensor, new_tensor: torch.Tensor):
    """
    Append new data to a memory-mapped tensor.

    Args:
        filepath (str): Path to the memory-mapped file.
        tensor (torch.Tensor): Existing memory-mapped tensor.
        new_data (torch.Tensor): New data to append.

    Returns:
        torch.Tensor: Updated memory-mapped tensor.
    """
    # Ensure new_data has the same shape except for the first dimension
    assert os.path.exists(filepath), f"File {filepath} does not exist."
    assert new_tensor.shape[1:] == tensor.shape[1:], (
        f"Shape mismatch between new tensor with shape {new_tensor.shape[1:]} and old tensor with shape {tensor.shape[1:]}. "
    )

    # Calculate the new size
    current_size = tensor.shape[0]
    new_size = current_size + new_tensor.shape[0]

    # Resize the file
    element_size = tensor.element_size()  # Size of each element in bytes
    assert element_size == new_tensor.element_size(), (
        f"Element size mismatch between new tensor with size {new_tensor.element_size()} and old tensor with size {element_size}."
    )
    with open(filepath, "ab") as f:
        f.truncate(new_size * tensor.shape[1:].numel() * element_size)

    # Update the storage and tensor
    storage = torch.FloatStorage.from_file(filepath, shared=True, size=new_size * tensor.shape[1:].numel())
    updated_tensor = torch.tensor(storage, dtype=tensor.dtype).reshape((new_size,) + tensor.shape[1:])

    # Copy the new data into the updated tensor
    updated_tensor[current_size:] = new_tensor

    return updated_tensor


def load_data(
    load_dirs: Union[str, list[str]],
    keywords: Optional[Union[str, list[str]]] = None,
    data_types: Optional[Union[str, list[str]]] = None,
    weights_only: bool = True,
    return_filenames: bool = False,
    error_if_not_found: bool = True,
) -> Union[Any, Tuple[Any, list[str]]]:
    """Load data from specified directories optionally filtering by keywords and data types.
    Args:
        load_dirs (Union[str, list[str]]): Directory or list of directories to load data from.
        keywords (Optional[Union[str, list[str]]], optional): Keywords to filter filenames. Defaults to None.
        data_types (Optional[Union[str, list[str]]], optional): Types of data to load. Defaults to None.
        weights_only (bool, optional): Whether to load only weights when loading torch tensors. Defaults to True.
    Returns:
        Any: Loaded data.
    """

    load_dirs, keywords, data_types = ensure_list(load_dirs, keywords, data_types)
    if data_types is not None:
        assert any(data_type in supported_data_types for data_type in data_types), (
            "data_types must be either 'npy', 'pkl', 'h5py' or 'pt'"
        )

    not_found = 0
    data = []
    all_filenames = []
    for load_dir in load_dirs:
        if not os.path.exists(load_dir) or not os.path.isdir(load_dir):
            print(f"Directory {load_dir} not found or is not a directory. Skipping...")
            not_found += 1
            continue
        filenames = _get_filenames(load_dir, keywords=keywords, data_types=data_types)
        if return_filenames:
            all_filenames.extend(filenames)
        for filename in filenames:
            file_path = os.path.join(load_dir, filename)
            if filename.endswith(".npy"):
                data.append(np.load(file_path, allow_pickle=True))
            elif filename.endswith(".pkl"):
                with open(file_path, "rb") as f:
                    data.append(pickle.load(f))
            elif filename.endswith(".h5py"):
                with h5py.File(file_path, "r") as f:
                    data.append({key: f[key][:] for key in f.keys()})
            elif filename.endswith(".pt"):
                data.append(torch.load(file_path, weights_only=weights_only))
    if not_found == len(load_dirs) and error_if_not_found:
        raise FileNotFoundError(f"Directories {load_dirs} not found or are not directories.")
    if len(data) == 0 and error_if_not_found:
        raise FileNotFoundError(f"No files found in {load_dirs} with keywords {keywords} and data types {data_types}.")
    if len(data) == 1:
        data = data[0]

    if return_filenames:
        return data, all_filenames
    return data


def save_data(
    save_dir: str,
    filenames: Union[str, list[str]],
    data: Union[Any, list[Any]],
    data_types: Optional[Union[str, list[str]]] = "pkl",
    overwrite: bool = False,
) -> None:
    """Save data to specified directories with specified filenames and datatypes.
    Args:
        save_dir (str): Directory to save data to.
        filenames (Union[str, list[str]]): Filename or list of filenames to save data as.
        data (Union[Any, list[Any]]): Data to save.
        data_types (Optional[Union[str, list[str]]], optional): Types of data to save. Defaults to None.
        overwrite (bool, optional): Whether to overwrite existing files. Defaults to False.
    """
    if isinstance(filenames, str):
        filenames = [filenames]
    if isinstance(data_types, str):
        data_types = [data_types]
    if not isinstance(data, list):
        data = [data]
    assert len(filenames) == len(data), "Number of filenames must match number of data items."
    assert any(data_type in supported_data_types for data_type in data_types), (
        "data_types must be either 'npy', 'pkl', 'h5py' or 'pt'"
    )
    if len(data_types) > 1:
        assert len(filenames) == len(data_types), (
            "Number of filenames must match number of data types when data_types is a list."
        )
    os.makedirs(save_dir, exist_ok=True)
    for i in range(len(filenames)):
        if len(data_types) == 1:
            data_type = data_types[0]
        else:
            data_type = data_types[i]
        if not filenames[i].endswith("." + data_type):
            filenames[i] = filenames[i] + "." + data_type
        # assert not filenames[i].endswith("." + data_type + "." + data_type), (
        #     "File saved with double extension. Please check the filenames."
        # )
        file_path = os.path.join(save_dir, filenames[i])
        if os.path.exists(file_path):
            if overwrite:
                os.remove(file_path)
            else:
                Warning(f"File {file_path} already exists. Use overwrite=True to replace it.")
                continue

        if file_path.endswith(".npy"):
            np.save(file_path, data[i], allow_pickle=True)
        elif file_path.endswith(".pkl"):
            with open(file_path, "wb") as f:
                pickle.dump(data[i], f)
        elif file_path.endswith(".h5py"):
            with h5py.File(file_path, "w") as f:
                for key, value in data[i].items():
                    f.create_dataset(key, data=value)
        elif file_path.endswith(".pt"):
            torch.save(data[i], file_path)


def prune_rollout_steps_inplace(rollout_obj: Dict[str, Any], required_keys: Iterable[str]) -> Dict[str, Any]:
    required_keys: Set[str] = set(required_keys)

    steps = rollout_obj.get("rollout", [])
    for step in steps:
        for k in list(step.keys()):
            if k not in required_keys and k != "action_pred" and k != "obs_embedding":
                del step[k]
            elif k == "terminal_eval_times":
                step[k] = np.array([0.])
            elif "terminal_velocities" in k:
                step[k] = step[k][:1]
    return rollout_obj

def load_raw_rollouts(
    load_dirs: Union[str, list[str]],
    data_types: Union[str, list[str]] = ["npy", "pkl"],
    keywords: Optional[Union[str, list[str]]] = ["episode", "eps", "rollout"],
    searched_filenames: Optional[Union[str, list[str]]] = [],
    with_filename: bool = False,
    raise_error_if_not_found: bool = True,
    required_keys: list[str] | None = None,
) -> list[Dict[str, Any]]:
    """Load raw rollouts from a specified directory.

    Args:
        load_dir(s) (str, list): Directory(s) where the rollouts are stored.
        data_types (str, optional): Type of data to load, either "npy", "pkl" or "h5py". If None, it will be inferred from the file extension. Defaults to None.
        keywords (Optional[Union[str, list[str]]], optional): Keywords to filter filenames. If None, all files will be loaded. Defaults to "episode".
        searched filenames (Optional[Union[str, list[str]]], optional): Specific filenames to load. If None, all files in the directory that contain the keywords and have a supported datatype will be loaded. Defaults to None.
        with_filename (bool, optional): Whether to include file name in the output. Defaults to False.

    Returns:
        rollouts (dict[list, dict]): Dict with keys: fileinfo and rollout.
    """

    data = []
    load_dirs, searched_filenames, data_types, keywords = ensure_list(load_dirs, searched_filenames, data_types, keywords)

    assert any(data_type in supported_data_types for data_type in data_types), (
        f"data_types must be in {supported_data_types}, but got {data_types}"
    )
    not_found = 0
    for load_dir in load_dirs:
        if not os.path.exists(load_dir) or not os.path.isdir(load_dir):
            print(f"Directory {load_dir} not found or is not a directory. Skipping...")
            not_found += 1
            continue
        filenames = _get_filenames(
            load_dir, keywords=keywords, data_types=data_types, searched_filenames=searched_filenames
        )

        for filename in filenames:
            file_path = os.path.join(load_dir, filename)
            if filename.endswith(".npy"):
                rollout = np.load(file_path, allow_pickle=True)
            elif filename.endswith(".pkl"):
                with open(file_path, "rb") as f:
                    rollout = pickle.load(f)
            elif filename.endswith(".h5py"):
                with h5py.File(file_path, "r") as f:
                    rollout = {key: f[key][:] for key in f.keys()}
            if with_filename:
                rollout = {"fileinfo": filename, "rollout": rollout}
            if required_keys:
                prune_rollout_steps_inplace(rollout, required_keys)
            data.append(rollout)
    if not_found == len(load_dirs) and raise_error_if_not_found:
        raise FileNotFoundError(f"Directories {load_dirs} not found or are not directories.")
    if len(data) == 0 and raise_error_if_not_found:
        raise FileNotFoundError(
            f"No files found in {load_dirs} with keywords {keywords} or filenames {searched_filenames} and data types {data_types}."
        )
    return data


def _get_filenames(
    load_dir: str,
    keywords: Optional[Union[str, list[str]]] = None,
    data_types: Union[str, list[str]] = ["pkl"],
    searched_filenames=None,
) -> list[str]:
    """Get a list of filenames in the specified directory that contain the keywords.

    Args:
        load_dir (str): Directory where the files are stored.
        keywords (Optional[Union[str, list[str]]], optional): Keywords to filter filenames. If None, all files will be loaded. Defaults to None.
        data_types (Union[str, list[str]], optional): Type of data to search for. If None, it will be inferred from the file extension. Defaults to ["pkl"].
        searched_filenames (Optional[Union[str, list[str]]], optional): Specific filenames to load. Takes precedence over keywords and data_types.

    Returns:
        list[str]: List of filtered filenames.
    """
    all_filenames = sorted(os.listdir(load_dir))
    keywords = ensure_list(keywords)
    # assert len(all_filenames) > 0, f"No files found in {load_dir}"

    filenames = []
    for filename in all_filenames:
        if searched_filenames is not None:
            if filename in searched_filenames:
                filenames.append(filename)
            continue
        if any(filename.endswith(dt) for dt in data_types):
            if keywords is None:
                filenames.append(filename)
            else:
                if any(keyword in filename for keyword in keywords):
                    filenames.append(filename)

    return sorted(filenames)


def _get_file_path(save_load_dir: str, filename: str, mode: str, data_type: str, overwrite=False) -> str:
    """Get the full file path for saving data.

    Args:
        save_dir (str): Directory where the file will be saved.
        filename (str): Name of the file.
        mode (str): Mode of operation, either "save" or "load".
        data_type (str): Type of data to save, either "pkl" or "h5py".
        overwrite (bool, optional): Whether to overwrite existing files. Defaults to False.

    Returns:
        str: Full file path.
    """
    assert data_type in ["pkl", "h5py"], "data_type must be either 'pkl' or 'h5py'"
    assert mode in ["save", "load"], "mode must be either 'save' or 'load'"
    if mode == "save":
        os.makedirs(save_load_dir, exist_ok=True)
        file_path = os.path.join(save_load_dir, filename)
        if os.path.exists(file_path):
            if overwrite:
                os.remove(file_path)
            else:
                Warning(f"File {file_path} already exists. Use overwrite=True to replace it.")
                return
    elif mode == "load":
        if not os.path.exists(save_load_dir):
            raise FileNotFoundError(f"Directory {save_load_dir} not found")
        if filename is None:
            filenames = sorted(
                [filename for filename in os.listdir(save_load_dir) if filename.endswith("." + data_type)]
            )
            if len(filenames) == 0:
                raise FileNotFoundError(f"No {data_type} files found in {save_load_dir}")
            file_path = os.path.join(save_load_dir, filenames[-1])
        else:
            file_path = os.path.join(save_load_dir, filename)
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File {file_path} not found")
    return file_path


def save_rollouts_pkl(save_dir: str, filename: str, rollouts: dict, dataset_metadata: dict, overwrite=False):
    """Save rollouts and metadata to a pickle file.

    Args:
        save_dir (str): Directory where the pickle file will be saved.
        filename (str): Name of the pickle file.
        rollouts (dict): Dictionary of rollouts with metadata.
        dataset_metadata (dict): Global metadata for the dataset.
        overwrite (bool, optional): Whether to overwrite existing files. Defaults to False.
    """
    file_path = _get_file_path(save_dir, filename, mode="save", data_type="pkl", overwrite=overwrite)

    with open(file_path, "wb") as f:
        pickle.dump({"rollouts": rollouts, "metadata": dataset_metadata}, f)
        print(f"Rollouts saved to {file_path}")


def load_rollouts_pkl(load_dir: str, filename: Optional[str] = None) -> tuple[dict, dict]:
    """Load rollouts and metadata from a pickle file.

    Args:
        load_dir (str): Directory where the pickle file is located.
        filename (str, optional): Name of the pickle file. If None, the latest file in the directory will be used.

    Returns:
        tuple: A tuple containing:
            - metadata (dict): Global metadata.
            - rollouts (dict): Dictionary of rollouts with metadata.
    """
    file_path = _get_file_path(load_dir, filename, mode="load", data_type="pkl")

    with open(file_path, "rb") as f:
        data = pickle.load(f)

    return data["metadata"], data["rollouts"]


def save_rollouts_hdf5(save_dir: str, filename: str, rollouts: dict, dataset_metadata: dict, overwrite=False):
    """Save rollouts and metadata to an HDF5 file.

    Args:
        save_dir (str): Directory where the HDF5 file will be saved.
        filename (str): Name of the HDF5 file.
        rollouts (dict): Dictionary of rollouts with metadata.
        dataset_metadata (dict): Global metadata for the dataset.
        overwrite (bool, optional): Whether to overwrite existing files. Defaults to False.
    """
    file_path = _get_file_path(save_dir, filename, mode="save", data_type="h5py", overwrite=overwrite)
    with h5py.File(file_path, "w") as f:
        # Save global metadata
        metadata_group = f.create_group("metadata")
        for key, value in dataset_metadata.items():
            metadata_group.attrs[key] = value

        # Save rollouts
        rollouts_group = f.create_group("rollouts")
        for rollout_id, rollout_data in rollouts.items():
            rollout_group = rollouts_group.create_group(rollout_id)
            rollout_group.create_dataset("rollout", data=np.array(rollout_data["rollout"], dtype=object))
            for meta_key, meta_value in rollout_data["metadata"].items():
                rollout_group.attrs[meta_key] = meta_value


def load_rollouts_hdf5(load_dir: str, filename: str = None) -> tuple[dict, dict]:
    """Load rollouts and metadata from an HDF5 file.

    Args:
        load_dir (str): Directory where the HDF5 file is located.
        filename (str, optional): Name of the HDF5 file. If None, the latest file in the directory will be used.
    Returns:
        tuple: A tuple containing:
            - metadata (dict): Global metadata.
            - rollouts (dict): Dictionary of rollouts with metadata.
    """
    file_path = _get_file_path(load_dir, filename, mode="load", data_type="h5py")
    with h5py.File(file_path, "r") as f:
        # Load global metadata
        metadata = {key: value for key, value in f["metadata"].attrs.items()}

        # Load rollouts
        rollouts = {}
        for rollout_id in f["rollouts"]:
            rollout_group = f["rollouts"][rollout_id]
            rollout = rollout_group["rollout"][:]
            rollout_metadata = {key: value for key, value in rollout_group.attrs.items()}
            rollouts[rollout_id] = {"rollout": rollout, "metadata": rollout_metadata}

    return metadata, rollouts
