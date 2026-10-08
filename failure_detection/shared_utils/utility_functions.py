import torch
import numpy as np
from typing import List, Tuple, Union
from omegaconf.listconfig import ListConfig
from omegaconf import OmegaConf, DictConfig


def compare_dicts(dict1: Union[dict, DictConfig], dict2: Union[dict, DictConfig], ignore_keys: list = ["overwrite"]) -> bool:
    """
    Compares two dictionaries or DictConfigs for equality, including nested structures.

    Args:
        dict1 (dict): First dictionary or DictConfig.
        dict2 (dict): Second dictionary or DictConfig.
        ignore_keys (list): List of keys to ignore during comparison.

    Returns:
        bool: True if the dictionaries are equal, False otherwise.
    """
    # Convert DictConfig to plain dictionaries if necessary
    
    if isinstance(dict1, DictConfig):
        dict1 = OmegaConf.to_container(dict1, resolve=True)
    if isinstance(dict2, DictConfig):
        dict2 = OmegaConf.to_container(dict2, resolve=True)

    if type(dict1) is not type(dict2):
        return False

    # Handle dictionaries
    if isinstance(dict1, dict) and isinstance(dict2, dict):
        if ignore_keys:
            # Remove ignored keys from both dictionaries
            dict1 = {k: v for k, v in dict1.items() if k not in ignore_keys}
            dict2 = {k: v for k, v in dict2.items() if k not in ignore_keys}
        # Check if keys are the same
        if set(dict1.keys()) != set(dict2.keys()):
            return False
        # Recursively compare values
        for key in dict1:
            if not compare_dicts(dict1[key], dict2[key], ignore_keys):
                return False
        return True

    # Handle lists
    if isinstance(dict1, list) and isinstance(dict2, list):
        if len(dict1) != len(dict2):
            return False
        return all(compare_dicts(item1, item2) for item1, item2 in zip(dict1, dict2))

    # Handle tuples
    if isinstance(dict1, tuple) and isinstance(dict2, tuple):
        if len(dict1) != len(dict2):
            return False
        return all(compare_dicts(item1, item2) for item1, item2 in zip(dict1, dict2))

    # Handle sets
    if isinstance(dict1, set) and isinstance(dict2, set):
        return dict1 == dict2

    # Handle other types (fallback to direct comparison)
    return dict1 == dict2


def ensure_list(*args: Union[str, List[str]]) -> Tuple[List[str]]:
    """
    Ensures that each argument is a list. If an argument is not a list, it converts it into a list.

    Args:
        *args: Arbitrary number of arguments.

    Returns:
        A tuple containing the same number of arguments, where each argument is a list.
    """
    arg_list = []
    for arg in args:
        if isinstance(arg, list):
            arg_list.append(arg)
        elif isinstance(arg, ListConfig):
            arg_list.append(list(arg))
        else:
            arg_list.append([arg])
    results = tuple(arg_list)
    # results = tuple([arg if isinstance(arg, list) else [arg] for arg in args])
    return results[0] if len(results) == 1 else results


def get_required_tensors(methods, base_config_path, method_overrides=None):
    """Gets the required and optional tensors for the given methods by checking all relevant config files.
    Args:
        methods (list): List of methods to check.
        base_config_path (str): Path to the base config directory.
        method_overrides (dict, optional): `{method: {key: value}}` experiment-level overrides of the
            per-method eval configs (see shared_utils.training_data.apply_method_cfg_overrides); a
            `required_tensors` entry there replaces the one in the config file.
    Returns:
        tuple: A tuple containing two lists: required tensors and optional tensors.
    """
    from shared_utils.hydra_utils import load_config
    from shared_utils.training_data import apply_method_cfg_overrides

    methods = ensure_list(methods)
    required_tensors = []
    optional_tensors = []
    for method in methods:
        cfg = load_config(module="eval", filename=method, return_only_subdict=True, base_config_dir=base_config_path)
        if method_overrides:
            apply_method_cfg_overrides(cfg, {"method_overrides": method_overrides}, method_name=method)
        if cfg.get("required_tensors", None) is not None:
            required_tensors.extend(cfg.required_tensors)
        if cfg.get("optional_tensors", None) is not None:
            optional_tensors.extend(cfg.optional_tensors)
    # Remove duplicates
    required_tensors = list(set(required_tensors))
    optional_tensors = list(set(optional_tensors))
    # Remove None values
    required_tensors = [tensor for tensor in required_tensors if tensor is not None]
    optional_tensors = [tensor for tensor in optional_tensors if tensor is not None]
    # Remove tensors from optional tensors that are in required tensors
    optional_tensors = [tensor for tensor in optional_tensors if tensor not in required_tensors]
    return required_tensors, optional_tensors

def list_to_tensor(list_with_numpy_arrays: list) -> torch.Tensor:
    """Convert a list of numpy arrays into a single tensor by stacking along the first dimension.

    Handles varying sizes by padding smaller arrays to match the largest size, with padding added at the end.

    Args:
        list_with_numpy_arrays (list): List of numpy arrays to convert.

    Returns:
        torch.Tensor: A tensor containing the data from the input list, padded to the largest size.
    """

    if list_with_numpy_arrays is None or not list_with_numpy_arrays:
        return None

    # Convert all numpy arrays to PyTorch tensors
    tensor_list = [torch.tensor(arr, dtype=torch.float32) for arr in list_with_numpy_arrays]

    # Check if all tensors have the same shape
    shapes = [t.shape for t in tensor_list]
    if all(shape == shapes[0] for shape in shapes):
        return torch.stack(tensor_list, dim=0)

    # Determine the maximum shape along each dimension
    max_shape = torch.tensor(shapes).max(dim=0).values

    # Pad tensors to the maximum shape
    padded_tensors = []
    for t in tensor_list:
        # Calculate padding for each dimension (pad only at the end)
        padding = []
        for i in range(len(t.shape) - 1, -1, -1):  # Reverse order for PyTorch's padding format
            diff = max_shape[i] - t.shape[i]
            padding.extend([0, diff])  # Pad only at the end (right side)

        # Apply padding
        padded_tensors.append(torch.nn.functional.pad(t, pad=padding, mode="constant", value=0))

    # Stack the padded tensors along the first dimension
    return torch.stack(padded_tensors, dim=0)


def list_to_tensor_padded(
    list_with_numpy_arrays: List[np.ndarray],
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Convert list of numpy arrays to a single padded tensor (N, *max_shape).
    Pads at the end of each dimension with zeros.

    Peak memory: ~1x output + one item at a time.
    """
    if not list_with_numpy_arrays:
        return None

    # Assume all arrays have the same ndim
    ndim = list_with_numpy_arrays[0].ndim
    for a in list_with_numpy_arrays:
        if a.ndim != ndim:
            raise ValueError(f"All arrays must have same ndim. Got {ndim} and {a.ndim}.")

    # Compute max shape in Python (no extra tensor allocation)
    max_shape = [0] * ndim
    for a in list_with_numpy_arrays:
        for d in range(ndim):
            if a.shape[d] > max_shape[d]:
                max_shape[d] = a.shape[d]

    n = len(list_with_numpy_arrays)
    if device is None:
        device = torch.device("cpu")

    # Allocate output once
    out = torch.zeros((n, *max_shape), dtype=dtype, device=device)

    # Fill
    for i, a in enumerate(list_with_numpy_arrays):
        # as_tensor avoids an extra copy if possible; dtype conversion only if needed
        t = torch.as_tensor(a, device=device)
        if t.dtype != dtype:
            t = t.to(dtype)

        # Build slices: [:s0, :s1, ...]
        sl = tuple(slice(0, s) for s in t.shape)
        out[i][sl] = t

    return out
