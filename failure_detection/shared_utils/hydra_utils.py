from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf, DictConfig
from shared_utils.utility_functions import ensure_list
import os
import pathlib
from typing import List, Dict, Union


def load_config(
    module: Union[str, List], filename: Union[str, List], return_only_subdict: bool = True, base_config_dir: str = None, as_dict: bool = False
) -> Union[Dict, DictConfig]:
    """Load a configuration file using Hydra.

    If Hydra is not initialized, it initializes it with the specified base config directory.
    If Hydra is already initialized, overrides the default config with the specified module and filename. If lists of modules and filenames are provided, it overrides all of them.
    If the filename is not found, it falls back to the base config in the specified module.
    It optionally returns only the subdictionary or (if multiple overrides) a dict that contains the subdictionaries of the requested modules.

    Args:
        module (str, list): The module name(s) where the config file is located.
        filename (str, list): The name(s) of the config file to overwrite from.
        return_only_subdict (bool, optional): Whether to return only the subdictionary. Defaults to True.
        base_config_dir (str, optional): The base directory for the config files. If not provided, it defaults to the "configs" directory in the current file's parent directory.
        as_dict (bool, optional): Whether to return the config as a dictionary (not DictConfig). Defaults to False.

    Returns:
        cfg (dict, dict[dict]): DictConfig(s). Returns either the complete config file or if return_only_subdict the subdictionary corresponding to a single module (if only a single module is provided) or a dictionary containing the subdictionaries to the list of modules.
    """
    # Check if the module and filename are strings or lists
    module, filename = ensure_list(module, filename)
    assert len(module) == len(filename), "Module and filename lists must be of the same length."
    # Check if hydra is initialized
    # If not initialized, initialize hydra with the base config directory
    if not GlobalHydra.instance().is_initialized():
        if base_config_dir is None:
            base_config_dir = os.path.join(str(pathlib.Path(__file__).parent.parent), "configs")
        initialize_config_dir(config_dir=base_config_dir, job_name="init")

    overides = []
    for m, f in zip(module, filename):
        if isinstance(m, str) and isinstance(f, str):
            overides.append(f"{m}={f}")
        else:
            raise ValueError("Module and filename must be strings.")
    if len(overides) == 0:
        raise ValueError("No overrides provided.")
    
    # Try to compose the config with the provided overrides
    # If it fails, try to compose with the base config in the module
    try:
        cfg = compose(config_name="default", overrides=overides)
    except Exception as _:
        cfg = compose(config_name="default", overrides=[f"{m}={'base'}"])

    if return_only_subdict:
        # If only one module and filename are provided, return the subdictionary
        if len(module) == 1 and len(filename) == 1:
            cfg = cfg.get(module[0])
        else:
            # If multiple modules and filenames are provided, return a dictionary of subdictionaries
            cfg = {m: cfg.get(m) for m in module}
    
    if as_dict:
        cfg = OmegaConf.to_container(cfg, resolve=True)
    return cfg


def reinit_load_config(
    config_path: str, config_name: str, reinitialize: bool = True, job_name: str = None, version_base="1.3"
) -> DictConfig:
    """Combined function that can either:
    1. Reinitialize hydra with an absolute config path and load a config file.
    2. Load a config file without reinitializing hydra by specifying a relative path ("" for same folder as the folder where hydra was initialzed).

    Args:
        config_path (str): The absolute (when reinitializing) or relative path to the config directory.
        config_name (str): The name of the config file to load.
        job_name (str, optional): The name of the job. Defaults to None.
        version_base (str, optional): The version base for hydra. Defaults to "1.3"."
    Returns:
        cfg (DictConfig): The loaded configuration file.
    """
    # Check if the config directory exists
    if not os.path.exists(config_path) or not os.path.exists(os.path.join(config_path, config_name)):
        raise FileNotFoundError(f"Config directory {config_path} does not exist.")
    if reinitialize:
        # Check if the config path is absolute or relative
        if not os.path.isabs(config_path):
            raise ValueError("config_path must be absolute when reinitializing.")
        job_name = config_name
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
        initialize_config_dir(config_dir=config_path, job_name=job_name, version_base=version_base)
    else:
        if len(config_path) > 0:
            config_name = os.path.join(config_path, config_name)

    if not isinstance(config_name, str):
        config_name = f"{config_name}"
    cfg = compose(config_name=config_name)
    return cfg
