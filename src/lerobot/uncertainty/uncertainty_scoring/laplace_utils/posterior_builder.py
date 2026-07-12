import copy
import logging
from pathlib import Path
from typing import Any, List, Optional

import torch
from laplace import Laplace
from laplace.baselaplace import BaseLaplace
from torch.nn.utils import vector_to_parameters
from torch.utils.data import DataLoader

from lerobot.configs.default import DatasetConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.utils import filter_libero_episodes
from lerobot.policies.common.flow_matching.adapter import BaseFlowMatchingAdapter
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.processor import PolicyProcessorPipeline
from lerobot.uncertainty.uncertainty_samplers.configuration_uncertainty_sampler import LaplaceConfig
from lerobot.utils.utils import format_big_number

from .laplace_wrappers.factory import make_laplace_wrapper
from .laplace_wrappers.laplace_wrapper import LaplaceWrapper


@torch.no_grad()
def create_laplace_calib_loader(
    laplace_wrapper: LaplaceWrapper,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    dataset_cfg: DatasetConfig,
    policy_cfg: PreTrainedConfig,
    calib_fraction: float,
    batch_size: int,
    episode_ids: list[int] | None = None,
) -> DataLoader:
    """
    Build a DataLoader for fitting a Laplace approximation around the policy's underlying model.

    Args:
        laplace_wrapper: Wrapper around the pre-trained model compatible with laplace-torch.
        preprocessor: Preprocessor to apply to raw dataset samples.
        dataset_cfg: Dataset configuration.
        calib_fraction: Fraction of the full dataset to reserve for calibration (between 0 and 1).
        batch_size: Number of samples per batch in the returned DataLoader.
    """
    # Extract a subset of the full dataset for calibration
    dataset = make_dataset(dataset_cfg=dataset_cfg, policy_cfg=policy_cfg)
    all_episode_ids = list(dataset.meta.episodes["episode_index"])
    if episode_ids is not None:
        episode_ids_to_use = episode_ids
    elif dataset.repo_id == "HuggingFaceVLA/libero" and dataset_cfg.libero_tasks is not None:
        episode_ids_to_use = filter_libero_episodes(
            dataset=dataset,
            tasks_to_use=dataset_cfg.libero_tasks,
        )
    else:
        episode_ids_to_use = all_episode_ids
    calib_sampler = EpisodeAwareSampler(
        dataset_from_indices=dataset.meta.episodes["dataset_from_index"],
        dataset_to_indices=dataset.meta.episodes["dataset_to_index"],
        episode_indices_to_use=sorted(episode_ids_to_use),
        drop_n_last_frames=getattr(policy_cfg, "drop_n_last_frames", 0),
        fraction=calib_fraction,
        shuffle=True,
    )

    logging.info(f"Total number of episodes: {dataset.num_episodes} and frames: {dataset.num_frames} ({format_big_number(len(dataset))})")
    if episode_ids is not None:
        logging.info(f"Number of selected episodes: {len(episode_ids_to_use)}")
    elif dataset.repo_id == "HuggingFaceVLA/libero" and dataset_cfg.libero_tasks is not None:
        logging.info(f"Number of selected episodes: {len(episode_ids_to_use)}")
    num_calib_frames = len(calib_sampler)
    logging.info(f"Number of Laplace calib frames: {num_calib_frames} ({format_big_number(num_calib_frames)}) for calibration fraction: {calib_fraction}")

    calib_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=0,
        batch_size=batch_size,
        shuffle=False,
        sampler=calib_sampler,
        drop_last=False,
        collate_fn=laplace_wrapper.build_collate_fn(preprocessor=preprocessor),
    )

    return calib_loader


def sample_from_posterior(
    laplace_posterior: BaseLaplace,
    uncertainty_adapter: BaseFlowMatchingAdapter,
    num_samples: int = 1,
    generator: Optional[torch.Generator] = None,
) -> List[BaseFlowMatchingAdapter]:
    """
    Draw one weight sample from a fitted Laplace posterior and return a cloned
    uncertainty adapter whose underlying model uses those sampled weights.

    Args:
        laplace_posterior: Fitted Laplace posterior over the model's trainable params.
        uncertainty_adapter: Source adapter whose model defines architecture
            and parameter ordering for writing sampled weights.
        generator: Optional RNG for reproducibility.

    Returns:
        An adapter with its model's trainable parameters replaced by one Monte Carlo sample from the Laplace posterior.
    """
    # Draw weights from the Laplace posterior
    laplace_model_weights = laplace_posterior.sample(
        n_samples=num_samples,
        generator=generator
    )

    laplace_adapters: List[BaseFlowMatchingAdapter] = []
    for i in range(num_samples):
        # Copy the MAP model so we never mutate the original
        laplace_adapter = copy.deepcopy(uncertainty_adapter)

        # Collect the parameters that were in the posterior
        target_params = [p for p in laplace_adapter.model.parameters() if p.requires_grad]

        # Consistency check to avoid silent weight mis-alignment
        n_expected = sum(p.numel() for p in target_params)
        if laplace_model_weights.shape[1] != n_expected:
            raise RuntimeError(
                f"[Laplace] Sample size mismatch: drew {laplace_model_weights.shape[1]} "
                f"weights but found {n_expected} trainable parameters in the copy."
            )

        # Write sampled parameters into the copied model (in-place assignment)
        vector_to_parameters(laplace_model_weights[i], target_params)

        # Move the model to the same device as sampled weights and switch to inference mode
        laplace_adapter.model = laplace_adapter.model.to(laplace_model_weights.device)
        laplace_adapter.model.eval()

        laplace_adapters.append(laplace_adapter)

    return laplace_adapters


def make_laplace_path(
    laplace_wrapper: LaplaceWrapper,
    pretrained_path: Path | str,
    calib_fraction: float,
) -> Path:
    """
    Build (and create) the on-disk path where we save/load a Laplace posterior.
    """
    scope_abbreviations = sorted([laplace_wrapper.scope_abbr[scope] for scope in laplace_wrapper.scopes])
    calib_fraction_pct = int(calib_fraction * 100)
    filename = f"laplace_{'_'.join(scope_abbreviations)}_frac{calib_fraction_pct}pct.bin"
    return Path(pretrained_path) / filename


def get_laplace_posterior(
    policy: PreTrainedPolicy,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    laplace_config: LaplaceConfig,
    dataset_cfg: DatasetConfig | None = None,
    episode_ids: list[int] | None = None,
    pretrained_path: Path | str | None = None,
) -> Laplace:
    """
    Construct or load a Laplace posterior for the underlying model of a policy.

    Builds a calibration DataLoader if needed and fits a new posterior,
    otherwise loads an existing one from disk.

    Returns:
        A fitted or loaded Laplace posterior.
    """
    # Wrap the flow matching model so it takes inputs and generates outputs compatible with Laplace
    laplace_wrapper = make_laplace_wrapper(policy=policy, scopes=laplace_config.scopes)

    laplace_path = make_laplace_path(
        laplace_wrapper=laplace_wrapper,
        pretrained_path=pretrained_path or policy.config.pretrained_path,
        calib_fraction=laplace_config.calib_fraction,
    )

    laplace_posterior = Laplace(
        laplace_wrapper,
        likelihood="regression",
        subset_of_weights="all",  # uses only params with requires_grad=True
        hessian_structure="diag",
    )

    if laplace_path.exists():
        logging.info(f"Loading Laplace posterior from {laplace_path}")
        laplace_posterior.load_state_dict(torch.load(laplace_path))
    else:
        logging.info("Create Laplace calibration loader.")
        if dataset_cfg is None:
            raise ValueError(
                "Dataset config must be provided to build Laplace calibration loader."
            )
        calib_loader = create_laplace_calib_loader(
            laplace_wrapper=laplace_wrapper,
            preprocessor=preprocessor,
            dataset_cfg=dataset_cfg,
            policy_cfg=policy.config,
            calib_fraction=laplace_config.calib_fraction,
            batch_size=laplace_config.batch_size,
            episode_ids=episode_ids,
        )

        logging.info("Fitting new Laplace posterior.")
        laplace_posterior.fit(calib_loader)

        logging.info(f"Save Laplace posterior to {laplace_path}")
        torch.save(laplace_posterior.state_dict(), laplace_path)

    return laplace_posterior
