#!/usr/bin/env python
"""
Test flow matching log-likelihood computation of a target sample x_1.

Examples:
    - Estimate the log-likelihood computation of target sample x_1 for a flow
    matching policy trained on the Push-T dataset:
    ```
    local$ python lerobot/scripts/compute_flow_matching_likelihood.py \
        -r lerobot/pusht \
        -p outputs/train/flow_matching_pusht/checkpoints/last/pretrained_model
    ```
"""
import argparse

import torch
from torch.distributions import Independent, Normal

from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.utils import cycle
from lerobot.policies.common.flow_matching.ode_solver import ODESolver
from lerobot.policies.flow_matching.modelling_flow_matching import FlowMatchingPolicy
from lerobot.policies.utils import get_device_from_parameters, get_dtype_from_parameters


def compute_log_likelihood(
    ds_repo_id: str,
    pretrained_flow_matching_path: str,
):
    # Initialize flow matching visualizer using pretrained flow matching policy.
    flow_matching_policy = FlowMatchingPolicy.from_pretrained(pretrained_flow_matching_path)
    flow_matching_model = flow_matching_policy.flow_matching

    device = get_device_from_parameters(flow_matching_model)
    dtype = get_dtype_from_parameters(flow_matching_model)

    train_cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id=ds_repo_id, episodes=[0]),
        policy=flow_matching_policy.config,
    )
    dataset = make_dataset(train_cfg)

    batch_size = 1
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=0,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=device != "cpu",
        drop_last=True,
    )
    dl_iter = cycle(dataloader)
    batch = next(dl_iter)

    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device, non_blocking=True)

    batch = flow_matching_policy.normalize_inputs(batch)
    if flow_matching_policy.config.image_features:
        batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
        batch["observation.images"] = torch.stack(
            [batch[key] for key in flow_matching_policy.config.image_features], dim=-4
        )
        if flow_matching_policy.config.n_obs_steps == 1:
            batch["observation.images"] = batch["observation.images"].unsqueeze(1)

    # Encode image features and concatenate them all together along with the state vector.
    global_cond = flow_matching_model._prepare_global_conditioning(batch)

    # Create a noise generator with a fixed seed for reproducibility
    seed = 42
    generator = torch.Generator(device=device).manual_seed(seed)

    # Sample noise prior.
    horizon = flow_matching_policy.config.horizon
    action_dim = flow_matching_policy.config.action_feature.shape[0]
    noise_sample = torch.randn(
        size=(batch_size, horizon, action_dim),
        dtype=dtype,
        device=device,
        generator=generator,
    )

    print(f"Noise sample x_0: {noise_sample}")

    # Noise distribution is an isotropic gaussian
    gaussian_log_density = Independent(
        Normal(
            loc = torch.zeros(horizon, action_dim, device=device),
            scale = torch.ones(horizon, action_dim, device=device),
        ),
        reinterpreted_batch_ndims=2
    ).log_prob

    ode_solver = ODESolver(velocity_model=flow_matching_model.unet)

    exact_divergence = False
    x_1, log_p_1_x_1_forward = ode_solver.sample_with_log_likelihood(
        x_init=noise_sample,
        time_grid=torch.tensor([0.0, 1.0], device=device, dtype=dtype),
        global_cond=global_cond,
        log_p_0 = gaussian_log_density,
        method=flow_matching_policy.config.ode_solver_method,
        step_size=flow_matching_policy.config.ode_step_size,
        atol=flow_matching_policy.config.atol,
        rtol=flow_matching_policy.config.rtol,
        exact_divergence=exact_divergence,
        generator=generator,
    )

    print("-------------------------------")
    print("Exact Divergence") if exact_divergence else print("Hutchinson")
    print("-------------------------------")

    print(f"Sample x_1: {x_1}")
    print(f"Forward log-likelihood of x_1: {log_p_1_x_1_forward}")

    x_0, log_p_1_x_1_reverse = ode_solver.sample_with_log_likelihood(
        x_init=x_1,
        time_grid=torch.tensor([1.0, 0.0], device=device, dtype=dtype),
        global_cond=global_cond,
        log_p_0=gaussian_log_density,
        method=flow_matching_policy.config.ode_solver_method,
        step_size=flow_matching_policy.config.ode_step_size,
        atol=flow_matching_policy.config.atol,
        rtol=flow_matching_policy.config.rtol,
        exact_divergence=exact_divergence,
        generator=generator,
    )

    print(f"Reconstructed x_0: {x_0}")
    print(f"Log-likelihood of x_1: {log_p_1_x_1_reverse}")

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-r", "--repo-id",
        type=str,
        required=True,
        help="Name of hugging face repository containing a LeRobotDataset dataset (e.g. `lerobot/pusht`).",
    )

    parser.add_argument(
        "-p", "--pretrained-fm-path",
        type=str,
        required=True,
        help="Path to a pretrained flow-matching policy checkpoint file.",
    )

    args = parser.parse_args()
    compute_log_likelihood(
        args.repo_id,
        args.pretrained_fm_path,
    )

if __name__ == "__main__":
    main()
