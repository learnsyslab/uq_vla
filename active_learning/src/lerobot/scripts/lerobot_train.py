#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import math
import time
from contextlib import nullcontext
from pprint import pformat
from typing import Any

import torch
from accelerate import Accelerator
from termcolor import colored
from torch.optim import Optimizer

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.compute_stats import compute_stats_for_episodes
from lerobot.datasets.factory import make_dataset, make_train_val_split
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.utils import cycle, filter_libero_episodes
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import close_envs
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.wandb_utils import WandBLogger
from lerobot.scripts.lerobot_eval import eval_policy_all
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.utils.utils import (
    format_big_number,
    has_method,
    init_logging,
)


def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    accelerator: Accelerator,
    lr_scheduler=None,
    lock=None,
) -> tuple[MetricsTracker, dict]:
    """
    Performs a single training step to update the policy's weights.

    This function executes the forward and backward passes, clips gradients, and steps the optimizer and
    learning rate scheduler. Accelerator handles mixed-precision training automatically.

    Args:
        train_metrics: A MetricsTracker instance to record training statistics.
        policy: The policy model to be trained.
        batch: A batch of training data.
        optimizer: The optimizer used to update the policy's parameters.
        grad_clip_norm: The maximum norm for gradient clipping.
        accelerator: The Accelerator instance for distributed training and mixed precision.
        lr_scheduler: An optional learning rate scheduler.
        lock: An optional lock for thread-safe optimizer updates.

    Returns:
        A tuple containing:
        - The updated MetricsTracker with new statistics for this step.
        - A dictionary of outputs from the policy's forward pass, for logging purposes.
    """
    start_time = time.perf_counter()
    policy.train()

    # Let accelerator handle mixed precision
    with accelerator.autocast():
        loss, output_dict = policy.forward(batch)
        # TODO(rcadene): policy.unnormalize_outputs(out_dict)

    # Use accelerator's backward method
    accelerator.backward(loss)

    # Clip gradients if specified
    if grad_clip_norm > 0:
        grad_norm = accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm)
    else:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            policy.parameters(), float("inf"), error_if_nonfinite=False
        )

    # Optimizer step
    with lock if lock is not None else nullcontext():
        optimizer.step()

    optimizer.zero_grad()

    # Step through pytorch scheduler at every batch instead of epoch
    if lr_scheduler is not None:
        lr_scheduler.step()

    # Update internal buffers if policy has update method
    if has_method(accelerator.unwrap_model(policy, keep_fp32_wrapper=True), "update"):
        accelerator.unwrap_model(policy, keep_fp32_wrapper=True).update()

    train_metrics.loss = loss.item()
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    return train_metrics, output_dict


@parser.wrap()
def train(cfg: TrainPipelineConfig, accelerator: Accelerator | None = None):
    """
    Main function to train a policy.

    This function orchestrates the entire training pipeline, including:
    - Setting up logging, seeding, and device configuration.
    - Creating the dataset, evaluation environment (if applicable), policy, and optimizer.
    - Handling resumption from a checkpoint.
    - Running the main training loop, which involves fetching data batches and calling `update_policy`.
    - Periodically logging metrics, saving model checkpoints, and evaluating the policy.
    - Pushing the final trained model to the Hugging Face Hub if configured.

    Args:
        cfg: A `TrainPipelineConfig` object containing all training configurations.
        accelerator: Optional Accelerator instance. If None, one will be created automatically.
    """
    cfg.validate()

    # Create Accelerator if not provided
    # It will automatically detect if running in distributed mode or single-process mode
    # We set step_scheduler_with_optimizer=False to prevent accelerate from adjusting the lr_scheduler steps based on the num_processes
    # We set find_unused_parameters=True to handle models with conditional computation
    if accelerator is None:
        from accelerate.utils import DistributedDataParallelKwargs

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        accelerator = Accelerator(step_scheduler_with_optimizer=False, kwargs_handlers=[ddp_kwargs])

    init_logging(accelerator=accelerator)
    if accelerator.num_processes > 1:
        # Under `accelerate launch` every rank must build its policy on its own GPU: a bare "cuda"
        # is cuda:0 for all of them, which loads N copies onto one device before `prepare` moves them.
        cfg.policy.device = str(accelerator.device)

    # Determine if this is the main process (for logging and checkpointing)
    # When using accelerate, only the main process should log to avoid duplicate outputs
    is_main_process = accelerator.is_main_process

    # Only log on main process
    if is_main_process:
        logging.info(pformat(cfg.to_dict()))

    # Initialize wandb only on main process
    if cfg.wandb.enable and cfg.wandb.project and is_main_process:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        if is_main_process:
            logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if cfg.seed is not None:
        set_seed(cfg.seed, accelerator=accelerator)

    # Use accelerator's device
    device = accelerator.device
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # Dataset loading synchronization: main process downloads first to avoid race conditions
    if is_main_process:
        logging.info("Creating dataset")
        full_dataset = make_dataset(
            dataset_cfg=cfg.dataset,
            policy_cfg=cfg.policy,
            num_workers=cfg.num_workers,
        )

    accelerator.wait_for_everyone()

    # Now all other processes can safely load the dataset
    if not is_main_process:
        full_dataset = make_dataset(
            dataset_cfg=cfg.dataset,
            policy_cfg=cfg.policy,
            num_workers=cfg.num_workers,
        )

    all_episode_ids = list(full_dataset.meta.episodes["episode_index"])
    if cfg.dataset.repo_id == "HuggingFaceVLA/libero" and cfg.dataset.libero_tasks is not None:
        episode_ids_to_use = filter_libero_episodes(
            dataset=full_dataset,
            tasks_to_use=cfg.dataset.libero_tasks,
        )
    else:
        episode_ids_to_use = all_episode_ids

    # Create dataloader for offline training
    train_sampler = None
    if episode_ids_to_use != all_episode_ids or getattr(cfg.policy, "drop_n_last_frames", 0) > 0:
        if cfg.dataset.streaming:
            raise RuntimeError(
                "Episode filtering or frame dropping requires an episode aware sampler, "
                "which is disallowed in streaming mode."
            )
        train_sampler = EpisodeAwareSampler(
            dataset_from_indices=full_dataset.meta.episodes["dataset_from_index"],
            dataset_to_indices=full_dataset.meta.episodes["dataset_to_index"],
            episode_indices=all_episode_ids,
            episode_indices_to_use=sorted(episode_ids_to_use),
            drop_n_last_frames=getattr(cfg.policy, "drop_n_last_frames", 0),
            shuffle=True,
        )

    train_shuffle = (train_sampler is None) and (not cfg.dataset.streaming)
    train_loader = torch.utils.data.DataLoader(
        full_dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        pin_memory=device.type == "cuda",
        drop_last=False,
        prefetch_factor=2,
    )

    # Effective sizes for logging and trackers
    effective_train_frames = len(train_sampler) if train_sampler is not None else len(full_dataset)
    effective_train_eps = len(episode_ids_to_use)

    if is_main_process:
        logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
        if cfg.env is not None:
            logging.info(f"{cfg.env.task=}")
        logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
        logging.info(f"Total number of episodes: {full_dataset.num_episodes} and frames: {full_dataset.num_frames} ({format_big_number(len(full_dataset))})")
        if cfg.dataset.repo_id == "HuggingFaceVLA/libero" and cfg.dataset.libero_tasks is not None:
            logging.info(f"Number of selected episodes: {effective_train_eps} and frames: {effective_train_frames} ({format_big_number(effective_train_frames)})")

    # Create environment used for evaluating checkpoints during training on simulation data.
    # On real-world data, no need to create an environment as evaluations are done outside train.py,
    # using the eval.py instead, with gym_dora environment and dora-rs.
    eval_env = None
    if cfg.eval_freq > 0 and cfg.env is not None:
        if is_main_process:
            logging.info("Creating env")
        eval_env = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)

    if is_main_process:
        logging.info("Creating policy")
    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=full_dataset.meta,
        rename_map=cfg.rename_map,
    )

    # Wait for all processes to finish policy creation before continuing
    accelerator.wait_for_everyone()

    if cfg.policy.pretrained_path and cfg.reinitialize_selected_layers:
        logging.info("Reinitialize selected layers")
        policy.reinitialize_selected_layers()

    if episode_ids_to_use != all_episode_ids:
        logging.info("Extracting dataset stats from training episodes")
        training_stats = compute_stats_for_episodes(full_dataset.meta.root, episode_ids_to_use)
    else:
        training_stats = full_dataset.meta.stats

    # Create processors - only provide dataset_stats if not resuming from saved processors
    processor_kwargs = {}
    postprocessor_kwargs = {}
    if (cfg.policy.pretrained_path and not cfg.resume) or not cfg.policy.pretrained_path:
        # Only provide dataset_stats when not resuming from saved processor state
        processor_kwargs["dataset_stats"] = training_stats

    if cfg.policy.pretrained_path is not None:
        processor_kwargs["preprocessor_overrides"] = {
            "device_processor": {"device": device.type},
            "normalizer_processor": {
                "stats": training_stats,
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
        }
        processor_kwargs["preprocessor_overrides"]["rename_observations_processor"] = {
            "rename_map": cfg.rename_map
        }
        if cfg.policy.type == "xvla" and cfg.env is not None and cfg.env.type == "libero":
            processor_kwargs["preprocessor_overrides"]["xvla_add_domain_id"] = {
                "domain_id": 3,
            }
        postprocessor_kwargs["postprocessor_overrides"] = {
            "unnormalizer_processor": {
                "stats": training_stats,
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
            },
        }

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        **processor_kwargs,
        **postprocessor_kwargs,
    )

    if is_main_process:
        logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)

    step = 0  # number of policy updates (forward + backward + optim)

    if cfg.resume:
        step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)

    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    if is_main_process:
        if cfg.env is not None:
            logging.info(f"{cfg.env.task=}")
            logging.info("Creating environment processors")
            env_preprocessor, env_postprocessor = make_env_pre_post_processors(
                env_cfg=cfg.env, policy_cfg=cfg.policy
            )
        logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
        logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    # Prepare everything with accelerator
    accelerator.wait_for_everyone()
    policy, optimizer, train_loader, lr_scheduler = accelerator.prepare(
        policy, optimizer, train_loader, lr_scheduler
    )
    dl_iter = cycle(train_loader)

    policy.train()

    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }

    # Use effective batch size for proper epoch calculation in distributed training
    effective_batch_size = cfg.batch_size * accelerator.num_processes
    train_tracker = MetricsTracker(
        batch_size=effective_batch_size,
        num_frames=effective_train_frames,
        num_episodes=effective_train_eps,
        metrics=train_metrics,
        initial_step=step,
    )

    if is_main_process:
        logging.info("Start offline training on a fixed dataset")

    for _ in range(step, cfg.steps):
        start_time = time.perf_counter()
        batch = next(dl_iter)
        batch = preprocessor(batch)
        train_tracker.dataloading_s = time.perf_counter() - start_time

        train_tracker, output_dict = update_policy(
            train_tracker,
            policy,
            batch,
            optimizer,
            cfg.optimizer.grad_clip_norm,
            accelerator=accelerator,
            lr_scheduler=lr_scheduler,
        )

        # Note: eval and checkpoint happens *after* the `step`th training update has completed, so we
        # increment `step` here.
        step += 1
        train_tracker.step()
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0 and is_main_process
        is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps
        is_eval_step = cfg.eval_freq > 0 and step % cfg.eval_freq == 0

        if is_log_step:
            extras = " ".join(f"{k}:{float(v):.3f}" for k, v in (output_dict or {}).items()
                                if isinstance(v, (int, float)) or (hasattr(v, "numel") and v.numel() == 1))
            logging.info(f"{train_tracker} {extras}" if extras else train_tracker)
            if wandb_logger:
                wandb_log_dict = train_tracker.to_dict()
                if output_dict:
                    wandb_log_dict.update(output_dict)
                wandb_logger.log_dict(wandb_log_dict, step)
            train_tracker.reset_averages()

        if cfg.save_checkpoint and is_saving_step:
            if is_main_process:
                logging.info(f"Checkpoint policy after step {step}")
                checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
                save_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    step=step,
                    cfg=cfg,
                    policy=accelerator.unwrap_model(policy),
                    optimizer=optimizer,
                    scheduler=lr_scheduler,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                )
                update_last_checkpoint(checkpoint_dir)
                if wandb_logger:
                    wandb_logger.log_policy(checkpoint_dir)

            accelerator.wait_for_everyone()

        if cfg.env and is_eval_step:
            if is_main_process:
                step_id = get_step_identifier(step, cfg.steps)
                logging.info(f"Eval policy at step {step}")
                with torch.no_grad(), accelerator.autocast():
                    eval_info = eval_policy_all(
                        envs=eval_env,  # dict[suite][task_id] -> vec_env
                        policy=accelerator.unwrap_model(policy),
                        env_preprocessor=env_preprocessor,
                        env_postprocessor=env_postprocessor,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        n_episodes=cfg.eval.n_episodes,
                        videos_dir=cfg.output_dir / "eval" / f"videos_step_{step_id}",
                        max_episodes_rendered=4,
                        start_seed=cfg.seed,
                        max_parallel_tasks=cfg.env.max_parallel_tasks,
                    )
                # overall metrics (suite-agnostic)
                aggregated = eval_info["overall"]

                # optional: per-suite logging
                for suite, suite_info in eval_info.items():
                    logging.info("Suite %s aggregated: %s", suite, suite_info)

                # meters/tracker
                eval_metrics = {
                    "avg_sum_reward": AverageMeter("∑rwrd", ":.3f"),
                    "pc_success": AverageMeter("success", ":.1f"),
                    "eval_s": AverageMeter("eval_s", ":.3f"),
                }
                eval_tracker = MetricsTracker(
                    cfg.batch_size,
                    effective_train_frames,
                    effective_train_eps,
                    eval_metrics,
                    initial_step=step,
                    accelerator=accelerator,
                )
                eval_tracker.eval_s = aggregated.pop("eval_s")
                eval_tracker.avg_sum_reward = aggregated.pop("avg_sum_reward")
                eval_tracker.pc_success = aggregated.pop("pc_success")
                if wandb_logger:
                    wandb_log_dict = {**eval_tracker.to_dict(), **eval_info}
                    wandb_logger.log_dict(wandb_log_dict, step, mode="eval")
                    wandb_logger.log_video(eval_info["overall"]["video_paths"][0], step, mode="eval")

            accelerator.wait_for_everyone()

    if eval_env:
        close_envs(eval_env)

    if is_main_process:
        logging.info("End of training")

        if cfg.policy.push_to_hub:
            unwrapped_policy = accelerator.unwrap_model(policy)
            unwrapped_policy.push_model_to_hub(cfg)
            preprocessor.push_to_hub(cfg.policy.repo_id)
            postprocessor.push_to_hub(cfg.policy.repo_id)

    # Properly clean up the distributed process group
    accelerator.wait_for_everyone()
    accelerator.end_training()


def main():
    register_third_party_plugins()
    train()


if __name__ == "__main__":
    main()
