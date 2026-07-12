"""Training loop for iterative active-learning fine-tuning rounds."""

from __future__ import annotations

import copy
import logging
import os
import subprocess
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from pprint import pformat
from typing import Any

import torch
from accelerate import Accelerator
from termcolor import colored
from torch.optim import Optimizer

from lerobot.configs.types import FeatureType
from lerobot.datasets.compute_stats import compute_stats_for_episodes
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.utils import cycle, dataset_to_policy_features, filter_libero_episodes, filter_metaworld_episodes
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import close_envs
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import validate_visual_features_consistency
from lerobot.rl.wandb_utils import WandBLogger
from lerobot.scripts.lerobot_eval import eval_policy_all
from lerobot.utils.constants import CHECKPOINTS_DIR, LAST_CHECKPOINT_LINK, PRETRAINED_MODEL_DIR
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.utils.utils import format_big_number, has_method, init_logging

from .config import IterativeFineTuningConfig
from .constants import (
    ITERATIVE_CONFIG_NAME,
    MEMBER_TRAINING_LOG_NAME,
    MEMBER_TRAINING_REQUEST_NAME,
    MEMBER_TRAINING_RESULT_NAME,
    POOL_ALL,
    POOL_HISTORY,
    POOL_NEW,
    ROUND_DIR_TEMPLATE,
    TRAINING_DIRNAME,
)
from .layout import RoundLayout, RunLayout
from .manifests import (
    TrainingManifest,
    TrainedEnsembleMember,
    load_trained_ensemble_member,
    make_timing_entry,
    save_dataclass_json,
    utc_timestamp,
)
from .sampling import MixtureFrameSampler, SamplingPool, frame_indices_for_episode_ids

logger = logging.getLogger(__name__)


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
    start_time = time.perf_counter()
    policy.train()

    with accelerator.autocast():
        loss, output_dict = policy.forward(batch)

    accelerator.backward(loss)

    if grad_clip_norm > 0:
        grad_norm = accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm)
    else:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            policy.parameters(),
            float("inf"),
            error_if_nonfinite=False,
        )

    with lock if lock is not None else nullcontext():
        optimizer.step()
    optimizer.zero_grad()

    if lr_scheduler is not None:
        lr_scheduler.step()

    if has_method(accelerator.unwrap_model(policy, keep_fp32_wrapper=True), "update"):
        accelerator.unwrap_model(policy, keep_fp32_wrapper=True).update()

    train_metrics.loss = loss.item()
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    return train_metrics, output_dict


def _merge_task_filters(*task_maps: dict[str, list[int]] | None) -> dict[str, list[int]] | None:
    merged: dict[str, set[int]] = {}
    for task_map in task_maps:
        if not task_map:
            continue
        for task_group, task_ids in task_map.items():
            merged.setdefault(task_group, set()).update(int(task_id) for task_id in task_ids)
    if not merged:
        return None
    return {task_group: sorted(task_ids) for task_group, task_ids in merged.items()}


def _build_training_dataset_cfg(cfg: IterativeFineTuningConfig):
    dataset_cfg = copy.deepcopy(cfg.dataset)
    merged_tasks = _merge_task_filters(
        cfg.dataset.metaworld_tasks if cfg.dataset.metaworld_tasks is not None else cfg.dataset.libero_tasks,
        cfg.replay.history_tasks,
    )
    if merged_tasks is not None:
        if cfg.dataset.metaworld_tasks is not None:
            dataset_cfg.metaworld_tasks = merged_tasks
        else:
            dataset_cfg.libero_tasks = merged_tasks
    return dataset_cfg


def _fixed_history_episode_ids(
    cfg: IterativeFineTuningConfig,
    *,
    dataset,
) -> list[int]:
    if cfg.replay.history_tasks is None:
        return []
    filter_fn = filter_metaworld_episodes if cfg.dataset.metaworld_tasks is not None else filter_libero_episodes
    return sorted(int(ep) for ep in filter_fn(dataset=dataset, tasks_to_use=cfg.replay.history_tasks))


def _make_round_policy(
    *,
    policy_cfg,
    ds_meta,
    model_reference: str,
    rename_map: dict[str, str],
) -> PreTrainedPolicy:
    round_policy_cfg = copy.deepcopy(policy_cfg)
    features = dataset_to_policy_features(ds_meta.features)
    round_policy_cfg.output_features = {
        key: feature for key, feature in features.items() if feature.type is FeatureType.ACTION
    }
    if not round_policy_cfg.input_features:
        round_policy_cfg.input_features = {
            key: feature
            for key, feature in features.items()
            if key not in round_policy_cfg.output_features
        }

    policy_cls = get_policy_class(round_policy_cfg.type)
    policy = policy_cls.from_pretrained(
        pretrained_name_or_path=model_reference,
        config=round_policy_cfg,
    )
    policy.to(round_policy_cfg.device)

    if not rename_map:
        validate_visual_features_consistency(round_policy_cfg, features)

    return policy


def _build_sampling_pools(
    cfg: IterativeFineTuningConfig,
    *,
    dataset,
    history_episode_ids: list[int],
    new_episode_ids: list[int],
) -> tuple[list[SamplingPool], dict[str, float], list[int], int]:
    drop_n_last_frames = getattr(cfg.policy, "drop_n_last_frames", 0)
    episodes = dataset.meta.episodes
    episode_indices = [int(item) for item in episodes["episode_index"]]
    dataset_from_indices = [int(item) for item in episodes["dataset_from_index"]]
    dataset_to_indices = [int(item) for item in episodes["dataset_to_index"]]

    pools = [
        SamplingPool(
            name=POOL_HISTORY,
            frame_indices=tuple(
                frame_indices_for_episode_ids(
                    dataset_from_indices=dataset_from_indices,
                    dataset_to_indices=dataset_to_indices,
                    episode_indices=episode_indices,
                    target_episode_ids=history_episode_ids,
                    drop_n_last_frames=drop_n_last_frames,
                )
            ),
            weight=cfg.replay.history_fraction,
        ),
        SamplingPool(
            name=POOL_NEW,
            frame_indices=tuple(
                frame_indices_for_episode_ids(
                    dataset_from_indices=dataset_from_indices,
                    dataset_to_indices=dataset_to_indices,
                    episode_indices=episode_indices,
                    target_episode_ids=new_episode_ids,
                    drop_n_last_frames=drop_n_last_frames,
                )
            ),
            weight=cfg.replay.new_fraction,
        ),
    ]

    active_pools = [pool for pool in pools if pool.weight > 0 and len(pool.frame_indices) > 0]
    if not active_pools:
        raise ValueError("No non-empty replay pools were available for training.")

    if cfg.replay.sampling_mode == "uniform_all_frames":
        combined_frame_indices = tuple(
            frame_index
            for pool in active_pools
            for frame_index in pool.frame_indices
        )
        active_pools = [
            SamplingPool(
                name=POOL_ALL,
                frame_indices=combined_frame_indices,
                weight=1.0,
            )
        ]
        normalized_weights = {POOL_ALL: 1.0}
    else:
        active_weight_total = sum(pool.weight for pool in active_pools)
        normalized_weights = {pool.name: pool.weight / active_weight_total for pool in active_pools}

    training_episode_ids = sorted(set(history_episode_ids) | set(new_episode_ids))

    default_samples = sum(len(pool.frame_indices) for pool in active_pools)
    num_samples = cfg.replay.samples_per_epoch or default_samples
    return active_pools, normalized_weights, training_episode_ids, num_samples


def _member_training_dir(round_layout: RoundLayout, member_index: int) -> Path:
    return round_layout.training_dir / f"member_{member_index:02d}"


def _member_final_model_path(training_dir: Path) -> Path:
    return training_dir / CHECKPOINTS_DIR / LAST_CHECKPOINT_LINK / PRETRAINED_MODEL_DIR


def _member_checkpoint_training_dir(
    cfg: IterativeFineTuningConfig,
    *,
    round_index: int,
    member_index: int,
) -> Path:
    return (
        cfg.checkpoint_run_dir()
        / ROUND_DIR_TEMPLATE.format(round_index=round_index)
        / TRAINING_DIRNAME
        / f"member_{member_index:02d}"
    )


def _member_seed(base_seed: int | None, round_index: int, member_index: int) -> int | None:
    if base_seed is None:
        return None
    return base_seed + (1000 * round_index) + member_index


@dataclass
class MemberTrainingRequest:
    config_snapshot_path: str
    round_dir: str
    round_index: int
    member_index: int
    model_reference: str
    history_episode_ids: list[int]
    selected_episode_ids: list[int]
    device: str = "cuda:0"
    result_path: str | None = None


def _member_request_path(training_dir: Path) -> Path:
    return training_dir / MEMBER_TRAINING_REQUEST_NAME


def _member_result_path(training_dir: Path) -> Path:
    return training_dir / MEMBER_TRAINING_RESULT_NAME


def _member_log_path(training_dir: Path) -> Path:
    return training_dir / MEMBER_TRAINING_LOG_NAME


def _round_config_snapshot_path(round_layout: RoundLayout) -> Path:
    return round_layout.round_dir.parent / ITERATIVE_CONFIG_NAME


def _normalize_worker_device(device: str) -> tuple[str, str | None]:
    normalized = str(device).strip()
    if normalized.startswith("cuda:"):
        return "cuda", normalized.split(":", 1)[1]
    return normalized, None


def _tail_log(path: Path, *, num_lines: int = 40) -> str:
    if not path.exists():
        return "<log file missing>"
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    tail = lines[-num_lines:]
    return "\n".join(tail) if tail else "<log file empty>"


def _load_member_result(path: Path) -> TrainedEnsembleMember:
    return load_trained_ensemble_member(path)


def _run_parallel_member_training(
    cfg: IterativeFineTuningConfig,
    *,
    round_index: int,
    model_references: list[str],
    history_episode_ids: list[int],
    selected_episode_ids: list[int],
    round_layout: RoundLayout,
) -> list[TrainedEnsembleMember]:
    config_snapshot_path = _round_config_snapshot_path(round_layout)
    if not config_snapshot_path.exists():
        raise FileNotFoundError(
            f"Could not find config snapshot required for worker launch at {config_snapshot_path}."
        )

    member_devices = cfg.resolved_member_devices(len(model_references))
    launches: list[tuple[int, subprocess.Popen, object, Path, Path]] = []

    for member_index, (model_reference, member_device) in enumerate(
        zip(model_references, member_devices, strict=True)
    ):
        training_dir = _member_training_dir(round_layout, member_index)
        training_dir.mkdir(parents=True, exist_ok=True)
        request_path = _member_request_path(training_dir)
        result_path = _member_result_path(training_dir)
        log_path = _member_log_path(training_dir)

        worker_device, visible_device = _normalize_worker_device(member_device)
        request = MemberTrainingRequest(
            config_snapshot_path=str(config_snapshot_path),
            round_dir=str(round_layout.round_dir),
            round_index=round_index,
            member_index=member_index,
            model_reference=model_reference,
            history_episode_ids=sorted(set(int(ep) for ep in history_episode_ids)),
            selected_episode_ids=[int(ep) for ep in selected_episode_ids],
            device=worker_device,
            result_path=str(result_path),
        )
        save_dataclass_json(request, request_path)

        env = os.environ.copy()
        if visible_device is not None:
            env["CUDA_VISIBLE_DEVICES"] = visible_device

        logger.info(
            "Launching ensemble member %s on device %s via worker process",
            member_index,
            member_device,
        )
        log_handle = open(log_path, "w", encoding="utf-8")
        process = subprocess.Popen(
            [sys.executable, "-m", "iterative_fine_tuning.train_worker", "--request_path", str(request_path)],
            cwd=str(Path.cwd()),
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        launches.append((member_index, process, log_handle, result_path, log_path))

    failures: list[tuple[int, int, Path]] = []
    for member_index, process, log_handle, result_path, log_path in launches:
        return_code = process.wait()
        log_handle.close()
        if return_code != 0:
            failures.append((member_index, return_code, log_path))
            continue
        if not result_path.exists():
            failures.append((member_index, -1, log_path))

    if failures:
        failure_messages = []
        for member_index, return_code, log_path in failures:
            failure_messages.append(
                f"member_{member_index:02d} failed with code {return_code}.\n"
                f"Last log lines from {log_path}:\n{_tail_log(log_path)}"
            )
        raise RuntimeError("\n\n".join(failure_messages))

    member_training_runs = [
        _load_member_result(result_path)
        for _, _, _, result_path, _ in launches
    ]
    member_training_runs.sort(key=lambda run: run.member_index)
    return member_training_runs


def _common_training_metadata(
    cfg: IterativeFineTuningConfig,
    *,
    selected_episode_ids: list[int],
) -> tuple[list[int], list[int], dict[str, float], list[int], int]:
    dataset = make_dataset(
        dataset_cfg=_build_training_dataset_cfg(cfg),
        policy_cfg=cfg.policy,
        num_workers=cfg.num_workers,
    )
    fixed_history_episode_ids = _fixed_history_episode_ids(cfg, dataset=dataset)
    selected_episode_ids = [int(ep) for ep in selected_episode_ids]
    _, sampling_weights, training_episode_ids, num_samples_per_epoch = _build_sampling_pools(
        cfg,
        dataset=dataset,
        history_episode_ids=fixed_history_episode_ids,
        new_episode_ids=selected_episode_ids,
    )
    return (
        fixed_history_episode_ids,
        selected_episode_ids,
        sampling_weights,
        training_episode_ids,
        num_samples_per_epoch,
    )


def _run_single_training_round(
    cfg: IterativeFineTuningConfig,
    *,
    round_index: int,
    member_index: int,
    model_reference: str,
    history_episode_ids: list[int],
    selected_episode_ids: list[int],
    round_layout: RoundLayout,
    accelerator: Accelerator | None = None,
) -> TrainedEnsembleMember:
    member_started_at = utc_timestamp()
    member_start_time = time.perf_counter()

    if accelerator is None:
        from accelerate.utils import DistributedDataParallelKwargs

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        accelerator = Accelerator(step_scheduler_with_optimizer=False, kwargs_handlers=[ddp_kwargs])

    training_dir = _member_training_dir(round_layout, member_index)
    training_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_training_dir = _member_checkpoint_training_dir(
        cfg,
        round_index=round_index,
        member_index=member_index,
    )
    checkpoint_training_dir.mkdir(parents=True, exist_ok=True)

    init_logging(accelerator=accelerator)
    is_main_process = accelerator.is_main_process

    member_seed = _member_seed(cfg.seed, round_index, member_index)
    if member_seed is not None:
        set_seed(member_seed, accelerator=accelerator)

    device = accelerator.device
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    training_dataset_cfg = _build_training_dataset_cfg(cfg)

    if is_main_process:
        logger.info(
            "Creating training dataset for iterative round %s, member %s",
            round_index,
            member_index,
        )
        full_dataset = make_dataset(
            dataset_cfg=training_dataset_cfg,
            policy_cfg=cfg.policy,
            num_workers=cfg.num_workers,
        )
    accelerator.wait_for_everyone()
    if not is_main_process:
        full_dataset = make_dataset(
            dataset_cfg=training_dataset_cfg,
            policy_cfg=cfg.policy,
            num_workers=cfg.num_workers,
        )

    fixed_history_episode_ids = sorted(set(int(ep) for ep in history_episode_ids))
    pools, sampling_weights, training_episode_ids, num_samples_per_epoch = _build_sampling_pools(
        cfg,
        dataset=full_dataset,
        history_episode_ids=fixed_history_episode_ids,
        new_episode_ids=[int(ep) for ep in selected_episode_ids],
    )
    sampler = MixtureFrameSampler(
        pools=pools,
        num_samples=num_samples_per_epoch,
        replacement=cfg.replay.replacement,
        seed=member_seed,
    )

    train_loader = torch.utils.data.DataLoader(
        full_dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=False,
        sampler=sampler,
        pin_memory=device.type == "cuda",
        drop_last=False,
        **({"prefetch_factor": 2} if cfg.num_workers > 0 else {}),
    )

    if is_main_process:
        logger.info(
            "Loading round-start policy for member %s from %s",
            member_index,
            model_reference,
        )
    policy = _make_round_policy(
        policy_cfg=cfg.policy,
        ds_meta=full_dataset.meta,
        model_reference=model_reference,
        rename_map=cfg.rename_map,
    )

    if cfg.reinitialize_selected_layers:
        logger.info("Reinitializing selected policy layers before round training")
        policy.reinitialize_selected_layers()

    all_episode_ids = [int(item) for item in full_dataset.meta.episodes["episode_index"]]
    if set(training_episode_ids) == set(all_episode_ids):
        training_stats = full_dataset.meta.stats
    else:
        training_stats = compute_stats_for_episodes(full_dataset.meta.root, training_episode_ids)

    processor_kwargs = {
        "dataset_stats": training_stats,
        "preprocessor_overrides": {
            "device_processor": {"device": device.type},
            "normalizer_processor": {
                "stats": training_stats,
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
            "rename_observations_processor": {"rename_map": cfg.rename_map},
        },
    }
    postprocessor_kwargs = {
        "postprocessor_overrides": {
            "unnormalizer_processor": {
                "stats": training_stats,
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
            },
        },
    }
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=model_reference,
        **processor_kwargs,
        **postprocessor_kwargs,
    )

    policy_cfg_for_logging = copy.deepcopy(policy.config)
    if Path(model_reference).exists():
        policy_cfg_for_logging.pretrained_path = Path(model_reference)
    train_cfg = cfg.to_train_config(
        output_dir=training_dir,
        round_index=round_index,
        policy_cfg=policy_cfg_for_logging,
    )
    train_cfg.job_name = f"{train_cfg.job_name}_member_{member_index:02d}"
    train_cfg.seed = member_seed
    train_cfg.dataset = copy.deepcopy(training_dataset_cfg)
    if not train_cfg.resume:
        train_cfg.wandb.run_id = None

    if is_main_process:
        logger.info(pformat(train_cfg.to_dict()))

    if train_cfg.wandb.enable and train_cfg.wandb.project and is_main_process:
        wandb_logger = WandBLogger(train_cfg)
    else:
        wandb_logger = None

    optimizer, lr_scheduler = make_optimizer_and_scheduler(train_cfg, policy)

    eval_env = None
    if train_cfg.eval_freq > 0 and train_cfg.env is not None:
        if is_main_process:
            logger.info("Creating evaluation environment")
        eval_env = make_env(
            train_cfg.env,
            n_envs=train_cfg.eval.batch_size,
            use_async_envs=train_cfg.eval.use_async_envs,
        )
        if is_main_process:
            env_preprocessor, env_postprocessor = make_env_pre_post_processors(
                env_cfg=train_cfg.env,
                policy_cfg=train_cfg.policy,
            )

    accelerator.wait_for_everyone()
    policy, optimizer, train_loader, lr_scheduler = accelerator.prepare(
        policy,
        optimizer,
        train_loader,
        lr_scheduler,
    )
    dl_iter = cycle(train_loader)

    effective_batch_size = cfg.batch_size * accelerator.num_processes
    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }
    train_tracker = MetricsTracker(
        batch_size=effective_batch_size,
        num_frames=len(sampler),
        num_episodes=len(training_episode_ids),
        metrics=train_metrics,
        initial_step=0,
    )

    if is_main_process:
        logger.info(colored("Iterative round training", "cyan", attrs=["bold"]))
        logger.info(
            "Round %s member %s output dir: %s",
            round_index,
            member_index,
            training_dir,
        )
        logger.info("Training episodes: %s", len(training_episode_ids))
        logger.info("Frames per epoch/sample cycle: %s", format_big_number(len(sampler)))
        logger.info("Sampling weights: %s", sampling_weights)

    step = 0
    for _ in range(train_cfg.steps):
        start_time = time.perf_counter()
        batch = next(dl_iter)
        batch = preprocessor(batch)
        train_tracker.dataloading_s = time.perf_counter() - start_time

        train_tracker, output_dict = update_policy(
            train_tracker,
            policy,
            batch,
            optimizer,
            train_cfg.optimizer.grad_clip_norm,
            accelerator=accelerator,
            lr_scheduler=lr_scheduler,
        )

        step += 1
        train_tracker.step()
        is_log_step = train_cfg.log_freq > 0 and step % train_cfg.log_freq == 0 and is_main_process
        is_saving_step = step % train_cfg.save_freq == 0 or step == train_cfg.steps
        is_eval_step = train_cfg.eval_freq > 0 and step % train_cfg.eval_freq == 0

        if is_log_step:
            logger.info(train_tracker)
            if wandb_logger is not None:
                wandb_log_dict = train_tracker.to_dict()
                if output_dict:
                    wandb_log_dict.update(output_dict)
                wandb_logger.log_dict(wandb_log_dict, step)
            train_tracker.reset_averages()

        if train_cfg.save_checkpoint and is_saving_step:
            if is_main_process:
                checkpoint_dir = get_step_checkpoint_dir(
                    checkpoint_training_dir,
                    train_cfg.steps,
                    step,
                )
                logger.info("Saving checkpoint at step %s -> %s", step, checkpoint_dir)
                save_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    step=step,
                    cfg=train_cfg,
                    policy=accelerator.unwrap_model(policy),
                    optimizer=optimizer,
                    scheduler=lr_scheduler,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                )
                update_last_checkpoint(checkpoint_dir)
                if wandb_logger is not None:
                    wandb_logger.log_policy(checkpoint_dir)
            accelerator.wait_for_everyone()

        if train_cfg.env is not None and is_eval_step:
            if is_main_process:
                step_id = get_step_identifier(step, train_cfg.steps)
                logger.info("Evaluating checkpoint at step %s", step)
                with torch.no_grad(), accelerator.autocast():
                    eval_info = eval_policy_all(
                        envs=eval_env,
                        policy=accelerator.unwrap_model(policy),
                        env_preprocessor=env_preprocessor,
                        env_postprocessor=env_postprocessor,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        n_episodes=train_cfg.eval.n_episodes,
                        videos_dir=train_cfg.output_dir / "eval" / f"videos_step_{step_id}",
                        max_episodes_rendered=4,
                        start_seed=train_cfg.seed,
                        max_parallel_tasks=train_cfg.env.max_parallel_tasks,
                    )
                if wandb_logger is not None:
                    aggregated = eval_info["overall"]
                    eval_metrics = {
                        "avg_sum_reward": aggregated["avg_sum_reward"],
                        "pc_success": aggregated["pc_success"],
                        "eval_s": aggregated["eval_s"],
                    }
                    wandb_logger.log_dict({**eval_metrics, **eval_info}, step, mode="eval")
                    if eval_info["overall"]["video_paths"]:
                        wandb_logger.log_video(eval_info["overall"]["video_paths"][0], step, mode="eval")
            accelerator.wait_for_everyone()

    if eval_env is not None:
        close_envs(eval_env)

    final_model_path = _member_final_model_path(checkpoint_training_dir)
    accelerator.wait_for_everyone()
    accelerator.end_training()
    return TrainedEnsembleMember(
        member_index=member_index,
        start_model_path=model_reference,
        final_model_path=str(final_model_path),
        training_dir=str(training_dir),
        seed=member_seed,
        timings={
            "total": make_timing_entry(
                started_at=member_started_at,
                duration_s=time.perf_counter() - member_start_time,
            )
        },
    )


def run_training_round(
    cfg: IterativeFineTuningConfig,
    *,
    round_index: int,
    model_references: list[str],
    selected_episode_ids: list[int],
    round_layout: RoundLayout,
) -> TrainingManifest:
    training_started_at = utc_timestamp()
    training_start_time = time.perf_counter()
    round_layout.training_dir.mkdir(parents=True, exist_ok=True)

    (
        fixed_history_episode_ids,
        selected_episode_ids,
        sampling_weights,
        training_episode_ids,
        num_samples_per_epoch,
    ) = _common_training_metadata(
        cfg,
        selected_episode_ids=selected_episode_ids,
    )

    if cfg.ensemble_training.parallel_members and len(model_references) > 1:
        member_training_runs = _run_parallel_member_training(
            cfg,
            round_index=round_index,
            model_references=model_references,
            history_episode_ids=fixed_history_episode_ids,
            selected_episode_ids=selected_episode_ids,
            round_layout=round_layout,
        )
    else:
        member_training_runs: list[TrainedEnsembleMember] = []
        for member_index, model_reference in enumerate(model_references):
            logger.info(
                "Training ensemble member %s/%s from %s",
                member_index + 1,
                len(model_references),
                model_reference,
            )
            member_training_runs.append(
                _run_single_training_round(
                    cfg,
                    round_index=round_index,
                    member_index=member_index,
                    model_reference=model_reference,
                    history_episode_ids=fixed_history_episode_ids,
                    selected_episode_ids=selected_episode_ids,
                    round_layout=round_layout,
                )
            )

    training_manifest = TrainingManifest(
        round_index=round_index,
        start_model_paths=[run.start_model_path for run in member_training_runs],
        final_model_paths=[run.final_model_path for run in member_training_runs],
        member_training_runs=member_training_runs,
        history_episode_ids=fixed_history_episode_ids,
        new_episode_ids=selected_episode_ids,
        all_training_episode_ids=training_episode_ids,
        sampling_weights=sampling_weights,
        num_samples_per_epoch=num_samples_per_epoch,
        batch_size=cfg.batch_size,
        steps=cfg.steps,
        seed=cfg.seed,
        timings={
            "total": make_timing_entry(
                started_at=training_started_at,
                duration_s=time.perf_counter() - training_start_time,
            )
        },
    )
    save_dataclass_json(training_manifest, round_layout.training_manifest_path)
    return training_manifest
