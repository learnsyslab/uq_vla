#!/usr/bin/env python3
"""Evaluate trained policies from an iterative fine-tuning run."""

from __future__ import annotations

import argparse
import logging
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from lerobot.utils.import_utils import register_third_party_plugins

from .config import IterativeFineTuningConfig
from .constants import STAGE_TRAIN
from .env import maybe_load_env_file, resolve_local_path
from .evaluation import (
    MemberEvaluationRequest,
    member_evaluation_log_path,
    member_evaluation_path,
    member_evaluation_request_path,
    member_training_dir,
    normalize_worker_device,
    normalized_evaluation_device,
    save_member_evaluation_request,
    tail_log,
)
from .layout import RoundLayout, RunLayout
from .manifests import (
    MemberEvaluation,
    RoundEvaluation,
    RoundMemberSummary,
    RoundTaskSummary,
    RunEvaluation,
    RunRoundSummary,
    TaskEvaluation,
    TaskSpec,
    TrainedEnsembleMember,
    TrainingManifest,
    aggregate_timing_entries,
    load_json,
    load_member_evaluation,
    load_run_state,
    load_training_manifest,
    save_dataclass_json,
    total_timing_entry,
)

logger = logging.getLogger(__name__)

DEFAULT_RUNS_ROOT = Path(os.environ.get("ITERATIVE_RUNS_ROOT", "outputs/iterative_fine_tuning"))
SLURM_ACTIVE_STATES = {
    "COMPLETING",
    "CONFIGURING",
    "PENDING",
    "REQUEUED",
    "RESIZING",
    "RUNNING",
    "SUSPENDED",
}
SLURM_SUCCESS_STATES = {"COMPLETED"}
SLURM_FAILURE_STATES = {
    "CANCELLED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "REVOKED",
    "TIMEOUT",
}


@dataclass(frozen=True)
class MemberEvaluationChunk:
    round_index: int
    member: TrainedEnsembleMember
    evaluation_path: Path
    request_path: Path
    log_path: Path


@dataclass
class ActiveMemberEvaluation:
    chunk: MemberEvaluationChunk
    assigned_device: str
    process: subprocess.Popen[str]
    log_handle: Any


@dataclass(frozen=True)
class EvaluationRunContext:
    run_dir: Path
    run_name: str
    config_data: dict[str, Any] | None
    expected_round_indices: set[int] | None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--config-path", type=Path, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--rounds", type=str, default="all")
    parser.add_argument("--members", type=str, default="all")
    parser.add_argument("--task-group", type=str, default=None)
    parser.add_argument("--task-ids", type=str, default="all")
    parser.add_argument("--n-rollouts", type=int, default=30)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--seed-start", type=int, default=100)
    parser.add_argument("--seed-step", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=520)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--member-devices", type=str, default=None)
    parser.add_argument("--workers-per-device", type=int, default=1)
    parser.add_argument("--poll-interval-seconds", type=float, default=5.0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--sync-envs", action="store_true")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--watch-job-id", type=str, default=None)
    parser.add_argument(
        "--cleanup-checkpoints-when-safe",
        action="store_true",
        help=(
            "Delete per-round checkpoint directories after the round is fully evaluated "
            "and the next round has already been trained. The final round is cleaned once "
            "all expected evals are complete or the watched train job has completed."
        ),
    )
    return parser.parse_args()


def _parse_index_spec(spec: str | None) -> list[int] | None:
    if spec is None:
        return None

    normalized = spec.strip().lower()
    if normalized in {"", "all"}:
        return None

    values: set[int] = set()
    for chunk in spec.split(","):
        item = chunk.strip()
        if not item:
            continue
        if "-" in item:
            start_str, end_str = item.split("-", 1)
            start = int(start_str)
            end = int(end_str)
            if end < start:
                raise ValueError("Round and member ranges must be ascending.")
            values.update(range(start, end + 1))
        else:
            values.add(int(item))
    return sorted(values)


def _parse_device_list(spec: str | None, fallback_device: str) -> list[str]:
    if spec is None or not spec.strip():
        return [fallback_device]

    devices = [item.strip() for item in spec.split(",") if item.strip()]
    if not devices:
        return [fallback_device]
    return devices


def _build_device_slots(devices: list[str], workers_per_device: int) -> list[str]:
    if workers_per_device <= 0:
        raise ValueError("--workers-per-device must be positive.")
    return [device for device in devices for _ in range(workers_per_device)]


def _load_iterative_cfg(config_path: Path) -> IterativeFineTuningConfig:
    cfg = IterativeFineTuningConfig.from_pretrained(str(config_path), cli_args=[])
    cfg.validate()
    return cfg


def _resolve_run_context(args: argparse.Namespace) -> EvaluationRunContext:
    config_data: dict[str, Any] | None = None
    expected_round_indices: set[int] | None = None
    run_name = args.run_name
    run_dir: Path | None = None

    if args.config_path is not None:
        config_path = resolve_local_path(args.config_path, base_dir=Path.cwd(), must_exist=True)
        cfg = _load_iterative_cfg(config_path)
        config_data = cfg.to_dict()
        expected_round_indices = set(range(cfg.iteration.total_rounds))
        if run_name is None:
            run_name = cfg.paths.run_name
        if args.run_dir is None:
            run_dir = cfg.run_dir()

    if args.run_dir is not None:
        run_dir = resolve_local_path(args.run_dir, base_dir=Path.cwd())
    elif run_dir is None:
        if not run_name:
            raise ValueError("Either --config-path, --run-dir, or --run-name must be provided.")
        runs_root = resolve_local_path(args.runs_root, base_dir=Path.cwd(), must_exist=True)
        run_dir = runs_root / run_name

    return EvaluationRunContext(
        run_dir=run_dir,
        run_name=run_name or run_dir.name,
        config_data=config_data,
        expected_round_indices=expected_round_indices,
    )


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _std(values: list[float]) -> float:
    return float(np.std(values)) if values else 0.0


def _compute_binomial_ci(successes: int, trials: int) -> tuple[float, float]:
    if trials <= 0:
        return 0.0, 0.0
    success_rate = successes / trials
    standard_error = math.sqrt(success_rate * (1.0 - success_rate) / trials)
    return (
        max(0.0, success_rate - 1.96 * standard_error),
        min(1.0, success_rate + 1.96 * standard_error),
    )


def _load_config_data(
    layout: RunLayout,
    *,
    fallback_config_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if layout.config_snapshot_path.exists():
        config_data = load_json(layout.config_snapshot_path)
    elif fallback_config_data is not None:
        config_data = fallback_config_data
    else:
        raise FileNotFoundError(
            f"Could not find saved config snapshot at {layout.config_snapshot_path}."
        )
    env_file = ((config_data.get("paths") or {}).get("env_file"))
    maybe_load_env_file(env_file, base_dir=Path.cwd())
    return config_data


def _task_specs_from_config(
    config_data: dict[str, Any],
    *,
    task_group_override: str | None,
    task_ids_override: list[int] | None,
) -> list[TaskSpec]:
    if task_ids_override is not None and task_group_override is None:
        raise ValueError("Task group must be provided when task ids are overridden.")

    dataset_cfg = config_data.get("dataset") or {}
    metaworld_tasks = dataset_cfg.get("metaworld_tasks") or {}
    libero_tasks = dataset_cfg.get("libero_tasks") or {}

    if metaworld_tasks:
        from lerobot.envs.metaworld import get_task_instruction
        tasks = metaworld_tasks
    elif libero_tasks:
        from lerobot.envs.libero import get_task_instruction
        tasks = libero_tasks
    else:
        raise ValueError(
            "Could not find dataset.libero_tasks or dataset.metaworld_tasks in the saved run config."
        )

    task_specs: list[TaskSpec] = []
    if task_group_override is not None:
        if task_group_override not in tasks:
            raise ValueError(f"Unknown task group {task_group_override!r}.")
        task_ids = task_ids_override or [int(task_id) for task_id in tasks[task_group_override]]
        for task_id in task_ids:
            task_specs.append(
                TaskSpec(
                    task_group=str(task_group_override),
                    task_id=int(task_id),
                    task_instruction=get_task_instruction(
                        task_group=str(task_group_override),
                        task_id=int(task_id),
                    ),
                )
            )
        return sorted(task_specs, key=lambda item: (item.task_group, item.task_id))

    for task_group, task_ids in tasks.items():
        for task_id in task_ids:
            task_specs.append(
                TaskSpec(
                    task_group=str(task_group),
                    task_id=int(task_id),
                    task_instruction=get_task_instruction(
                        task_group=str(task_group),
                        task_id=int(task_id),
                    ),
                )
            )
    return sorted(task_specs, key=lambda item: (item.task_group, item.task_id))


def _evaluation_env_type_from_config(config_data: dict[str, Any]) -> str:
    dataset_cfg = config_data.get("dataset") or {}
    if dataset_cfg.get("metaworld_tasks"):
        return "metaworld"
    if dataset_cfg.get("libero_tasks"):
        return "libero"
    raise ValueError(
        "Could not infer evaluation env type: neither dataset.libero_tasks nor "
        "dataset.metaworld_tasks is set."
    )


def _initial_member_runs(config_data: dict[str, Any], run_layout: RunLayout) -> list[TrainedEnsembleMember]:
    selection_cfg = config_data.get("selection") or {}
    model_paths = list(dict.fromkeys(selection_cfg.get("ensemble_model_paths") or []))
    if not model_paths:
        policy_cfg = config_data.get("policy") or {}
        pretrained_path = policy_cfg.get("pretrained_path")
        if pretrained_path:
            model_paths = [pretrained_path]

    return [
        TrainedEnsembleMember(
            member_index=member_index,
            start_model_path=str(model_path),
            final_model_path=str(model_path),
            training_dir=str(run_layout.initial_member_dir(member_index)),
            seed=None,
        )
        for member_index, model_path in enumerate(model_paths)
    ]


def _training_manifest_path(round_record, round_layout: RoundLayout) -> Path:
    if round_record.training_manifest_path is not None:
        return resolve_local_path(round_record.training_manifest_path, must_exist=True)
    return round_layout.training_manifest_path


def _task_keys(task_specs: list[TaskSpec | TaskEvaluation]) -> list[tuple[str, int]]:
    return [(task.task_group, task.task_id) for task in task_specs]


def _can_reuse_member_evaluation(
    evaluation: MemberEvaluation,
    *,
    round_index: int,
    member_index: int,
    policy_type: str,
    env_type: str,
    task_specs: list[TaskSpec],
    n_rollouts: int,
    eval_batch_size: int,
    seed_start: int,
    seed_step: int,
    max_steps: int,
    device: str,
    use_async_envs: bool,
) -> bool:
    return (
        evaluation.round_index == round_index
        and evaluation.member_index == member_index
        and evaluation.policy_type == policy_type
        and getattr(evaluation, "env_type", "libero") == env_type
        and normalized_evaluation_device(evaluation.device) == normalized_evaluation_device(device)
        and evaluation.eval_batch_size == eval_batch_size
        and evaluation.n_rollouts_per_task == n_rollouts
        and evaluation.seed_start == seed_start
        and evaluation.seed_step == seed_step
        and evaluation.max_steps == max_steps
        and evaluation.use_async_envs == use_async_envs
        and _task_keys(evaluation.per_task) == _task_keys(task_specs)
    )


def _summarize_round(
    *,
    run_dir: Path,
    round_index: int,
    member_evaluations: list[tuple[MemberEvaluation, Path]],
) -> RoundEvaluation:
    member_summaries = [
        RoundMemberSummary(
            member_index=evaluation.member_index,
            model_path=evaluation.model_path,
            macro_avg_success_rate=evaluation.macro_avg_success_rate,
            pooled_success_rate=evaluation.pooled_success_rate,
            macro_avg_reward=evaluation.macro_avg_reward,
            macro_avg_steps=evaluation.macro_avg_steps,
            total_successes=evaluation.total_successes,
            total_rollouts=evaluation.total_rollouts,
            result_path=str(path),
            timings=evaluation.timings,
        )
        for evaluation, path in member_evaluations
    ]

    task_summaries: list[RoundTaskSummary] = []
    task_keys = sorted(
        {
            (task.task_group, task.task_id)
            for evaluation, _ in member_evaluations
            for task in evaluation.per_task
        }
    )
    for task_group, task_id in task_keys:
        matching_tasks = [
            task
            for evaluation, _ in member_evaluations
            for task in evaluation.per_task
            if task.task_group == task_group and task.task_id == task_id
        ]
        total_successes = sum(task.successes for task in matching_tasks)
        total_rollouts = sum(task.n_rollouts for task in matching_tasks)
        ci_lower, ci_upper = _compute_binomial_ci(total_successes, total_rollouts)
        task_summaries.append(
            RoundTaskSummary(
                task_group=task_group,
                task_id=task_id,
                task_instruction=matching_tasks[0].task_instruction,
                num_members=len(matching_tasks),
                total_successes=total_successes,
                total_rollouts=total_rollouts,
                pooled_success_rate=(total_successes / total_rollouts) if total_rollouts > 0 else 0.0,
                pooled_success_rate_ci_lower=ci_lower,
                pooled_success_rate_ci_upper=ci_upper,
                mean_member_success_rate=_mean([task.success_rate for task in matching_tasks]),
                std_member_success_rate=_std([task.success_rate for task in matching_tasks]),
                mean_member_avg_reward=_mean([task.avg_reward for task in matching_tasks]),
                std_member_avg_reward=_std([task.avg_reward for task in matching_tasks]),
                mean_member_avg_steps=_mean([task.avg_steps for task in matching_tasks]),
                std_member_avg_steps=_std([task.avg_steps for task in matching_tasks]),
            )
        )

    best_member = max(member_summaries, key=lambda item: item.macro_avg_success_rate)
    timings = {}
    total_timing = aggregate_timing_entries(
        [total_timing_entry(evaluation.timings) for evaluation, _ in member_evaluations]
    )
    if total_timing is not None:
        timings["total"] = total_timing

    return RoundEvaluation(
        run_dir=str(run_dir),
        round_index=round_index,
        num_members_evaluated=len(member_summaries),
        mean_member_macro_success_rate=_mean([item.macro_avg_success_rate for item in member_summaries]),
        std_member_macro_success_rate=_std([item.macro_avg_success_rate for item in member_summaries]),
        mean_member_pooled_success_rate=_mean([item.pooled_success_rate for item in member_summaries]),
        std_member_pooled_success_rate=_std([item.pooled_success_rate for item in member_summaries]),
        mean_member_macro_avg_reward=_mean([item.macro_avg_reward for item in member_summaries]),
        std_member_macro_avg_reward=_std([item.macro_avg_reward for item in member_summaries]),
        mean_member_macro_avg_steps=_mean([item.macro_avg_steps for item in member_summaries]),
        std_member_macro_avg_steps=_std([item.macro_avg_steps for item in member_summaries]),
        best_member_index=best_member.member_index,
        best_member_macro_success_rate=best_member.macro_avg_success_rate,
        member_results=member_summaries,
        per_task=task_summaries,
        timings=timings,
    )


def _summarize_run(
    *,
    run_name: str,
    run_dir: Path,
    task_specs: list[TaskSpec],
    round_evaluations: list[tuple[RoundEvaluation, Path]],
) -> RunEvaluation:
    timings = {}
    total_timing = aggregate_timing_entries(
        [total_timing_entry(evaluation.timings) for evaluation, _ in round_evaluations]
    )
    if total_timing is not None:
        timings["total"] = total_timing

    return RunEvaluation(
        run_name=run_name,
        run_dir=str(run_dir),
        num_rounds_evaluated=len(round_evaluations),
        tasks=task_specs,
        rounds=[
            RunRoundSummary(
                round_index=evaluation.round_index,
                num_members_evaluated=evaluation.num_members_evaluated,
                mean_member_macro_success_rate=evaluation.mean_member_macro_success_rate,
                mean_member_pooled_success_rate=evaluation.mean_member_pooled_success_rate,
                mean_member_macro_avg_reward=evaluation.mean_member_macro_avg_reward,
                mean_member_macro_avg_steps=evaluation.mean_member_macro_avg_steps,
                best_member_index=evaluation.best_member_index,
                best_member_macro_success_rate=evaluation.best_member_macro_success_rate,
                result_path=str(path),
                timings=evaluation.timings,
            )
            for evaluation, path in round_evaluations
        ],
        timings=timings,
    )


def _member_runs(training_manifest: TrainingManifest) -> list[TrainedEnsembleMember]:
    return sorted(training_manifest.member_training_runs, key=lambda item: item.member_index)


def _evaluation_chunk(
    *,
    round_index: int,
    member_run: TrainedEnsembleMember,
    evaluation_path: Path,
    request_path: Path,
    log_path: Path,
) -> MemberEvaluationChunk:
    return MemberEvaluationChunk(
        round_index=round_index,
        member=member_run,
        evaluation_path=evaluation_path,
        request_path=request_path,
        log_path=log_path,
    )


def _launch_member_evaluation(
    *,
    chunk: MemberEvaluationChunk,
    assigned_device: str,
    policy_type: str,
    env_type: str,
    task_specs: list[TaskSpec],
    n_rollouts: int,
    eval_batch_size: int,
    seed_start: int,
    seed_step: int,
    max_steps: int,
    use_async_envs: bool,
    env_file: str | None,
) -> ActiveMemberEvaluation:
    worker_device, visible_device = normalize_worker_device(assigned_device)
    request = MemberEvaluationRequest(
        round_index=chunk.round_index,
        member_index=chunk.member.member_index,
        model_path=chunk.member.final_model_path,
        policy_type=policy_type,
        task_specs=task_specs,
        n_rollouts=n_rollouts,
        eval_batch_size=eval_batch_size,
        seed_start=seed_start,
        seed_step=seed_step,
        max_steps=max_steps,
        device=worker_device,
        use_async_envs=use_async_envs,
        env_type=env_type,
        env_file=env_file,
        result_path=str(chunk.evaluation_path),
    )
    save_member_evaluation_request(request, chunk.request_path)

    env = os.environ.copy()
    if visible_device is not None:
        env["CUDA_VISIBLE_DEVICES"] = visible_device

    logger.info(
        "Launching evaluation for round %s member %s on device %s",
        chunk.round_index,
        chunk.member.member_index,
        assigned_device,
    )
    log_handle = open(chunk.log_path, "w", encoding="utf-8")
    process = subprocess.Popen(
        [sys.executable, "-m", "iterative_fine_tuning.eval_worker", "--request_path", str(chunk.request_path)],
        cwd=str(Path.cwd()),
        env=env,
        stdout=log_handle,
        stderr=log_handle,
        text=True,
    )
    return ActiveMemberEvaluation(
        chunk=chunk,
        assigned_device=assigned_device,
        process=process,
        log_handle=log_handle,
    )


def _cleanup_active_launches(active_launches: list[ActiveMemberEvaluation]) -> None:
    for launch in active_launches:
        if launch.process.poll() is None:
            launch.process.terminate()
        if not launch.log_handle.closed:
            launch.log_handle.close()


def _append_error_to_log(log_path: Path, message: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("\n[orchestrator-error]\n")
        handle.write(message.rstrip())
        handle.write("\n")


def _normalize_slurm_state(raw_state: str | None) -> str | None:
    if raw_state is None:
        return None
    normalized = str(raw_state).strip().upper()
    if not normalized:
        return None
    return normalized.split()[0]


def _coalesce_slurm_states(raw_states: list[str]) -> str | None:
    states = [
        normalized
        for normalized in (_normalize_slurm_state(state) for state in raw_states)
        if normalized is not None
    ]
    if not states:
        return None
    for state in states:
        if state in SLURM_ACTIVE_STATES:
            return state
    for state in states:
        if state in SLURM_FAILURE_STATES:
            return state
    for state in states:
        if state in SLURM_SUCCESS_STATES:
            return state
    return states[0]


def _query_slurm_job_state(job_id: str | None) -> str | None:
    if not job_id:
        return None

    try:
        squeue_result = subprocess.run(
            ["squeue", "-h", "-j", str(job_id), "-o", "%T"],
            capture_output=True,
            text=True,
            check=False,
        )
        if squeue_result.returncode == 0:
            squeue_state = _coalesce_slurm_states(squeue_result.stdout.splitlines())
            if squeue_state is not None:
                return squeue_state
    except OSError:
        logger.debug("Could not query squeue for job %s", job_id, exc_info=True)

    try:
        sacct_result = subprocess.run(
            ["sacct", "-n", "-P", "-j", str(job_id), "--format=State"],
            capture_output=True,
            text=True,
            check=False,
        )
        if sacct_result.returncode == 0:
            states = [
                line.split("|", 1)[0]
                for line in sacct_result.stdout.splitlines()
                if line.strip()
            ]
            return _coalesce_slurm_states(states)
    except OSError:
        logger.debug("Could not query sacct for job %s", job_id, exc_info=True)

    return None


def _queue_member_evaluation(
    *,
    round_index: int,
    member_run: TrainedEnsembleMember,
    evaluation_path: Path,
    request_path: Path,
    log_path: Path,
    scheduled_member_keys: set[tuple[int, int]],
    round_member_evaluations: dict[int, list[tuple[MemberEvaluation, Path]]],
    pending_chunks: list[MemberEvaluationChunk],
    policy_type: str,
    env_type: str,
    task_specs: list[TaskSpec],
    n_rollouts: int,
    eval_batch_size: int,
    seed_start: int,
    seed_step: int,
    max_steps: int,
    expected_eval_device: str,
    use_async_envs: bool,
    skip_existing: bool,
) -> bool:
    member_key = (round_index, member_run.member_index)
    if member_key in scheduled_member_keys:
        return False

    if skip_existing and evaluation_path.exists():
        member_evaluation = load_member_evaluation(evaluation_path)
        if _can_reuse_member_evaluation(
            member_evaluation,
            round_index=round_index,
            member_index=member_run.member_index,
            policy_type=policy_type,
            env_type=env_type,
            task_specs=task_specs,
            n_rollouts=n_rollouts,
            eval_batch_size=eval_batch_size,
            seed_start=seed_start,
            seed_step=seed_step,
            max_steps=max_steps,
            device=expected_eval_device,
            use_async_envs=use_async_envs,
        ):
            logger.info(
                "Reusing cached evaluation for round %s member %s from %s",
                round_index,
                member_run.member_index,
                evaluation_path,
            )
            round_member_evaluations.setdefault(round_index, []).append(
                (member_evaluation, evaluation_path)
            )
            scheduled_member_keys.add(member_key)
            return True

        logger.info(
            "Ignoring cached evaluation for round %s member %s because it does not match the requested settings.",
            round_index,
            member_run.member_index,
        )

    pending_chunks.append(
        _evaluation_chunk(
            round_index=round_index,
            member_run=member_run,
            evaluation_path=evaluation_path,
            request_path=request_path,
            log_path=log_path,
        )
    )
    scheduled_member_keys.add(member_key)
    return True


def _discover_available_work(
    *,
    run_layout: RunLayout,
    config_data: dict[str, Any],
    round_filter: list[int] | None,
    member_filter: list[int] | None,
    include_initial_models: bool,
    policy_type: str,
    env_type: str,
    task_specs: list[TaskSpec],
    args: argparse.Namespace,
    expected_eval_device: str,
    use_async_envs: bool,
    pending_chunks: list[MemberEvaluationChunk],
    round_layouts: dict[int, RoundLayout],
    round_member_evaluations: dict[int, list[tuple[MemberEvaluation, Path]]],
    expected_member_counts: dict[int, int],
    scheduled_member_keys: set[tuple[int, int]],
    initial_round_index: int,
) -> tuple[RunState | None, bool]:
    discovered_new_work = False

    if include_initial_models:
        initial_member_runs = _initial_member_runs(config_data, run_layout)
        if initial_member_runs:
            round_member_evaluations.setdefault(initial_round_index, [])
            for member_run in initial_member_runs:
                if member_filter is not None and member_run.member_index not in member_filter:
                    continue
                member_dir = run_layout.initial_member_dir(member_run.member_index)
                evaluation_path = run_layout.initial_member_evaluation_path(member_run.member_index)
                discovered_new_work |= _queue_member_evaluation(
                    round_index=initial_round_index,
                    member_run=member_run,
                    evaluation_path=evaluation_path,
                    request_path=member_evaluation_request_path(member_dir),
                    log_path=member_evaluation_log_path(member_dir),
                    scheduled_member_keys=scheduled_member_keys,
                    round_member_evaluations=round_member_evaluations,
                    pending_chunks=pending_chunks,
                    policy_type=policy_type,
                    env_type=env_type,
                    task_specs=task_specs,
                    n_rollouts=args.n_rollouts,
                    eval_batch_size=max(1, args.eval_batch_size),
                    seed_start=args.seed_start,
                    seed_step=args.seed_step,
                    max_steps=args.max_steps,
                    expected_eval_device=expected_eval_device,
                    use_async_envs=use_async_envs,
                    skip_existing=args.skip_existing,
                )

    if not run_layout.run_state_path.exists():
        return None, discovered_new_work

    run_state = load_run_state(run_layout.run_state_path)
    for round_record in run_state.rounds:
        if round_filter is not None and round_record.round_index not in round_filter:
            continue
        if STAGE_TRAIN not in round_record.completed_stages and round_record.training_manifest_path is None:
            continue

        round_layout = run_layout.round_layout(round_record.round_index)
        training_manifest_path = _training_manifest_path(round_record, round_layout)
        if not training_manifest_path.exists():
            logger.debug(
                "Round %s is not ready for evaluation yet because %s is missing.",
                round_record.round_index,
                training_manifest_path,
            )
            continue

        training_manifest = load_training_manifest(training_manifest_path)
        round_layouts[round_record.round_index] = round_layout
        round_member_evaluations.setdefault(round_record.round_index, [])

        filtered_member_runs = [
            member_run
            for member_run in _member_runs(training_manifest)
            if member_filter is None or member_run.member_index in member_filter
        ]
        if filtered_member_runs:
            expected_member_counts[round_record.round_index] = len(filtered_member_runs)

        unscheduled_members = [
            member_run
            for member_run in filtered_member_runs
            if (round_record.round_index, member_run.member_index) not in scheduled_member_keys
        ]
        if unscheduled_members:
            logger.info(
                "Discovered round %s for evaluation across %s new ensemble member(s)",
                round_record.round_index,
                len(unscheduled_members),
            )
        for member_run in filtered_member_runs:
            training_dir = member_training_dir(
                round_layout,
                model_path=member_run.final_model_path,
                member_index=member_run.member_index,
            )
            evaluation_path = member_evaluation_path(
                round_layout,
                model_path=member_run.final_model_path,
                member_index=member_run.member_index,
            )
            discovered_new_work |= _queue_member_evaluation(
                round_index=round_record.round_index,
                member_run=member_run,
                evaluation_path=evaluation_path,
                request_path=member_evaluation_request_path(training_dir),
                log_path=member_evaluation_log_path(training_dir),
                scheduled_member_keys=scheduled_member_keys,
                round_member_evaluations=round_member_evaluations,
                pending_chunks=pending_chunks,
                policy_type=policy_type,
                env_type=env_type,
                task_specs=task_specs,
                n_rollouts=args.n_rollouts,
                eval_batch_size=max(1, args.eval_batch_size),
                seed_start=args.seed_start,
                seed_step=args.seed_step,
                max_steps=args.max_steps,
                expected_eval_device=expected_eval_device,
                use_async_envs=use_async_envs,
                skip_existing=args.skip_existing,
            )

    return run_state, discovered_new_work


def _fully_evaluated_round_indices(
    *,
    expected_member_counts: dict[int, int],
    round_member_evaluations: dict[int, list[tuple[MemberEvaluation, Path]]],
) -> set[int]:
    return {
        round_index
        for round_index, expected_member_count in expected_member_counts.items()
        if expected_member_count > 0
        and len(round_member_evaluations.get(round_index, [])) >= expected_member_count
    }


def _save_evaluation_summaries(
    *,
    run_name: str,
    run_dir: Path,
    run_layout: RunLayout,
    task_specs: list[TaskSpec],
    round_layouts: dict[int, RoundLayout],
    round_member_evaluations: dict[int, list[tuple[MemberEvaluation, Path]]],
    initial_round_index: int,
) -> None:
    round_evaluations: list[tuple[RoundEvaluation, Path]] = []
    initial_member_evaluations = sorted(
        round_member_evaluations.get(initial_round_index, []),
        key=lambda item: item[0].member_index,
    )
    if initial_member_evaluations:
        initial_evaluation = _summarize_round(
            run_dir=run_dir,
            round_index=initial_round_index,
            member_evaluations=initial_member_evaluations,
        )
        save_dataclass_json(initial_evaluation, run_layout.initial_evaluation_path)
        round_evaluations.append((initial_evaluation, run_layout.initial_evaluation_path))

    for round_index in sorted(round_layouts):
        member_evaluations = sorted(
            round_member_evaluations.get(round_index, []),
            key=lambda item: item[0].member_index,
        )
        if not member_evaluations:
            continue

        round_layout = round_layouts[round_index]
        round_evaluation = _summarize_round(
            run_dir=run_dir,
            round_index=round_index,
            member_evaluations=member_evaluations,
        )
        save_dataclass_json(round_evaluation, round_layout.round_evaluation_path)
        round_evaluations.append((round_evaluation, round_layout.round_evaluation_path))

    run_evaluation = _summarize_run(
        run_name=run_name,
        run_dir=run_dir,
        task_specs=task_specs,
        round_evaluations=round_evaluations,
    )
    save_dataclass_json(run_evaluation, run_layout.run_evaluation_path)


def _checkpoint_dirs_from_training_manifest(training_manifest_path: Path) -> list[Path]:
    if not training_manifest_path.exists():
        return []
    training_manifest = load_training_manifest(training_manifest_path)
    checkpoint_dirs: list[Path] = []
    for member_run in _member_runs(training_manifest):
        final_model_path = Path(member_run.final_model_path)
        for parent in final_model_path.parents:
            if parent.name == "checkpoints":
                checkpoint_dirs.append(parent)
                break
    return sorted(set(checkpoint_dirs))


def _cleanup_checkpoints_for_safe_rounds(
    *,
    completed_round_indices: set[int],
    round_layouts: dict[int, RoundLayout],
    cleaned_round_indices: set[int],
    final_cleanup: bool = False,
) -> None:
    for round_index in sorted(completed_round_indices):
        if round_index in cleaned_round_indices:
            continue
        round_layout = round_layouts.get(round_index)
        if round_layout is None:
            continue
        next_round_layout = round_layouts.get(round_index + 1)
        next_round_trained = (
            next_round_layout is not None
            and next_round_layout.training_manifest_path.exists()
        )
        if not final_cleanup and not next_round_trained:
            continue

        deleted_any = False
        for checkpoint_dir in _checkpoint_dirs_from_training_manifest(
            round_layout.training_manifest_path
        ):
            if checkpoint_dir.exists():
                logger.info(
                    "Deleting checkpoints for fully evaluated round %s: %s",
                    round_index,
                    checkpoint_dir,
                )
                shutil.rmtree(checkpoint_dir)
                deleted_any = True
        if deleted_any:
            cleaned_round_indices.add(round_index)


def main() -> None:
    register_third_party_plugins()
    args = _parse_args()

    run_context = _resolve_run_context(args)
    run_dir = run_context.run_dir
    run_layout = RunLayout(root=run_dir)
    logger.info("Evaluating iterative fine-tuning run in %s", run_dir)
    config_data = _load_config_data(
        run_layout,
        fallback_config_data=run_context.config_data,
    )
    run_state = load_run_state(run_layout.run_state_path) if run_layout.run_state_path.exists() else None
    round_filter = _parse_index_spec(args.rounds)
    member_filter = _parse_index_spec(args.members)
    task_ids_override = _parse_index_spec(args.task_ids)
    eval_batch_size = max(1, args.eval_batch_size)
    use_async_envs = not args.sync_envs
    member_devices = _parse_device_list(args.member_devices, args.device)
    device_slots = _build_device_slots(member_devices, args.workers_per_device)
    expected_eval_device = member_devices[0]
    task_specs = _task_specs_from_config(
        config_data,
        task_group_override=args.task_group,
        task_ids_override=task_ids_override,
    )
    env_type = _evaluation_env_type_from_config(config_data)
    policy_type = str(((config_data.get("policy") or {}).get("type")) or "smolvla")
    env_file = ((config_data.get("paths") or {}).get("env_file"))
    expected_round_indices = run_context.expected_round_indices
    if expected_round_indices is not None and round_filter is not None:
        expected_round_indices = {
            round_index for round_index in expected_round_indices if round_index in round_filter
        }
    watch_mode = bool(args.watch or args.watch_job_id is not None)

    pending_chunks: list[MemberEvaluationChunk] = []
    round_layouts: dict[int, RoundLayout] = {}
    round_member_evaluations: dict[int, list[tuple[MemberEvaluation, Path]]] = {}
    expected_member_counts: dict[int, int] = {}
    scheduled_member_keys: set[tuple[int, int]] = set()
    initial_round_index = -1
    include_initial_models = round_filter is None
    active_launches: list[ActiveMemberEvaluation] = []
    available_devices = list(device_slots)
    poll_interval_seconds = max(0.1, args.poll_interval_seconds)
    cleaned_checkpoint_round_indices: set[int] = set()
    logger.info(
        "Evaluation scheduler configured with %s device(s), %s worker slot(s) per device, %s total slot(s)",
        len(member_devices),
        args.workers_per_device,
        len(device_slots),
    )

    try:
        while True:
            run_state, discovered_new_work = _discover_available_work(
                run_layout=run_layout,
                config_data=config_data,
                round_filter=round_filter,
                member_filter=member_filter,
                include_initial_models=include_initial_models,
                policy_type=policy_type,
                env_type=env_type,
                task_specs=task_specs,
                args=args,
                expected_eval_device=expected_eval_device,
                use_async_envs=use_async_envs,
                pending_chunks=pending_chunks,
                round_layouts=round_layouts,
                round_member_evaluations=round_member_evaluations,
                expected_member_counts=expected_member_counts,
                scheduled_member_keys=scheduled_member_keys,
                initial_round_index=initial_round_index,
            )
            if run_state is not None:
                logger.debug("Observed run state with %s round record(s)", len(run_state.rounds))

            if discovered_new_work:
                pending_chunks.sort(key=lambda chunk: (chunk.round_index, chunk.member.member_index))
                _save_evaluation_summaries(
                    run_name=(run_state.run_name if run_state is not None and run_state.run_name else run_context.run_name),
                    run_dir=run_dir,
                    run_layout=run_layout,
                    task_specs=task_specs,
                    round_layouts=round_layouts,
                    round_member_evaluations=round_member_evaluations,
                    initial_round_index=initial_round_index,
                )

            while pending_chunks and available_devices:
                assigned_device = available_devices.pop(0)
                chunk = pending_chunks.pop(0)
                active_launches.append(
                    _launch_member_evaluation(
                        chunk=chunk,
                        assigned_device=assigned_device,
                        policy_type=policy_type,
                        env_type=env_type,
                        task_specs=task_specs,
                        n_rollouts=args.n_rollouts,
                        eval_batch_size=eval_batch_size,
                        seed_start=args.seed_start,
                        seed_step=args.seed_step,
                        max_steps=args.max_steps,
                        use_async_envs=use_async_envs,
                        env_file=env_file,
                    )
                )

            completed_round_indices = _fully_evaluated_round_indices(
                expected_member_counts=expected_member_counts,
                round_member_evaluations=round_member_evaluations,
            )
            if args.cleanup_checkpoints_when_safe:
                _cleanup_checkpoints_for_safe_rounds(
                    completed_round_indices=completed_round_indices,
                    round_layouts=round_layouts,
                    cleaned_round_indices=cleaned_checkpoint_round_indices,
                )
            if (
                expected_round_indices is not None
                and expected_round_indices.issubset(completed_round_indices)
                and not pending_chunks
                and not active_launches
            ):
                if args.cleanup_checkpoints_when_safe:
                    _cleanup_checkpoints_for_safe_rounds(
                        completed_round_indices=completed_round_indices,
                        round_layouts=round_layouts,
                        cleaned_round_indices=cleaned_checkpoint_round_indices,
                        final_cleanup=True,
                    )
                logger.info("All expected rounds have been evaluated.")
                break

            watched_job_state = _query_slurm_job_state(args.watch_job_id)
            if (
                watched_job_state is not None
                and watched_job_state not in SLURM_ACTIVE_STATES
                and not pending_chunks
                and not active_launches
            ):
                missing_rounds = (
                    sorted(expected_round_indices - completed_round_indices)
                    if expected_round_indices is not None
                    else []
                )
                if missing_rounds:
                    raise RuntimeError(
                        "Watched training job "
                        f"{args.watch_job_id} ended with state {watched_job_state}, "
                        f"but rounds {missing_rounds} were never fully evaluated."
                    )
                if watched_job_state not in SLURM_SUCCESS_STATES:
                    raise RuntimeError(
                        f"Watched training job {args.watch_job_id} ended with state {watched_job_state}."
                    )
                if args.cleanup_checkpoints_when_safe:
                    _cleanup_checkpoints_for_safe_rounds(
                        completed_round_indices=completed_round_indices,
                        round_layouts=round_layouts,
                        cleaned_round_indices=cleaned_checkpoint_round_indices,
                        final_cleanup=True,
                    )
                logger.info(
                    "Watched training job %s finished with state %s and no evaluation work remains.",
                    args.watch_job_id,
                    watched_job_state,
                )
                break

            if not active_launches:
                if not watch_mode and not pending_chunks:
                    break
                time.sleep(poll_interval_seconds)
                continue

            time.sleep(poll_interval_seconds)
            still_running: list[ActiveMemberEvaluation] = []
            finished_any_evaluations = False
            for launch in active_launches:
                return_code = launch.process.poll()
                if return_code is None:
                    still_running.append(launch)
                    continue

                launch.log_handle.close()
                available_devices.append(launch.assigned_device)
                if return_code != 0:
                    failure_message = (
                        f"Evaluation failed for round {launch.chunk.round_index} member "
                        f"{launch.chunk.member.member_index} with code {return_code}.\n"
                        f"Last log lines from {launch.chunk.log_path}:\n"
                        f"{tail_log(launch.chunk.log_path)}"
                    )
                    _append_error_to_log(launch.chunk.log_path, failure_message)
                    _cleanup_active_launches(still_running)
                    raise RuntimeError(failure_message)
                if not launch.chunk.evaluation_path.exists():
                    failure_message = (
                        f"Evaluation finished for round {launch.chunk.round_index} member "
                        f"{launch.chunk.member.member_index}, but no result file was written to "
                        f"{launch.chunk.evaluation_path}.\n"
                        f"Last log lines from {launch.chunk.log_path}:\n"
                        f"{tail_log(launch.chunk.log_path)}"
                    )
                    _append_error_to_log(launch.chunk.log_path, failure_message)
                    _cleanup_active_launches(still_running)
                    raise RuntimeError(failure_message)

                member_evaluation = load_member_evaluation(launch.chunk.evaluation_path)
                round_member_evaluations[launch.chunk.round_index].append(
                    (member_evaluation, launch.chunk.evaluation_path)
                )
                finished_any_evaluations = True
                logger.info(
                    "Finished evaluation for round %s member %s -> %s",
                    launch.chunk.round_index,
                    launch.chunk.member.member_index,
                    launch.chunk.evaluation_path,
                )

            active_launches = still_running
            if finished_any_evaluations:
                _save_evaluation_summaries(
                    run_name=(run_state.run_name if run_state is not None and run_state.run_name else run_context.run_name),
                    run_dir=run_dir,
                    run_layout=run_layout,
                    task_specs=task_specs,
                    round_layouts=round_layouts,
                    round_member_evaluations=round_member_evaluations,
                    initial_round_index=initial_round_index,
                )
    finally:
        _cleanup_active_launches(active_launches)

    _save_evaluation_summaries(
        run_name=(run_state.run_name if run_state is not None and run_state.run_name else run_context.run_name),
        run_dir=run_dir,
        run_layout=run_layout,
        task_specs=task_specs,
        round_layouts=round_layouts,
        round_member_evaluations=round_member_evaluations,
        initial_round_index=initial_round_index,
    )
    logger.info("Saved run evaluation summary to %s", run_layout.run_evaluation_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
