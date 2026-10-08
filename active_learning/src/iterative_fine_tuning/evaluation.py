"""Shared evaluation helpers for iterative fine-tuning."""

from __future__ import annotations

import logging
import math
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from lerobot.envs.configs import LiberoEnv
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.utils.random_utils import set_seed

from .constants import MEMBER_EVALUATION_LOG_NAME, MEMBER_EVALUATION_REQUEST_NAME
from .env import maybe_load_env_file, resolve_model_reference
from .layout import RoundLayout
from .manifests import (
    MemberEvaluation,
    TaskEvaluation,
    TaskSpec,
    load_json,
    make_timing_entry,
    save_dataclass_json,
    utc_timestamp,
)

logger = logging.getLogger(__name__)


@dataclass
class MemberEvaluationRequest:
    round_index: int
    member_index: int
    model_path: str
    policy_type: str
    task_specs: list[TaskSpec] = field(default_factory=list)
    n_rollouts: int = 0
    eval_batch_size: int = 1
    seed_start: int = 0
    seed_step: int = 1
    max_steps: int = 0
    device: str = "cuda"
    use_async_envs: bool = True
    env_type: str = "libero"
    env_file: str | None = None
    result_path: str | None = None
    # Rollouts per env seed. With >1, each seed (= initial state) is rolled out this many
    # times inside the same batch, differing only through policy sampling noise, which
    # yields a per-seed success rate (calibration studies). n_rollouts then counts seeds.
    rollouts_per_seed: int = 1


def member_evaluation_request_path(training_dir: Path) -> Path:
    return training_dir / MEMBER_EVALUATION_REQUEST_NAME


def member_evaluation_log_path(training_dir: Path) -> Path:
    return training_dir / MEMBER_EVALUATION_LOG_NAME


def load_member_evaluation_request(path: Path) -> MemberEvaluationRequest:
    data = load_json(path)
    return MemberEvaluationRequest(
        round_index=int(data["round_index"]),
        member_index=int(data["member_index"]),
        model_path=str(data["model_path"]),
        policy_type=str(data["policy_type"]),
        task_specs=[TaskSpec(**item) for item in data.get("task_specs", [])],
        n_rollouts=int(data["n_rollouts"]),
        eval_batch_size=int(data["eval_batch_size"]),
        seed_start=int(data["seed_start"]),
        seed_step=int(data["seed_step"]),
        max_steps=int(data["max_steps"]),
        device=str(data.get("device", "cuda")),
        use_async_envs=bool(data.get("use_async_envs", True)),
        env_type=str(data.get("env_type", "libero")),
        env_file=data.get("env_file"),
        result_path=data.get("result_path"),
        rollouts_per_seed=int(data.get("rollouts_per_seed", 1)),
    )


def save_member_evaluation_request(request: MemberEvaluationRequest, path: Path) -> None:
    save_dataclass_json(request, path)


def normalize_worker_device(device: str) -> tuple[str, str | None]:
    normalized = str(device).strip()
    if normalized.isdigit():
        normalized = f"cuda:{normalized}"
    if normalized.startswith("cuda:"):
        return "cuda", normalized.split(":", 1)[1]
    return normalized, None


def normalized_evaluation_device(device: str) -> str:
    return normalize_worker_device(device)[0]


def tail_log(path: Path, *, num_lines: int = 40) -> str:
    if not path.exists():
        return "<log file missing>"
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    tail = lines[-num_lines:]
    return "\n".join(tail) if tail else "<log file empty>"


def member_evaluation_path(round_layout: RoundLayout, model_path: str, member_index: int) -> Path:
    training_dir = _member_training_dir(round_layout, model_path=model_path, member_index=member_index)
    return training_dir / round_layout.member_evaluation_path(member_index).name


def member_training_dir(round_layout: RoundLayout, model_path: str, member_index: int) -> Path:
    return _member_training_dir(round_layout, model_path=model_path, member_index=member_index)


def _member_training_dir(round_layout: RoundLayout, *, model_path: str, member_index: int) -> Path:
    return round_layout.member_training_dir(member_index)


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _compute_binomial_ci(successes: int, trials: int) -> tuple[float, float]:
    if trials <= 0:
        return 0.0, 0.0
    success_rate = successes / trials
    standard_error = math.sqrt(success_rate * (1.0 - success_rate) / trials)
    return (
        max(0.0, success_rate - 1.96 * standard_error),
        min(1.0, success_rate + 1.96 * standard_error),
    )


def _extract_batch_successes(info: dict[str, Any], batch_size: int) -> list[bool]:
    if "final_info" in info:
        final_info = info["final_info"]
        if isinstance(final_info, dict) and "is_success" in final_info:
            raw_success = final_info["is_success"]
            if isinstance(raw_success, np.ndarray):
                return [bool(item) for item in raw_success.tolist()]
            if isinstance(raw_success, list):
                return [bool(item) for item in raw_success]
        if isinstance(final_info, list):
            return [bool((item or {}).get("is_success", False)) for item in final_info]

    if "is_success" in info:
        raw_success = info["is_success"]
        if isinstance(raw_success, np.ndarray):
            return [bool(item) for item in raw_success.tolist()]
        if isinstance(raw_success, list):
            return [bool(item) for item in raw_success]

    return [False] * batch_size


def _rollout_batch(
    *,
    env,
    policy,
    preprocessor,
    postprocessor,
    env_preprocessor,
    env_postprocessor,
    task_instruction: str,
    max_steps: int,
    seeds: list[int] | None,
    use_amp: bool,
    device_type: str,
) -> list[dict[str, Any]]:
    from lerobot.envs.utils import preprocess_observation

    batch_size = env.num_envs
    policy.reset()
    obs, _ = env.reset(seed=seeds)
    done = np.zeros(batch_size, dtype=bool)
    step = 0
    steps = np.zeros(batch_size, dtype=np.int64)
    total_rewards = np.zeros(batch_size, dtype=np.float64)
    successes = np.zeros(batch_size, dtype=bool)
    task_batch = [task_instruction] * batch_size

    while not np.all(done) and step < max_steps:
        observation = preprocess_observation(obs)
        observation["task"] = task_batch
        observation = env_preprocessor(observation)
        observation = preprocessor(observation)

        autocast_ctx = (
            torch.autocast(device_type=device_type)
            if use_amp and device_type in {"cuda", "cpu"}
            else nullcontext()
        )
        with torch.inference_mode(), autocast_ctx:
            action = policy.select_action(observation)

        action = postprocessor(action)
        action_dict = {"action": action}
        action_dict = env_postprocessor(action_dict)
        raw_action = action_dict["action"]
        if isinstance(raw_action, torch.Tensor):
            # Cast to float32 first: bfloat16 policies (e.g. X-VLA) have no
            # numpy equivalent. No-op for float32 policies like SmolVLA.
            raw_action = raw_action.detach().to(torch.float32).cpu().numpy()
        if raw_action.ndim == 1:
            raw_action = raw_action[None, :]

        obs, reward, terminated, truncated, info = env.step(raw_action)
        active_mask = ~done
        reward_array = np.asarray(reward, dtype=np.float64)
        terminated_array = np.asarray(terminated, dtype=bool)
        truncated_array = np.asarray(truncated, dtype=bool)
        batch_successes = np.asarray(_extract_batch_successes(info, batch_size), dtype=bool)

        total_rewards[active_mask] += reward_array[active_mask]
        steps[active_mask] += 1
        successes[active_mask] |= batch_successes[active_mask]
        done = done | terminated_array | truncated_array
        step += 1

    return [
        {
            "success": bool(successes[env_index]),
            "steps": int(steps[env_index]),
            "total_reward": float(total_rewards[env_index]),
        }
        for env_index in range(batch_size)
    ]


def _evaluate_task(
    *,
    task_spec: TaskSpec,
    env_type: str,
    policy,
    preprocessor,
    postprocessor,
    policy_cfg,
    n_rollouts: int,
    eval_batch_size: int,
    seed_start: int,
    seed_step: int,
    max_steps: int,
    use_async_envs: bool,
    use_amp: bool,
    device_type: str,
    rollouts_per_seed: int = 1,
) -> TaskEvaluation:
    task_started_at = utc_timestamp()
    task_start_time = time.perf_counter()
    rollouts_per_seed = max(1, int(rollouts_per_seed))
    if rollouts_per_seed > 1 and eval_batch_size % rollouts_per_seed != 0:
        # All repeats of one seed must share a batch: set_seed() is called once per batch,
        # so the same env seed in two batches would reproduce identical rollouts.
        raise ValueError(
            f"eval_batch_size ({eval_batch_size}) must be a multiple of rollouts_per_seed "
            f"({rollouts_per_seed})."
        )
    if env_type == "libero":
        env_cfg = LiberoEnv(task=task_spec.task_group, task_ids=[task_spec.task_id])
    elif env_type == "pusht":
        from lerobot.envs.configs import PushtEnv

        # Push-T is single-task; make_env normalizes to {"pusht": {0: vec}}.
        env_cfg = PushtEnv()
    else:
        raise ValueError(f"Unsupported evaluation env_type {env_type!r}.")
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=env_cfg,
        policy_cfg=policy_cfg,
    )
    # n_rollouts counts distinct seeds; each seed is rolled out rollouts_per_seed times.
    seeds_per_batch = min(max(1, eval_batch_size // rollouts_per_seed), max(1, n_rollouts))
    batch_size = seeds_per_batch * rollouts_per_seed
    envs = make_env(env_cfg, n_envs=batch_size, use_async_envs=use_async_envs)
    env = envs[task_spec.task_group][task_spec.task_id]

    rollout_results: list[dict[str, Any]] = []
    try:
        for batch_start in tqdm(
            range(0, n_rollouts, seeds_per_batch),
            desc=f"{task_spec.task_group}/{task_spec.task_id}",
            leave=False,
        ):
            # Seeds beyond n_rollouts only pad the final batch; their results are dropped.
            batch_seed_group = [
                seed_start + (seed_index * seed_step)
                for seed_index in range(batch_start, batch_start + seeds_per_batch)
            ]
            batch_seeds = [seed for seed in batch_seed_group for _ in range(rollouts_per_seed)]
            batch_repeats = [repeat for _ in batch_seed_group for repeat in range(rollouts_per_seed)]
            if hasattr(env, "envs"):
                for i, sub_env in enumerate(env.envs):
                    base_env = getattr(sub_env, "unwrapped", sub_env)
                    init_states = getattr(base_env, "_init_states", None)
                    if init_states is not None:
                        base_env._init_state_id = (batch_start + i) % len(init_states)
            set_seed(batch_seeds[0])
            batch_rollout_results = _rollout_batch(
                env=env,
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                env_preprocessor=env_preprocessor,
                env_postprocessor=env_postprocessor,
                task_instruction=task_spec.task_instruction,
                max_steps=max_steps,
                seeds=batch_seeds,
                use_amp=use_amp,
                device_type=device_type,
            )
            remaining = (n_rollouts - batch_start) * rollouts_per_seed
            for rollout_seed, rollout_repeat, rollout_result in zip(
                batch_seeds[:remaining],
                batch_repeats[:remaining],
                batch_rollout_results[:remaining],
                strict=True,
            ):
                rollout_results.append(
                    {
                        **rollout_result,
                        "seed": int(rollout_seed),
                        **({"repeat": int(rollout_repeat)} if rollouts_per_seed > 1 else {}),
                    }
                )
    finally:
        env.close()

    successes = sum(int(item["success"]) for item in rollout_results)
    ci_lower, ci_upper = _compute_binomial_ci(successes, len(rollout_results))

    return TaskEvaluation(
        task_group=task_spec.task_group,
        task_id=task_spec.task_id,
        task_instruction=task_spec.task_instruction,
        n_rollouts=len(rollout_results),
        seed_start=seed_start,
        seed_step=seed_step,
        successes=successes,
        success_rate=(successes / len(rollout_results)) if rollout_results else 0.0,
        success_rate_ci_lower=ci_lower,
        success_rate_ci_upper=ci_upper,
        avg_steps=_mean([float(item["steps"]) for item in rollout_results]),
        avg_reward=_mean([float(item["total_reward"]) for item in rollout_results]),
        per_rollout_results=rollout_results,
        timings={
            "total": make_timing_entry(
                started_at=task_started_at,
                duration_s=time.perf_counter() - task_start_time,
            )
        },
        rollouts_per_seed=rollouts_per_seed,
    )


def evaluate_member_request(request: MemberEvaluationRequest) -> MemberEvaluation:
    evaluation_started_at = utc_timestamp()
    evaluation_start_time = time.perf_counter()
    if request.result_path is None:
        raise ValueError("Member evaluation request is missing result_path.")

    maybe_load_env_file(request.env_file, base_dir=Path.cwd())

    model_path = resolve_model_reference(request.model_path, base_dir=Path.cwd())
    logger.info(
        "Evaluating round %s member %s from %s on %s (async_envs=%s)",
        request.round_index,
        request.member_index,
        model_path,
        request.device,
        request.use_async_envs,
    )

    policy_cls = get_policy_class(request.policy_type)
    policy = policy_cls.from_pretrained(pretrained_name_or_path=str(model_path))
    policy.to(request.device)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(model_path),
        preprocessor_overrides={"device_processor": {"device": request.device}},
    )

    device_type = torch.device(request.device).type
    use_amp = bool(getattr(policy.config, "use_amp", False))
    per_task = [
        _evaluate_task(
            task_spec=task_spec,
            env_type=request.env_type,
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            policy_cfg=policy.config,
            n_rollouts=request.n_rollouts,
            eval_batch_size=request.eval_batch_size,
            seed_start=request.seed_start,
            seed_step=request.seed_step,
            max_steps=task_spec.max_steps or request.max_steps,
            use_async_envs=request.use_async_envs,
            use_amp=use_amp,
            device_type=device_type,
            rollouts_per_seed=request.rollouts_per_seed,
        )
        for task_spec in request.task_specs
    ]

    total_successes = sum(task.successes for task in per_task)
    total_rollouts = sum(task.n_rollouts for task in per_task)
    ci_lower, ci_upper = _compute_binomial_ci(total_successes, total_rollouts)

    evaluation = MemberEvaluation(
        round_index=request.round_index,
        member_index=request.member_index,
        model_path=str(model_path),
        policy_type=request.policy_type,
        env_type=request.env_type,
        device=request.device,
        eval_batch_size=request.eval_batch_size,
        n_rollouts_per_task=request.n_rollouts,
        seed_start=request.seed_start,
        seed_step=request.seed_step,
        max_steps=request.max_steps,
        use_async_envs=request.use_async_envs,
        num_tasks=len(request.task_specs),
        macro_avg_success_rate=_mean([task.success_rate for task in per_task]),
        macro_avg_reward=_mean([task.avg_reward for task in per_task]),
        macro_avg_steps=_mean([task.avg_steps for task in per_task]),
        total_successes=total_successes,
        total_rollouts=total_rollouts,
        pooled_success_rate=(total_successes / total_rollouts) if total_rollouts > 0 else 0.0,
        pooled_success_rate_ci_lower=ci_lower,
        pooled_success_rate_ci_upper=ci_upper,
        per_task=per_task,
        timings={
            "total": make_timing_entry(
                started_at=evaluation_started_at,
                duration_s=time.perf_counter() - evaluation_start_time,
            )
        },
        rollouts_per_seed=max(1, int(request.rollouts_per_seed)),
    )
    save_dataclass_json(evaluation, Path(request.result_path))
    return evaluation
