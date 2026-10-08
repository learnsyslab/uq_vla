"""AMF-style selection: request a demonstration for the task that minimizes posterior covariance.

Baseline reproducing *Active Fine-Tuning of Multi-Task Policies* (https://arxiv.org/abs/2410.05026)
inside the iterative fine-tuning pipeline.

Each demonstration acquired so far is embedded through last-layer loss gradients. Those embeddings
induce a linear (NTK) kernel, hence a Gaussian posterior over the expert policy, and conditioning on
a candidate demonstration shrinks that posterior. Each round greedily requests demonstrations for the
tasks whose demonstrations shrink the task-uniform posterior variance the most.

Because a demonstration cannot be inspected before it is requested, a task is represented by the
demonstrations already acquired for it. The first rounds therefore prefill one demonstration per
task, spending the regular per-round episode budget.

Data loading reuses ``selection._selection_backend``: each candidate task gets its own per-task
dataset, and episode ids are reported in the same global index space as every other strategy.
"""

from __future__ import annotations

import copy
import logging
import random
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data.dataloader import default_collate

from lerobot.datasets.factory import make_dataset
from lerobot.uncertainty.uncertainty_scoring.loss_gradient_embeddings import (
    compute_loss_gradient_embeddings,
    is_supported_policy_type,
)

from .config import IterativeFineTuningConfig
from .layout import RoundLayout
from .manifests import (
    EpisodeSelection,
    SelectionManifest,
    TaskSelection,
    make_timing_entry,
    save_dataclass_json,
    utc_timestamp,
)

logger = logging.getLogger(__name__)

TaskKey = tuple[str, int]


# --------------------------------------------------------------------------------------
# Acquisition criterion. Pure functions over a Gram matrix, so they are testable in isolation.
#
# Demonstration ``i`` owns rows ``[i * block_size, (i + 1) * block_size)`` of ``gram``, one row per
# frame sampled from that demonstration.
# --------------------------------------------------------------------------------------


@dataclass
class AMFPick:
    """One greedy acquisition decision."""

    task_key: TaskKey
    proxy_demo: int
    objective: float
    variance_reduction: float


def _demo_rows(demo_indices: list[int], block_size: int) -> np.ndarray:
    return np.concatenate([np.arange(i * block_size, (i + 1) * block_size) for i in demo_indices])


def posterior_traces(
    gram: np.ndarray,
    block_size: int,
    conditioned_demos: list[int],
    noise: float,
) -> np.ndarray:
    """Trace of every demonstration's posterior covariance after conditioning on ``conditioned_demos``.

    The posterior of block ``R`` given conditioning rows ``C`` is
    ``G[R, R] - G[R, C] (G[C, C] + noise I)^-1 G[C, R]``; only its trace is needed.
    """
    num_demos = gram.shape[0] // block_size
    prior = np.diag(gram).reshape(num_demos, block_size).sum(axis=1)
    if not conditioned_demos:
        return prior

    rows = _demo_rows(conditioned_demos, block_size)
    conditioning = gram[np.ix_(rows, rows)] + noise * np.eye(len(rows))
    cross = gram[rows]
    reduction = (cross * np.linalg.solve(conditioning, cross)).sum(axis=0)
    return prior - reduction.reshape(num_demos, block_size).sum(axis=1)


def task_uniform_objective(traces: np.ndarray, task_demos: dict[TaskKey, list[int]]) -> float:
    """Posterior variance averaged within each task, then uniformly across tasks."""
    return float(np.mean([traces[demos].mean() for demos in task_demos.values()]))


def greedy_task_picks(
    gram: np.ndarray,
    block_size: int,
    task_demos: dict[TaskKey, list[int]],
    *,
    budget: int,
    noise: float,
    capacity: dict[TaskKey, int],
) -> tuple[list[AMFPick], dict[TaskKey, float]]:
    """Greedily pick the ``budget`` tasks that minimize the task-uniform posterior variance.

    A task is scored by the mean, over the demonstrations acquired for it, of the objective obtained
    when conditioning on that demonstration. The proxy achieving the minimum joins the conditioning
    set, so later picks in the same round see the variance already removed by earlier ones.

    Returns the picks and, for reporting, the first-step variance reduction of every task.
    """
    remaining = dict(capacity)
    conditioned: list[int] = []
    objective = task_uniform_objective(posterior_traces(gram, block_size, [], noise), task_demos)
    picks: list[AMFPick] = []
    first_step_reductions: dict[TaskKey, float] = {}

    for step in range(budget):
        scored: list[tuple[float, TaskKey, int]] = []
        for task_key, demos in task_demos.items():
            per_proxy = [
                task_uniform_objective(
                    posterior_traces(gram, block_size, [*conditioned, demo], noise), task_demos
                )
                for demo in demos
            ]
            best_proxy = demos[int(np.argmin(per_proxy))]
            scored.append((float(np.mean(per_proxy)), task_key, best_proxy))

        scored.sort(key=lambda item: (item[0], item[1]))
        if step == 0:
            first_step_reductions = {task_key: objective - score for score, task_key, _ in scored}

        available = next((item for item in scored if remaining.get(item[1], 0) > 0), None)
        if available is None:
            break

        score, task_key, proxy_demo = available
        remaining[task_key] -= 1
        picks.append(
            AMFPick(
                task_key=task_key,
                proxy_demo=proxy_demo,
                objective=score,
                variance_reduction=objective - score,
            )
        )
        conditioned.append(proxy_demo)
        objective = task_uniform_objective(
            posterior_traces(gram, block_size, conditioned, noise), task_demos
        )

    return picks, first_step_reductions


def prefill_task_slots(
    task_keys: list[TaskKey],
    *,
    round_index: int,
    episodes_per_round: int,
    demos_per_task: int,
    rng: random.Random,
) -> list[TaskKey]:
    """Slice of the prefill schedule belonging to ``round_index``.

    The schedule lists every task ``demos_per_task`` times in a seeded order and is consumed
    ``episodes_per_round`` entries at a time, so the per-round budget is never exceeded.
    """
    order = list(task_keys)
    rng.shuffle(order)
    schedule = [task_key for task_key in order for _ in range(demos_per_task)]
    start = round_index * episodes_per_round
    return schedule[start : start + episodes_per_round]


# --------------------------------------------------------------------------------------
# Per-task dataset loading. Mirrors `selection.run_selection_round`'s per-task loop so AMF
# addresses the same episode-id space as every other strategy.
# --------------------------------------------------------------------------------------


@dataclass
class _TaskContext:
    task_group: str
    task_id: int
    instruction: str
    dataset: object
    offset: int
    episode_ids: list[int]  # global ids


def _build_task_contexts(cfg: IterativeFineTuningConfig) -> list[_TaskContext]:
    # Deferred import: `selection.py` imports `run_amf_selection_round` from this module at
    # module load time, so importing `selection` back at module level here would be circular.
    from .selection import _selection_backend

    backend, get_instruction, filter_episodes = _selection_backend(cfg)

    contexts: list[_TaskContext] = []
    for task_group, task_ids in cfg.selection.candidate_tasks.items():
        for task_id in task_ids:
            dataset_cfg = copy.deepcopy(cfg.dataset)
            if backend == "pusht":
                dataset_cfg.pusht_tasks = {task_group: [task_id]}
            else:
                dataset_cfg.libero_tasks = {task_group: [task_id]}
            dataset = make_dataset(dataset_cfg=dataset_cfg, policy_cfg=cfg.policy)

            instruction = get_instruction(task_group=task_group, task_id=task_id)

            offset = 0
            local_episode_ids = [
                int(episode_id)
                for episode_id in filter_episodes(
                    dataset=dataset, tasks_to_use={task_group: [task_id]}
                )
            ]
            contexts.append(
                _TaskContext(
                    task_group=task_group,
                    task_id=int(task_id),
                    instruction=instruction,
                    dataset=dataset,
                    offset=offset,
                    episode_ids=[local_id + offset for local_id in local_episode_ids],
                )
            )
    return contexts


def _episode_frame_bounds(dataset) -> dict[int, tuple[int, int]]:
    episodes = dataset.meta.episodes
    return {
        int(episode_id): (int(start), int(end))
        for episode_id, start, end in zip(
            episodes["episode_index"],
            episodes["dataset_from_index"],
            episodes["dataset_to_index"],
            strict=True,
        )
    }


def _index_task_contexts(
    contexts: list[_TaskContext],
) -> tuple[dict[int, TaskKey], dict[TaskKey, _TaskContext], dict[int, tuple[int, int]]]:
    """Build global lookup tables from the per-task contexts.

    ``frame_bounds`` is keyed by global episode id but stores *local* frame bounds (the frame
    index space of that task's own per-task dataset), matching the ``frame_index`` convention
    used elsewhere in this pipeline (episode-level ids are global, frame indices
    inside a selection manifest are local to the per-task repo).
    """
    task_of_episode: dict[int, TaskKey] = {}
    context_of_task: dict[TaskKey, _TaskContext] = {}
    frame_bounds: dict[int, tuple[int, int]] = {}
    for context in contexts:
        task_key = (context.task_group, context.task_id)
        context_of_task[task_key] = context
        local_bounds = _episode_frame_bounds(context.dataset)
        for global_id in context.episode_ids:
            task_of_episode[global_id] = task_key
            frame_bounds[global_id] = local_bounds[global_id - context.offset]
    return task_of_episode, context_of_task, frame_bounds


def _demo_frame_indices(bounds: tuple[int, int], frames_per_demo: int) -> np.ndarray:
    """Evenly spaced frames, repeating frames if the episode is shorter than ``frames_per_demo``."""
    start, end = bounds
    return np.linspace(start, end - 1, frames_per_demo).round().astype(int)


def _acquired_demos_by_task(
    acquired_episode_ids: list[int],
    task_of_episode: dict[int, TaskKey],
    *,
    max_demos_per_task: int | None,
) -> dict[TaskKey, list[int]]:
    """Group acquired episodes by task, preserving acquisition order and keeping the most recent."""
    demos: dict[TaskKey, list[int]] = {}
    for episode_id in acquired_episode_ids:
        task_key = task_of_episode.get(int(episode_id))
        if task_key is not None:
            demos.setdefault(task_key, []).append(int(episode_id))
    if max_demos_per_task is not None:
        demos = {task_key: eps[-max_demos_per_task:] for task_key, eps in demos.items()}
    return demos


def _compute_gram_matrix(
    cfg: IterativeFineTuningConfig,
    *,
    context_of_task: dict[TaskKey, _TaskContext],
    task_of_episode: dict[int, TaskKey],
    policy,
    preprocessor,
    demo_episode_ids: list[int],
    frame_bounds: dict[int, tuple[int, int]],
    round_index: int,
) -> np.ndarray:
    """Embed every demonstration and return the Gram matrix of the stacked frame embeddings."""
    amf_cfg = cfg.selection.amf
    device = cfg.selection.device

    # Each demo lives in its task's dataset, so frames are addressed as (dataset, local frame
    # index) pairs rather than a single flat index array into one shared dataset.
    frame_entries: list[tuple[object, int]] = []
    for episode_id in demo_episode_ids:
        dataset = context_of_task[task_of_episode[episode_id]].dataset
        for local_frame_index in _demo_frame_indices(frame_bounds[episode_id], amf_cfg.frames_per_demo):
            frame_entries.append((dataset, int(local_frame_index)))

    embeddings: list[torch.Tensor] = []
    for start in range(0, len(frame_entries), cfg.selection.observation_batch_size):
        chunk = frame_entries[start : start + cfg.selection.observation_batch_size]
        batch = default_collate([dataset[frame_index] for dataset, frame_index in chunk])
        batch = preprocessor(batch)
        batch = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
        # Average the gradient over several (time, noise) draws to reduce Monte-Carlo variance.
        samples = [
            compute_loss_gradient_embeddings(
                policy,
                batch,
                seed=hash((cfg.selection.seed, round_index, start, sample_index)) % (2**31),
                scope=amf_cfg.scope,
            )
            for sample_index in range(amf_cfg.num_time_samples)
        ]
        embeddings.append(torch.stack(samples).mean(dim=0))

    stacked = torch.cat(embeddings)
    gram = (stacked @ stacked.T).double().cpu().numpy()

    mean_diagonal = float(np.mean(np.diag(gram)))
    logger.info(
        "AMF round %s: %s demonstrations, embedding dim %s, mean kernel diagonal %.4g",
        round_index,
        len(demo_episode_ids),
        stacked.shape[1],
        mean_diagonal,
    )
    if amf_cfg.normalize_embeddings:
        if mean_diagonal <= 0:
            raise ValueError("Loss-gradient embeddings are degenerate: the kernel diagonal vanished.")
        gram = gram / mean_diagonal
    return gram


def _load_policy_and_preprocessor(cfg: IterativeFineTuningConfig, model_reference: str):
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    policy_cls = get_policy_class(cfg.policy.type)
    policy = policy_cls.from_pretrained(pretrained_name_or_path=model_reference)
    policy.to(cfg.selection.device)
    policy.eval()
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=model_reference,
    )
    return policy, preprocessor


def run_amf_selection_round(
    cfg: IterativeFineTuningConfig,
    *,
    round_index: int,
    sampler_model_reference: str,
    acquired_episode_ids: list[int],
    round_layout: RoundLayout,
) -> SelectionManifest:
    started_at = utc_timestamp()
    start_time = time.perf_counter()
    amf_cfg = cfg.selection.amf

    if not is_supported_policy_type(cfg.policy.type):
        raise NotImplementedError(
            f"AMF selection needs loss-gradient embeddings, which are not implemented for "
            f"policy type {cfg.policy.type!r}."
        )

    contexts = _build_task_contexts(cfg)
    task_of_episode, context_of_task, frame_bounds = _index_task_contexts(contexts)
    episodes_by_task = {
        (context.task_group, context.task_id): context.episode_ids for context in contexts
    }
    task_keys = sorted(episodes_by_task)

    acquired = [int(episode_id) for episode_id in acquired_episode_ids]
    available = {
        task_key: set(episode_ids) - set(acquired)
        for task_key, episode_ids in episodes_by_task.items()
    }

    acquisition_rng = _rng(cfg, f"amf_acquisition:{round_index}")

    def acquire(task_key: TaskKey) -> int | None:
        pool = available[task_key]
        if not pool:
            logger.warning("AMF round %s: task %s has no unselected episodes left.", round_index, task_key)
            return None
        episode_id = acquisition_rng.choice(sorted(pool))
        pool.discard(episode_id)
        return episode_id

    budget = min(cfg.iteration.episodes_per_round, sum(len(pool) for pool in available.values()))
    selections: list[tuple[TaskKey, int, str, float | None]] = []  # task, episode, source, score

    prefill_slots = prefill_task_slots(
        task_keys,
        round_index=round_index,
        episodes_per_round=cfg.iteration.episodes_per_round,
        demos_per_task=amf_cfg.prefill_demos_per_task,
        rng=_rng(cfg, "amf_prefill_task_order"),
    )[:budget]
    for task_key in prefill_slots:
        episode_id = acquire(task_key)
        if episode_id is not None:
            selections.append((task_key, episode_id, "prefill", None))

    task_reductions: dict[TaskKey, float] = {}
    proxy_demos: dict[TaskKey, list[int]] = {}
    embedding_timing: dict[str, object] | None = None

    if len(selections) < budget:
        # Demonstrations prefilled earlier in *this* round already exist by the time the
        # criterion runs, so they must be eligible proxies too -- not just prior rounds' history.
        # This only matters when prefill finishes mid-round (num_tasks * prefill_demos_per_task is
        # not a multiple of episodes_per_round, e.g. 3 tasks / 10 episodes-per-round); with round
        # counts that divide evenly (e.g. LIBERO's 10 tasks / 5 episodes-per-round) prefill always
        # exactly fills whole rounds and this list is empty.
        prefill_this_round = [
            episode_id for _, episode_id, source, _ in selections if source == "prefill"
        ]
        proxy_demos = _acquired_demos_by_task(
            acquired + prefill_this_round, task_of_episode, max_demos_per_task=amf_cfg.max_demos_per_task
        )
        if not proxy_demos:
            raise ValueError(
                f"AMF round {round_index} has no acquired demonstrations to represent any task. "
                "Increase iteration.episodes_per_round or check the prefill schedule."
            )

        policy, preprocessor = _load_policy_and_preprocessor(cfg, sampler_model_reference)
        demo_episode_ids = [episode_id for demos in proxy_demos.values() for episode_id in demos]
        embedding_started_at = utc_timestamp()
        embedding_start_time = time.perf_counter()
        gram = _compute_gram_matrix(
            cfg,
            context_of_task=context_of_task,
            task_of_episode=task_of_episode,
            policy=policy,
            preprocessor=preprocessor,
            demo_episode_ids=demo_episode_ids,
            frame_bounds=frame_bounds,
            round_index=round_index,
        )
        embedding_timing = make_timing_entry(
            started_at=embedding_started_at,
            duration_s=time.perf_counter() - embedding_start_time,
        )

        demo_index = {episode_id: index for index, episode_id in enumerate(demo_episode_ids)}
        task_demos = {
            task_key: [demo_index[episode_id] for episode_id in demos]
            for task_key, demos in proxy_demos.items()
        }
        picks, task_reductions = greedy_task_picks(
            gram,
            amf_cfg.frames_per_demo,
            task_demos,
            budget=budget - len(selections),
            noise=amf_cfg.noise,
            capacity={task_key: len(pool) for task_key, pool in available.items()},
        )
        for pick in picks:
            episode_id = acquire(pick.task_key)
            if episode_id is not None:
                selections.append((pick.task_key, episode_id, "amf", pick.variance_reduction))

    manifest = _build_manifest(
        cfg,
        round_index=round_index,
        sampler_model_reference=sampler_model_reference,
        selections=selections,
        context_of_task=context_of_task,
        episodes_by_task=episodes_by_task,
        available=available,
        proxy_demos=proxy_demos,
        task_reductions=task_reductions,
        frame_bounds=frame_bounds,
        acquired=acquired,
        round_layout=round_layout,
        timings={
            "total": make_timing_entry(
                started_at=started_at, duration_s=time.perf_counter() - start_time
            ),
            **({"loss_gradient_embeddings": embedding_timing} if embedding_timing else {}),
        },
    )
    logger.info(
        "AMF round %s selected %s episodes (%s prefill) across %s tasks",
        round_index,
        len(manifest.selected_episodes),
        sum(1 for _, _, source, _ in selections if source == "prefill"),
        len(manifest.selected_tasks),
    )
    return manifest


def _rng(cfg: IterativeFineTuningConfig, stream_name: str) -> random.Random:
    return random.Random(f"{cfg.selection.seed}:{stream_name}")


def _build_manifest(
    cfg: IterativeFineTuningConfig,
    *,
    round_index: int,
    sampler_model_reference: str,
    selections: list[tuple[TaskKey, int, str, float | None]],
    context_of_task: dict[TaskKey, _TaskContext],
    episodes_by_task: dict[TaskKey, list[int]],
    available: dict[TaskKey, set[int]],
    proxy_demos: dict[TaskKey, list[int]],
    task_reductions: dict[TaskKey, float],
    frame_bounds: dict[int, tuple[int, int]],
    acquired: list[int],
    round_layout: RoundLayout,
    timings: dict[str, dict],
) -> SelectionManifest:
    selected_by_task: dict[TaskKey, list[tuple[int, str, float | None]]] = {}
    for task_key, episode_id, source, score in selections:
        selected_by_task.setdefault(task_key, []).append((episode_id, source, score))

    task_summaries: list[TaskSelection] = []
    selected_episodes: list[EpisodeSelection] = []
    for task_key in sorted(episodes_by_task):
        task_group, task_id = task_key
        chosen = selected_by_task.get(task_key, [])
        instruction = context_of_task[task_key].instruction
        task_summaries.append(
            TaskSelection(
                task_group=task_group,
                task_id=task_id,
                instruction=instruction,
                uncertainty=task_reductions.get(task_key),
                candidate_episode_count=len(available[task_key]) + len(chosen),
                selected_episode_count=len(chosen),
                selected_episode_ids=sorted(episode_id for episode_id, _, _ in chosen),
                metadata={
                    "selection_level": "task_summary",
                    "uncertainty_aggregation": "posterior_variance_reduction",
                    "proxy_demo_count": len(proxy_demos.get(task_key, [])),
                    "proxy_episode_ids": list(proxy_demos.get(task_key, [])),
                },
            )
        )
        for episode_id, source, score in chosen:
            selected_episodes.append(
                EpisodeSelection(
                    episode_id=episode_id,
                    task_group=task_group,
                    task_id=task_id,
                    instruction=instruction,
                    frame_index=frame_bounds[episode_id][0],
                    uncertainty=score,
                    metadata={
                        "selection_level": "episode",
                        "acquisition": source,
                        "task_variance_reduction": task_reductions.get(task_key),
                    },
                )
            )

    if cfg.execution.save_candidate_scores:
        save_dataclass_json(task_summaries, round_layout.candidate_scores_path)

    manifest = SelectionManifest(
        round_index=round_index,
        strategy="amf",
        uncertainty_method="amf",
        scoring_metric=None,
        sampler_model_path=sampler_model_reference,
        scorer_model_paths=[],
        candidate_count=len(task_summaries),
        candidate_episode_count=sum(task.candidate_episode_count for task in task_summaries),
        excluded_episode_ids=sorted(set(acquired)),
        selected_tasks=[task for task in task_summaries if task.selected_episode_count],
        selected_episodes=selected_episodes,
        candidate_scores_path=(
            str(round_layout.candidate_scores_path) if cfg.execution.save_candidate_scores else None
        ),
        timings=timings,
    )
    save_dataclass_json(manifest, round_layout.selection_manifest_path)
    return manifest
