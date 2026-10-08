"""Episode-level selection for iterative active-learning fine-tuning."""

from __future__ import annotations

import copy
import json
import logging
import random
import time
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from lerobot.datasets.factory import make_dataset
from lerobot.datasets.utils import filter_libero_episodes
from lerobot.policies.factory import (
    get_policy_class,
    make_pre_post_processors,
    make_uncertainty_sampler,
)
from lerobot.uncertainty.uncertainty_samplers.configuration_uncertainty_sampler import (
    ACESamplerConfig,
    CrossBayesianSamplerConfig,
    EntropySamplerConfig,
    ScoringMetricConfig,
    UncertaintySamplerConfig,
)
from lerobot.uncertainty.uncertainty_scoring.scorer_artifacts import (
    build_scorer_artifacts_for_uncertainty_sampler,
)

from .amf_selection import run_amf_selection_round
from .config import IterativeFineTuningConfig
from .env import resolve_local_path
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


@dataclass
class _TaskCandidatePool:
    task_group: str
    task_id: int
    instruction: str | None
    candidates: list[EpisodeSelection]
    uncertainty: float | None
    historical_selected_count: int = 0
    scored_episode_count: int = 0


def _iter_candidate_tasks(tasks: dict[str, list[int]]) -> Iterable[TaskKey]:
    for task_group, task_ids in tasks.items():
        for task_id in task_ids:
            yield task_group, task_id


def _selection_backend(cfg: IterativeFineTuningConfig):
    if cfg.dataset.repo_id == "HuggingFaceVLA/libero":
        from lerobot.envs.libero import get_task_instruction

        return "libero", get_task_instruction, filter_libero_episodes

    if cfg.dataset.pusht_tasks is not None or cfg.dataset.repo_id == "lerobot/pusht":
        from lerobot.datasets.utils import filter_pusht_episodes
        from lerobot.envs.pusht import get_task_instruction

        return "pusht", get_task_instruction, filter_pusht_episodes

    raise NotImplementedError("The iterative active-learning selector supports LIBERO and Push-T.")


def _build_uncertainty_sampler_config(
    cfg: IterativeFineTuningConfig,
    *,
    scorer_model_references: list[str],
) -> UncertaintySamplerConfig:
    if cfg.selection.uncertainty_method == "entropy":
        return UncertaintySamplerConfig(
            type="entropy",
            entropy_sampler=EntropySamplerConfig(
                num_action_samples=cfg.selection.num_action_samples,
            ),
        )
    if cfg.selection.uncertainty_method == "ace":
        return UncertaintySamplerConfig(
            type="ace",
            ace_sampler=ACESamplerConfig(
                num_action_samples=cfg.selection.num_action_samples,
            ),
        )

    return UncertaintySamplerConfig(
        type="cross_bayesian",
        cross_bayesian_sampler=CrossBayesianSamplerConfig(
            scorer_type=cfg.selection.scorer_type,
            num_action_samples=cfg.selection.num_action_samples,
            scoring_metric=ScoringMetricConfig(metric_type=cfg.selection.scoring_metric),
            ensemble_model_paths=list(scorer_model_references),
            laplace_config=copy.deepcopy(cfg.selection.laplace),
        ),
    )


def _load_policy(model_reference: str, policy_type: str, device: str):
    policy_cls = get_policy_class(policy_type)
    policy = policy_cls.from_pretrained(pretrained_name_or_path=model_reference)
    policy.to(device)
    policy.eval()
    return policy


def _underlying_flow_matching_model(policy):
    """Flow-matching module backing a policy.

    FlowMatchingPolicy exposes it as `.flow_matching`; the VLA policies
    (smolvla/xvla/fastwam) expose it as `.model`.
    """
    if policy.config.type == "flow_matching":
        return policy.flow_matching
    return policy.model


def _initial_frame_index(dataset, episode_id: int) -> int:
    starts = getattr(dataset, "_al_episode_start_index", None)
    if starts is None:
        episodes = dataset.meta.episodes
        starts = {
            int(ep): int(start)
            for ep, start in zip(episodes["episode_index"], episodes["dataset_from_index"], strict=True)
        }
        dataset._al_episode_start_index = starts  # cached: the episode table is immutable for a loaded dataset
    try:
        return starts[int(episode_id)]
    except KeyError:
        raise ValueError(f"Episode {episode_id} was not found in the current dataset view.") from None


class _FrameSubset(torch.utils.data.Dataset):
    """Raw dataset items for a fixed list of frame indices; used to decode candidate frames in parallel."""

    def __init__(self, dataset, frame_indices: list[int]):
        self.dataset = dataset
        self.frame_indices = list(frame_indices)

    def __len__(self) -> int:
        return len(self.frame_indices)

    def __getitem__(self, idx: int):
        return self.dataset[self.frame_indices[idx]]


def _iter_candidate_items(dataset, frame_indices: list[int], num_workers: int):
    """Yield dataset[frame_index] in order. Video decoding dominates candidate scoring, so
    decode in DataLoader workers; the preprocessor is applied by the caller."""
    if num_workers <= 0 or len(frame_indices) < 2:
        for frame_index in frame_indices:
            yield dataset[frame_index]
        return
    # Each worker holds file descriptors for the video shards it decodes, and the default "file_descriptor"
    # sharing strategy passes tensors between workers as fds too. With many workers -- or two jobs sharing a
    # cluster node -- that exhausts the per-process limit and the loader dies with "Too many open files".
    # The file_system strategy shares through /dev/shm instead and is the documented remedy.
    try:
        torch.multiprocessing.set_sharing_strategy("file_system")
    except (RuntimeError, AttributeError):  # already set, or unavailable on this platform
        pass
    loader = torch.utils.data.DataLoader(
        _FrameSubset(dataset, frame_indices),
        batch_size=None,
        shuffle=False,
        num_workers=min(num_workers, len(frame_indices)),
        prefetch_factor=2,
        persistent_workers=False,
    )
    yield from loader


def _prepare_observation_for_sampler(
    processed_sample: dict[str, torch.Tensor],
    device: str,
) -> dict[str, torch.Tensor]:
    observation: dict[str, torch.Tensor] = {}

    for key, value in processed_sample.items():
        if key == "task" and isinstance(value, (str, list, tuple)):
            # Policies that consume the raw instruction (FastWAM looks it up in its text-context table)
            # get it as a per-sample list (the batch processor may already have wrapped the string);
            # tensor-only consumers ignore non-tensor entries.
            observation[key] = [value] if isinstance(value, str) else [str(v) for v in value]
            continue
        if not isinstance(value, torch.Tensor):
            continue

        if "language.tokens" in key or "language.attention_mask" in key:
            if value.dim() == 2 and value.shape[0] == 1:
                observation[key] = value.to(device)
            else:
                observation[key] = value.unsqueeze(0).to(device)
        elif "images" in key and "_is_pad" not in key and "_padding_mask" not in key:
            observation[key] = value.unsqueeze(0).to(device)
        elif "state" in key and "_is_pad" not in key:
            observation[key] = value.unsqueeze(0).to(device)
        else:
            observation[key] = (
                value.unsqueeze(0).to(device) if value.dim() > 0 else value.view(1).to(device)
            )

    return observation


def _prepare_observation_batch_for_sampler(
    processed_samples: list[dict[str, torch.Tensor]],
    device: str,
) -> dict[str, torch.Tensor]:
    single_observations = [
        _prepare_observation_for_sampler(processed_sample, device)
        for processed_sample in processed_samples
    ]
    if not single_observations:
        raise ValueError("At least one processed sample is required to build an observation batch.")

    observation_batch: dict[str, torch.Tensor] = {}
    for key in single_observations[0]:
        values = [observation[key] for observation in single_observations]
        if isinstance(values[0], list):
            observation_batch[key] = [item for value in values for item in value]
            continue
        if ("language.tokens" in key or "language.attention_mask" in key) and values[0].dim() == 2:
            # Policies that pad language to "longest" (e.g. SmolVLA) yield per-sample token lengths
            # when candidates carry different instructions. Right-pad to the batch maximum;
            # padded positions are masked out via the attention mask, so the pad id is irrelevant.
            max_len = max(value.shape[1] for value in values)
            values = [
                torch.nn.functional.pad(value, (0, max_len - value.shape[1]), value=0) for value in values
            ]
        observation_batch[key] = torch.cat(values, dim=0)
    return observation_batch


def _aggregate_task_uncertainty(candidates: list[EpisodeSelection]) -> float | None:
    uncertainties = [candidate.uncertainty for candidate in candidates if candidate.uncertainty is not None]
    if not uncertainties:
        return None
    return float(np.mean(uncertainties))


def _task_sort_key(task_key: TaskKey) -> tuple[str, int]:
    return task_key[0], int(task_key[1])


def _episode_sort_key(candidate: EpisodeSelection) -> tuple[float, str, int, int]:
    return (
        -(candidate.uncertainty if candidate.uncertainty is not None else float("-inf")),
        candidate.task_group or "",
        candidate.task_id if candidate.task_id is not None else -1,
        int(candidate.episode_id),
    )


def _task_selection_limits(
    cfg: IterativeFineTuningConfig,
    *,
    task_pools: dict[TaskKey, _TaskCandidatePool],
) -> dict[TaskKey, int] | None:
    if cfg.selection.strategy != "constrained_top_k" or cfg.selection.max_selected_per_task is None:
        return None

    max_total_per_task = int(cfg.selection.max_selected_per_task)
    return {
        task_key: min(max_total_per_task, len(pool.candidates))
        for task_key, pool in task_pools.items()
    }


def _selection_rng(
    cfg: IterativeFineTuningConfig,
    *,
    round_index: int,
    stream_name: str,
) -> random.Random:
    # Derive an independent, reproducible RNG stream per round so resumed runs
    # reproduce exactly while later rounds do not replay round-0 samples.
    return random.Random(f"{cfg.selection.seed}:{stream_name}:{int(round_index)}")


def _select_candidates_in_order(
    ordered_candidates: Iterable[EpisodeSelection],
    *,
    total_budget: int,
    per_task_limits: dict[TaskKey, int] | None = None,
    preselected_candidates: list[EpisodeSelection] | None = None,
) -> list[EpisodeSelection]:
    selected_candidates = list(preselected_candidates or [])
    selected_episode_ids = {int(candidate.episode_id) for candidate in selected_candidates}
    selected_per_task = Counter(
        (str(candidate.task_group), int(candidate.task_id))
        for candidate in selected_candidates
    )

    for candidate in ordered_candidates:
        if len(selected_candidates) >= total_budget:
            break

        candidate_id = int(candidate.episode_id)
        task_key = (str(candidate.task_group), int(candidate.task_id))
        if candidate_id in selected_episode_ids:
            continue
        if per_task_limits is not None and selected_per_task[task_key] >= per_task_limits.get(task_key, 0):
            continue

        selected_candidates.append(candidate)
        selected_episode_ids.add(candidate_id)
        selected_per_task[task_key] += 1

    return selected_candidates


def _sample_task_counts_by_uncertainty(
    cfg: IterativeFineTuningConfig,
    *,
    task_pools: dict[TaskKey, _TaskCandidatePool],
    total_budget: int,
    round_index: int = 0,
    per_task_limits: dict[TaskKey, int] | None = None,
) -> Counter[TaskKey]:
    rng = _selection_rng(
        cfg,
        round_index=round_index,
        stream_name="task_weighted_top_k_task_sampling",
    )
    sampled_task_counts: Counter[TaskKey] = Counter()
    task_capacities = {
        task_key: min(
            len(pool.candidates),
            per_task_limits.get(task_key, len(pool.candidates))
            if per_task_limits is not None
            else len(pool.candidates),
        )
        for task_key, pool in task_pools.items()
    }

    while sum(sampled_task_counts.values()) < total_budget:
        eligible_tasks = [
            task_key
            for task_key in sorted(task_pools.keys(), key=_task_sort_key)
            if sampled_task_counts[task_key] < task_capacities[task_key]
        ]
        if not eligible_tasks:
            break

        raw_task_scores = np.array(
            [
                float(
                    task_pools[task_key].uncertainty
                    if task_pools[task_key].uncertainty is not None
                    else 0.0
                )
                for task_key in eligible_tasks
            ],
            dtype=np.float64,
        )
        if cfg.selection.scoring_metric == "ensemble_terminal_variance":
            finite_scores = raw_task_scores[np.isfinite(raw_task_scores)]
            if len(finite_scores):
                score_min = float(np.min(finite_scores))
                score_max = float(np.max(finite_scores))
                score_range = score_max - score_min
                task_scores = raw_task_scores - score_min + 0.01 * score_range
                task_scores = np.where(np.isfinite(task_scores), task_scores, 0.0)
            else:
                task_scores = np.zeros_like(raw_task_scores)
        else:
            task_scores = np.maximum(
                np.where(np.isfinite(raw_task_scores), raw_task_scores, 0.0),
                0.0,
            )
        if float(cfg.selection.task_selection_temperature) == 0.0:
            task_weights = np.full(len(task_scores), 1.0 / len(task_scores), dtype=np.float64)
        else:
            powered_task_scores = np.power(
                task_scores,
                float(cfg.selection.task_selection_temperature),
            )
            task_score_sum = float(np.sum(powered_task_scores))
            if task_score_sum > 0:
                task_weights = powered_task_scores / task_score_sum
            else:
                task_weights = np.full(len(task_scores), 1.0 / len(task_scores), dtype=np.float64)
        sampled_task = rng.choices(eligible_tasks, weights=task_weights.tolist(), k=1)[0]
        sampled_task_counts[sampled_task] += 1

    return sampled_task_counts


def _sample_task_weighted_candidates(
    cfg: IterativeFineTuningConfig,
    *,
    task_pools: dict[TaskKey, _TaskCandidatePool],
    sampled_task_counts: Counter[TaskKey],
    round_index: int,
) -> list[EpisodeSelection]:
    selected_candidates: list[EpisodeSelection] = []
    if cfg.selection.task_weighted_episode_selection == "top_k":
        for task_key in sorted(sampled_task_counts.keys(), key=_task_sort_key):
            selected_candidates.extend(
                sorted(task_pools[task_key].candidates, key=_episode_sort_key)[
                    : sampled_task_counts[task_key]
                ]
            )
        return selected_candidates

    rng = _selection_rng(
        cfg,
        round_index=round_index,
        stream_name="task_weighted_top_k_episode_sampling",
    )
    for task_key in sorted(sampled_task_counts.keys(), key=_task_sort_key):
        task_candidates = list(task_pools[task_key].candidates)
        draw_count = sampled_task_counts[task_key]
        if draw_count <= 0:
            continue
        selected_candidates.extend(rng.sample(task_candidates, k=draw_count))

    return selected_candidates


def _select_episode_candidates(
    cfg: IterativeFineTuningConfig,
    *,
    task_pools: dict[TaskKey, _TaskCandidatePool],
    round_index: int = 0,
) -> list[EpisodeSelection]:
    all_candidates = [
        candidate
        for task_key in sorted(task_pools.keys(), key=_task_sort_key)
        for candidate in task_pools[task_key].candidates
    ]
    total_budget = min(cfg.iteration.episodes_per_round, len(all_candidates))
    if total_budget <= 0:
        return []

    per_task_limits = _task_selection_limits(cfg, task_pools=task_pools)
    if cfg.selection.strategy == "random":
        rng = _selection_rng(cfg, round_index=round_index, stream_name="random_episode_shuffle")
        shuffled_candidates = list(all_candidates)
        rng.shuffle(shuffled_candidates)
        return _select_candidates_in_order(
            shuffled_candidates,
            total_budget=total_budget,
            per_task_limits=per_task_limits,
        )

    ranked_candidates = sorted(all_candidates, key=_episode_sort_key)
    selected_candidates: list[EpisodeSelection] = []
    if cfg.selection.strategy == "balanced_top_k":
        # Seed the round with the strongest remaining episode from each task before
        # filling the rest of the budget from the global ranking.
        best_per_task = sorted(
            [
                min(pool.candidates, key=_episode_sort_key)
                for task_key, pool in sorted(task_pools.items(), key=lambda item: _task_sort_key(item[0]))
                if pool.candidates
                and (per_task_limits is None or per_task_limits.get(task_key, 0) > 0)
            ],
            key=_episode_sort_key,
        )
        selected_candidates = _select_candidates_in_order(
            best_per_task,
            total_budget=total_budget,
            per_task_limits=per_task_limits,
        )
    elif cfg.selection.strategy == "task_weighted_top_k":
        sampled_task_counts = _sample_task_counts_by_uncertainty(
            cfg,
            task_pools=task_pools,
            total_budget=total_budget,
            round_index=round_index,
            per_task_limits=per_task_limits,
        )
        selected_candidates = _sample_task_weighted_candidates(
            cfg,
            task_pools=task_pools,
            sampled_task_counts=sampled_task_counts,
            round_index=round_index,
        )
        return sorted(selected_candidates, key=_episode_sort_key)

    return _select_candidates_in_order(
        ranked_candidates,
        total_budget=total_budget,
        per_task_limits=per_task_limits,
        preselected_candidates=selected_candidates,
    )


def _task_summary_from_pool(
    pool: _TaskCandidatePool,
    *,
    selected_episode_ids: list[int],
    aggregation_name: str,
) -> TaskSelection:
    return TaskSelection(
        task_group=pool.task_group,
        task_id=pool.task_id,
        instruction=pool.instruction,
        uncertainty=pool.uncertainty,
        candidate_episode_count=len(pool.candidates),
        selected_episode_count=len(selected_episode_ids),
        selected_episode_ids=sorted(int(episode_id) for episode_id in selected_episode_ids),
        metadata={
            "selection_level": "task_summary",
            "uncertainty_aggregation": aggregation_name,
            "episode_uncertainty_count": int(pool.scored_episode_count),
            "historical_selected_episode_count": int(pool.historical_selected_count),
        },
    )


def _build_selection_outputs(
    task_pools: dict[TaskKey, _TaskCandidatePool],
    selected_candidates: list[EpisodeSelection],
) -> tuple[list[TaskSelection], list[TaskSelection], list[EpisodeSelection]]:
    all_task_summaries: list[TaskSelection] = []
    selected_tasks: list[TaskSelection] = []
    selected_episodes: list[EpisodeSelection] = []
    selected_by_task: dict[TaskKey, list[EpisodeSelection]] = {}

    for candidate in selected_candidates:
        task_key = (str(candidate.task_group), int(candidate.task_id))
        selected_by_task.setdefault(task_key, []).append(candidate)

    for task_key in sorted(task_pools.keys(), key=_task_sort_key):
        pool = task_pools[task_key]
        chosen_candidates = sorted(selected_by_task.get(task_key, []), key=_episode_sort_key)
        chosen_episode_ids = [int(candidate.episode_id) for candidate in chosen_candidates]

        task_summary = _task_summary_from_pool(
            pool,
            selected_episode_ids=chosen_episode_ids,
            aggregation_name="mean_initial_frame_uncertainty",
        )
        all_task_summaries.append(task_summary)
        if chosen_candidates:
            selected_tasks.append(task_summary)

        for candidate in chosen_candidates:
            metadata = dict(candidate.metadata)
            if candidate.uncertainty is not None:
                metadata["episode_uncertainty"] = float(candidate.uncertainty)
            metadata["selection_level"] = "episode"
            metadata["task_uncertainty"] = pool.uncertainty
            selected_episodes.append(
                EpisodeSelection(
                    episode_id=int(candidate.episode_id),
                    task_group=candidate.task_group,
                    task_id=candidate.task_id,
                    instruction=candidate.instruction,
                    frame_index=candidate.frame_index,
                    uncertainty=candidate.uncertainty,
                    metadata=metadata,
                )
            )

    all_task_summaries.sort(
        key=lambda item: (
            -(item.uncertainty if item.uncertainty is not None else float("-inf")),
            item.task_group,
            item.task_id,
        )
    )
    selected_tasks.sort(
        key=lambda item: (
            -(item.uncertainty if item.uncertainty is not None else float("-inf")),
            item.task_group,
            item.task_id,
        )
    )
    selected_episodes.sort(
        key=lambda item: (
            -(item.uncertainty if item.uncertainty is not None else float("-inf")),
            -(
                item.metadata.get("task_uncertainty")
                if item.metadata.get("task_uncertainty") is not None
                else float("-inf")
            ),
            item.task_group or "",
            item.task_id if item.task_id is not None else -1,
            int(item.episode_id),
        )
    )
    return all_task_summaries, selected_tasks, selected_episodes


def _run_predefined_ranking_selection(
    cfg: IterativeFineTuningConfig,
    *,
    round_index: int,
    excluded_episode_ids: set[int],
    round_layout: RoundLayout,
) -> SelectionManifest:
    """Pick the next ``episodes_per_round`` items from a precomputed episode ranking.

    Used by the diversity baseline (paper Appendix B.4): the ranking is computed
    offline (e.g. SigLIP k-center-greedy via ``scripts/active_learning/k_greedy.py``);
    each round pops the next ``n_e`` not-yet-selected episodes.
    """
    selection_started_at = utc_timestamp()
    selection_start_time = time.perf_counter()

    ranking_path = resolve_local_path(
        cfg.selection.predefined_ranking_path,
        base_dir=Path.cwd(),
        must_exist=True,
    )
    with open(ranking_path, "r", encoding="utf-8") as handle:
        ranking = json.load(handle)
    if not isinstance(ranking, list):
        raise ValueError(
            f"Predefined ranking at {ranking_path} must be a JSON list of episode entries."
        )

    n_to_select = int(cfg.iteration.episodes_per_round)
    selected_episodes: list[EpisodeSelection] = []
    seen_in_round: set[int] = set()
    for entry in ranking:
        if not isinstance(entry, dict) or "episode_id" not in entry:
            raise ValueError(
                "Each ranking entry must be a JSON object with at least an 'episode_id' field."
            )
        episode_id = int(entry["episode_id"])
        if episode_id in excluded_episode_ids or episode_id in seen_in_round:
            continue
        seen_in_round.add(episode_id)
        selected_episodes.append(
            EpisodeSelection(
                episode_id=episode_id,
                task_group=str(entry.get("task_group", "")),
                task_id=int(entry.get("task_id", -1)),
                instruction=str(entry.get("instruction", "")),
                frame_index=int(entry.get("frame_index", 0)),
                uncertainty=None,
                metadata={"selection_level": "predefined_ranking", "round_index": round_index},
            )
        )
        if len(selected_episodes) >= n_to_select:
            break

    if not selected_episodes:
        raise RuntimeError(
            "Predefined ranking exhausted: no candidate episodes remain after excluding "
            f"{len(excluded_episode_ids)} prior selections."
        )

    by_task: dict[TaskKey, list[EpisodeSelection]] = {}
    for ep in selected_episodes:
        by_task.setdefault((ep.task_group, ep.task_id), []).append(ep)
    selected_tasks = [
        TaskSelection(
            task_group=task_group,
            task_id=task_id,
            uncertainty=None,
            selected_episode_count=len(eps),
            selected_episode_ids=[ep.episode_id for ep in eps],
        )
        for (task_group, task_id), eps in by_task.items()
    ]

    selection_completed_at = utc_timestamp()
    selection_duration = time.perf_counter() - selection_start_time
    manifest = SelectionManifest(
        round_index=round_index,
        strategy="predefined_ranking",
        uncertainty_method="predefined_ranking",
        scoring_metric=None,
        sampler_model_path=None,
        scorer_model_paths=[],
        candidate_count=len(ranking),
        candidate_episode_count=len(ranking),
        excluded_episode_ids=sorted(int(ep_id) for ep_id in excluded_episode_ids),
        selected_tasks=selected_tasks,
        selected_episodes=selected_episodes,
        candidate_scores_path=None,
        timings={
            "total": {
                "started_at": selection_started_at,
                "completed_at": selection_completed_at,
                "duration_s": selection_duration,
            },
        },
    )
    save_dataclass_json(manifest, round_layout.selection_manifest_path)
    return manifest


def run_selection_round(
    cfg: IterativeFineTuningConfig,
    *,
    round_index: int,
    sampler_model_reference: str,
    scorer_model_references: list[str],
    excluded_episode_ids: set[int],
    round_layout: RoundLayout,
    acquired_episode_ids: list[int] | None = None,
) -> SelectionManifest:
    if cfg.selection.strategy == "predefined_ranking":
        return _run_predefined_ranking_selection(
            cfg,
            round_index=round_index,
            excluded_episode_ids=excluded_episode_ids,
            round_layout=round_layout,
        )
    if cfg.selection.strategy == "amf":
        return run_amf_selection_round(
            cfg,
            round_index=round_index,
            sampler_model_reference=sampler_model_reference,
            acquired_episode_ids=list(acquired_episode_ids or []),
            round_layout=round_layout,
        )
    selection_started_at = utc_timestamp()
    selection_start_time = time.perf_counter()
    uncertainty_started_at: str | None = None
    uncertainty_completed_at: str | None = None
    uncertainty_duration_s = 0.0

    policy = None
    preprocessor = None
    sampler = None
    scorer_artifacts = None
    policy_cfg = cfg.policy
    scorer_model_references = list(scorer_model_references) or [sampler_model_reference]
    if cfg.selection.strategy != "random":
        policy = _load_policy(sampler_model_reference, cfg.policy.type, cfg.selection.device)
        preprocessor, _ = make_pre_post_processors(
            policy_cfg=policy.config,
            pretrained_path=sampler_model_reference,
        )

        uncertainty_cfg = _build_uncertainty_sampler_config(
            cfg,
            scorer_model_references=scorer_model_references,
        )
        scorer_artifacts = build_scorer_artifacts_for_uncertainty_sampler(
            uncertainty_sampler_cfg=uncertainty_cfg,
            policy=policy,
            preprocessor=preprocessor,
            dataset_cfg=cfg.dataset,
        )
        sampler = make_uncertainty_sampler(
            uncertainty_sampler_config=uncertainty_cfg,
            policy_config=policy.config,
            model=_underlying_flow_matching_model(policy),
            scorer_artifacts=scorer_artifacts,
        )
        policy_cfg = policy.config

    task_pools: dict[TaskKey, _TaskCandidatePool] = {}

    backend, _get_instruction, _filter_episodes = _selection_backend(cfg)

    # make_dataset() ignores the per-benchmark task maps (episodes are filtered below via
    # _filter_episodes), so every task shares one loaded dataset.
    shared_dataset = None
    for task_group, task_id in _iter_candidate_tasks(cfg.selection.candidate_tasks):
        dataset_cfg = copy.deepcopy(cfg.dataset)
        if backend == "pusht":
            dataset_cfg.pusht_tasks = {task_group: [task_id]}
        else:
            dataset_cfg.libero_tasks = {task_group: [task_id]}
        if shared_dataset is None:
            shared_dataset = make_dataset(dataset_cfg=dataset_cfg, policy_cfg=policy_cfg)
        dataset = shared_dataset

        instruction = _get_instruction(task_group=task_group, task_id=task_id)
        episode_ids = [
            int(episode_id)
            for episode_id in _filter_episodes(
                dataset=dataset,
                tasks_to_use={task_group: [task_id]},
            )
        ]
        historical_selected_count = sum(
            int(episode_id) in excluded_episode_ids for episode_id in episode_ids
        )
        selectable_episode_ids = list(episode_ids)
        if cfg.selection.max_candidates_per_task is not None:
            selectable_episode_ids = selectable_episode_ids[: cfg.selection.max_candidates_per_task]

        scored_candidates: list[EpisodeSelection] = []
        for episode_id in tqdm(episode_ids, desc=f"{task_group}/{task_id}", leave=False):
            frame_index = _initial_frame_index(dataset, episode_id)
            scored_candidates.append(
                EpisodeSelection(
                    episode_id=int(episode_id),
                    task_group=task_group,
                    task_id=int(task_id),
                    instruction=instruction,
                    frame_index=int(frame_index),
                    uncertainty=None,
                    metadata={"selection_level": "task_candidate_episode"},
                )
            )

        if cfg.selection.strategy != "random":
            if preprocessor is None or sampler is None:
                raise RuntimeError("Random-selection fast path was not bypassed correctly.")

            # With a candidate cap only the selectable candidates are scored; the rest keep
            # uncertainty=None (aggregation ignores them). Without a cap everything is scored.
            if cfg.selection.max_candidates_per_task is not None:
                capped = set(selectable_episode_ids)
                candidates_to_score = [c for c in scored_candidates if c.episode_id in capped]
            else:
                candidates_to_score = scored_candidates
            item_iter = _iter_candidate_items(
                dataset,
                [candidate.frame_index for candidate in candidates_to_score],
                num_workers=int(getattr(cfg, "num_workers", 0) or 0),
            )
            for batch_start in tqdm(
                range(0, len(candidates_to_score), cfg.selection.observation_batch_size),
                desc=f"{task_group}/{task_id} scoring",
                leave=False,
            ):
                candidate_batch = candidates_to_score[
                    batch_start : batch_start + cfg.selection.observation_batch_size
                ]
                processed_batch = [preprocessor(next(item_iter)) for _ in candidate_batch]
                observation_batch = _prepare_observation_batch_for_sampler(
                    processed_batch,
                    cfg.selection.device,
                )
                batch_started_at = utc_timestamp()
                batch_start_time = time.perf_counter()
                with torch.no_grad():
                    _, batch_uncertainties = sampler.conditional_sample_with_uncertainty_batch(
                        observation=observation_batch,
                        generator=None,
                    )
                uncertainty_duration_s += time.perf_counter() - batch_start_time
                if uncertainty_started_at is None:
                    uncertainty_started_at = batch_started_at
                uncertainty_completed_at = utc_timestamp()

                for candidate, uncertainty in zip(candidate_batch, batch_uncertainties.tolist(), strict=True):
                    candidate.uncertainty = float(uncertainty)

        selectable_episode_id_set = set(selectable_episode_ids)
        candidates = [
            candidate
            for candidate in scored_candidates
            if int(candidate.episode_id) in selectable_episode_id_set
        ]
        if not candidates:
            continue

        task_pools[(task_group, int(task_id))] = _TaskCandidatePool(
            task_group=task_group,
            task_id=int(task_id),
            instruction=instruction,
            candidates=candidates,
            uncertainty=(
                None
                if cfg.selection.strategy == "random"
                else _aggregate_task_uncertainty(scored_candidates)
            ),
            historical_selected_count=historical_selected_count,
            scored_episode_count=(
                sum(candidate.uncertainty is not None for candidate in scored_candidates)
                if cfg.selection.strategy != "random"
                else 0
            ),
        )

    if not task_pools:
        raise ValueError(
            "No selectable tasks were found after applying the candidate-task filter."
        )

    selected_candidates = _select_episode_candidates(
        cfg,
        task_pools=task_pools,
        round_index=round_index,
    )
    all_task_summaries, selected_tasks, selected_episodes = _build_selection_outputs(
        task_pools,
        selected_candidates,
    )

    if cfg.execution.save_candidate_scores:
        save_dataclass_json(all_task_summaries, round_layout.candidate_scores_path)

    timings = {
        "total": make_timing_entry(
            started_at=selection_started_at,
            duration_s=time.perf_counter() - selection_start_time,
        )
    }
    if uncertainty_started_at is not None and uncertainty_completed_at is not None:
        timings["uncertainty_quantification"] = make_timing_entry(
            started_at=uncertainty_started_at,
            completed_at=uncertainty_completed_at,
            duration_s=uncertainty_duration_s,
        )

    manifest = SelectionManifest(
        round_index=round_index,
        strategy=cfg.selection.strategy,
        uncertainty_method=cfg.selection.uncertainty_method,
        scoring_metric=(
            None
            if cfg.selection.strategy == "random"
            or cfg.selection.uncertainty_method in {"entropy", "ace"}
            else cfg.selection.scoring_metric
        ),
        sampler_model_path=sampler_model_reference,
        scorer_model_paths=(
            []
            if cfg.selection.strategy == "random"
            or cfg.selection.uncertainty_method in {"entropy", "ace"}
            or cfg.selection.scorer_type != "ensemble"
            else scorer_model_references
        ),
        candidate_count=len(all_task_summaries),
        candidate_episode_count=sum(task_summary.candidate_episode_count for task_summary in all_task_summaries),
        excluded_episode_ids=sorted(int(ep) for ep in excluded_episode_ids),
        selected_tasks=selected_tasks,
        selected_episodes=selected_episodes,
        candidate_scores_path=(
            str(round_layout.candidate_scores_path) if cfg.execution.save_candidate_scores else None
        ),
        timings=timings,
    )
    save_dataclass_json(manifest, round_layout.selection_manifest_path)
    logger.info(
        "Round %s selected %s new episodes across %s tasks with strategy=%s",
        round_index,
        len(manifest.selected_episodes),
        len(manifest.selected_tasks),
        cfg.selection.strategy,
    )
    # Release the scoring models before the training workers start: on a shared node the sampler
    # policy, the ensemble members and the allocator cache would otherwise keep ~10 GB of GPU 0
    # (FastWAM) for the whole round, which the member training on that GPU cannot spare.
    del sampler, scorer_artifacts, policy
    import gc

    try:  # FastWAM shares frozen base weights process-wide; the cache would keep them resident.
        from lerobot.policies.fastwam.modeling_fastwam import clear_frozen_base_cache

        clear_frozen_base_cache()
    except ImportError:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return manifest
