"""Structured manifests for iterative fine-tuning runs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .constants import STAGE_SELECT, STAGE_TRAIN


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return {k: _to_jsonable(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_to_jsonable(v) for v in value]
    return value


def save_dataclass_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_to_jsonable(obj), handle, indent=2, sort_keys=False)


def load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


@dataclass
class MultiSeedChildRun:
    seed: int
    selection_seed: int | None = None
    run_name: str = ""
    run_dir: str = ""
    run_state_path: str = ""
    run_evaluation_path: str = ""
    pretrained_path: str | None = None
    ensemble_model_paths: list[str] = field(default_factory=list)
    latest_model_paths: list[str] = field(default_factory=list)
    completed_round_indices: list[int] = field(default_factory=list)


@dataclass
class MultiSeedRunManifest:
    created_at: str = field(default_factory=utc_timestamp)
    run_name: str = ""
    run_dir: str = ""
    seeds: list[int] = field(default_factory=list)
    child_runs: list[MultiSeedChildRun] = field(default_factory=list)


def _normalize_timing_entry(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}

    normalized: dict[str, Any] = {}
    started_at = value.get("started_at")
    completed_at = value.get("completed_at")
    duration_s = value.get("duration_s")
    if isinstance(started_at, str) and started_at:
        normalized["started_at"] = started_at
    if isinstance(completed_at, str) and completed_at:
        normalized["completed_at"] = completed_at
    if duration_s is not None:
        normalized["duration_s"] = float(duration_s)
    return normalized


def normalize_timings(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, dict):
        return {}
    return {
        str(name): _normalize_timing_entry(entry)
        for name, entry in value.items()
        if isinstance(entry, dict)
    }


def make_timing_entry(
    *,
    started_at: str,
    duration_s: float,
    completed_at: str | None = None,
) -> dict[str, Any]:
    return {
        "started_at": started_at,
        "completed_at": completed_at or utc_timestamp(),
        "duration_s": float(duration_s),
    }


def total_timing_entry(timings: dict[str, dict[str, Any]] | None) -> dict[str, Any] | None:
    if not timings:
        return None
    entry = timings.get("total")
    normalized = _normalize_timing_entry(entry)
    return normalized or None


def aggregate_timing_entries(entries: list[dict[str, Any] | None]) -> dict[str, Any] | None:
    normalized_entries = []
    for entry in entries:
        normalized = _normalize_timing_entry(entry)
        if "started_at" not in normalized or "completed_at" not in normalized:
            continue
        normalized_entries.append(normalized)

    if not normalized_entries:
        return None

    started_at = min(
        datetime.fromisoformat(str(entry["started_at"])) for entry in normalized_entries
    )
    completed_at = max(
        datetime.fromisoformat(str(entry["completed_at"])) for entry in normalized_entries
    )
    return make_timing_entry(
        started_at=started_at.isoformat(),
        completed_at=completed_at.isoformat(),
        duration_s=max((completed_at - started_at).total_seconds(), 0.0),
    )


@dataclass
class EpisodeSelection:
    episode_id: int
    task_group: str | None = None
    task_id: int | None = None
    instruction: str | None = None
    frame_index: int | None = None
    uncertainty: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskSelection:
    task_group: str
    task_id: int
    instruction: str | None = None
    uncertainty: float | None = None
    candidate_episode_count: int = 0
    selected_episode_count: int = 0
    selected_episode_ids: list[int] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SelectionManifest:
    created_at: str = field(default_factory=utc_timestamp)
    round_index: int = 0
    strategy: str = "top_k"
    uncertainty_method: str = "cross_bayesian"
    scoring_metric: str | None = None
    sampler_model_path: str | None = None
    scorer_model_paths: list[str] = field(default_factory=list)
    candidate_count: int = 0
    candidate_episode_count: int = 0
    excluded_episode_ids: list[int] = field(default_factory=list)
    selected_tasks: list[TaskSelection] = field(default_factory=list)
    selected_episodes: list[EpisodeSelection] = field(default_factory=list)
    candidate_scores_path: str | None = None
    timings: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def episode_ids(self) -> list[int]:
        return [episode.episode_id for episode in self.selected_episodes]


@dataclass
class TrainedEnsembleMember:
    member_index: int
    start_model_path: str
    final_model_path: str
    training_dir: str
    seed: int | None = None
    timings: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class TrainingManifest:
    created_at: str = field(default_factory=utc_timestamp)
    round_index: int = 0
    start_model_paths: list[str] = field(default_factory=list)
    final_model_paths: list[str] = field(default_factory=list)
    member_training_runs: list[TrainedEnsembleMember] = field(default_factory=list)
    history_episode_ids: list[int] = field(default_factory=list)
    new_episode_ids: list[int] = field(default_factory=list)
    all_training_episode_ids: list[int] = field(default_factory=list)
    sampling_weights: dict[str, float] = field(default_factory=dict)
    num_samples_per_epoch: int = 0
    batch_size: int = 0
    steps: int = 0
    seed: int | None = None
    timings: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class RoundRecord:
    round_index: int
    start_model_paths: list[str] = field(default_factory=list)
    selected_episode_ids: list[int] = field(default_factory=list)
    selection_manifest_path: str | None = None
    training_manifest_path: str | None = None
    final_model_paths: list[str] = field(default_factory=list)
    completed_stages: list[str] = field(default_factory=list)
    timings: dict[str, dict[str, Any]] = field(default_factory=dict)

    def mark_selected(
        self,
        manifest_path: Path,
        episode_ids: list[int],
        timing: dict[str, Any] | None = None,
    ) -> None:
        self.selection_manifest_path = str(manifest_path)
        self.selected_episode_ids = [int(ep) for ep in episode_ids]
        if timing is not None:
            self.timings["selection"] = _normalize_timing_entry(timing)
        if STAGE_SELECT not in self.completed_stages:
            self.completed_stages.append(STAGE_SELECT)

    def mark_trained(
        self,
        manifest_path: Path,
        final_model_paths: list[Path],
        timing: dict[str, Any] | None = None,
    ) -> None:
        self.training_manifest_path = str(manifest_path)
        self.final_model_paths = [str(path) for path in final_model_paths]
        if timing is not None:
            self.timings["training"] = _normalize_timing_entry(timing)
        if STAGE_TRAIN not in self.completed_stages:
            self.completed_stages.append(STAGE_TRAIN)


@dataclass
class RunState:
    created_at: str = field(default_factory=utc_timestamp)
    run_name: str = ""
    run_dir: str = ""
    bootstrap_episode_ids: list[int] = field(default_factory=list)
    rounds: list[RoundRecord] = field(default_factory=list)

    def get_round(self, round_index: int) -> RoundRecord | None:
        for record in self.rounds:
            if record.round_index == round_index:
                return record
        return None

    def upsert_round(self, record: RoundRecord) -> None:
        current = self.get_round(record.round_index)
        if current is None:
            self.rounds.append(record)
            self.rounds.sort(key=lambda item: item.round_index)
            return

        current.start_model_paths = list(record.start_model_paths)
        current.selected_episode_ids = record.selected_episode_ids
        current.selection_manifest_path = record.selection_manifest_path
        current.training_manifest_path = record.training_manifest_path
        current.final_model_paths = list(record.final_model_paths)
        current.completed_stages = record.completed_stages
        current.timings = normalize_timings(record.timings)

    def cumulative_selected_episode_ids(self, *, include_round: int | None = None) -> list[int]:
        selected = [int(ep) for ep in self.bootstrap_episode_ids]
        for record in self.rounds:
            if include_round is not None and record.round_index > include_round:
                continue
            selected.extend(int(ep) for ep in record.selected_episode_ids)
        return selected

    def latest_trained_model_paths(self) -> list[str] | None:
        trained_rounds = [record for record in self.rounds if record.final_model_paths]
        if not trained_rounds:
            return None
        latest = max(trained_rounds, key=lambda item: item.round_index)
        return list(latest.final_model_paths)


def load_selection_manifest(path: Path) -> SelectionManifest:
    data = load_json(path)
    selected_episodes = [EpisodeSelection(**item) for item in data.get("selected_episodes", [])]
    selected_tasks = [TaskSelection(**item) for item in data.get("selected_tasks", [])]
    return SelectionManifest(
        created_at=data.get("created_at", utc_timestamp()),
        round_index=data["round_index"],
        strategy=data["strategy"],
        uncertainty_method=data.get("uncertainty_method", "cross_bayesian"),
        scoring_metric=data.get("scoring_metric"),
        sampler_model_path=data.get("sampler_model_path"),
        scorer_model_paths=list(data.get("scorer_model_paths", [])),
        candidate_count=data.get("candidate_count", len(selected_tasks) or len(selected_episodes)),
        candidate_episode_count=data.get("candidate_episode_count", len(selected_episodes)),
        excluded_episode_ids=list(data.get("excluded_episode_ids", [])),
        selected_tasks=selected_tasks,
        selected_episodes=selected_episodes,
        candidate_scores_path=data.get("candidate_scores_path"),
        timings=normalize_timings(data.get("timings")),
    )


def _trained_ensemble_member_from_data(data: dict[str, Any]) -> TrainedEnsembleMember:
    return TrainedEnsembleMember(
        member_index=int(data["member_index"]),
        start_model_path=str(data["start_model_path"]),
        final_model_path=str(data["final_model_path"]),
        training_dir=str(data.get("training_dir", "")),
        seed=data.get("seed"),
        timings=normalize_timings(data.get("timings")),
    )


def load_trained_ensemble_member(path: Path) -> TrainedEnsembleMember:
    return _trained_ensemble_member_from_data(load_json(path))


def load_training_manifest(path: Path) -> TrainingManifest:
    data = load_json(path)
    start_model_paths = list(data.get("start_model_paths", []))
    final_model_paths = list(data.get("final_model_paths", []))
    legacy_start = data.get("start_model_path")
    legacy_final = data.get("final_model_path")
    if not start_model_paths and legacy_start:
        start_model_paths = [legacy_start]
    if not final_model_paths and legacy_final:
        final_model_paths = [legacy_final]

    member_training_runs = [
        _trained_ensemble_member_from_data(item) for item in data.get("member_training_runs", [])
    ]
    if not member_training_runs and start_model_paths and final_model_paths:
        member_training_runs = [
            TrainedEnsembleMember(
                member_index=index,
                start_model_path=start_path,
                final_model_path=final_path,
                training_dir="",
                seed=data.get("seed"),
                timings={},
            )
            for index, (start_path, final_path) in enumerate(
                zip(start_model_paths, final_model_paths, strict=True)
            )
        ]

    return TrainingManifest(
        created_at=data.get("created_at", utc_timestamp()),
        round_index=data.get("round_index", 0),
        start_model_paths=start_model_paths,
        final_model_paths=final_model_paths,
        member_training_runs=member_training_runs,
        history_episode_ids=list(data.get("history_episode_ids", [])),
        new_episode_ids=list(data.get("new_episode_ids", [])),
        all_training_episode_ids=list(data.get("all_training_episode_ids", [])),
        sampling_weights=dict(data.get("sampling_weights", {})),
        num_samples_per_epoch=data.get("num_samples_per_epoch", 0),
        batch_size=data.get("batch_size", 0),
        steps=data.get("steps", 0),
        seed=data.get("seed"),
        timings=normalize_timings(data.get("timings")),
    )


def load_run_state(path: Path) -> RunState:
    data = load_json(path)
    rounds = []
    for item in data.get("rounds", []):
        start_model_paths = list(item.get("start_model_paths", []))
        final_model_paths = list(item.get("final_model_paths", []))
        if not start_model_paths and item.get("start_model_path"):
            start_model_paths = [item["start_model_path"]]
        if not final_model_paths and item.get("final_model_path"):
            final_model_paths = [item["final_model_path"]]
        rounds.append(
            RoundRecord(
                round_index=item["round_index"],
                start_model_paths=start_model_paths,
                selected_episode_ids=list(item.get("selected_episode_ids", [])),
                selection_manifest_path=item.get("selection_manifest_path"),
                training_manifest_path=item.get("training_manifest_path"),
                final_model_paths=final_model_paths,
                completed_stages=list(item.get("completed_stages", [])),
                timings=normalize_timings(item.get("timings")),
            )
        )
    return RunState(
        created_at=data.get("created_at", utc_timestamp()),
        run_name=data.get("run_name", ""),
        run_dir=data.get("run_dir", ""),
        bootstrap_episode_ids=list(data.get("bootstrap_episode_ids", [])),
        rounds=rounds,
    )


def extract_episode_ids_from_payload(data: Any) -> list[int]:
    if isinstance(data, list):
        return [int(item) for item in data]

    if not isinstance(data, dict):
        raise ValueError(f"Unsupported manifest payload type: {type(data)!r}")

    if "selected_episodes" in data:
        return [int(item["episode_id"]) for item in data["selected_episodes"]]
    if "selected_episode_ids" in data:
        return [int(item) for item in data["selected_episode_ids"]]
    if "episode_indices" in data:
        return [int(item) for item in data["episode_indices"]]
    if "rows" in data:
        row_ids = []
        for row in data["rows"]:
            if "episode_id" in row:
                row_ids.append(int(row["episode_id"]))
        if row_ids:
            return row_ids

    raise ValueError("Could not extract episode ids from manifest payload.")


def load_episode_ids_from_manifest(path: Path) -> list[int]:
    return extract_episode_ids_from_payload(load_json(path))


@dataclass(frozen=True)
class TaskSpec:
    task_group: str
    task_id: int
    task_instruction: str


@dataclass
class TaskEvaluation:
    task_group: str
    task_id: int
    task_instruction: str
    n_rollouts: int
    seed_start: int
    seed_step: int
    successes: int
    success_rate: float
    success_rate_ci_lower: float
    success_rate_ci_upper: float
    avg_steps: float
    avg_reward: float
    per_rollout_results: list[dict[str, Any]] = field(default_factory=list)
    timings: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class MemberEvaluation:
    created_at: str = field(default_factory=utc_timestamp)
    round_index: int = 0
    member_index: int = 0
    model_path: str = ""
    policy_type: str = ""
    env_type: str = "libero"
    device: str = ""
    eval_batch_size: int = 1
    n_rollouts_per_task: int = 0
    seed_start: int = 0
    seed_step: int = 1
    max_steps: int = 0
    use_async_envs: bool = False
    num_tasks: int = 0
    macro_avg_success_rate: float = 0.0
    macro_avg_reward: float = 0.0
    macro_avg_steps: float = 0.0
    total_successes: int = 0
    total_rollouts: int = 0
    pooled_success_rate: float = 0.0
    pooled_success_rate_ci_lower: float = 0.0
    pooled_success_rate_ci_upper: float = 0.0
    per_task: list[TaskEvaluation] = field(default_factory=list)
    timings: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class RoundMemberSummary:
    member_index: int
    model_path: str
    macro_avg_success_rate: float
    pooled_success_rate: float
    macro_avg_reward: float
    macro_avg_steps: float
    total_successes: int
    total_rollouts: int
    result_path: str
    timings: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class RoundTaskSummary:
    task_group: str
    task_id: int
    task_instruction: str
    num_members: int
    total_successes: int
    total_rollouts: int
    pooled_success_rate: float
    pooled_success_rate_ci_lower: float
    pooled_success_rate_ci_upper: float
    mean_member_success_rate: float
    std_member_success_rate: float
    mean_member_avg_reward: float
    std_member_avg_reward: float
    mean_member_avg_steps: float
    std_member_avg_steps: float


@dataclass
class RoundEvaluation:
    created_at: str = field(default_factory=utc_timestamp)
    run_dir: str = ""
    round_index: int = 0
    num_members_evaluated: int = 0
    mean_member_macro_success_rate: float = 0.0
    std_member_macro_success_rate: float = 0.0
    mean_member_pooled_success_rate: float = 0.0
    std_member_pooled_success_rate: float = 0.0
    mean_member_macro_avg_reward: float = 0.0
    std_member_macro_avg_reward: float = 0.0
    mean_member_macro_avg_steps: float = 0.0
    std_member_macro_avg_steps: float = 0.0
    best_member_index: int | None = None
    best_member_macro_success_rate: float | None = None
    member_results: list[RoundMemberSummary] = field(default_factory=list)
    per_task: list[RoundTaskSummary] = field(default_factory=list)
    timings: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class RunRoundSummary:
    round_index: int
    num_members_evaluated: int
    mean_member_macro_success_rate: float
    mean_member_pooled_success_rate: float
    mean_member_macro_avg_reward: float
    mean_member_macro_avg_steps: float
    best_member_index: int | None
    best_member_macro_success_rate: float | None
    result_path: str
    timings: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class RunEvaluation:
    created_at: str = field(default_factory=utc_timestamp)
    run_name: str = ""
    run_dir: str = ""
    num_rounds_evaluated: int = 0
    tasks: list[TaskSpec] = field(default_factory=list)
    rounds: list[RunRoundSummary] = field(default_factory=list)
    timings: dict[str, dict[str, Any]] = field(default_factory=dict)


def _task_evaluation_from_data(data: dict[str, Any]) -> TaskEvaluation:
    return TaskEvaluation(
        task_group=str(data["task_group"]),
        task_id=int(data["task_id"]),
        task_instruction=str(data["task_instruction"]),
        n_rollouts=int(data["n_rollouts"]),
        seed_start=int(data["seed_start"]),
        seed_step=int(data["seed_step"]),
        successes=int(data["successes"]),
        success_rate=float(data["success_rate"]),
        success_rate_ci_lower=float(data["success_rate_ci_lower"]),
        success_rate_ci_upper=float(data["success_rate_ci_upper"]),
        avg_steps=float(data["avg_steps"]),
        avg_reward=float(data["avg_reward"]),
        per_rollout_results=list(data.get("per_rollout_results", [])),
        timings=normalize_timings(data.get("timings")),
    )


def load_member_evaluation(path: Path) -> MemberEvaluation:
    data = load_json(path)
    return MemberEvaluation(
        created_at=data.get("created_at", utc_timestamp()),
        round_index=int(data["round_index"]),
        member_index=int(data["member_index"]),
        model_path=str(data["model_path"]),
        policy_type=str(data["policy_type"]),
        env_type=str(data.get("env_type", "libero")),
        device=str(data["device"]),
        eval_batch_size=int(data.get("eval_batch_size", 1)),
        n_rollouts_per_task=int(data["n_rollouts_per_task"]),
        seed_start=int(data["seed_start"]),
        seed_step=int(data["seed_step"]),
        max_steps=int(data["max_steps"]),
        use_async_envs=bool(data.get("use_async_envs", False)),
        num_tasks=int(data["num_tasks"]),
        macro_avg_success_rate=float(data["macro_avg_success_rate"]),
        macro_avg_reward=float(data["macro_avg_reward"]),
        macro_avg_steps=float(data["macro_avg_steps"]),
        total_successes=int(data["total_successes"]),
        total_rollouts=int(data["total_rollouts"]),
        pooled_success_rate=float(data["pooled_success_rate"]),
        pooled_success_rate_ci_lower=float(data["pooled_success_rate_ci_lower"]),
        pooled_success_rate_ci_upper=float(data["pooled_success_rate_ci_upper"]),
        per_task=[_task_evaluation_from_data(item) for item in data.get("per_task", [])],
        timings=normalize_timings(data.get("timings")),
    )
