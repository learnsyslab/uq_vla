import json
from pathlib import Path

import pytest

from iterative_fine_tuning.config import SelectionConfig
from iterative_fine_tuning.env import resolve_local_path
from iterative_fine_tuning.evaluate_run import (
    _coalesce_slurm_states,
    _fully_evaluated_round_indices,
    _normalize_slurm_state,
)
from iterative_fine_tuning.layout import RunLayout
from iterative_fine_tuning.manifests import (
    extract_episode_ids_from_payload,
    load_member_evaluation,
    load_run_state,
    load_selection_manifest,
    load_training_manifest,
)


def test_extract_episode_ids_supports_legacy_payloads():
    payload = {
        "rows": [
            {"episode_id": 11},
            {"episode_id": 7},
            {"episode_id": 11},
        ]
    }

    assert extract_episode_ids_from_payload(payload) == [11, 7, 11]


def test_resolve_local_path_rejects_unresolved_env_vars(tmp_path: Path):
    with pytest.raises(ValueError, match="unresolved environment variable"):
        resolve_local_path("${MISSING_ITERATIVE_ROOT}/run", base_dir=tmp_path)


def test_selection_config_rejects_unknown_task_weighted_episode_selection():
    cfg = SelectionConfig(
        ensemble_model_paths=["/tmp/model_a", "/tmp/model_b"],
        strategy="task_weighted_top_k",
        task_weighted_episode_selection="invalid",
    )

    with pytest.raises(ValueError, match="task_weighted_episode_selection"):
        cfg.validate()


def test_load_training_manifest_supports_legacy_single_model_shape(tmp_path: Path):
    manifest_path = tmp_path / "training_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "round_index": 2,
                "start_model_path": "/tmp/start",
                "final_model_path": "/tmp/final",
                "history_episode_ids": [1, 2],
            }
        ),
        encoding="utf-8",
    )

    manifest = load_training_manifest(manifest_path)

    assert manifest.start_model_paths == ["/tmp/start"]
    assert manifest.final_model_paths == ["/tmp/final"]
    assert manifest.member_training_runs[0].member_index == 0


def test_load_run_state_supports_legacy_single_model_shape(tmp_path: Path):
    state_path = tmp_path / "run_state.json"
    state_path.write_text(
        json.dumps(
            {
                "run_name": "legacy",
                "rounds": [
                    {
                        "round_index": 0,
                        "start_model_path": "/tmp/start_a",
                        "final_model_path": "/tmp/final_a",
                        "selected_episode_ids": [3],
                    },
                    {
                        "round_index": 1,
                        "start_model_path": "/tmp/start_b",
                        "final_model_path": "/tmp/final_b",
                        "selected_episode_ids": [4],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    state = load_run_state(state_path)

    assert state.rounds[0].start_model_paths == ["/tmp/start_a"]
    assert state.latest_trained_model_paths() == ["/tmp/final_b"]


def test_run_state_cumulative_selected_episode_ids_preserve_reselections(tmp_path: Path):
    state_path = tmp_path / "run_state.json"
    state_path.write_text(
        json.dumps(
            {
                "run_name": "reselected",
                "bootstrap_episode_ids": [1, 1],
                "rounds": [
                    {
                        "round_index": 0,
                        "selected_episode_ids": [2, 3],
                    },
                    {
                        "round_index": 1,
                        "selected_episode_ids": [3, 4],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    state = load_run_state(state_path)

    assert state.cumulative_selected_episode_ids(include_round=1) == [1, 1, 2, 3, 3, 4]


def test_load_selection_manifest_supports_task_level_metadata(tmp_path: Path):
    manifest_path = tmp_path / "selection_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "round_index": 0,
                "strategy": "top_k",
                "candidate_count": 10,
                "candidate_episode_count": 100,
                "selected_tasks": [
                    {
                        "task_group": "libero_10",
                        "task_id": 3,
                        "uncertainty": 1.25,
                        "candidate_episode_count": 10,
                        "selected_episode_count": 2,
                        "selected_episode_ids": [30, 31],
                    }
                ],
                "selected_episodes": [
                    {
                        "episode_id": 30,
                        "task_group": "libero_10",
                        "task_id": 3,
                        "uncertainty": 1.25,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    manifest = load_selection_manifest(manifest_path)

    assert manifest.candidate_count == 10
    assert manifest.candidate_episode_count == 100
    assert manifest.selected_tasks[0].task_id == 3
    assert manifest.selected_tasks[0].selected_episode_ids == [30, 31]
    assert manifest.episode_ids == [30]


def test_load_selection_manifest_preserves_timings(tmp_path: Path):
    manifest_path = tmp_path / "selection_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "round_index": 0,
                "strategy": "top_k",
                "timings": {
                    "total": {
                        "started_at": "2026-01-01T00:00:00+00:00",
                        "completed_at": "2026-01-01T00:00:05+00:00",
                        "duration_s": 5.0,
                    },
                    "uncertainty_quantification": {
                        "started_at": "2026-01-01T00:00:01+00:00",
                        "completed_at": "2026-01-01T00:00:04+00:00",
                        "duration_s": 2.5,
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    manifest = load_selection_manifest(manifest_path)

    assert manifest.timings["total"]["duration_s"] == pytest.approx(5.0)
    assert manifest.timings["uncertainty_quantification"]["duration_s"] == pytest.approx(2.5)


def test_load_training_manifest_preserves_member_timings(tmp_path: Path):
    manifest_path = tmp_path / "training_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "round_index": 1,
                "start_model_paths": ["/tmp/start"],
                "final_model_paths": ["/tmp/final"],
                "timings": {
                    "total": {
                        "started_at": "2026-01-01T00:00:00+00:00",
                        "completed_at": "2026-01-01T00:10:00+00:00",
                        "duration_s": 600.0,
                    }
                },
                "member_training_runs": [
                    {
                        "member_index": 0,
                        "start_model_path": "/tmp/start",
                        "final_model_path": "/tmp/final",
                        "training_dir": "/tmp/member_00",
                        "timings": {
                            "total": {
                                "started_at": "2026-01-01T00:00:00+00:00",
                                "completed_at": "2026-01-01T00:09:00+00:00",
                                "duration_s": 540.0,
                            }
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    manifest = load_training_manifest(manifest_path)

    assert manifest.timings["total"]["duration_s"] == pytest.approx(600.0)
    assert manifest.member_training_runs[0].timings["total"]["duration_s"] == pytest.approx(540.0)


def test_load_member_evaluation_preserves_task_timings(tmp_path: Path):
    evaluation_path = tmp_path / "member_evaluation.json"
    evaluation_path.write_text(
        json.dumps(
            {
                "round_index": 0,
                "member_index": 1,
                "model_path": "/tmp/model",
                "policy_type": "smolvla",
                "device": "cuda",
                "n_rollouts_per_task": 10,
                "seed_start": 100,
                "seed_step": 1,
                "max_steps": 520,
                "num_tasks": 1,
                "macro_avg_success_rate": 0.5,
                "macro_avg_reward": 1.0,
                "macro_avg_steps": 42.0,
                "total_successes": 5,
                "total_rollouts": 10,
                "pooled_success_rate": 0.5,
                "pooled_success_rate_ci_lower": 0.1,
                "pooled_success_rate_ci_upper": 0.9,
                "timings": {
                    "total": {
                        "started_at": "2026-01-01T00:00:00+00:00",
                        "completed_at": "2026-01-01T00:02:00+00:00",
                        "duration_s": 120.0,
                    }
                },
                "per_task": [
                    {
                        "task_group": "libero_10",
                        "task_id": 0,
                        "task_instruction": "open the drawer",
                        "n_rollouts": 10,
                        "seed_start": 100,
                        "seed_step": 1,
                        "successes": 5,
                        "success_rate": 0.5,
                        "success_rate_ci_lower": 0.1,
                        "success_rate_ci_upper": 0.9,
                        "avg_steps": 42.0,
                        "avg_reward": 1.0,
                        "per_rollout_results": [],
                        "timings": {
                            "total": {
                                "started_at": "2026-01-01T00:00:10+00:00",
                                "completed_at": "2026-01-01T00:01:00+00:00",
                                "duration_s": 50.0,
                            }
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    evaluation = load_member_evaluation(evaluation_path)

    assert evaluation.timings["total"]["duration_s"] == pytest.approx(120.0)
    assert evaluation.per_task[0].timings["total"]["duration_s"] == pytest.approx(50.0)


def test_run_layout_exposes_initial_evaluation_paths(tmp_path: Path):
    layout = RunLayout(root=tmp_path / "example_run")

    assert layout.initial_evaluation_dir == layout.root / "initial_evaluation"
    assert layout.initial_evaluation_path == layout.initial_evaluation_dir / "initial_evaluation.json"
    assert layout.initial_member_dir(2) == layout.initial_evaluation_dir / "member_02"
    assert (
        layout.initial_member_evaluation_path(2)
        == layout.initial_member_dir(2) / "member_evaluation.json"
    )


def test_normalize_and_coalesce_slurm_states():
    assert _normalize_slurm_state("cancelled by 12345") == "CANCELLED"
    assert _coalesce_slurm_states(["COMPLETED", "FAILED"]) == "FAILED"
    assert _coalesce_slurm_states(["PENDING", "COMPLETED"]) == "PENDING"


def test_fully_evaluated_round_indices_counts_members():
    round_member_evaluations = {
        0: [(object(), Path("/tmp/member0.json"))],
        1: [
            (object(), Path("/tmp/member1a.json")),
            (object(), Path("/tmp/member1b.json")),
        ],
    }

    completed = _fully_evaluated_round_indices(
        expected_member_counts={0: 2, 1: 2},
        round_member_evaluations=round_member_evaluations,
    )

    assert completed == {1}
