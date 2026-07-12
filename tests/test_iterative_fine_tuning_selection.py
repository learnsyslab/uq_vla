from iterative_fine_tuning.manifests import EpisodeSelection
from iterative_fine_tuning.selection import _TaskCandidatePool, _build_selection_outputs, _select_episode_candidates


class _Cfg:
    class Iteration:
        def __init__(self, episodes_per_round: int):
            self.episodes_per_round = episodes_per_round

    class Selection:
        def __init__(
            self,
            strategy: str,
            seed: int = 0,
            max_selected_per_task: int | None = None,
            task_selection_temperature: float = 1.0,
            task_weighted_episode_selection: str = "top_k",
        ):
            self.strategy = strategy
            self.seed = seed
            self.max_selected_per_task = max_selected_per_task
            self.task_selection_temperature = task_selection_temperature
            self.task_weighted_episode_selection = task_weighted_episode_selection

    def __init__(
        self,
        *,
        strategy: str,
        episodes_per_round: int,
        seed: int = 0,
        max_selected_per_task: int | None = None,
        task_selection_temperature: float = 1.0,
        task_weighted_episode_selection: str = "top_k",
    ):
        self.iteration = self.Iteration(episodes_per_round)
        self.selection = self.Selection(
            strategy=strategy,
            seed=seed,
            max_selected_per_task=max_selected_per_task,
            task_selection_temperature=task_selection_temperature,
            task_weighted_episode_selection=task_weighted_episode_selection,
        )


def _candidate(episode_id: int, task_id: int, uncertainty: float) -> EpisodeSelection:
    return EpisodeSelection(
        episode_id=episode_id,
        task_group="libero_10",
        task_id=task_id,
        uncertainty=uncertainty,
    )


def _pool(
    task_id: int,
    episode_specs: list[tuple[int, float]],
    *,
    historical_selected_count: int = 0,
    task_uncertainty: float | None = None,
    scored_episode_count: int | None = None,
) -> _TaskCandidatePool:
    candidates = [_candidate(episode_id, task_id, uncertainty) for episode_id, uncertainty in episode_specs]
    return _TaskCandidatePool(
        task_group="libero_10",
        task_id=task_id,
        instruction=None,
        candidates=candidates,
        uncertainty=(
            task_uncertainty
            if task_uncertainty is not None
            else sum(uncertainty for _, uncertainty in episode_specs) / len(episode_specs)
        ),
        historical_selected_count=historical_selected_count,
        scored_episode_count=(
            scored_episode_count if scored_episode_count is not None else len(episode_specs)
        ),
    )


def test_top_k_selects_highest_uncertainty_episodes_globally():
    cfg = _Cfg(strategy="top_k", episodes_per_round=3)
    task_pools = {
        ("libero_10", 0): _pool(0, [(100, 0.4), (101, 0.95)]),
        ("libero_10", 1): _pool(1, [(200, 0.9), (201, 0.2)]),
    }

    selected = _select_episode_candidates(cfg, task_pools=task_pools)

    assert [candidate.episode_id for candidate in selected] == [101, 200, 100]


def test_balanced_top_k_seeds_one_episode_per_task_before_global_fill():
    cfg = _Cfg(strategy="balanced_top_k", episodes_per_round=4)
    task_pools = {
        ("libero_10", 0): _pool(0, [(100, 0.99), (101, 0.98), (102, 0.97)]),
        ("libero_10", 1): _pool(1, [(200, 0.7)]),
        ("libero_10", 2): _pool(2, [(300, 0.6)]),
    }

    selected = _select_episode_candidates(cfg, task_pools=task_pools)

    assert [candidate.episode_id for candidate in selected] == [100, 200, 300, 101]


def test_constrained_top_k_respects_per_round_task_caps():
    cfg = _Cfg(strategy="constrained_top_k", episodes_per_round=4, max_selected_per_task=2)
    task_pools = {
        ("libero_10", 0): _pool(0, [(100, 0.99), (101, 0.98)], historical_selected_count=1),
        ("libero_10", 1): _pool(1, [(200, 0.97), (201, 0.96)], historical_selected_count=2),
        ("libero_10", 2): _pool(2, [(300, 0.95), (301, 0.94)], historical_selected_count=0),
    }

    selected = _select_episode_candidates(cfg, task_pools=task_pools)

    assert [candidate.episode_id for candidate in selected] == [100, 101, 200, 201]


def test_task_weighted_top_k_can_repeat_the_highest_weight_task():
    cfg = _Cfg(
        strategy="task_weighted_top_k",
        episodes_per_round=3,
        seed=0,
        task_selection_temperature=1.0,
    )
    task_pools = {
        ("libero_10", 0): _pool(0, [(100, 100.0), (101, 99.0), (102, 98.0)]),
        ("libero_10", 1): _pool(1, [(200, 1.0), (201, 0.9), (202, 0.8)]),
    }

    selected = _select_episode_candidates(cfg, task_pools=task_pools)

    assert [candidate.episode_id for candidate in selected] == [100, 101, 102]


def test_task_weighted_top_k_can_sample_uniform_random_episodes_within_a_task():
    cfg = _Cfg(
        strategy="task_weighted_top_k",
        episodes_per_round=3,
        seed=0,
        task_selection_temperature=10.0,
        task_weighted_episode_selection="uniform_random",
    )
    task_pools = {
        ("libero_10", 0): _pool(0, [(100, 100.0), (101, 99.0), (102, 98.0), (103, 97.0)]),
        ("libero_10", 1): _pool(1, [(200, 1.0), (201, 0.9), (202, 0.8), (203, 0.7)]),
    }

    selected = _select_episode_candidates(cfg, task_pools=task_pools)

    assert [candidate.episode_id for candidate in selected] != [100, 101, 102]
    assert len({candidate.episode_id for candidate in selected}) == 3
    assert all(candidate.task_id == 0 for candidate in selected)


def test_task_weighted_top_k_uses_task_uncertainty_from_all_task_episodes():
    cfg = _Cfg(
        strategy="task_weighted_top_k",
        episodes_per_round=2,
        seed=0,
        task_selection_temperature=1.0,
    )
    task_pools = {
        ("libero_10", 0): _pool(
            0,
            [(100, 0.1)],
            historical_selected_count=5,
            task_uncertainty=100.0,
            scored_episode_count=6,
        ),
        ("libero_10", 1): _pool(
            1,
            [(200, 10.0), (201, 9.0)],
            historical_selected_count=0,
            task_uncertainty=1.0,
            scored_episode_count=2,
        ),
    }

    selected = _select_episode_candidates(cfg, task_pools=task_pools)

    assert [candidate.episode_id for candidate in selected] == [200, 100]


def test_task_weighted_top_k_renormalizes_after_a_task_runs_out_of_candidates():
    cfg = _Cfg(
        strategy="task_weighted_top_k",
        episodes_per_round=3,
        seed=0,
        task_selection_temperature=1.0,
    )
    task_pools = {
        ("libero_10", 0): _pool(0, [(100, 100.0)]),
        ("libero_10", 1): _pool(1, [(200, 10.0), (201, 9.0), (202, 8.0)]),
        ("libero_10", 2): _pool(2, [(300, 1.0), (301, 0.9), (302, 0.8)]),
    }

    selected = _select_episode_candidates(cfg, task_pools=task_pools)

    assert [candidate.episode_id for candidate in selected] == [100, 200, 201]


def test_task_weighted_top_k_temperature_changes_the_task_distribution():
    uniform_cfg = _Cfg(
        strategy="task_weighted_top_k",
        episodes_per_round=3,
        seed=0,
        task_selection_temperature=0.0,
    )
    sharp_cfg = _Cfg(
        strategy="task_weighted_top_k",
        episodes_per_round=3,
        seed=0,
        task_selection_temperature=10.0,
    )
    task_pools = {
        ("libero_10", 0): _pool(0, [(100, 9.0), (101, 9.0), (102, 9.0)]),
        ("libero_10", 1): _pool(1, [(200, 1.0), (201, 1.0), (202, 1.0)]),
    }

    uniform_selected = _select_episode_candidates(uniform_cfg, task_pools=task_pools)
    sharp_selected = _select_episode_candidates(sharp_cfg, task_pools=task_pools)

    assert [candidate.episode_id for candidate in uniform_selected] == [100, 200, 201]
    assert [candidate.episode_id for candidate in sharp_selected] == [100, 101, 102]


def test_task_weighted_top_k_uses_a_distinct_rng_stream_per_round():
    cfg = _Cfg(
        strategy="task_weighted_top_k",
        episodes_per_round=4,
        seed=0,
        task_selection_temperature=0.0,
    )
    task_pools = {
        ("libero_10", 0): _pool(0, [(100, 9.0), (101, 8.9), (102, 8.8)]),
        ("libero_10", 1): _pool(1, [(200, 1.0), (201, 0.9), (202, 0.8)]),
        ("libero_10", 2): _pool(2, [(300, 1.0), (301, 0.9), (302, 0.8)]),
        ("libero_10", 3): _pool(3, [(400, 1.0), (401, 0.9), (402, 0.8)]),
    }

    round_zero = _select_episode_candidates(cfg, task_pools=task_pools, round_index=0)
    repeated_round_zero = _select_episode_candidates(cfg, task_pools=task_pools, round_index=0)
    round_one = _select_episode_candidates(cfg, task_pools=task_pools, round_index=1)

    assert [candidate.episode_id for candidate in round_zero] == [
        candidate.episode_id for candidate in repeated_round_zero
    ]
    assert [candidate.episode_id for candidate in round_zero] != [
        candidate.episode_id for candidate in round_one
    ]


def test_random_selection_uses_a_distinct_rng_stream_per_round():
    cfg = _Cfg(
        strategy="random",
        episodes_per_round=4,
        seed=0,
    )
    task_pools = {
        ("libero_10", 0): _pool(0, [(100, 9.0), (101, 8.9), (102, 8.8)]),
        ("libero_10", 1): _pool(1, [(200, 1.0), (201, 0.9), (202, 0.8)]),
        ("libero_10", 2): _pool(2, [(300, 1.0), (301, 0.9), (302, 0.8)]),
        ("libero_10", 3): _pool(3, [(400, 1.0), (401, 0.9), (402, 0.8)]),
    }

    round_zero = _select_episode_candidates(cfg, task_pools=task_pools, round_index=0)
    repeated_round_zero = _select_episode_candidates(cfg, task_pools=task_pools, round_index=0)
    round_one = _select_episode_candidates(cfg, task_pools=task_pools, round_index=1)

    assert [candidate.episode_id for candidate in round_zero] == [
        candidate.episode_id for candidate in repeated_round_zero
    ]
    assert [candidate.episode_id for candidate in round_zero] != [
        candidate.episode_id for candidate in round_one
    ]


def test_task_summaries_are_derived_from_episode_level_selection():
    task_pools = {
        ("libero_10", 0): _pool(0, [(100, 0.9), (101, 0.8)]),
        ("libero_10", 1): _pool(1, [(200, 0.7)]),
    }
    selected_candidates = [
        task_pools[("libero_10", 0)].candidates[0],
        task_pools[("libero_10", 1)].candidates[0],
    ]

    all_task_summaries, selected_tasks, selected_episodes = _build_selection_outputs(
        task_pools,
        selected_candidates,
    )

    assert [(task.task_id, task.selected_episode_ids) for task in all_task_summaries] == [
        (0, [100]),
        (1, [200]),
    ]
    assert [episode.metadata["selection_level"] for episode in selected_episodes] == ["episode", "episode"]
    assert [task.task_id for task in selected_tasks] == [0, 1]
    assert [task.metadata["episode_uncertainty_count"] for task in all_task_summaries] == [2, 1]
