from types import SimpleNamespace

import pytest

from iterative_fine_tuning.config import ReplayMixingConfig
from iterative_fine_tuning.sampling import (
    MixtureFrameSampler,
    SamplingPool,
    frame_indices_for_episode_ids,
)
from iterative_fine_tuning.training import _build_sampling_pools


def test_frame_indices_for_episode_ids_use_true_episode_ids():
    frame_indices = frame_indices_for_episode_ids(
        dataset_from_indices=[0, 4, 8],
        dataset_to_indices=[4, 8, 12],
        episode_indices=[100, 200, 300],
        target_episode_ids=[200, 300],
        drop_n_last_frames=1,
    )

    assert frame_indices == [4, 5, 6, 8, 9, 10]


def test_frame_indices_for_episode_ids_preserve_episode_multiplicity():
    frame_indices = frame_indices_for_episode_ids(
        dataset_from_indices=[0, 4, 8],
        dataset_to_indices=[4, 8, 12],
        episode_indices=[100, 200, 300],
        target_episode_ids=[200, 200, 300],
        drop_n_last_frames=1,
    )

    assert frame_indices == [4, 5, 6, 4, 5, 6, 8, 9, 10]


def test_mixture_sampler_without_replacement_respects_requested_length():
    sampler = MixtureFrameSampler(
        pools=[
            SamplingPool(name="history", frame_indices=(0, 1, 2), weight=0.5),
            SamplingPool(name="new", frame_indices=(10, 11, 12), weight=0.5),
        ],
        num_samples=4,
        replacement=False,
        seed=7,
    )

    draws = list(iter(sampler))

    assert len(draws) == 4
    assert len(set(draws)) == 4
    assert all(draw in {0, 1, 2, 10, 11, 12} for draw in draws)


def test_replay_mixing_config_rejects_unknown_sampling_mode():
    cfg = ReplayMixingConfig(history_tasks={"libero_10": [0]}, sampling_mode="invalid")

    with pytest.raises(ValueError, match="sampling_mode"):
        cfg.validate()


def test_build_sampling_pools_keep_overlap_weight_in_uniform_all_frames_mode():
    cfg = SimpleNamespace(
        replay=ReplayMixingConfig(
            history_tasks={"libero_10": [0]},
            history_fraction=0.25,
            new_fraction=0.75,
            sampling_mode="uniform_all_frames",
        ),
        policy=SimpleNamespace(drop_n_last_frames=0),
    )
    dataset = SimpleNamespace(
        meta=SimpleNamespace(
            episodes={
                "episode_index": [10, 20, 30],
                "dataset_from_index": [0, 3, 6],
                "dataset_to_index": [3, 6, 9],
            }
        )
    )

    pools, sampling_weights, training_episode_ids, num_samples = _build_sampling_pools(
        cfg,
        dataset=dataset,
        history_episode_ids=[10, 20],
        new_episode_ids=[20, 30],
    )

    assert len(pools) == 1
    assert pools[0].name == "all"
    assert pools[0].frame_indices == (0, 1, 2, 3, 4, 5, 3, 4, 5, 6, 7, 8)
    assert sampling_weights == {"all": 1.0}
    assert training_episode_ids == [10, 20, 30]
    assert num_samples == 12
