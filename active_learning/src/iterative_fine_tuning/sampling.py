"""Sampling utilities for mixing replay pools during iterative fine-tuning."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import torch
from torch.utils.data import Sampler


def frame_indices_for_episode_ids(
    *,
    dataset_from_indices: Sequence[int],
    dataset_to_indices: Sequence[int],
    episode_indices: Sequence[int],
    target_episode_ids: Sequence[int],
    drop_n_first_frames: int = 0,
    drop_n_last_frames: int = 0,
) -> list[int]:
    """Map episode ids to frame indices, preserving repeated episode ids as extra weight."""
    target_id_counts = Counter(int(episode_id) for episode_id in target_episode_ids)
    frame_indices: list[int] = []

    for episode_id, start_index, end_index in zip(
        episode_indices,
        dataset_from_indices,
        dataset_to_indices,
        strict=True,
    ):
        count = target_id_counts.get(int(episode_id), 0)
        if count <= 0:
            continue
        start = int(start_index) + drop_n_first_frames
        end = int(end_index) - drop_n_last_frames
        if end > start:
            episode_frame_indices = list(range(start, end))
            for _ in range(count):
                frame_indices.extend(episode_frame_indices)

    return frame_indices


@dataclass(frozen=True)
class SamplingPool:
    name: str
    frame_indices: tuple[int, ...]
    weight: float


class MixtureFrameSampler(Sampler[int]):
    """Sample frame indices from multiple pools with configurable proportions."""

    def __init__(
        self,
        pools: Sequence[SamplingPool],
        *,
        num_samples: int,
        replacement: bool = True,
        seed: int | None = None,
    ):
        if num_samples <= 0:
            raise ValueError("num_samples must be positive.")

        active_pools = [
            pool for pool in pools
            if pool.weight > 0 and len(pool.frame_indices) > 0
        ]
        if not active_pools:
            raise ValueError("At least one non-empty sampling pool with positive weight is required.")

        total_weight = sum(pool.weight for pool in active_pools)
        self.pools = active_pools
        self.normalized_weights = torch.tensor(
            [pool.weight / total_weight for pool in active_pools],
            dtype=torch.float32,
        )
        self.num_samples = num_samples
        self.replacement = replacement
        self.seed = seed
        self._epoch = 0

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator()
        if self.seed is not None:
            generator.manual_seed(self.seed + self._epoch)
        self._epoch += 1

        pool_indices = torch.multinomial(
            self.normalized_weights,
            num_samples=self.num_samples,
            replacement=True,
            generator=generator,
        )

        if self.replacement:
            for pool_idx in pool_indices.tolist():
                pool = self.pools[pool_idx]
                sample_idx = torch.randint(
                    low=0,
                    high=len(pool.frame_indices),
                    size=(1,),
                    generator=generator,
                ).item()
                yield pool.frame_indices[sample_idx]
            return

        counts = torch.bincount(pool_indices, minlength=len(self.pools)).tolist()
        shuffled_draws: list[list[int]] = []
        for pool, count in zip(self.pools, counts, strict=True):
            if count > len(pool.frame_indices):
                raise ValueError(
                    f"Pool {pool.name!r} only has {len(pool.frame_indices)} frames but "
                    f"{count} draws were requested without replacement."
                )
            perm = torch.randperm(len(pool.frame_indices), generator=generator)[:count].tolist()
            shuffled_draws.append([pool.frame_indices[idx] for idx in perm])

        draw_offsets = [0 for _ in self.pools]
        for pool_idx in pool_indices.tolist():
            frame_indices = shuffled_draws[pool_idx]
            offset = draw_offsets[pool_idx]
            yield frame_indices[offset]
            draw_offsets[pool_idx] += 1
