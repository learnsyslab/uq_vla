"""Keep the oversampled episodes' frames decoded in memory instead of re-decoding them every step.

Training samples between the episodes selected so far (5 per round) and the history episodes. Every
frame of the selected half is therefore drawn dozens of times per round, and each draw costs a random
seek into a shared mp4 plus a decode from the preceding key frame, which can leave the GPU idle.

Decoding those episodes once, sequentially, into shared-memory uint8 tensors removes that half of the
decode load; the history half keeps the normal path, since its frames are seen about once each and
caching them would cost hundreds of GB.

The cached frames are bit-identical to the decoded ones: `decode_video_frames` returns exactly
`uint8 / 255`, so storing uint8 and dividing on read reproduces it.
"""

from __future__ import annotations

import logging
import time

import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.video_utils import _default_decoder_cache

logger = logging.getLogger(__name__)


def _episode_frame_span(dataset: LeRobotDataset, ep_idx: int, vid_key: str) -> tuple[int, int]:
    """Frame range this episode occupies inside its shared mp4 (episodes are concatenated per file)."""
    row = dataset._episode_row(ep_idx)
    start = round(float(row[f"videos/{vid_key}/from_timestamp"]) * dataset.meta.fps)
    return start, start + int(row["length"])


def _decode_range_torchcodec(path, start: int, stop: int) -> torch.Tensor:
    from torchcodec.decoders import VideoDecoder

    decoder = VideoDecoder(str(path), seek_mode="approximate")
    # One sequential pass over the episode, rather than a seek per sample.
    frames = decoder.get_frames_at(indices=list(range(start, stop))).data
    del decoder  # close the file handle before any loader forks
    return frames.to(torch.uint8)


def _decode_range_pyav(path, start: int, stop: int) -> torch.Tensor:
    """Same frames via PyAV, for machines whose torchcodec has no FFmpeg libraries."""
    import av
    import numpy as np

    out = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for i, frame in enumerate(container.decode(stream)):
            if i >= stop:
                break
            if i >= start:
                out.append(torch.from_numpy(np.ascontiguousarray(frame.to_ndarray(format="rgb24"))))
    return torch.stack(out).permute(0, 3, 1, 2).contiguous()  # T,H,W,C -> T,C,H,W uint8


def _decode_range(path, start: int, stop: int) -> torch.Tensor:
    try:
        return _decode_range_torchcodec(path, start, stop)
    except (ImportError, RuntimeError, OSError):
        return _decode_range_pyav(path, start, stop)


def build_frame_cache(
    dataset: LeRobotDataset, episode_ids: list[int]
) -> tuple[dict[tuple[str, int], torch.Tensor], float, float]:
    """Decode every frame of `episode_ids` into shared-memory uint8 tensors.

    Returns the cache, the seconds spent building it, and its size in GB.
    """
    cache: dict[tuple[str, int], torch.Tensor] = {}
    started = time.perf_counter()
    total_bytes = 0
    for ep_idx in dict.fromkeys(int(e) for e in episode_ids):  # ids may repeat as sampling weight
        for vid_key in dataset.meta.video_keys:
            path = dataset.root / dataset.meta.get_video_file_path(ep_idx, vid_key)
            start, stop = _episode_frame_span(dataset, ep_idx, vid_key)
            buf = _decode_range(path, start, stop).contiguous()
            buf.share_memory_()  # inherited by fork()ed loader workers, not copied per worker
            cache[(vid_key, ep_idx)] = buf
            total_bytes += buf.numel()
    return cache, time.perf_counter() - started, total_bytes / 1e9


def attach_frame_cache(dataset: LeRobotDataset, cache: dict[tuple[str, int], torch.Tensor]) -> None:
    """Serve cached episodes from memory, leaving every other episode on the decode path."""
    fps = dataset.meta.fps
    original_query_videos = dataset._query_videos

    def cached_query_videos(query_timestamps: dict[str, list[float]], ep_idx: int) -> dict:
        item = {}
        for vid_key, query_ts in query_timestamps.items():
            buf = cache.get((vid_key, int(ep_idx)))
            if buf is None:
                item.update(original_query_videos({vid_key: query_ts}, ep_idx))
                continue
            indices = [min(max(round(ts * fps), 0), buf.shape[0] - 1) for ts in query_ts]
            item[vid_key] = (buf[indices].to(torch.float32) / 255.0).squeeze(0)
        return item

    dataset._query_videos = cached_query_videos
    # Decoders opened while building the cache must not be inherited by loader workers: sharing one
    # file handle across processes corrupts the reads ("Could not push packet to decoder").
    _default_decoder_cache.clear()


def enable_frame_cache(dataset: LeRobotDataset, episode_ids: list[int]) -> None:
    """Build and attach the cache, logging what it cost. Falls back to plain decoding on failure."""
    try:
        cache, seconds, gigabytes = build_frame_cache(dataset, episode_ids)
    except Exception:  # a cache is an optimisation; never fail a round over it
        logger.exception("Frame cache could not be built; falling back to decoding every sample")
        return
    attach_frame_cache(dataset, cache)
    logger.info(
        "Frame cache: %d episodes, %d tensors, %.2f GB, built in %.1f s",
        len({ep for _, ep in cache}),
        len(cache),
        gigabytes,
        seconds,
    )
