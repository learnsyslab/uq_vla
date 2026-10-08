import numpy as np
import torch
from torch import Tensor

def find_eval_time_indices(
    cfg_times: np.ndarray,
    recorded_times: np.ndarray,
) -> np.ndarray:
    """
    Match 'cfg_times' to 'recorded_times' and returns indices of matches.
    """
    recorded_times = [round(float(t), 3) for t in recorded_times]
    cfg_times = [round(float(t), 3) for t in cfg_times]

    recorded_times_idx_map = {float(time): idx for idx, time in enumerate(recorded_times)}
    matching_idx = []
    missing_times = []
    for time in cfg_times:
        if time in recorded_times_idx_map:
            matching_idx.append(recorded_times_idx_map[time])
        else:
            missing_times.append(time)

    if missing_times:
        print(
            f"Could not find the following evaluation times in the recorded times {missing_times}."
        )
    return np.asarray(matching_idx)

def build_eval_time_mask(
    cfg_times: Tensor,
    recorded_times: Tensor,
) -> torch.Tensor:
    """
    Return a boolean mask over 'recorded_times' indicating which entries match
    any of the values in 'cfg_times'.
    """
    mask = torch.isin(recorded_times, cfg_times)

    # Warn about missing times
    missing_times = cfg_times[~torch.isin(cfg_times, recorded_times)]
    if missing_times.numel() > 0:
        print(
            f"Could not find the following evaluation times in the recorded times {missing_times.tolist()}."
        )

    return mask

def add_multi_scorer_axis(x: Tensor, num_scorers: int, scorer_indices=None) -> Tensor:
    """Normalize to (timesteps, num_scorer, num_samples, horizon, action_dim) by inserting a singleton scorer axis when needed.

    `scorer_indices` selects *which* ensemble members to score with, instead of the first
    `num_scorers` of them. Use it to isolate one member (`scorer_indices: [1]`); left unset,
    the behaviour is exactly the prefix truncation it has always been.
    """
    if scorer_indices is not None:
        scorer_indices = [int(i) for i in scorer_indices]
        if x.dim() == 4:
            return x.unsqueeze(1).repeat(1, len(scorer_indices), 1, 1, 1)
        if x.dim() == 5:
            bad = [i for i in scorer_indices if not -x.size(1) <= i < x.size(1)]
            if bad:
                raise ValueError(
                    f"scorer_indices {bad} out of range for a tensor with {x.size(1)} scorers."
                )
            return x[:, scorer_indices, ...]
        raise ValueError(f"Expected 4D or 5D tensor, got shape {tuple(x.shape)} (ndim={x.dim()}).")

    if x.dim() == 4:
        return x.unsqueeze(1).repeat(1, num_scorers, 1, 1, 1)
    
    if x.dim() == 5:
        if x.size(1) > num_scorers:
            return x[:, :num_scorers, ...]
        if x.size(1) < num_scorers:
            raise ValueError(
                f"Cannot upsample scorers for a tensor. Shape={tuple(x.shape)}."
            )
        return x
    
    raise ValueError(f"Expected 4D or 5D tensor, got shape {tuple(x.shape)} (ndim={x.dim()}).")