import numpy as np
import os
import pathlib
import scipy.stats as stats
import sys

from typing import Dict, List, Union

ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)


def _calculate_accuracy(TP: int, TN: int, FP: int, FN: int) -> tuple[float, float, float, float]:
    """Calculate accuracy metrics from confusion matrix values.

    Returns:
        TPR: True Positive Rate (Sensitivity)

        TNR: True Negative Rate (Specificity)

        accuracy: Overall accuracy

        balanced_accuracy: Balanced accuracy
    """
    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
    TNR = TN / (FP + TN) if (FP + TN) > 0 else 0
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    balanced_accuracy = (TPR + TNR) / 2
    return TPR, TNR, accuracy, balanced_accuracy


def _calculate_confusion_matrix(
    failures_detected: Union[list, np.ndarray], successful_rollouts: Union[list, np.ndarray]
):
    """Calculate confusion matrix values from detected failures and successful rollouts.

    Args:
        failures_detected: List or numpy array of detected failures in rollouts.
        successful_rollouts: List or numpy array of successful rollouts.

    Returns:
        TP, TN, FP, FN: True Positive, True Negative, False Positive, False Negative counts.
    """
    failures_detected = np.asarray(failures_detected, dtype=bool)
    successful_rollouts = np.asarray(successful_rollouts, dtype=bool)

    if failures_detected.shape != successful_rollouts.shape:
        raise ValueError("Shape of failures_detected and successful_rollouts must be the same.")

    # Predicted "failure" but actually success => FP
    FP = int(np.sum(failures_detected & successful_rollouts))
    TN = int(np.sum(~failures_detected & successful_rollouts))
    TP = int(np.sum(failures_detected & ~successful_rollouts))
    FN = int(np.sum(~failures_detected & ~successful_rollouts))
    return TP, TN, FP, FN


def _calculate_twa(detected_failure_in_episode, successful_rollouts, detection_times):
    """
    Calculate the timestep-wise accuracy metrics.

    Args:
        detected_failure_in_episode: List of booleans indicating if a failure was detected in the episode.
        successful_rollouts: List of booleans indicating if the episode was successful.
        detected_times: List of indices where the failure was detected.

    Returns:
        timestep_wise_accuracy: Timestep-wise accuracy metric.
    """
    num_successful_rollouts = np.sum(successful_rollouts)
    num_failed_rollouts = len(successful_rollouts) - num_successful_rollouts
    # Calculate the timestep-wise accuracy
    timestep_wise_accuracy = 0.0
    # Counter for the detection times as they are only for failed rollouts that were detected
    count = 0
    for i in range(len(detected_failure_in_episode)):
        if detected_failure_in_episode[i] and not successful_rollouts[i]:
            timestep_wise_accuracy += (1 - detection_times[count]) / num_failed_rollouts
            count += 1
        elif not detected_failure_in_episode[i] and successful_rollouts[i]:
            timestep_wise_accuracy += 1 / num_successful_rollouts
    return timestep_wise_accuracy * 0.5


def translate_scores(uncertainty_scores):
    """Translate uncertainty scores to a dictionary with keys as step indices."""
    scores_for_all_steps = {}
    for episode in uncertainty_scores:
        for i, step in enumerate(episode):
            if i not in scores_for_all_steps:
                scores_for_all_steps[i] = []
            scores_for_all_steps[i].append(step)
    return scores_for_all_steps


def compute_thresholds(
    scores_per_episode: List[np.ndarray],
    threshold_strategy: str,
    quantile: float,
    window_size: int,
) -> np.ndarray:
    """
    Compute per-step thresholds for the given strategy.

    Strategies:
      - "ct_quantile": Constant threshold, quantile of episode-wise maxima.
      - "ct_mean_std" : Constant threshold, mean + z*std of episode-wise maxima.
      - "tvt_quantile": Time-varying local per-step quantile blended with a
            constant global quantile baseline via a window-fill weight.
      - "tvt_mean_std": Time-varying local per-step mean+z*std blended with a
            constant global mean+z*std baseline via a window-fill weight.
      - "tvt_cp_band": Time-varying local CP band blended with a constant global
            quantile baseline via a window-fill weight.

    Args:
        scores_per_episode: Per-episode score sequences.
        threshold_strategy: Strategy to compute threshold.
        quantile: Target upper quantile in (0, 1). 
        window_size: Rolling window size used to compute blending weights..

    Returns:
        Per-step thresholds.
    """
    if threshold_strategy == "ct_quantile":
        # Constant threshold via quantile of episode maxima
        threshold = compute_ct_quantile_threshold(
            scores_per_episode=scores_per_episode,
            quantile=quantile,
        )
    elif threshold_strategy == "ct_mean_std":
        # Constant threshold via mean+z*std of episode maxima
        threshold = compute_ct_mean_std_threshold(
            scores_per_episode=scores_per_episode,
            quantile=quantile,
        )
    elif threshold_strategy == "tvt_quantile":
        # Local: per-step quantile; Global: quantile of episode maxima
        global_treshold = compute_ct_quantile_threshold(
            scores_per_episode=scores_per_episode,
            quantile=quantile,
        )
        local_threshold = compute_tv_quantile_threshold(
            scores_per_episode=scores_per_episode,
            quantile=quantile,
        )
        threshold = blend_incomplete_windows_with_global_threshold(
            local_threshold=local_threshold,
            global_threshold=global_treshold,
            window_size=window_size
        )
        threshold = blend_low_coverage_with_global_threshold(
            local_threshold=threshold,
            global_threshold=global_treshold,
            scores_per_episode=scores_per_episode,
        )
    elif threshold_strategy == "tvt_mean_std":
        # Local: per-step mean+z*std; Global: mean+z*std of episode maxima
        global_treshold = compute_ct_mean_std_threshold(
            scores_per_episode=scores_per_episode,
            quantile=quantile,
        )
        local_threshold = compute_tv_mean_std_threshold(
            scores_per_episode=scores_per_episode,
            quantile=quantile,
        )
        threshold = blend_incomplete_windows_with_global_threshold(
            local_threshold=local_threshold,
            global_threshold=global_treshold,
            window_size=window_size
        )
        threshold = blend_low_coverage_with_global_threshold(
            local_threshold=threshold,
            global_threshold=global_treshold,
            scores_per_episode=scores_per_episode,
        )
    elif threshold_strategy == "tvt_cp_band":
        # Local: CP band upper curve; Global: quantile of episode maxima
        global_treshold = compute_ct_quantile_threshold(
            scores_per_episode=scores_per_episode,
            quantile=quantile,
        )
        local_threshold = compute_cp_band_threshold(
            scores_per_episode=scores_per_episode.copy(),
            quantile=quantile
        )
        threshold = blend_incomplete_windows_with_global_threshold(
            local_threshold=local_threshold,
            global_threshold=global_treshold,
            window_size=window_size
        )
        threshold = blend_low_coverage_with_global_threshold(
            local_threshold=threshold,
            global_threshold=global_treshold,
            scores_per_episode=scores_per_episode,
        )
    else:
        raise ValueError(f"Unknown threshold strategy: {threshold_strategy}")
    return threshold

def blend_low_coverage_with_global_threshold(
    local_threshold: np.ndarray,
    global_threshold: np.ndarray,
    scores_per_episode: List[np.ndarray],
) -> np.ndarray:
    """
    Compute a time-varying threshold curve by blending:
      - a global baseline threshold, and
      - a local per-step estimate.

    The blending weight at each step t is based on coverage, i.e. the fraction of
    calibration episodes that contain a score at step t:

        w_t = n_t / N

    where n_t is the number of episodes with data at step t and N is the total
    number of calibration episodes.

    Args:
        local_threshold: Per-step local thresholds.
        global_threshold: Sequence of per-step threshold values where every entry is the same constant threshold.
        scores_per_episode: List of per-episode score arrays, used to calculate coverage n_t at each step.

    Returns:
        1D numpy array of blended per-step thresholds.
    """
    # Ensure global_threshold is constant across all steps
    if not np.allclose(global_threshold, global_threshold[0]):
        raise ValueError(
            "All entries in global_threshold must be identical. "
            f"Got min={global_threshold.min()}, max={global_threshold.max()}."
        )
    
    # Map: step to array of scores at that step across all episodes
    scores_per_step: Dict[int, np.ndarray] = translate_scores(scores_per_episode)
    # Compute coverage per step: n_t episodes contain a score at step t
    coverage_counts = np.array([len(scores) for scores in scores_per_step.values()])

    blending_weights = coverage_counts / coverage_counts[0]

    # Linear interpolation between global and local thresholds
    return blending_weights * local_threshold + (1.0 - blending_weights) * global_threshold[0]

def blend_incomplete_windows_with_global_threshold(
    local_threshold: np.ndarray,
    global_threshold: np.ndarray,
    window_size: int,
) -> np.ndarray:
    """
    Compute a time-varying threshold curve. Each step's threshold is a blend of:
    - a global baseline, and
    - a local per-step estimate.
    The blending weight increases with the effective window size, so early steps lean on the global
    baseline, while later steps rely more on the local estimate as the effective window size approaches
    the actual window size.

    Args:
        local_threshold: Per-step local thresholds.
        global_threshold: Sequence of per-step threshold values where every entry is the same constant threshold.
        window_size: Size of the rolling window.

    Returns:
        Per-step blended threshold values.
    """
    # Ensure global_threshold is constant across all steps
    if not np.allclose(global_threshold, global_threshold[0]):
        raise ValueError(
            "All entries in global_threshold must be identical. "
            f"Got min={global_threshold.min()}, max={global_threshold.max()}."
        )

    # Blending weights based on effective window length at each step
    blending_weights = np.minimum(np.arange(local_threshold.size) + 1, window_size) / window_size

    # Linear interpolation between global and local thresholds
    return blending_weights * local_threshold + (1 - blending_weights) * global_threshold

def compute_ct_quantile_threshold(
    scores_per_episode: np.ndarray,
    quantile: float,
) -> np.ndarray:
    """
    Compute a constant threshold from the episode-wise maximum scores using an empirical quantile.

    Args:
        scores_per_episode: Per-episode score sequences.
        quantile: Desired upper quantile in (0, 1). For example, 0.95 returns the 95th
            percentile of the episode maxima.

    Returns:
        Sequence of per-step threshold values where every entry is the same constant threshold.
    """
    # Get the maximum uncertainty scores in each episode
    max_scores_per_episode = [np.nanmax(ep_scores) for ep_scores in scores_per_episode]

    # Global threshold based on episode maxima
    threshold = np.quantile(max_scores_per_episode, quantile)

    # Repeat threshold for each step in the longest episode
    max_ep_len = max(len(ep_scores) for ep_scores in scores_per_episode)

    return np.full(max_ep_len, threshold)

def compute_ct_mean_std_threshold(
    scores_per_episode: np.ndarray,
    quantile: float,
) -> np.ndarray:
    """
    Compute a constant threshold using mean + z * std of the maximum uncertainty score per episode.

    Args:
        scores_per_episode: Per-episode score sequences.
        quantile: One-sided upper quantile in (0, 1), determines the value of the z-score based on
            the standard normal percent point function.

    Returns:
        Sequence of per-step threshold values where every entry is the same constant threshold.
    """
    # Get the maximum uncertainty scores in each episode
    max_scores_per_episode = [np.nanmax(ep_scores) for ep_scores in scores_per_episode]

    # Global threshold based on episode maxima
    threshold = np.mean(max_scores_per_episode) + stats.norm.ppf(quantile) * np.std(max_scores_per_episode)

    # Repeat threshold for each step in the longest episode
    max_ep_len = max(len(ep_scores) for ep_scores in scores_per_episode)

    return np.full(max_ep_len, threshold)

def compute_tv_mean_std_threshold(
    scores_per_episode: np.ndarray,
    quantile: float,
) -> np.ndarray:
    """
    Compute a time-varying threshold curve using a mean + z * std rule where the z-score based on the 
    standard normal percent point function.

    Args:
        scores_per_episode: Per-episode score sequences.
        quantile: One-sided upper quantile in (0, 1), determines the value of the z-score based on the 
            standard normal percent point function.

    Returns:
        Sequence of per-step threshold values.
    """
    # Map step to array of scores at that step across all episodes
    scores_per_step: Dict[int, np.ndarray] = translate_scores(scores_per_episode)

    tv_threshold: List[np.ndarray] = []
    for step in scores_per_step:
        # Local per-step threshold: mean + z * std across episodes at this step
        local_threshold = np.mean(scores_per_step[step]) + stats.norm.ppf(quantile) * np.std(scores_per_step[step])
        tv_threshold.append(local_threshold)

    return np.array(tv_threshold)

def compute_tv_quantile_threshold(
    scores_per_episode: np.ndarray,
    quantile: float,
) -> np.ndarray:
    """
    Compute a time-varying threshold curve using empirical quantiles.

    Args:
        scores_per_episode: Per-episode score sequences.
        quantile: Desired quantile in (0, 1). For example, 0.95 corresponds to the
            95th percentile of the score distribution.

    Returns:
        Sequence of per-step threshold values.
    """
    # Map: step to array of scores at that step across all episodes
    scores_per_step: Dict[int, np.ndarray] = translate_scores(scores_per_episode)

    tv_threshold: List[np.ndarray] = []
    for step in scores_per_step:
        # Local per-step threshold (empirical quantile across episodes at this step)
        local_threshold = np.quantile(scores_per_step[step], quantile)
        tv_threshold.append(local_threshold)

    return np.array(tv_threshold)

def compute_cp_band_threshold(scores_per_episode: np.ndarray, quantile: float) -> np.ndarray:
    """
    Compute a conformal prediction band threshold from episode-wise uncertainty scores.

    The method follows the procedure described in:
      Chen Xu et al., "Can We Detect Failures Without Failure Data? 
      Uncertainty-Aware Runtime Failure Detection for Imitation Learning Policies"
      (https://arxiv.org/abs/2503.08558).

    Args:
        scores_per_episode: Per-episode score sequences.
        quantile: Desired upper quantile in (0, 1) for computing both gamma and h.

    Returns:
        Sequence of per-step threshold values.
    """
    if len(scores_per_episode) == 0:
        return np.array([])

    global_threshold = compute_ct_quantile_threshold(scores_per_episode, quantile)

    # Get longest episode and use it in set 1 (so that the mean trajectory is as long as possible)
    longest_index = max(range(len(scores_per_episode)), key=lambda i: len(scores_per_episode[i]))
    longest_episode = scores_per_episode.pop(longest_index)
    max_episode_length = len(longest_episode)

    # Split the uncertainty scores into two disjoint sets (N1 = N2 + 1)
    N1 = len(scores_per_episode) // 2
    N2 = len(scores_per_episode) - N1
    uncertainty_scores_1 = [longest_episode] + scores_per_episode[:N1]
    uncertainty_scores_2 = scores_per_episode[N1:]
    N1 += 1

    # Translate the scores for set 1
    scores_for_steps_1 = translate_scores(uncertainty_scores_1)

    # Compute the mean trajectory on set 1
    mean_trajectory_1 = [np.mean(scores_for_steps_1[t]) for t in scores_for_steps_1.keys()]

    # Determine set H
    if (N1 + 1) * quantile > N1:
        H = list(range(N1))
    else:
        # Determine gamma
        deviations = np.array(
            [
                np.nanmax(
                    [
                        abs(uncertainty_scores_1[m][t] - mean_trajectory_1[t])
                        for t in range(len(uncertainty_scores_1[m]))
                    ]
                )
                for m in range(N1)
            ]
        )
        gamma = np.quantile(deviations, quantile)
        # Determine set H
        H = np.where(deviations <= gamma)[0].tolist()

    # Compute the modulation function scalA(t)
    filtered_uncertainty_scores_1 = [uncertainty_scores_1[j] for j in H]
    filtered_scores_for_steps_1 = translate_scores(filtered_uncertainty_scores_1)
    scalA = []
    for t in filtered_scores_for_steps_1:
        scalA.append(max(abs(np.array(filtered_scores_for_steps_1[t]) - mean_trajectory_1[t])))

    # Fill remainder with the mean of scalA
    if len(scalA) == 0 or not np.isfinite(np.nanmean(scalA)):
        return np.full(max_episode_length, global_threshold)
    if len(scalA) < len(mean_trajectory_1):
        scalA.extend([np.nanmean(scalA)] * (len(mean_trajectory_1) - len(scalA)))
    scalA = [np.nanmean(scalA) if val < 1e-7 else val for val in scalA]

    # Compute the max deviations Dj for set 2
    deviations = []
    for episode in uncertainty_scores_2:
        episode_deviations = [
            (mean_trajectory_1[t] - episode[t]) / scalA[t]
            for t in range(min(len(episode), len(scalA), len(mean_trajectory_1)))
            if np.isfinite(scalA[t]) and abs(scalA[t]) >= 1e-12
        ]
        if episode_deviations:
            max_deviation = np.nanmax(episode_deviations)
            if np.isfinite(max_deviation):
                deviations.append(max_deviation)

    # Compute the band width h as the quantile of deviations
    if len(deviations) == 0:
        return np.full(max_episode_length, global_threshold)
    h = np.quantile(deviations, quantile)

    # Compute the time-varying thresholds as upper bounds
    threshold = [mean_trajectory_1[t] + h * scalA[t] for t in range(min(len(mean_trajectory_1), len(scalA)))]
    return np.array(threshold)

def normalize_scores_by_threshold(
    uncertainty_scores,
    thresholds,
    cfg,
):
    """Normalize the uncertainty scores by the thresholds.
    Args:
        uncertainty_scores: List of uncertainty scores for each episode.
        thresholds: Thresholds for normalization. Can be a list, numpy array, float, or int.
        extend_thresholds: How to extend the thresholds if they are shorter than the scores.
            "last": Extend the last threshold to match the length of the scores.
            "mean": Extend the mean of the thresholds to match the length of the scores.
    """
    info = dict(cfg.get("handle_zero_thresholds", {"style": "cond"}))
    max_episode_length = max([len(scores) for scores in uncertainty_scores])
    # convert constant thresholds to lists for uniformity
    if isinstance(thresholds, (float, int)):
        thresholds = [thresholds] * max_episode_length
    # convert lists to numpy arrays for uniformity
    if len(thresholds) < max_episode_length:
        if cfg.get("extend_thresholds", "mean") == "last":
            # repeat the last valid non-NaN value
            valid = ~np.isnan(thresholds)
            if valid.any():
                fill_val = thresholds[np.where(valid)[0][-1]]
            else:
                fill_val = np.nan
        elif cfg.get("extend_thresholds", "mean") == "mean":
            fill_val = np.nanmean(thresholds) if not np.all(np.isnan(thresholds)) else 0.0
        else:
            raise ValueError("extend_thresholds must be 'last' or 'mean'.")
    
        thresholds = np.concatenate(
            [thresholds, np.full(max_episode_length - len(thresholds), fill_val)]
        )

    if info["style"] == "cap_norm_scores":
        scores_by_threshold = normalize_scores_with_cap(
            uncertainty_scores, thresholds, info["cap_norm_scores"], info["clip_threshold"]
        )
    elif info["style"] == "clip_threshold":
        thresholds = np.clip(thresholds, info["clip_threshold"], None)
        scores_by_threshold = [
            np.array(scores) / thresholds[: len(scores)] if len(scores) > 0 else np.array(scores)
            for scores in uncertainty_scores
        ]
    elif info["style"] == "add_small_score":
        assert min(thresholds) > 0, "Thresholds must be greater than 0."
        scores_by_threshold = [
            np.array(scores) / thresholds[: len(scores)] if len(scores) > 0 else np.array(scores)
            for scores in uncertainty_scores
        ]
    elif info["style"] == "standard":
        scores_by_threshold = normalize_scores_standard(
            uncertainty_scores, thresholds
        )
    else:
        scores_by_threshold = normalize_scores_conditionally(
            uncertainty_scores, thresholds, info["small_threshold"], info["cap_norm_scores"]
        )

    return scores_by_threshold


def normalize_scores_standard(
    scores: np.ndarray, thresholds: np.ndarray
) -> list[np.ndarray]:
    """
    Threshold-centered, std-scaled normalization:
        norm_t = 1 + (score_t - threshold_t) / std_t
    where std_t is the per-timestep standard deviation computed across all episodes.
    """
    max_ep_len = max(len(ep_scores) for ep_scores in scores)

    # Per-timestep std across episodes (robust to variable episode lengths)
    score_stds = np.zeros(max_ep_len, dtype=float)
    for t in range(max_ep_len):
        scores_t = np.array([ep_scores[t] for ep_scores in scores if len(ep_scores) > t], dtype=float)
        score_stds[t] = np.std(scores_t)
    eps = 1e-6
    score_stds = np.maximum(score_stds, eps)

    thresholds = np.asarray(thresholds, dtype=float)

    normalized_scores = []
    for ep_scores in scores:
        ep_len = len(ep_scores)
        normalized_scores.append(1.0 + (ep_scores - thresholds[:ep_len]) / score_stds[:ep_len])
    return normalized_scores

def normalize_scores_with_cap(scores, thresholds, cap=5.0, clip_threshold=1e-6):
    """Normalize scores by thresholds with an upper cap on the normalized values."""
    thresholds = np.clip(thresholds, clip_threshold, None)
    # Normalize the scores by the thresholds
    scores_by_threshold = [np.array(s) / thresholds[: len(s)] if len(s) > 0 else np.array(s) for s in scores]
    # cap the normalized scores
    scores_by_threshold = [
        np.clip(scores, 0, cap) if len(scores) > 0 else np.array(scores) for scores in scores_by_threshold
    ]
    return scores_by_threshold


def normalize_scores_conditionally(scores, thresholds, small_threshold=1e-3, cap=3):
    """Normalize scores conditionally to avoid distortion from small thresholds."""
    normalized_scores = []
    for episode in scores:
        episode_norm = []
        for t in range(len(episode)):
            score = episode[t]
            threshold = thresholds[t]
            if threshold > score:
                new_score = score / threshold
            # Handle zero thresholds
            elif threshold <= 1e-8:
                new_score = 0.0 if score <= 1e-8 else cap
            # Handle small thresholds
            elif threshold <= small_threshold and score >= small_threshold * 5:
                new_score = cap
            elif score > 30 * threshold:
                new_score = cap * 2
            else:
                new_score = score / threshold
            episode_norm.append(new_score)
        normalized_scores.append(np.array(episode_norm))
    return normalized_scores


def _get_avg_episode_stats_by_type(scores_by_threshold: list, dataset_stats: dict, alt=True):
    """Calculate average episode statistics by type (ID/ood) and success/failure.
    Args:
        scores_by_threshold: List of normalized uncertainty scores for each episode.
        dataset_stats: Dictionary containing dataset statistics, including:
            - id_rollouts: List of booleans indicating ID rollouts.
            - ood_rollouts: List of booleans indicating OOD rollouts.
            - successful_rollouts: List of booleans indicating successful rollouts.
    Returns:
        avg_episode_stats_by_type: Dictionary containing average episode statistics by type.
    """
    if alt:
        return _get_avg_episode_stats_by_type_alt(scores_by_threshold, dataset_stats)
    id_rollouts = dataset_stats["id_rollouts"]
    ood_rollouts = dataset_stats["ood_rollouts"]
    successful_rollouts = dataset_stats["successful_rollouts"]
    avg_episode_stats_by_type = {}

    def mask_scores(scores, mask, norm_factor=1):
        """Helper function to mask scores based on a boolean mask."""
        if norm_factor == 0:
            norm_factor = 1
        return np.concatenate([score / norm_factor for score, m in zip(scores, mask) if m])

    # Step 1: Calculate mean scores
    if sum(id_rollouts) > 0:
        avg_episode_stats_by_type["id_success"] = {
            "mean": np.mean(mask_scores(scores_by_threshold, id_rollouts & successful_rollouts))
        }
        avg_episode_stats_by_type["id_failure"] = {
            "mean": np.mean(mask_scores(scores_by_threshold, id_rollouts & ~successful_rollouts))
        }

    if sum(ood_rollouts) > 0:
        avg_episode_stats_by_type["ood_success"] = {
            "mean": np.mean(mask_scores(scores_by_threshold, ood_rollouts & successful_rollouts))
        }
        avg_episode_stats_by_type["ood_failure"] = {
            "mean": np.mean(mask_scores(scores_by_threshold, ood_rollouts & ~successful_rollouts))
        }

    # Step 2: Normalize mean scores by the maximum mean score
    max_mean_score = max(
        [
            avg_episode_stats_by_type[key]["mean"]
            for key in avg_episode_stats_by_type.keys()
            if avg_episode_stats_by_type[key]["mean"] > 0
        ]
    )
    if max_mean_score > 0:
        for key in avg_episode_stats_by_type.keys():
            avg_episode_stats_by_type[key]["mean"] /= max_mean_score

    # # Step 3: Calculate uncertainty values (std and percentiles) based on normalized scores
    # def calculate_uncertainty(scores: np.ndarray):
    #     """Helper function to calculate std and percentiles."""
    #     return {
    #         "std": np.std(scores),
    #         "percentiles": {
    #             10: np.percentile(scores, 10),
    #             25: np.percentile(scores, 25),
    #             50: np.percentile(scores, 50),  # Median
    #             75: np.percentile(scores, 75),
    #             90: np.percentile(scores, 90),
    #         },
    #     }

    # if sum(id_rollouts) > 0:
    #     avg_episode_stats_by_type["id_success"].update(
    #         calculate_uncertainty(mask_scores(scores_by_threshold, id_rollouts & successful_rollouts, max_mean_score))
    #     )
    #     avg_episode_stats_by_type["id_failure"].update(
    #         calculate_uncertainty(mask_scores(scores_by_threshold, id_rollouts & ~successful_rollouts, max_mean_score))
    #     )

    # if sum(ood_rollouts) > 0:
    #     avg_episode_stats_by_type["ood_success"].update(
    #         calculate_uncertainty(mask_scores(scores_by_threshold, ood_rollouts & successful_rollouts, max_mean_score))
    #     )
    #     avg_episode_stats_by_type["ood_failure"].update(
    #         calculate_uncertainty(mask_scores(scores_by_threshold, ood_rollouts & ~successful_rollouts, max_mean_score))
    #     )

    return avg_episode_stats_by_type


# def _get_avg_episode_stats_by_type_alt(scores_by_threshold: list, dataset_stats: dict):
#     """Calculate average episode statistics by type (ID/ood) and success/failure.
#     Args:
#         scores_by_threshold: List of normalized uncertainty scores for each episode.
#         dataset_stats: Dictionary containing dataset statistics, including:
#             - id_rollouts: List of booleans indicating ID rollouts.
#             - ood_rollouts: List of booleans indicating OOD rollouts.
#             - successful_rollouts: List of booleans indicating successful rollouts.
#     Returns:
#         avg_episode_stats_by_type: Dictionary containing average episode statistics by type.
#     """
#     id_rollouts = dataset_stats["id_rollouts"]
#     ood_rollouts = dataset_stats["ood_rollouts"]
#     successful_rollouts = dataset_stats["successful_rollouts"]
#     avg_episode_stats_by_type = {}

#     def calculate_stats(scores):
#         """Helper function to calculate mean, std, and percentiles."""
#         scores_array = np.concatenate(scores)
#         return {
#             "mean": np.mean(scores_array),
#             "std": np.std(scores_array),
#             "percentiles": {
#                 10: np.percentile(scores_array, 10),
#                 25: np.percentile(scores_array, 25),
#                 50: np.percentile(scores_array, 50),  # Median
#                 75: np.percentile(scores_array, 75),
#                 90: np.percentile(scores_array, 90),
#             },
#         }

#     if sum(id_rollouts) > 0:
#         id_success_scores = [
#             score for score, mask in zip(scores_by_threshold, id_rollouts & successful_rollouts) if mask
#         ]
#         id_failure_scores = [
#             score for score, mask in zip(scores_by_threshold, id_rollouts & ~successful_rollouts) if mask
#         ]
#         avg_episode_stats_by_type["id_success"] = calculate_stats(id_success_scores)
#         avg_episode_stats_by_type["id_failure"] = calculate_stats(id_failure_scores)

#     if sum(ood_rollouts) > 0:
#         ood_success_scores = [
#             score for score, mask in zip(scores_by_threshold, ood_rollouts & successful_rollouts) if mask
#         ]
#         ood_failure_scores = [
#             score for score, mask in zip(scores_by_threshold, ood_rollouts & ~successful_rollouts) if mask
#         ]
#         avg_episode_stats_by_type["ood_success"] = calculate_stats(ood_success_scores)
#         avg_episode_stats_by_type["ood_failure"] = calculate_stats(ood_failure_scores)

#     # Normalize the stats by the maximum mean entry
#     max_mean_score = max(
#         [
#             avg_episode_stats_by_type[key]["mean"]
#             for key in avg_episode_stats_by_type.keys()
#             if avg_episode_stats_by_type[key]["mean"] > 0
#         ]
#     )
#     if max_mean_score > 0:
#         for key in avg_episode_stats_by_type.keys():
#             avg_episode_stats_by_type[key]["mean"] /= max_mean_score
#             avg_episode_stats_by_type[key]["std"] /= max_mean_score
#             avg_episode_stats_by_type[key]["percentiles"] = {
#                 k: v / max_mean_score for k, v in avg_episode_stats_by_type[key]["percentiles"].items()
#             }

#     return avg_episode_stats_by_type


def _get_avg_episode_stats_by_type_alt(scores_by_threshold: list, dataset_stats: dict):
    """Calculate average episode statistics by type (ID/ood) and success/failure.

    Args:
        scores_by_threshold: List of normalized uncertainty scores for each episode.
        dataset_stats: Dictionary containing dataset statistics, including:
            - id_rollouts: List of booleans indicating ID rollouts.
            - ood_rollouts: List of booleans indicating OOD rollouts.
            - successful_rollouts: List of booleans indicating successful rollouts.

    Returns:
        avg_episode_stats_by_type: Dictionary containing average episode statistics by type.
    """
    id_rollouts = dataset_stats["id_rollouts"]
    ood_rollouts = dataset_stats["ood_rollouts"]
    successful_rollouts = dataset_stats["successful_rollouts"]
    avg_episode_stats_by_type = {}

    def calculate_stats(scores):
        """Helper function to calculate mean, std, and percentiles."""
        scores = [score for score in scores if len(score) > 0]
        if len(scores) == 0:
            return {
                "mean": 0,
                "std": 0,
                "percentiles": {10: 0, 25: 0, 50: 0, 75: 0, 90: 0},
            }
        scores_array = np.concatenate(scores)
        return {
            "mean": np.mean(scores_array),
            "std": np.std(scores_array),
            "percentiles": {
                10: np.percentile(scores_array, 10),
                25: np.percentile(scores_array, 25),
                50: np.percentile(scores_array, 50),  # Median
                75: np.percentile(scores_array, 75),
                90: np.percentile(scores_array, 90),
            },
        }

    def extract_scores(scores_by_threshold, rollouts_mask):
        """Helper function to extract scores for a given rollout type."""
        return [score for score, mask in zip(scores_by_threshold, rollouts_mask) if mask]

    # Calculate statistics for each rollout type
    if sum(id_rollouts) > 0:
        avg_episode_stats_by_type["id_success"] = calculate_stats(
            extract_scores(scores_by_threshold, id_rollouts & successful_rollouts)
        )
        avg_episode_stats_by_type["id_failure"] = calculate_stats(
            extract_scores(scores_by_threshold, id_rollouts & ~successful_rollouts)
        )

    if sum(ood_rollouts) > 0:
        avg_episode_stats_by_type["ood_success"] = calculate_stats(
            extract_scores(scores_by_threshold, ood_rollouts & successful_rollouts)
        )
        avg_episode_stats_by_type["ood_failure"] = calculate_stats(
            extract_scores(scores_by_threshold, ood_rollouts & ~successful_rollouts)
        )

    # Normalize mean and std by the maximum mean score
    max_mean_score = max(
        [
            avg_episode_stats_by_type[key]["mean"]
            for key in avg_episode_stats_by_type.keys()
            if avg_episode_stats_by_type[key]["mean"] > 0
        ],
        default=0,
    )
    if max_mean_score > 0:
        for key in avg_episode_stats_by_type.keys():
            # Normalize mean and std
            avg_episode_stats_by_type[key]["mean"] /= max_mean_score
            avg_episode_stats_by_type[key]["std"] /= max_mean_score

            # Recalculate percentiles on normalized scores
            if key == "id_success":
                normalized_scores = [
                    np.array(score) / max_mean_score
                    for score in extract_scores(scores_by_threshold, id_rollouts & successful_rollouts)
                ]
            elif key == "id_failure":
                normalized_scores = [
                    np.array(score) / max_mean_score
                    for score in extract_scores(scores_by_threshold, id_rollouts & ~successful_rollouts)
                ]
            elif key == "ood_success":
                normalized_scores = [
                    np.array(score) / max_mean_score
                    for score in extract_scores(scores_by_threshold, ood_rollouts & successful_rollouts)
                ]
            elif key == "ood_failure":
                normalized_scores = [
                    np.array(score) / max_mean_score
                    for score in extract_scores(scores_by_threshold, ood_rollouts & ~successful_rollouts)
                ]
            else:
                continue

            normalized_scores = [score for score in normalized_scores if len(score) > 0]
            if len(normalized_scores) == 0:
                continue
            scores_array = np.concatenate(normalized_scores)
            avg_episode_stats_by_type[key]["percentiles"] = {
                10: np.percentile(scores_array, 10),
                25: np.percentile(scores_array, 25),
                50: np.percentile(scores_array, 50),  # Median
                75: np.percentile(scores_array, 75),
                90: np.percentile(scores_array, 90),
            }

    return avg_episode_stats_by_type

def _compute_detection_flags_and_times(
    scores_by_threshold: List[np.ndarray],
    successful_rollouts: Union[List[bool], np.ndarray],
    max_episode_length: int,
    detection_patience: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      detected_failure_in_episode: (N,) bool array
      detection_times: (K,) float array of normalized detection times for correctly detected failures
    """
    successful_rollouts = np.asarray(successful_rollouts, dtype=bool)
    detection_patience = max(0, int(detection_patience))

    denom = max(1, int(max_episode_length) - 1)

    detected_failure_in_episode: List[bool] = []
    detection_times: List[float] = []

    for scores, success in zip(scores_by_threshold, successful_rollouts):
        # Check whether a failure was detected
        scores_above_threshold = np.asarray(scores) > 1
        detected_failure_in_episode.append(np.sum(scores_above_threshold) > detection_patience)

        # Append detection times if a failure was correctly detected
        if detected_failure_in_episode[-1] and not success:
            detection_time = np.where(scores_above_threshold)[0][detection_patience] / denom
            detection_times.append(float(detection_time))

    return np.asarray(detected_failure_in_episode, dtype=bool), np.asarray(detection_times, dtype=float)

def _calculate_scalar_metrics_for_subset(
    scores_by_threshold: list,
    successful_rollouts: np.ndarray,
    max_episode_length: int, 
    detection_patience = 0
) -> dict[str, float]:
    """Calculate the evaluation metrics for a list of episode infos.

    Args:
        scores_by_threshold: List of lists of uncertainty scores for each episode normalized by thresholds.
        successful_rollouts: List of booleans indicating whether the episode was successful or not.
        dataset_stats: Dictionary containing the dataset statistics including:
            - max_episode_length: Maximum episode length.
            - id_rollouts: List of booleans indicating whether the episode is an ID test rollout.
            - ood_rollouts: List of booleans indicating whether the episode is an OOD test rollout.
            - successful_rollouts: List of booleans indicating whether the episode was successful.
    Returns:
        metrics: Dictionary containing the evaluation metrics.
    """
    detected_failure_in_episode, detection_times = _compute_detection_flags_and_times(
        scores_by_threshold=scores_by_threshold,
        successful_rollouts=successful_rollouts,
        max_episode_length=max_episode_length,
        detection_patience=detection_patience,
    )

    # Calculate the number of true positives, true negatives, false positives, and false negatives
    TP, TN, FP, FN = _calculate_confusion_matrix(detected_failure_in_episode, successful_rollouts)
    # calculate metrics
    TPR, TNR, accuracy, balanced_accuracy = _calculate_accuracy(TP, TN, FP, FN)
    avg_detection_time = np.mean(detection_times) if len(detection_times) > 0 else 1.0
    std_detection_time = np.std(detection_times) if len(detection_times) > 0 else 0.0

    timestep_wise_accuracy = _calculate_twa(detected_failure_in_episode, successful_rollouts, detection_times)

    return {
        "TPR": float(TPR),
        "TNR": float(TNR),
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "avg_detection_time": avg_detection_time,
        "std_detection_time": std_detection_time,
        "timestep_wise_accuracy": timestep_wise_accuracy,
        # Optional but often useful for debugging/reporting:
        "TP": float(TP),
        "TN": float(TN),
        "FP": float(FP),
        "FN": float(FN),
        "num_episodes": float(successful_rollouts.size),
    }

def _subset_inputs_by_mask(
    scores_by_threshold: List[np.ndarray],
    dataset_stats: dict,
    mask: np.ndarray,
) -> tuple[List[np.ndarray], dict]:
    """Filter scores and dataset_stats arrays by a boolean mask."""

    subset_scores = [s for s, m in zip(scores_by_threshold, mask) if m]

    # Filter any per-episode boolean arrays in dataset_stats
    subset_stats = dict(dataset_stats)
    for k in ["id_rollouts", "ood_rollouts", "successful_rollouts"]:
        if k in subset_stats:
            subset_stats[k] = np.asarray(subset_stats[k], dtype=bool)[mask]

    # Keep max_episode_length as-is (global normalization), unless you want subset-based normalization
    subset_stats["max_episode_length"] = dataset_stats["max_episode_length"]
    return subset_scores, subset_stats

def calculate_metrics(scores_by_threshold: list, dataset_stats: dict, detection_patience: int = 0) -> dict[str, float]:
    max_episode_length = dataset_stats["max_episode_length"]
    successful_rollouts = np.asarray(dataset_stats["successful_rollouts"], dtype=bool)
    detection_patience = max(0, int(detection_patience))

    detected_failure_in_episode, detection_times = _compute_detection_flags_and_times(
        scores_by_threshold=scores_by_threshold,
        successful_rollouts=successful_rollouts,
        max_episode_length=max_episode_length,
        detection_patience=detection_patience,
    )

    TP, TN, FP, FN = _calculate_confusion_matrix(detected_failure_in_episode, successful_rollouts)
    TPR, TNR, accuracy, balanced_accuracy = _calculate_accuracy(TP, TN, FP, FN)

    avg_detection_time = np.mean(detection_times) if len(detection_times) > 0 else 1.0
    std_detection_time = np.std(detection_times) if len(detection_times) > 0 else 0.0

    avg_episode_stats_by_type = _get_avg_episode_stats_by_type(scores_by_threshold, dataset_stats)
    timestep_wise_accuracy = _calculate_twa(detected_failure_in_episode, successful_rollouts, detection_times)

    metrics = {
        "TPR": float(TPR),
        "TNR": float(TNR),
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "avg_detection_time": float(avg_detection_time),
        "std_detection_time": float(std_detection_time),
        "timestep_wise_accuracy": float(timestep_wise_accuracy),
        "avg_episode_stats_by_type": avg_episode_stats_by_type,
    }

    id_mask = np.asarray(dataset_stats.get("id_rollouts", []), dtype=bool)
    ood_mask = np.asarray(dataset_stats.get("ood_rollouts", []), dtype=bool)

    metrics_by_type = {}

    if id_mask.size == len(scores_by_threshold) and id_mask.any():
        id_scores, id_stats = _subset_inputs_by_mask(scores_by_threshold, dataset_stats, id_mask)
        metrics_by_type["id"] = _calculate_scalar_metrics_for_subset(
            scores_by_threshold=id_scores,
            successful_rollouts=id_stats["successful_rollouts"],
            max_episode_length=id_stats["max_episode_length"],
            detection_patience=detection_patience,
        )
    else:
        metrics_by_type["id"] = None

    if ood_mask.size == len(scores_by_threshold) and ood_mask.any():
        ood_scores, ood_stats = _subset_inputs_by_mask(scores_by_threshold, dataset_stats, ood_mask)
        metrics_by_type["ood"] = _calculate_scalar_metrics_for_subset(
            scores_by_threshold=ood_scores,
            successful_rollouts=ood_stats["successful_rollouts"],
            max_episode_length=ood_stats["max_episode_length"],
            detection_patience=detection_patience,
        )
    else:
        metrics_by_type["ood"] = None

    metrics["metrics_by_type"] = metrics_by_type
    return metrics
