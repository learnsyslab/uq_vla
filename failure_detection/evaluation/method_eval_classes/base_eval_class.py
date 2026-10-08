import os
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union
import time
import numpy as np
from omegaconf import DictConfig
from sklearn.metrics import roc_auc_score
from evaluation.utils import (
    calculate_metrics,
    compute_thresholds,
    normalize_scores_by_threshold,
)
from rollout_data.rollout_datasets import ProcessedRolloutDataset
from shared_utils.calibration_split import (
    get_calibration_split_indices,
    splits_calibration_set,
)


class BaseEvalClass(ABC):
    """
    Abstract base class that provides the common functionality of the evaluation methods.
    It handles the treshold calculation via calibration rollouts, method evaluation via test rollouts, and metrics calculation.
    Subclasses (one for each method) must only implement the following methods:
    - calculate_uncertainty_score
    When loading a model, the subclass must also implement the load_model method.
    When preprocessing based on calibration and/or test rollouts is required, the subclass must implement the _execute_preprocessing method.
    A subclass that fits a model on the calibration rollouts must set
    trains_on_calibration_data = True, which makes the base class hold the training
    episodes back from threshold calibration (see shared_utils.calibration_split).
    """

    # Whether this method fits a model on the calibration rollouts. Detectors that do
    # (RND-*, logpZO) get the episode-level calibration split applied; scorers that do
    # not (VFD, ACE, STAC) calibrate on all calibration episodes.
    trains_on_calibration_data = False

    def __init__(
        self,
        cfg: DictConfig,
        method_name: str,
        device: str,
        task_data_path: Union[str, Path],
        dataset: ProcessedRolloutDataset,
    ):
        """
        Initialize the base evaluation class.

        Args:
            method_name (str): Name of the evaluation method.
            cfg (dict): Configuration dictionary or object.
            policy (BaseImagePolicy): Trained diffusion policy.
            device (str): Device to use for evaluation.
            task_data_path (str): Base path to the data "data_all/{task}.
        """
        self.method_name = method_name
        self.cfg = cfg
        self.device = device
        self.task_data_path = task_data_path
        self.dataset = dataset
        self.required_tensors = list(cfg.get("required_tensors", []))
        self.optional_tensors = list(cfg.get("optional_tensors", []))
        self.required_actions = list(cfg.get("required_actions", []))
        self.optional_actions = list(cfg.get("optional_actions", []))
        self.normalize_tensors = dict(cfg.get("normalize_tensors", {}))
        # Create the results directory
        self.results_dir = os.path.join(task_data_path, "results", method_name)
        os.makedirs(self.results_dir, exist_ok=True)
        # Load the model (must be implemented in the subclass, not all methods require a model)
        self.load_model()

    @abstractmethod
    def calculate_uncertainty_score(self, rollout_tensor_dict: dict, **kwargs) -> float:
        """
        Calculate the uncertainty score for a rollout step and the inference time in doing so. Must be implemented in the subclass.

        Args:
            rollout_step: Dict that includes the requested tensors for one step of the rollout.

        Returns:
            uncertainty_score : Uncertainty score for one step.
        """
        pass

    def load_model(self):
        """
        Loads models, modules that are required for the evaluation. Optional, not all methods require a model.

        Returns:
            Nothing. Main model is stored in self.model, other models as other class attributes.
        """
        self.model = None

    def _execute_preprocessing(self):
        """Placeholder for preprocessing steps that are required before the evaluation.
        Should be implemented in the subclass if needed. The method can use the dataset class to access the data."""
        pass

    def _compound_uncertainty_scores(self, score_list: list) -> float:
        """Compunds multiple uncertainty scores if a rollout is recorded with multiple robots.

        Args:
            score_list: List of uncertainty scores. For each robot/arm, there is one score.
        Returns:
            uncertainty_score: Compounded uncertainty score.
        """
        assert len(score_list) >= 1
        if len(score_list) == 1:
            return score_list[-1]

        if self.cfg.score_compounding_method == "mean":
            return np.mean(score_list)
        if self.cfg.score_compounding_method == "max":
            return np.max(score_list)
        if self.cfg.score_compounding_method == "mult":
            return np.prod(score_list)
        # Default
        return np.max(score_list)

    def _get_thresholds(self, uncertainty_scores_by_window_size: dict) -> tuple[dict, dict]:
        """Get the thresholds for the quantiles and window sizes defined in self.cfg.

        Args:
            uncertainty_scores_by_window_size: Dictionary containing the uncertainty scores and success labels for each window size.

        Returns:
            (thresholds, time_varying_thresholds, scores_by_threshold, scores_by_threshold_tvt):
                - Dictionary containing the constant thresholds for each quantile and window size.
                - Dictionary containing the lists of time-varying thresholds for each quantile and window size.
                - Dictionary containing the normalized uncertainty scores for each quantile and window size.
                - Dictionary containing the normalized uncertainty scores for each quantile and window size using time-varying thresholds.
        """
        # Set up the dictionaries to store the thresholds and scores
        thresholds = {}
        scores_by_threshold = {}
        for threshold_style in self.cfg.thresholds:
            if self.cfg.thresholds[threshold_style]:
                thresholds[threshold_style] = {}
                scores_by_threshold[threshold_style] = {}

        for threshold_style in thresholds.keys():
            for quantile in self.cfg.quantiles:
                thresholds[threshold_style][quantile] = {}
                scores_by_threshold[threshold_style][quantile] = {}
                for window_size in self.cfg.window_sizes:
                    # Prefer successful calibration episodes, but allow tiny smoke
                    # datasets to proceed when none of the calibration rollouts
                    # succeed for a task.
                    successful_uncertainty_scores = [
                        episode_info["uncertainty_scores"]
                        for episode_info in uncertainty_scores_by_window_size[window_size]
                        if episode_info["successful"]
                    ]
                    uncertainty_scores = successful_uncertainty_scores or [
                        episode_info["uncertainty_scores"]
                        for episode_info in uncertainty_scores_by_window_size[window_size]
                    ]
                    thresholds[threshold_style][quantile][window_size] = compute_thresholds(
                        scores_per_episode=uncertainty_scores,
                        threshold_strategy=threshold_style,
                        quantile=quantile,
                        window_size=window_size,
                    )

                    # Calculate the scores by threshold (also for failed episodes)
                    uncertainty_scores = [
                        episode_info["uncertainty_scores"]
                        for episode_info in uncertainty_scores_by_window_size[window_size]
                    ]
                    scores_by_threshold[threshold_style][quantile][window_size] = normalize_scores_by_threshold(
                        uncertainty_scores,
                        thresholds[threshold_style][quantile][window_size],
                        cfg=self.cfg,
                    )

        return thresholds, scores_by_threshold

    def _get_metrics(self, thresholds: dict, test_scores: dict) -> tuple[dict, dict]:
        """Evaluate the test results using the constant and time-varying tresholds in the calibration results."""
        # Obtain some dataset statistics
        dataset_stats = {
            "max_episode_length": max(self.dataset.data["metadata"]["episode_lengths"]),
            "id_rollouts": self.dataset.get_rollout_subtypes(subset="test", subsubset="id"),
            "ood_rollouts": self.dataset.get_rollout_subtypes(subset="test", subsubset="ood"),
            "successful_rollouts": self.dataset.get_rollout_types(
                subset=["test", "successful"], reduce_mask_to=["test"]
            ),
        }
        scores_by_threshold_test = {}
        metrics = {}
        for threshold_style in thresholds.keys():
            scores_by_threshold_test[threshold_style] = {}
            metrics[threshold_style] = {}
            for quantile in self.cfg.quantiles:
                scores_by_threshold_test[threshold_style][quantile] = {}
                metrics[threshold_style][quantile] = {}
                for window_size in self.cfg.window_sizes:
                    # Extract the uncertainty scores from the test scores
                    uncertainty_scores = [
                        episode_info["uncertainty_scores"] for episode_info in test_scores[window_size]
                    ]
                    # List of successful episodes
                    # successful_rollouts = [episode_info["successful"] for episode_info in test_scores[window_size]]
                    # Normalize the test scores by the thresholds
                    scores_by_threshold_test[threshold_style][quantile][window_size] = normalize_scores_by_threshold(
                        uncertainty_scores,
                        thresholds[threshold_style][quantile][window_size],
                        cfg=self.cfg,
                    )
                    # Calculate the metrics using the normalized scores and the successful rollouts
                    metrics[threshold_style][quantile][window_size] = calculate_metrics(
                        scores_by_threshold_test[threshold_style][quantile][window_size],
                        dataset_stats, self.cfg.detection_patience,
                    )

        return metrics, scores_by_threshold_test

    def evaluate(self) -> dict:
        """Evaluate a method."""

        # Full freedom to use the dataset class
        self._execute_preprocessing()

        # Process the calibration rollouts
        uncertainty_scores_by_window_size_calibration, avg_inference_time_calibration = self._process_rollouts(
            subset="calibration"
        )
        # Thresholds are dicts with structure: thresholds[threshold_style][quantile][window_size] = threshold, where threshold is a float or a list of floats (for time-varying thresholds)
        thresholds, scores_by_threshold_calibration = self._get_thresholds(
            uncertainty_scores_by_window_size_calibration
        )
        # Process the test rollouts
        uncertainty_scores_by_window_size_test, avg_inference_time_test = self._process_rollouts(subset="test")

        # Calculate the metrics for the test rollouts
        metrics, scores_by_threshold_test = self._get_metrics(thresholds, uncertainty_scores_by_window_size_test)

        # Threshold-free AUROC: per task, rank test rollouts by
        # one scalar per rollout -- the max score over the rollout ("auroc") and the
        # mean score ("auroc_mean") -- and separate failed (positive) from successful
        # (negative) rollouts. Macro-averaged across tasks at reporting time.
        self._inject_auroc(metrics, self._compute_auroc(uncertainty_scores_by_window_size_test))

        eval_results = {
            "method": self.method_name,
            "quantiles": self.cfg.quantiles,
            "window_sizes": self.cfg.window_sizes,
            "calibration_uncertainty_scores": uncertainty_scores_by_window_size_calibration,
            "calibration_thresholds": thresholds,
            "calibration_scores_by_threshold": scores_by_threshold_calibration,
            "test_uncertainty_scores": uncertainty_scores_by_window_size_test,
            "test_metrics": metrics,
            "test_scores_by_threshold": scores_by_threshold_test,
            "avg_inference_time": np.mean([avg_inference_time_test, avg_inference_time_calibration]),
            "cfg": self.cfg,
            "max_episode_length": max(self.dataset.data["metadata"]["episode_lengths"]),
            "successful_test_rollouts": self.dataset.get_rollout_types(
                subset=["test", "successful"], reduce_mask_to=["test"]
            ),
            "id_test_rollouts": self.dataset.get_rollout_subtypes(subset="test", subsubset="id"),
            "ood_test_rollouts": self.dataset.get_rollout_subtypes(subset="test", subsubset="ood"),
        }
        self._save_pickle(self.results_dir, eval_results, "eval_results.pkl")
        return eval_results

    #: How a rollout's per-step scores are reduced to the one scalar AUROC ranks.
    #: "max" is the peak score (the value FIPER's alarm rule effectively looks at);
    #: "mean" is the average level over the rollout. They can differ a lot for spiky
    #: scores, e.g. a detector that peaks near the end of *successful* episodes.
    AUROC_AGGREGATIONS = {"max": np.max, "mean": np.mean}

    #: Prefixes of each rollout, as fractions of its own length, for the horizon AUROC:
    #: one AUROC per prefix from the max score seen *so far*, then averaged over the
    #: prefixes ("horizon_mean"). It rewards detectors that already rank episodes
    #: correctly early on, where a late-firing alarm is worth little. The 1.0 prefix is
    #: the whole rollout, so its value equals the plain max-AUROC by construction.
    AUROC_HORIZONS = (0.25, 0.5, 0.75, 1.0)

    def _compute_auroc(self, test_scores_by_window_size: dict) -> dict:
        """Per-task, threshold-free AUROC for each window size and episode aggregation.

        For every window size, each test rollout is reduced to one scalar per
        aggregation in :attr:`AUROC_AGGREGATIONS` (NaN steps ignored): "max" gives
        s_bar_T = max_t s_t, "mean" the average score over the rollout. The AUROC then
        measures how well that scalar separates failed rollouts (positive class,
        label 1) from successful ones (negative class, label 0).

        A third value, "horizon_mean", averages the AUROCs obtained from the max score
        over the first 25/50/75/100 % of each rollout's steps (:attr:`AUROC_HORIZONS`);
        the individual values are returned too, as "max@25%" ... "max@100%".

        Returns ``{window_size: {aggregation: auroc}}``, with NaN for a window whose
        test set has only one class present (AUROC is undefined); such tasks are
        skipped when macro-averaging across tasks.
        """
        horizon_keys = [f"max@{int(round(h * 100))}%" for h in self.AUROC_HORIZONS]
        subtype_keys = ["max_id", "max_ood"]
        all_keys = list(self.AUROC_AGGREGATIONS) + ["horizon_mean"] + horizon_keys + subtype_keys
        auroc_by_window = {}
        for window_size, episodes in test_scores_by_window_size.items():
            labels = []
            per_episode = {name: [] for name in list(self.AUROC_AGGREGATIONS) + horizon_keys}
            # Peak score and label per subtype, for the id-only / ood-only AUROCs.
            by_subtype: dict[str, tuple[list, list]] = {"id": ([], []), "ood": ([], [])}
            for episode_info in episodes:
                scores = np.asarray(episode_info["uncertainty_scores"], dtype=np.float64)
                scores = scores[~np.isnan(scores)]
                if scores.size == 0:
                    continue
                labels.append(0 if episode_info["successful"] else 1)
                for name, reduce_fn in self.AUROC_AGGREGATIONS.items():
                    per_episode[name].append(float(reduce_fn(scores)))
                for horizon, key in zip(self.AUROC_HORIZONS, horizon_keys):
                    # Prefix of this rollout, at least one step; 1.0 -> the whole rollout.
                    n_steps = max(1, int(np.ceil(horizon * scores.size)))
                    per_episode[key].append(float(np.max(scores[:n_steps])))
                subtype = episode_info.get("rollout_subtype")
                if subtype in by_subtype:
                    by_subtype[subtype][0].append(0 if episode_info["successful"] else 1)
                    by_subtype[subtype][1].append(float(np.max(scores)))
            labels = np.asarray(labels, dtype=int)
            if labels.size == 0 or np.unique(labels).size < 2:
                auroc_by_window[window_size] = {name: float("nan") for name in all_keys}
                continue
            per_agg = {
                name: float(roc_auc_score(labels, np.asarray(values)))
                for name, values in per_episode.items()
            }
            per_agg["horizon_mean"] = float(np.mean([per_agg[k] for k in horizon_keys]))
            for subtype, (sub_labels, sub_scores) in by_subtype.items():
                key = f"max_{subtype}"
                per_agg[key] = (float(roc_auc_score(sub_labels, np.asarray(sub_scores)))
                                if len(set(sub_labels)) > 1 else float("nan"))
            auroc_by_window[window_size] = per_agg
        return auroc_by_window

    @staticmethod
    def _inject_auroc(metrics: dict, auroc_by_window: dict) -> None:
        """Write the AUROC values into every metrics entry of their window size.

        AUROC depends only on the window size, not on the threshold style or quantile,
        so the same value goes into every entry; it then flows through results_manager
        as an ordinary per-(task, window) column. Keys: "auroc" (max aggregation, the
        historical name), "auroc_mean", "auroc_horizon_mean" and the per-horizon values
        under "auroc_max@25%" ... "auroc_max@100%", plus "auroc_id" / "auroc_ood" (max aggregation,
        restricted to the in- / out-of-distribution test rollouts).
        """
        for threshold_style in metrics:
            for quantile in metrics[threshold_style]:
                for window_size in metrics[threshold_style][quantile]:
                    per_agg = auroc_by_window.get(window_size, {})
                    entry = metrics[threshold_style][quantile][window_size]
                    entry["auroc"] = per_agg.get("max", float("nan"))
                    entry["auroc_mean"] = per_agg.get("mean", float("nan"))
                    entry["auroc_horizon_mean"] = per_agg.get("horizon_mean", float("nan"))
                    entry["auroc_id"] = per_agg.get("max_id", float("nan"))
                    entry["auroc_ood"] = per_agg.get("max_ood", float("nan"))
                    for key, value in per_agg.items():
                        if key.startswith("max@"):
                            entry[f"auroc_{key}"] = value

    def _process_one_rollout(self, rollout_dict: dict) -> tuple[list[float], float]:
        """
        Obtain the uncertainty scores and average inference time for one rollout.

        Args:
            rollout_dict: One rollout that is a dict with the required tensors and one success label as entries.

        Returns:
            uncertainty_scores: List of uncertainty scores for each step in a rollout.
            avg_inference_time: Average inference time.
        """
        # rollout_dict contains the success labels and the tensors for the episode
        rollout_length = rollout_dict[self.required_tensors[0]].shape[0]
        num_robots = self.dataset.data["metadata"].get("num_robots", 1)
        inference_times = []
        uncertainty_scores_one_rollout = []
        for i in range(rollout_length):
            new_dict = {}
            for key in rollout_dict.keys():
                if key == "successful":
                    continue
                else:
                    new_dict[key] = rollout_dict[key][i]
            start_time = time.time()
            uncertainty_score = self.calculate_uncertainty_score(rollout_tensor_dict=new_dict)
            inference_times.append(time.time() - start_time)
            if self.cfg.get("handle_zero_thresholds", {"style": "whatever"})["style"] == "add_small_score":
                # Add a small value to avoid division by zero
                uncertainty_score = uncertainty_score + 1e-6
            uncertainty_scores_one_rollout.append(uncertainty_score)

        # If the dataset contains multiple robots, compound the uncertainty scores
        if num_robots > 1:
            # Split the uncertainty scores into sublists, each sublist corresponds to one step
            sublists = [
                uncertainty_scores_one_rollout[i : i + num_robots]
                for i in range(0, len(uncertainty_scores_one_rollout), num_robots)
            ]
            uncertainty_scores_one_rollout = []
            for sublist in sublists:
                # Compound the uncertainty scores for each robot
                compounded_score = self._compound_uncertainty_scores(sublist)
                uncertainty_scores_one_rollout.append(compounded_score)
        assert len(uncertainty_scores_one_rollout) == rollout_length // num_robots
        avg_inference_time = np.mean(inference_times)
        return uncertainty_scores_one_rollout, avg_inference_time

    def _should_process_rollout(self, subset: str, episode_idx: int, rollout_dict: dict) -> bool:
        """Whether an episode contributes to this method's scores.

        For methods that train on the calibration rollouts, the calibration episodes
        used for fitting are excluded here so that thresholds are computed only on the
        held-out half. Test episodes are always processed.
        """
        if subset != "calibration" or not self.trains_on_calibration_data:
            return True
        if not splits_calibration_set(self.cfg):
            return True

        _, threshold_indices = get_calibration_split_indices(self.dataset, self.cfg)
        return episode_idx in threshold_indices

    def _calibration_train_episode_indices(self) -> set[int]:
        """Calibration episode indices this method may fit on (all of them if unsplit)."""
        train_indices, _ = get_calibration_split_indices(self.dataset, self.cfg)
        return train_indices

    def _subset_subtypes(self, subset: str) -> list[str | None] | None:
        """The id/ood label of each episode of `subset`, in the order iterate_episodes yields them.

        The dataset metadata carries `id_rollout_labels` / `ood_rollout_labels` over all rollouts
        (calibration first, then test, see TaskManager._convert_raw_rollouts), so the subset's own
        labels are those arrays masked by `<subset>_rollout_labels`. Returns None when the labels are
        missing or inconsistent, in which case the per-subtype AUROCs are simply not computed.
        """
        metadata = self.dataset.get_metadata()
        id_labels = np.asarray(metadata.get("id_rollout_labels", []), dtype=bool)
        ood_labels = np.asarray(metadata.get("ood_rollout_labels", []), dtype=bool)
        if id_labels.size == 0 or id_labels.size != ood_labels.size:
            return None
        if subset in ("calibration", "test"):
            mask = np.asarray(metadata.get(f"{subset}_rollout_labels", []), dtype=bool)
            if mask.size != id_labels.size:
                return None
            id_labels, ood_labels = id_labels[mask], ood_labels[mask]
        return ["id" if is_id else ("ood" if is_ood else None)
                for is_id, is_ood in zip(id_labels, ood_labels)]

    def _process_rollouts(self, subset: str) -> tuple[dict, float]:
        """Processes the rollouts in the dataset of the given subset and returns the uncertainty scores for each window size and the average inference time.

        Returns:
            (uncertainty_scores_by_window_size, avg_inference_time):
                - uncertainty_scores_by_window_size: A dictionary containing the uncertainty scores for each window size and sucess labels.
                - avg_inference_times: Average inference time during rollout procession.
        """
        inference_times = []
        subtypes = self._subset_subtypes(subset)
        uncertainty_scores_by_window_size = {}
        for window_size in self.cfg.window_sizes:
            uncertainty_scores_by_window_size[window_size] = []

        for episode_idx, rollout_dict in enumerate(self.dataset.iterate_episodes(
            subset=subset,
            required_tensors=self.required_tensors,
            optional_tensors=self.optional_tensors,
            required_actions=self.required_actions,
            optional_actions=self.optional_actions,
            with_success_labels=True,
            normalize_tensors=self.normalize_tensors,
            history=self.cfg.history_length,
        )):
            if not self._should_process_rollout(subset, episode_idx, rollout_dict):
                continue

            # Process the rollout
            uncertainty_scores_one_rollout, inference_time = self._process_one_rollout(rollout_dict)

            inference_times.append(inference_time)
            for window_size in self.cfg.window_sizes:
                uncertainty_scores = self._apply_windowing(uncertainty_scores_one_rollout, window_size)
                episode_info = {
                    "successful": rollout_dict["successful"],
                    "uncertainty_scores": uncertainty_scores,
                    # For the per-subtype AUROCs; None when the recording carries no subtype.
                    "rollout_subtype": subtypes[episode_idx] if subtypes is not None else None,
                }
                uncertainty_scores_by_window_size[window_size].append(episode_info)
        # Calculate the average inference time
        avg_inference_time = np.mean(inference_times)
        return uncertainty_scores_by_window_size, avg_inference_time

    def _apply_windowing(self, scores: list, window_size: int | str) -> np.ndarray:
        """
        Apply windowing to a single rollout's uncertainty scores, handling NaNs and
        shrinking windows when requested size exceeds rollout length.

        Ignore NaNs inside each window and rescale so windows with NaNs stay on the 
        same scale as full windows. If window_size > rollout length, use window_size = rollout
        length.

        Args:
            scores: Per-step scores of a single rollout.
            window_size: "all" for cumulative window, or an integer.

        Returns:
            Array of windowed scores with NaN if a window has no valid values.
        """
        if len(scores) == 0:
            return np.zeros(0, dtype=np.float64)

        scores = np.asarray(scores, dtype=np.float64)
        rollout_length = len(scores)

        # Cumulative window case: Sum up all values [0 : end_idx] for each step
        if window_size == "all":
            windowed_scores = np.empty(rollout_length, dtype=np.float64)
            for end_idx in range(rollout_length):
                # Current cumulative window values and uniform weights
                window_vals = scores[:end_idx + 1]
                cur_weights = np.ones(len(window_vals), dtype=np.float64)
                
                # Mask valid entries (ignore NaNs) and compute effective weight
                valid_indices = ~np.isnan(window_vals)
                total_valid_weight = cur_weights[valid_indices].sum()
                if total_valid_weight == 0:
                    # All-NaN window results in undefined score
                    windowed_scores[end_idx] = np.nan
                    continue
                total_weight = cur_weights.sum()
                windowed_score = np.nansum(window_vals * cur_weights) * (total_weight / total_valid_weight)
                windowed_scores[end_idx] = windowed_score
            return windowed_scores

        # Fixed-size window case
        window_size = int(window_size)

        # Build base weights for the maximum window length
        window_weighting = getattr(self.cfg, "window_weighting", "uniform")
        if window_weighting == "linear":
            slope = self.cfg.window_weighting_strategy.linear.get("slope", 0.2)
            base_weights = 1.0 + slope * np.arange(window_size, dtype=np.float64)
        elif window_weighting == "exponential":
            power = self.cfg.window_weighting_strategy.exponential.get("power", 1.1)
            base_weights = np.power(1 + np.arange(window_size, dtype=np.float64), power)
        else:
            base_weights = np.ones(window_size, dtype=np.float64)

        windowed_scores = np.empty(rollout_length, dtype=np.float64)

        for end_idx in range(rollout_length):
            # Current window length cannot exceed available prefix
            cur_window_len = min(end_idx + 1, window_size)
            start_idx = end_idx - cur_window_len + 1

            # Slice values and align the trailing portion of the weights
            window_vals = scores[start_idx : end_idx + 1]
            cur_weights = base_weights[-cur_window_len:]

            # Ignore NaNs but keep scale by rescaling with valid vs total weight
            valid_indices = ~np.isnan(window_vals)
            total_valid_weight = cur_weights[valid_indices].sum()
            total_weight = cur_weights.sum()
            if total_valid_weight == 0:
                # All-NaN window results in undefined score
                windowed_scores[end_idx] = np.nan
                continue
            # Weighted sum with debiasing due to NaNs
            windowed_score = np.nansum(window_vals * cur_weights) * (total_weight / total_valid_weight)
            windowed_scores[end_idx] = windowed_score

        return windowed_scores

    def _save_pickle(self, save_dir, data, filename):
        if not filename.endswith(".pkl"):
            filename += ".pkl"
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, filename), "wb") as f:
            pickle.dump(data, f)

    def _load_pickle(self, load_dir, filename):
        if not filename.endswith(".pkl"):
            filename += ".pkl"
        with open(os.path.join(load_dir, filename), "rb") as f:
            data = pickle.load(f)
        return data
