import os
import torch
import hydra
import numpy as np
from shared_utils import ensure_list, save_data, load_data, load_config
from .method_eval_classes import BaseEvalClass, ENTROPYEval, TCEval, RNDEval
from omegaconf import OmegaConf, DictConfig
import matplotlib.pyplot as plt
from evaluation.utils import calculate_metrics
from sklearn.metrics import roc_auc_score
from shared_utils.training_data import apply_method_cfg_overrides
from rollout_data.rollout_datasets import ProcessedRolloutDataset


def combine_two_methods(combination_config, total_results):
    """Logically combine two failure-prediction methods ('and' = per-step minimum of the two
    threshold-normalised scores, so an alarm needs both detectors over their own thresholds at the
    same step; 'or' = maximum). Module-level so it can also run offline on saved results
    (pipeline.py `combine_from_saved`). Adds a threshold-dependent AUROC per (quantile, window):
    the AUROC of max_t of the combined normalised score, over all / ID / OOD test rollouts.
    """
    method1 = combination_config["m1"]["name"]
    method2 = combination_config["m2"]["name"]
    operation = combination_config["operation"]
    method_keyword = method1 + "_" + operation + "_" + method2

    # Get the results of the two methods
    method1_results = total_results[method1]
    method2_results = total_results[method2]
    # Merge the configuration of the two methods
    method1_cfg = OmegaConf.to_container(method1_results["cfg"], resolve=True)
    method2_cfg = OmegaConf.to_container(method2_results["cfg"], resolve=True)
    combined_config = OmegaConf.merge(OmegaConf.create(method1_cfg), OmegaConf.create(method2_cfg))

    # Get the combined quantiles and window sizes
    quantiles1: list = combination_config["m1"].get("quantiles", None)
    if quantiles1 is None:
        quantiles1: list = method1_results["quantiles"]
    window_sizes1 = combination_config["m1"].get("window_sizes", None)
    if window_sizes1 is None:
        window_sizes1 = method1_results["window_sizes"]
    quantiles2: list = combination_config["m2"].get("quantiles", None)
    if quantiles2 is None:
        quantiles2: list = method2_results["quantiles"]
    window_sizes2 = combination_config["m2"].get("window_sizes", None)
    if window_sizes2 is None:
        window_sizes2: list = method2_results["window_sizes"]

    # Combine quantiles and window sizes
    if quantiles1 != quantiles2:
        new_quantiles1, new_quantiles2 = [], []
        for i in range(len(quantiles1)):
            for j in range(len(quantiles2)):
                new_quantiles1.append(quantiles1[i])
                new_quantiles2.append(quantiles2[j])
        quantiles1, quantiles2 = new_quantiles1, new_quantiles2

    new_window_sizes1, new_window_sizes2 = [], []
    for i in range(len(window_sizes1)):
        for j in range(len(window_sizes2)):
            new_window_sizes1.append(window_sizes1[i])
            new_window_sizes2.append(window_sizes2[j])
    window_sizes1, window_sizes2 = new_window_sizes1, new_window_sizes2

    total_results[method_keyword] = {
        "method": method_keyword,
        "quantiles": None,
        "window_sizes": None,
        "calibration_uncertainty_scores": {
            method1: method1_results["calibration_uncertainty_scores"],
            method2: method2_results["calibration_uncertainty_scores"],
        },
        "test_uncertainty_scores": {
            method1: method1_results["test_uncertainty_scores"],
            method2: method2_results["test_uncertainty_scores"],
        },
        "calibration_thresholds": {
            method1: method1_results["calibration_thresholds"],
            method2: method2_results["calibration_thresholds"],
        },
        "test_scores_by_threshold": {},
        "test_metrics": {},
        "avg_inference_time": method1_results["avg_inference_time"] + method2_results["avg_inference_time"],
        "max_episode_length": method1_results["max_episode_length"],
        "successful_test_rollouts": method1_results["successful_test_rollouts"],
        "id_test_rollouts": method1_results["id_test_rollouts"],
        "ood_test_rollouts": method1_results["ood_test_rollouts"],
        "cfg": combined_config,
    }

    dataset_stats = {
        "id_rollouts": method1_results["id_test_rollouts"],
        "ood_rollouts": method1_results["ood_test_rollouts"],
        "successful_rollouts": method1_results["successful_test_rollouts"],
        "max_episode_length": method1_results["max_episode_length"],
    }
    threshold_styles = list(method1_results["calibration_thresholds"].keys())

    comb_quantiles, test_metrics, test_scores_by_threshold = [], {}, {}
    for threshold_style in threshold_styles:
        test_metrics[threshold_style] = {}
        test_scores_by_threshold[threshold_style] = {}
        for quantile1, quantile2 in zip(quantiles1, quantiles2):
            comb_quantile = quantile1 if quantile1 == quantile2 else f"{quantile1}/{quantile2}"
            comb_quantiles.append(comb_quantile)
            test_metrics[threshold_style][comb_quantile] = {}
            test_scores_by_threshold[threshold_style][comb_quantile] = {}

            comb_window_sizes = []
            for window_size1, window_size2 in zip(window_sizes1, window_sizes2):
                comb_window_size = (
                    window_size1 if window_size1 == window_size2 else f"{window_size1}/{window_size2}"
                )
                comb_window_sizes.append(comb_window_size)

                scores_by_threshold_m1 = method1_results["test_scores_by_threshold"][threshold_style][quantile1][
                    window_size1
                ]
                scores_by_threshold_m2 = method2_results["test_scores_by_threshold"][threshold_style][quantile2][
                    window_size2
                ]

                scores_by_threshold = []
                for episode in range(len(scores_by_threshold_m1)):
                    if operation == "and":
                        scores_by_threshold.append(
                            np.minimum(scores_by_threshold_m1[episode], scores_by_threshold_m2[episode])
                        )
                    elif operation == "or":
                        scores_by_threshold.append(
                            np.maximum(scores_by_threshold_m1[episode], scores_by_threshold_m2[episode])
                        )

                test_metrics[threshold_style][comb_quantile][comb_window_size] = calculate_metrics(
                    scores_by_threshold, dataset_stats, combined_config["detection_patience"]
                )
                _inject_combined_auroc(
                    test_metrics[threshold_style][comb_quantile][comb_window_size], scores_by_threshold, dataset_stats
                )
                test_scores_by_threshold[threshold_style][comb_quantile][comb_window_size] = scores_by_threshold

    total_results[method_keyword]["test_metrics"] = test_metrics
    total_results[method_keyword]["test_scores_by_threshold"] = test_scores_by_threshold
    total_results[method_keyword]["quantiles"] = comb_quantiles
    total_results[method_keyword]["window_sizes"] = comb_window_sizes
    return total_results


def _inject_combined_auroc(entry: dict, scores_by_threshold: list, dataset_stats: dict) -> None:
    """AUROC of the combined rule: each test rollout is reduced to the peak of its combined
    normalised score (NaN steps ignored), failed rollouts are the positive class. Unlike the
    per-method AUROC this depends on the quantile (the normalisation does); NaN when a subset has
    a single class. Keys match the per-method ones so results_manager picks them up unchanged."""
    labels = 1 - np.asarray(dataset_stats["successful_rollouts"], dtype=int)
    peaks = np.asarray([np.nanmax(np.asarray(sc, dtype=np.float64)) if np.asarray(sc).size else np.nan
                        for sc in scores_by_threshold], dtype=np.float64)
    ok = ~np.isnan(peaks)
    def _auc(mask):
        mask = np.asarray(mask, dtype=bool) & ok
        return float(roc_auc_score(labels[mask], peaks[mask])) if np.unique(labels[mask]).size > 1 else float("nan")
    entry["auroc"] = _auc(np.ones_like(ok))
    entry["auroc_mean"] = float("nan")
    entry["auroc_horizon_mean"] = float("nan")
    entry["auroc_id"] = _auc(dataset_stats.get("id_rollouts", np.zeros_like(ok)))
    entry["auroc_ood"] = _auc(dataset_stats.get("ood_rollouts", np.zeros_like(ok)))


class EvaluationManager:
    def __init__(
        self,
        config_path: str,
        task_data_path: str,
        dataset: ProcessedRolloutDataset,
        device: str = None,
        method_cfg_overrides: dict | None = None,
        **kwargs,
    ):
        """The EvaluationManager class provides the inferface of the evaluation of the failure prediction methods.
        It loads the method-specific configuration and then calls the method-specific evaluation classes.

        Important:
        - Task selection is done via given dataset, config_path, and task_data_path.
        - To run the evaluation, call the evaluate method with the methods to evaluate.
        - Evaluation results are returned as a dictionary with the method names as keys.

        Returns:
            evaluation_manager: An EvaluationManager object

        Args:
            config_path (str): The path to the base configuration folder.
            task_data_path (str): The base path to the data (task-specific).
            dataset (ProcessedRolloutDataset): The dataset to use for evaluation.
            device (str, optional): The device ["cuda", "cpu"] to use for evaluation. If not provided, "cuda" is used if available.
        """
        self.base_config_path = config_path
        self.task_data_path = task_data_path
        self.base_cfg = self._load_config()
        if not isinstance(dataset, ProcessedRolloutDataset):
            raise ValueError("dataset must be of type ProcessedRolloutDataset.")
        self.dataset = dataset
        # Pooled-training extras (pipeline.py, training_data=calibration_pooled) go only to the
        # learned detectors; the base class rejects unknown keywords.
        self.pooled_kwargs = {k: kwargs.pop(k) for k in ("pooled_datasets", "pooled_models_dir") if k in kwargs}
        self.kwargs = kwargs
        # Experiment-level keys (training_data, demo_embeddings_file) applied to every method cfg;
        # kept out of self.kwargs, which is forwarded verbatim to the eval classes.
        self.method_cfg_overrides = method_cfg_overrides

        if device is not None and device in ["cpu", "cuda:0"]:
            self.device = device if torch.cuda.is_available() else "cpu"
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def evaluate(self, methods, combine_methods: bool = False, combined_methods: dict = None):
        """
        Evaluate the given methods.

        Args:
            methods (list): The list of methods to evaluate.
            visualize (bool, optional): If True, visualize the results. Defaults to False.
            combine_methods (bool, optional): If True, logically combine the methods given in the combined_methods dict. Defaults to False.
            combined_methods (dict, optional): Dictionary containing the methods to combine. Defaults to None.

        Returns:
            dict: A dictionary containing the evaluation results for each method.
        """
        total_results = {}  # Dictionary to store results for all methods

        for method in methods:
            # Load the configuration for the method
            cfg = self._load_config(method_name=method)
            apply_method_cfg_overrides(cfg, self.method_cfg_overrides, method_name=method)

            # Get the method-specific evaluation class
            methodEvalClass: BaseEvalClass = self._get_method_eval_class(method, cfg)

            # Run evaluation for this method
            evaluation_output = methodEvalClass.evaluate()

            if isinstance(evaluation_output, list):
                # The method produced results for multiple configurations
                for config_result in evaluation_output:
                    # Each config_result must contain a unique identifier in the "method" field
                    method_key = config_result.get("method")
                    total_results[method_key] = config_result
            else:
                # Standard case: single configuration result
                total_results[method] = evaluation_output

        # Combine methods if specified
        if combine_methods:
            for key in combined_methods:
                total_results = self._combine_two_methods(combined_methods[key], total_results)

        return total_results

    def _combine_two_methods(self, combination_config, total_results):
        """See module-level `combine_two_methods`."""
        return combine_two_methods(combination_config, total_results)

    def _get_method_eval_class(self, method_name: str, cfg):
        """Get the method-specific evaluation class. The classes are subclasses from the base evaluation class and only implement the model-specific functions."""
        # Get the class name from the method name (normalized version use the same class as the non-normalized version)
        class_name = method_name
        # All RND method used the same evaluation class
        if "rnd" in class_name:
            class_name = "rnd"

        class_name = f"{class_name.replace('_', '').upper()}Eval"  # Class name follows the pattern {METHOD}Eval where METHOD is the method name in uppercase without underscores
        module_name = "evaluation.method_eval_classes."  # Base module name for method_eval_classes

        # Import the method_eval_classes module (which exposes all classes in its __init__.py)
        method_eval_class: BaseEvalClass = hydra.utils.get_class(module_name + class_name)

        class_obj = method_eval_class(
            cfg=cfg,
            method_name=method_name,
            device=self.device,
            task_data_path=self.task_data_path,
            dataset=self.dataset,
            **self.kwargs,
            **(self.pooled_kwargs if class_name in ("RNDEval", "LOGPZOEval") else {}),
        )
        return class_obj

    def _load_config(self, method_name: str = "base") -> DictConfig:
        """Load the configuration file for the respective method_name with hydra."""
        cfg = load_config(
            module="eval", filename=method_name, return_only_subdict=True, base_config_dir=self.base_config_path
        )
        return cfg
