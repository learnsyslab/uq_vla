from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch

from ..base_eval_class import BaseEvalClass
from rollout_data.rollout_datasets import ProcessedRolloutDataset


class ScoringBasedEval(BaseEvalClass, ABC):
    """
    Abstract base class for evaluation methods that rely on scoring metrics.
    """
    def __init__(
        self,
        cfg: Dict[str, Any],
        method_name: str,
        device: torch.device,
        task_data_path: Union[str, Path],
        dataset: ProcessedRolloutDataset,
        **kwargs
    ):
        super().__init__(
            cfg=cfg,
            method_name=method_name,
            device=device,
            task_data_path=task_data_path, 
            dataset=dataset,
            **kwargs
        )
    
    @abstractmethod
    def get_scorer_types(self) -> List[str]:
        """
        Return the scorer types to evaluate (e.g., ["ensemble", "laplace"]). 
        If not applicable, return an empty list.
        """
        raise NotImplementedError
    
    @abstractmethod
    def configure_run(self, scorer_type: str, scoring_metric: str) -> Tuple[str, str]:
        """
        Configure subclass state for the current scorer/metric pair and return 
        (method_tag, results_dir).
        """
        raise NotImplementedError
    
    def evaluate(self) -> List[Dict[str, Any]]:
        """
        Generic evaluation loop over across all combinations of scorer types and
        scoring metrics. Subclasses can override _get_scorer_types to return [] if
        scorer types are not applicable.

        For each (scorer_type, scoring_metric) pair, thresholds are computed from
        the calibration rollouts and failure prediction performance is evaluated for
        the configured Bayesian methods. Results are saved to disk.

        Returns:
            List of dictionaries, each containing evaluation results for one 
            (scorer_type, scoring_metric) combination.
        """
        # Perform preprocessing before evaluation
        self._execute_preprocessing()
        
        results_list: List[Dict[str, Any]] = []

        scorer_types = self.get_scorer_types()
        if not scorer_types:
            # Still run over scoring metrics without scorer types
            scorer_types = [None]

        # Iterate over all scorer types (e.g., ensemble, laplace)
        for scorer_type in scorer_types:
            for scoring_metric in self.cfg.scoring_metrics:
                method_tag, results_dir = self.configure_run(scorer_type, scoring_metric)

                # Process calibration rollouts to compute uncertainty scores
                uncertainty_scores_by_window_size_calibration, avg_inference_time_calibration = self._process_rollouts(
                    subset="calibration"
                )
                
                # Compute thresholds from calibration scores
                thresholds, scores_by_threshold_calibration = self._get_thresholds(
                    uncertainty_scores_by_window_size_calibration
                )

                # Process test rollouts to compute uncertainty scores
                uncertainty_scores_by_window_size_test, avg_inference_time_test = self._process_rollouts(subset="test")

                # Evaluate metrics on test rollouts using calibration thresholds
                metrics, scores_by_threshold_test = self._get_metrics(thresholds, uncertainty_scores_by_window_size_test)

                # Threshold-free AUROC under both episode aggregations (max and mean);
                # one value per window size, injected into every metrics entry so it
                # flows through results_manager like the other per-(task, window)
                # metrics (see BaseEvalClass.evaluate for the same injection).
                self._inject_auroc(metrics, self._compute_auroc(uncertainty_scores_by_window_size_test))

                # Collect results for this (scorer_type, scoring_metric) combination
                eval_results = {
                    "method": method_tag,
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

                # Save results and append to the list
                self._save_pickle(results_dir, eval_results, "eval_results.pkl")
                results_list.append(eval_results)
            
        return results_list