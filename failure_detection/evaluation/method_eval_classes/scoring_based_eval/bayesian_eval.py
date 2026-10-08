from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor

from .scoring_based_eval import ScoringBasedEval
from .scoring_metrics import inter_vel_diff, likelihood, mode_distance
from rollout_data.rollout_datasets import ProcessedRolloutDataset


class BAYESIANEval(ScoringBasedEval):
    """
    Evaluation class for cross-Bayesian failure prediction method.
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

        # The current Bayesian scorer model being evaluated (e.g., ensemble, laplace)
        self.scorer_type: Optional[str] = None
        # The current scoring metric being evaluated (e.g., likelihood, mode_distance, inter_vel_diff)
        self.scoring_metric: Optional[str] = None

    def get_scorer_types(self) -> List[str]:
        return self.cfg.scorer_types

    def configure_run(self, scorer_type: str, scoring_metric: str) -> Tuple[str, str]:
        self.scorer_type = scorer_type
        self.scoring_metric = scoring_metric
        method_tag = f"{self.method_name}_{scorer_type}_{scoring_metric}"
        results_dir = f"{self.results_dir}_{scorer_type}_{scoring_metric}"
        return method_tag, results_dir

    def calculate_uncertainty_score(self, rollout_tensor_dict: Dict[str, Tensor]) -> float:
        """
        Compute the uncertainty score for a rollout step based on the configured 
        scoring metric.

        Args:
            rollout_tensor_dict: Dictionary containing data from the action generation 
                process at a rollout step. The required keys depend on the selected 
                scoring metric.

        Returns:
            Scalar uncertainty score as a float.
        """
        if self.scoring_metric == "likelihood":
            return likelihood(
                log_likelihood=rollout_tensor_dict[f"{self.scorer_type}_log_likelihood"]
            )
        elif self.scoring_metric == "mode_distance":
            return mode_distance(
                terminal_vels=rollout_tensor_dict[f"{self.scorer_type}_terminal_velocities"],
                eval_times=rollout_tensor_dict["terminal_eval_times"],
                config=self.cfg,
            )
        elif self.scoring_metric == "inter_vel_diff":
            return inter_vel_diff(
                ref_velocities=rollout_tensor_dict["velocities"],
                cmp_velocities=rollout_tensor_dict[f"{self.scorer_type}_velocities"],
                ode_eval_times=rollout_tensor_dict["ode_eval_times"],
                vel_diff_scaling_factors=rollout_tensor_dict["vel_diff_scaling"],
                config=self.cfg,
            )
        elif self.scoring_metric == "inter_vel_diff_2way":
            return float(rollout_tensor_dict[f"{self.scorer_type}_inter_vel_diff_2way_scores"].item())
        else:
            raise ValueError(
                f"Unknown scoring metric: {self.scoring_metric}."
            )
