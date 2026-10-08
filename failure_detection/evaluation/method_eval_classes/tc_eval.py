from .base_eval_class import BaseEvalClass
import numpy as np
import os
import time
from typing import Optional, Union
from sklearn import metrics
import torch


class TCEval(BaseEvalClass):
    def __init__(
        self,
        cfg,
        method_name: str,
        device: str,
        task_data_path: str,
        dataset,
        **kwargs,
    ):
        super().__init__(cfg, method_name, device, task_data_path, dataset, **kwargs)

        failure_function = kwargs.get("failure_function", None)
        self.failure_functions = {
            "mmd_rbf": self._mmd_rbf,
        }
        if failure_function is not None:
            self.failure_function = self.failure_functions.get(failure_function, self._mmd_rbf)
        else:
            self.failure_function = self._mmd_rbf
        # Determine the action prediction horizon and execution horizon and the maximum number of backtrack steps
        self.exec_horizon = self.dataset.data["metadata"]["actions"].get("action_execution_horizon", 4)
        if self.exec_horizon is None:
            self.exec_horizon = 4
        pred_horizon = self.dataset.get_tensor_shape("action_preds")[-2]
        self.backtrack_steps = self.cfg.get("backtrack_steps", 1)
        self.backtrack_steps = min(self.backtrack_steps, pred_horizon // self.exec_horizon - 1)

    def calculate_uncertainty_score(self, rollout_tensor_dict, **kwargs):
        # Tensors are of shape [n_history, tensor_shape]
        action_preds: torch.Tensor = rollout_tensor_dict["action_preds"]
        assert action_preds.ndim == 4, "Invalid action prediction shape"
        all_zero = torch.all(action_preds == 0, dim=(1, 2, 3))
        num_non_zero = torch.sum(~all_zero)

        if num_non_zero <= 1:
            uncertainty_score = 0.0
        else:
            # Determine how many steps to backtrack
            backtrack_steps = min(self.backtrack_steps, num_non_zero - 1)
            curr_action_pred: torch.Tensor = action_preds[-1]
            prev_action_pred: torch.Tensor = action_preds[-(backtrack_steps + 1)]

            # Extract overlapping action predictions
            prev = prev_action_pred[:, self.exec_horizon * backtrack_steps :]
            prev = np.array(prev.reshape(prev.shape[0], -1))
            # prev = np.array(prev.reshape(-1, prev.shape[1] * prev.shape[2]))
            curr = curr_action_pred[:, : -self.exec_horizon * backtrack_steps]
            curr = np.array(curr.reshape(curr.shape[0], -1))
            # curr = np.array(curr.reshape(-1, curr.shape[1] * curr.shape[2]))

            if self.cfg.error_function.name in self.failure_functions.keys():
                # Calculate uncertainty score using the specified error function
                uncertainty_score = self.failure_functions[self.cfg.error_function.name](prev, curr)
            else:
                # Default to MMD with RBF kernel
                uncertainty_score = self._mmd_rbf(prev, curr)
        return uncertainty_score

    def _mmd_rbf(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> np.float64:
        """MMD using rbf (gaussian) kernel (i.e., k(x,y) = exp(-gamma * ||x-y||^2 / 2))

        Args:
            x: [N, D] matrix.
            y: [M, D] matrix.
            gamma: rbf kernel parameter.

        Returns:
            MMD value.
        """
        assert x.ndim == 2 and y.ndim == 2

        gamma = self.cfg.error_function.mmd_rbf.gamma
        if isinstance(gamma, str):
            if gamma == "median":
                z = np.vstack([x, y])
                distances = np.sum((z[:, np.newaxis, :] - z[np.newaxis, :, :]) ** 2, axis=2)
                gamma = 1.0 / (2 * np.median(distances[distances > 0]))
            elif gamma == "max_eig":
                z = np.vstack([x, y])
                cov = np.cov(z.T)
                max_eig = np.max(np.linalg.eigvalsh(cov))
                gamma = 1.0 / max_eig
            else:
                raise ValueError(f"Gamma {gamma} is not supported.")

        xx = metrics.pairwise.rbf_kernel(x, x, gamma)
        yy = metrics.pairwise.rbf_kernel(y, y, gamma)
        xy = metrics.pairwise.rbf_kernel(x, y, gamma)
        return xx.mean() + yy.mean() - 2 * xy.mean()
