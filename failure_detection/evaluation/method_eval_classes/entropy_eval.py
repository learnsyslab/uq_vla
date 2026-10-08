from .base_eval_class import BaseEvalClass
import numpy as np
from scipy.stats import entropy
import torch


class ENTROPYEval(BaseEvalClass):
    def __init__(self, cfg, method_name, device, task_data_path, dataset, **kwargs):
        self.cellsize_factor = kwargs.get("cellsize_factor", None)
        if self.cellsize_factor is None:
            self.cellsize_factor = cfg.get("cellsize_factor", 0.01)

        super().__init__(cfg, method_name, device, task_data_path, dataset, **kwargs)

    def _execute_preprocessing(self):
        # Extract the action predictions from the dataset for the calibration subset and only the position action
        action_preds: torch.Tensor = self.dataset.get_subset(
            subset="calibration",
            required_tensors=self.required_tensors,
            optional_tensors=self.optional_tensors,
            required_actions=self.required_actions,
            optional_actions=self.optional_actions,
        )["action_preds"]
        # Flatten the action_preds (keep the last dimension)
        positions = action_preds.view(-1, action_preds.shape[-1])

        ranges = torch.max(positions, dim=0).values - torch.min(positions, dim=0).values
        max_range_value = torch.max(ranges)

        if self.cfg.single_cellsize:
            # Set all ranges to the maximum range value
            ranges = torch.full_like(ranges, max_range_value)
        else:
            # Set zero ranges to the maximum range value
            ranges = torch.where(ranges == 0, max_range_value, ranges)

        # Calculate the cell size based on the maximum working area and a configurable factor

        self.cell_size = np.array(ranges) * self.cellsize_factor

    def calculate_uncertainty_score(self, rollout_tensor_dict, **kwargs):
        action_preds: torch.Tensor = rollout_tensor_dict["action_preds"]

        action_preds = np.array(action_preds)
        # action_preds shape: (batch_size, num_steps, num_actions)
        # actions already filtered by the action mapping
        entropy_values = []
        test_factor = 1
        for i in range(action_preds.shape[-2] // test_factor):
            new_value, _ = self._entropy_endpoints(action_preds[:, i * test_factor, :])
            entropy_values.append(new_value)

        entropy = sum(entropy_values) / len(entropy_values)
        return entropy

    def _entropy_endpoints(self, endpoints):
        """
        endpoints shape (number of endpoints, 3)
        """
        cell_size = self.cell_size
        # Determine dynamic grid limits based on endpoint distribution
        x_min, x_max = endpoints[:, 0].min(), endpoints[:, 0].max()
        y_min, y_max = endpoints[:, 1].min(), endpoints[:, 1].max()
        z_min, z_max = endpoints[:, 2].min(), endpoints[:, 2].max()

        # Add a small buffer to the limits
        x_buffer = 0.01 * (x_max - x_min)
        y_buffer = 0.01 * (y_max - y_min)
        z_buffer = 0.01 * (z_max - z_min)
        x_min -= x_buffer
        x_max += x_buffer
        y_min -= y_buffer
        y_max += y_buffer
        z_min -= z_buffer
        z_max += z_buffer

        # Create the grid based on dynamic limits and cell size
        x_grid = np.arange(x_min, x_max + cell_size[0], cell_size[0])
        y_grid = np.arange(y_min, y_max + cell_size[1], cell_size[1])
        z_grid = np.arange(z_min, z_max + cell_size[2], cell_size[2])

        # Count endpoints in each cell
        cell_indices_x = np.digitize(endpoints[:, 0], x_grid)
        cell_indices_y = np.digitize(endpoints[:, 1], y_grid)
        cell_indices_z = np.digitize(endpoints[:, 2], z_grid)

        # Adjust indices to prevent out-of-bounds error
        cell_indices_x -= 1
        cell_indices_y -= 1
        cell_indices_z -= 1

        # Initialize counts
        num_cells_x = max(len(x_grid) - 1, 1)
        num_cells_y = max(len(y_grid) - 1, 1)
        num_cells_z = max(len(z_grid) - 1, 1)
        counts = np.zeros((num_cells_x, num_cells_y, num_cells_z), dtype=int)
        for i in range(len(endpoints)):
            counts[cell_indices_x[i], cell_indices_y[i], cell_indices_z[i]] += 1

        # Compute the Shannon entropy with base 2
        counts_vector = counts.flatten()
        d_entropy = entropy(counts_vector, base=2)

        return d_entropy, counts_vector.shape[0]
