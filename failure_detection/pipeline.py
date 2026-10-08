"""
This module serves as the main entry point for the failure prediction pipeline.

The pipeline consists of the following main components:

1. **TaskManager**:
   - Interfaces with task environments.
   - Initializes and manages the `ProcessedRolloutDataset` from raw rollouts.

2. **ProcessedRolloutDataset**:
   - Handles the data for evaluation, training, and results generation.
   - Provides utilities for loading, normalizing, and iterating over rollouts.

3. **RNDTrainer**:
   - Trains Random Network Distillation (RND) models for failure prediction.

4. **EvaluationManager**:
   - Interfaces with method-specific evaluation classes.
   - Evaluates failure prediction methods and generates metrics.

5. **ResultsManager**:
   - Combines evaluation results.
   - Creates summaries and visualizations of the results.

### Configuration:
- Experiments: `/configs/pusht.yaml`, `/configs/libero_plus.yaml` (base: `/configs/default.yaml`)
- Base Evaluation settings: `/configs/eval/base.yaml`
- Method-specific settings: `/configs/eval/{method}.yaml`
- Task-specific settings: `/configs/task/{task}.yaml`

Each stage can be executed independently or as part of the complete pipeline.
"""

import sys
import os
import shutil
import pathlib
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
import torch

# Add the root directory to the path
ROOT_DIR = str(pathlib.Path(__file__).resolve().parent)
sys.path.insert(0, ROOT_DIR)
os.chdir(ROOT_DIR)


from rnd import RNDTrainer
from evaluation import EvaluationManager, ResultsManager
from tasks import TaskManager

from shared_utils.hydra_utils import load_config
from shared_utils.utility_functions import get_required_tensors
from shared_utils.training_data import METHOD_CFG_OVERRIDE_KEYS


# Determine Config Path
base_config_path = os.path.join(ROOT_DIR, "configs")
# FD_DATA_ROOT sets the data root (<root>/<task> input links, <root>/results output), so several pipeline runs
# can proceed side by side (e.g. one per ensemble) without sharing data or results.
base_data_path = os.environ.get("FD_DATA_ROOT", os.path.join(ROOT_DIR, "data"))


# Add the config directory to the path
@hydra.main(config_path=base_config_path, config_name="default.yaml", version_base="1.1")
def main(cfg: DictConfig):
    # Tasks to be processed
    tasks = cfg.get("tasks", [])
    suite_to_task_ids = cfg.get("suite_to_task_ids", {})
    # Methods to be evaluated
    rnd_models = cfg.get("rnd_models", [])
    methods = cfg.get("methods", [])
    methods.extend(rnd_models)

    # Logical combination of methods
    combine_methods = cfg.get("combine_methods", False)
    combined_methods = cfg.get("combined_methods", OmegaConf.create({}))
    combined_methods = OmegaConf.to_container(combined_methods, resolve=True)
    with open_dict(cfg):
        cfg["combined_methods"] = combined_methods

    train_rnd = cfg.get("train_rnd", True)
    # Read before the loop: load_config below recomposes the *default* config with only
    # the task overridden, which discards experiment-level settings from `cfg`.
    max_calibration_rollouts = cfg.get("max_calibration_rollouts", None)
    # Which N of the available calibration rollouts (None -> the first N, see TaskManager).
    calibration_subset_seed = cfg.get("calibration_subset_seed", None)
    # Same reason: where learned detectors get their training data from is an experiment-level
    # choice that must be pushed into every method's eval config explicitly.
    method_cfg_overrides = {k: cfg[k] for k in METHOD_CFG_OVERRIDE_KEYS if k in cfg}
    # Per-method settings (`method_overrides: {bayesian: {num_scorers: 3, ...}}`) travel the same way.
    if cfg.get("method_overrides", None):
        method_cfg_overrides["method_overrides"] = OmegaConf.to_container(cfg.method_overrides, resolve=True)
    # Check if the pipeline inputs are valid
    check_inputs(
        cfg,
        tasks,
        methods,
        combined_methods,
        combine_methods,
        base_config_path,
    )
    # Check which tensors are required for the methods
    required_tensors, optional_tensors = get_required_tensors(
        methods, base_config_path, method_overrides=method_cfg_overrides.get("method_overrides")
    )
    total_results = {}
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def task_entries():
        for task in tasks:
            task_ids = suite_to_task_ids.get(task, [])
            for task_id in (task_ids if task_ids else [None]):
                task_data_path = os.path.join(base_data_path, task)
                task_label = task
                if task_id is not None:
                    task_data_path = os.path.join(task_data_path, f"task{task_id:02d}")
                    task_label += f"_task_{task_id:02d}"
                yield task, task_id, task_data_path, task_label

    def build_dataset(cfg, task, task_data_path):
        taskmanager = TaskManager(
            cfg,
            task,
            base_config_path,
            task_data_path,
            required_tensors=required_tensors,
            optional_tensors=optional_tensors,
            device=device,
            max_calibration_rollouts=max_calibration_rollouts,
            calibration_subset_seed=calibration_subset_seed,
        )
        # Create or load or update the dataset
        return taskmanager.get_rollout_dataset(load_dataset_if_exists=False)

    # training_data=calibration_pooled: one logpZO / RND model per policy, fit on the union of the
    # training halves of every task below. All task datasets are built first and handed to the
    # trainers/evaluators (`pooled_datasets`); the shared models live in <root>/pooled_models and
    # are cleared once per run so the first task trains them and the others reuse them.
    pooled = str(cfg.get("training_data", "calibration")) == "calibration_pooled"
    detector_kwargs = {}
    if pooled:
        pooled_models_dir = os.path.join(base_data_path, "pooled_models")
        if os.path.isdir(pooled_models_dir):
            shutil.rmtree(pooled_models_dir)
        os.makedirs(pooled_models_dir, exist_ok=True)
        pooled_datasets = {}
        for task, task_id, task_data_path, task_label in task_entries():
            print(f"[pooled] loading dataset {task_label}")
            pooled_datasets[task_label] = build_dataset(load_config("task", task, return_only_subdict=False), task, task_data_path)
        detector_kwargs = {"pooled_datasets": pooled_datasets, "pooled_models_dir": pooled_models_dir}

    # Iterate over the task
    for task, task_id, task_data_path, task_label in task_entries():
            print(f"-------------- Task: {task}" + (f" | Task ID: {task_id}" if task_id is not None else "") + " ----------------")
            # Load the complete config with only the task overridden
            cfg = load_config("task", task, return_only_subdict=False)

            # Initial Data Gathering and Processing
            dataset = pooled_datasets[task_label] if pooled else build_dataset(cfg, task, task_data_path)

            # Train RND models if required
            if train_rnd:
                rndtrainer = RNDTrainer(
                    base_config_path, task_data_path, dataset, device=device, task_cfg=cfg,
                    method_cfg_overrides=method_cfg_overrides, **detector_kwargs,
                )
                rndtrainer.train(rnd_models)
            else:
                pass

            # Initialize the evaluation interface
            evaluationmanager = EvaluationManager(
                base_config_path, task_data_path, dataset, device=device, method_cfg_overrides=method_cfg_overrides,
                **detector_kwargs,
            )
            # Evaluate the methods including the combined methods
            results = evaluationmanager.evaluate(methods, combine_methods, combined_methods)
            # Save the results for the task
            total_results[task_label] = results

            if not pooled:
                dataset.data.clear()

    # Extract the method names from the results (required due to combined methods)
    method_names = [method for method in total_results[task_label].keys()]
    # Create the results manager
    resultsmanager = ResultsManager(base_config_path, base_data_path)
    # Update the saved results with the new results
    resultsmanager.combine_results(total_results, method_names=method_names)
    # Summarize the results as defined in results config. Saves the summary(ies) as csv file(s).
    resultsmanager.create_summary()
    # # Generate uncertainty plots for the results as defined in results config. Saves the plots as pdf files.
    # resultsmanager.generate_uncertainty_plots()


def check_inputs(cfg: DictConfig, tasks, methods, combined_methods, combine_methods, base_config_path):
    """
    Check the inputs for the pipeline.
    """
    available_tasks = cfg.get("available_tasks", [])
    assert all(task in available_tasks for task in tasks)
    implemented_methods = cfg.get("implemented_methods", [])
    assert all(method in implemented_methods for method in methods)
    available_rnd_models = cfg.get("available_rnd_models", [])
    assert all(model in available_rnd_models for model in methods if model.startswith("rnd_"))
    if combine_methods:
        assert all(
            combined_methods[key]["m1"]["name"] in methods
            and combined_methods[key]["m2"]["name"] in methods
            and combined_methods[key]["operation"] in ["or", "and"]
            for key in combined_methods.keys()
        )

    # Check if the config files exist
    config_files = os.listdir(os.path.join(base_config_path, "eval"))
    assert all(f"{method}.yaml" in config_files for method in methods), (
        f"Some config files are missing for the methods: {methods}"
    )
    config_files = os.listdir(os.path.join(base_config_path, "task"))
    assert all(f"{task}.yaml" in config_files for task in tasks), (
        f"Some config files are missing for the tasks: {tasks}"
    )
    return


if __name__ == "__main__":
    main()
