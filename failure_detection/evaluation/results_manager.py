import os
import torch
import numpy as np
from shared_utils import save_data, load_config, load_data, compare_dicts
from shared_utils.data_management import _get_filenames, ensure_list
from omegaconf import OmegaConf, DictConfig
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, TypeVar, Union


class ResultsManager:
    def __init__(
        self,
        config_path: str,
        base_data_path: str,
        **kwargs,
    ):
        """The ResultsManager class is responsible for managing the results of the evaluation.

        It saves/loads the individual results for all tasks/methods/hparams, combines them, and generates the final results using the result configuration file.

        It optionally visualizes the results.

        Results structure: nested dictionary with the following structure: {task_name: {method_name: hyperparameter_id: {window_size: {quantile: {metrics}}}}}}

        The results are saved in {base_data_path}/results/.

        Returns:
            results_manager: An EvaluationManager object

        Args:
            config_path (str): The path to the base configuration folder.
            base_data_path (str): The base path to the data.
        """
        self.base_config_path = config_path
        self.base_data_path = base_data_path
        self.results_dir = os.path.join(self.base_data_path, "results")
        os.makedirs(self.results_dir, exist_ok=True)
        self.summary_dir = os.path.join(self.results_dir, "summaries")
        os.makedirs(self.summary_dir, exist_ok=True)
        self.method_results_dir = os.path.join(self.results_dir, "method_results")
        os.makedirs(self.method_results_dir, exist_ok=True)

        self.cfg = self._load_config()
        self.kwargs = kwargs

    def _load_config(self, filename: str = "base") -> DictConfig:
        """Load the results configuration file with hydra."""
        cfg = load_config(
            module="results", filename=filename, return_only_subdict=True, base_config_dir=self.base_config_path
        )
        return cfg

    def _load_results(self, method_names: Optional[List[str]] = None) -> dict:
        """Load results for specific methods or all methods if no method_names are provided."""
        results = {}
        filenames = _get_filenames(self.method_results_dir, keywords="_results", data_types="pkl")
        for filename in filenames:
            method_name = filename.split("_results.pkl")[0]
            if method_names and method_name not in method_names:
                continue
            method_results = load_data(self.method_results_dir, keywords=filename, data_types="pkl", error_if_not_found=False)
            if isinstance(method_results, list) and len(method_results) > 0:
                method_results = method_results[0]
            if isinstance(method_results, dict):
                results[method_name] = method_results
        return results

    def _save_results(self, results: dict, method_names: Optional[List[str]] = None) -> None:
        """Save results for specific methods or all methods if no method_names are provided."""
        for method, method_results in results.items():
            if method_names and method not in method_names:
                continue
            filename = f"{method}_results.pkl"
            save_data(self.method_results_dir, filename, method_results, data_types="pkl", overwrite=True)

    def _save_dataframe(self, df: pd.DataFrame, filename="complete_results") -> None:
        if not filename.endswith(".csv"):
            filename += ".csv"
        df.to_csv(os.path.join(self.results_dir, filename), index=False)

    def _save_summary(self, df: pd.DataFrame, filename="summary") -> None:
        if not filename.endswith(".csv"):
            filename += ".csv"
        df.to_csv(os.path.join(self.summary_dir, filename), index=False)

    def _load_dataframe(self, filename="complete_results") -> pd.DataFrame:
        """Load the results of the evaluation from the results directory."""
        if not filename.endswith(".csv"):
            filename += ".csv"
        df = pd.read_csv(os.path.join(self.results_dir, filename))
        return df

    def _convert_new_results(self, new_results: dict = {}, method_names: list = [None]) -> dict:
        """Convert the new results to the format of the complete results.

        The new results are in the format {task_name: {method_name: {metrics}}
        The complete results are in the format {method_name: hyperparameter_id: {task_name: {metrics}}}

        Args:
            new_results (dict): The results of the evaluation.
            method_name (str): The name of the method to extract.

        Returns:
            dict: The results of the specified method.
        """
        # Check if the new results are empty
        if not new_results or not method_names:
            return {}

        # Check if the new results are a dictionary
        if not isinstance(new_results, dict):
            raise ValueError("New results must be a dictionary.")

        # Convert the new results to the format of the complete results
        converted_results = {}
        for method_name in method_names:
            method_results = {}
            method_results[0] = {}
            hparams = {}
            for task in new_results.keys():
                if method_name not in new_results[task]:
                    continue
                method_results[0][task], hparams = self._filter_new_results(new_results[task][method_name])
            if hparams:
                method_results[0]["hparams"] = hparams
            converted_results[method_name] = method_results
        return converted_results

    def _filter_new_results(self, new_results: dict) -> Tuple[dict, dict]:
        hparams = new_results["cfg"]["hparams"]
        new_results.pop("cfg")
        return new_results, hparams

    def _create_complete_df(self) -> pd.DataFrame:
        """Create a complete DataFrame from the results.

        The DataFrame contains the following columns:
            - Task
            - Method
            - HID
            - Window
            - Quantile
            - Metric 1
            - Metric 2
            - ...
        """
        # Create a DataFrame from the results
        complete_df = pd.DataFrame()
        # Obtain the available results
        filenames = _get_filenames(self.method_results_dir, keywords="_results", data_types="pkl")
        methods = [filename.split("_results.pkl")[0] for filename in filenames]
        for method in methods:
            method_results = self._load_results(method_names=[method]).get(method, {})
            if not method_results:
                continue
            for hparam_id in method_results.keys():
                for task in method_results[hparam_id].keys():
                    if task == "hparams":
                        continue
                    for window_size in method_results[hparam_id][task]["window_sizes"]:
                        for quantile in method_results[hparam_id][task]["quantiles"]:
                            for threshold_style in method_results[hparam_id][task]["test_metrics"].keys():
                                metrics = method_results[hparam_id][task]["test_metrics"][threshold_style][quantile][
                                    window_size
                                ]
                                row = {
                                    "Method": method,
                                    "Task": task,
                                    "TWA": float(metrics["timestep_wise_accuracy"]),
                                    "Accuracy": float(metrics["balanced_accuracy"]),
                                    "Det. Time": float(metrics["avg_detection_time"]),
                                    "HID": int(hparam_id),
                                    "Window": str(window_size),
                                    "Quantile": float(quantile),
                                    "TPR": float(metrics["TPR"]),
                                    "TNR": float(metrics["TNR"]),
                                    "Threshold": threshold_style,
                                    "M": float(0.7 * metrics["balanced_accuracy"] + 0.3 * (1 - metrics["avg_detection_time"])),
                                    "AUROC": float(metrics.get("auroc", float("nan"))),
                                    # Same ranking, mean instead of max over the rollout's steps.
                                    "AUROC (mean)": float(metrics.get("auroc_mean", float("nan"))),
                                    # Mean of the AUROCs from the max score within the first
                                    # 25/50/75/100 % of each rollout (per-horizon values in the pkl).
                                    "AUROC (horizon)": float(metrics.get("auroc_horizon_mean", float("nan"))),
                                    # Max-aggregation AUROC within each rollout subtype.
                                    "ID AUROC": float(metrics.get("auroc_id", float("nan"))),
                                    "OOD AUROC": float(metrics.get("auroc_ood", float("nan"))),
                                }

                                metrics_by_type = metrics.get("metrics_by_type", {})

                                id_metrics = metrics_by_type.get("id", None)
                                if id_metrics is not None:
                                    row.update({
                                        "ID TWA": float(id_metrics["timestep_wise_accuracy"]),
                                        "ID Accuracy": float(id_metrics["balanced_accuracy"]),
                                        "ID Det. Time": float(id_metrics["avg_detection_time"]),
                                        "ID TPR": float(id_metrics["TPR"]),
                                        "ID TNR": float(id_metrics["TNR"]),
                                    })

                                ood_metrics = metrics_by_type.get("ood", None)
                                if ood_metrics is not None:
                                    row.update({
                                        "OOD TWA": float(ood_metrics["timestep_wise_accuracy"]),
                                        "OOD Accuracy": float(ood_metrics["balanced_accuracy"]),
                                        "OOD Det. Time": float(ood_metrics["avg_detection_time"]),
                                        "OOD TPR": float(ood_metrics["TPR"]),
                                        "OOD TNR": float(ood_metrics["TNR"]),
                                    })


                                if "avg_episode_stats_by_type" in metrics:
                                    for key, value in metrics["avg_episode_stats_by_type"].items():
                                        if key == "id_success":
                                            keyword = "ID S"
                                        elif key == "ood_success":
                                            keyword = "OOD S"
                                        elif key == "id_failure":
                                            keyword = "ID F"
                                        elif key == "ood_failure":
                                            keyword = "OOD F"
                                        else:
                                            continue

                                        row[keyword] = float(value["mean"])
                                        # row[keyword + " STD"] = value["std"]
                                        # for k, v in value.get("percentiles", {}).items():
                                        #     row[keyword + " P" + str(k)] = v

                                complete_df = pd.concat([complete_df, pd.DataFrame([row])], ignore_index=True)

        complete_df = complete_df.round(3)
        # complete_df = complete_df.sort_values(by=["TWA", "Accuracy", "Det. Time"], ascending=[False, False, True])
        return complete_df

    def combine_results(self, new_results: dict = {}, method_names: list = [], **kwargs) -> None:
        """Combine the new results with the existing results."""
        if not new_results or not method_names:
            return

        if not isinstance(new_results, dict):
            raise ValueError("New results must be a dictionary.")

        cfg = self.cfg.copy()
        cfg.update(kwargs)

        for method_name in method_names:
            new_method_results = self._convert_new_results(new_results, [method_name]).get(method_name, {})
            if cfg.overwrite_data:
                old_method_results = {}
            else:
                old_method_results = self._load_results([method_name]).get(method_name, {})

            combined_method_results = old_method_results.copy()

            for new_hid in new_method_results.keys():
                if not old_method_results:
                    combined_method_results = {new_hid: new_method_results[new_hid]}
                else:
                    match_found = False
                    new_hyperparamter_dict = new_method_results[new_hid]["hparams"]
                    for old_hid in old_method_results.keys():
                        old_hyperparamter_dict = old_method_results[old_hid]["hparams"]
                        if compare_dicts(new_hyperparamter_dict, old_hyperparamter_dict):
                            combined_method_results[old_hid] = self._update_dictionary_recursively(
                                combined_method_results[old_hid], new_method_results[new_hid]
                            )
                            match_found = True
                            break

                    if not match_found:
                        combined_method_results[max(combined_method_results.keys(), default=-1) + 1] = new_method_results[
                            new_hid
                        ]
            # Save the combined results for the method
            self._save_results({method_name: combined_method_results}, method_names=[method_name])


        complete_df: pd.DataFrame = self._create_complete_df()
        self._save_dataframe(complete_df, filename="complete_results")

    def create_summary(self, **kwargs):
        """Create a summary of the results.

        The summary is saved as "{base_data_path}/results/{summary_name}.csv".

        Args:
            new_results (dict): The results of the current evaluation. Only used if only_latest is set to True.
            kwargs (dict): Additional arguments to control the summary generation. Overrides the config file.
        """
        cfg = self.cfg.copy()
        cfg.update(kwargs)
        if not cfg.create_summary:
            return
        # Load the complete DataFrame
        summary_df = self._load_dataframe()

        cfg.metric_columns = OmegaConf.to_container(cfg.metric_columns, resolve=True)
        # metric_columns: dict = dict(metric_columns)
        # Filter the DataFrame based on the filter columns and values
        summary_df = self._filter_dataframe(
            df=summary_df,
            cfg=cfg,
        )

        # Average the DataFrame over the specified columns
        summary_df = self._average_dataframe(
            df=summary_df,
            cfg=cfg,
        )

        summary_df = self._sort_and_reorder_dataframe(
            df=summary_df,
            cfg=cfg,
        )
        # Print the summary DataFrame
        print(summary_df)
        # Print the hyperparameters if desired
        hp_df_save_dict = {}
        if "Method" in summary_df.columns and "HID" in summary_df.columns:
            if cfg.print_hyperparameters or cfg.save_hyperparameters:
                for method in summary_df["Method"].unique():
                    reduced_df = summary_df[summary_df["Method"] == method]
                    # All hyperparameter IDs
                    hparam_ids = [int(hparam_id) for hparam_id in reduced_df["HID"].unique()]
                    # Calculate the best hyperparameter ID
                    best_hparam_id = self._get_best_hyperparameter_id(reduced_df, cfg)

                    # Create a DataFrame for the hyperparameters
                    hp_df, hp_dict = self._create_hyperparameter_dataframe(method, hparam_ids, cfg)

                    # Differences between the hyperparameters
                    hp_df_diff = self._get_hyperparameter_differences(hp_df)

                    # Print and save the hyperparameters
                    hp_df_save_dict[method] = self._output_hyperparameters(
                        method=method,
                        best_hparam_id=best_hparam_id,
                        hp_df=hp_df,
                        hp_df_diff=hp_df_diff,
                        hp_dict=hp_dict,
                        hparam_ids=hparam_ids,
                        cfg=cfg,
                    )

        # Save the summary DataFrame to a CSV file

        # Get existing summaries and hparam summaries
        filenames = _get_filenames(self.summary_dir, keywords="summary", data_types="csv")
        # Filter out the hparams
        filenames_s = [filename for filename in filenames if "hparams" not in filename]
        # Remove complete_summary
        # Determine new filename
        if len(filenames_s) == 0:
            new_filename = "summary_00.csv"
        else:
            overwrite = cfg.overwrite_summary
            if overwrite == "latest":
                new_filename = filenames_s[-1]
            elif overwrite == "all":
                for filename in filenames:
                    os.remove(os.path.join(self.summary_dir, filename))
                new_filename = "summary_00.csv"
            elif overwrite == "oldest":
                new_filename = filenames_s[0]
            else:
                # Get the latest summary file
                latest_filename = sorted(filenames_s)[-1]
                # Extract the number from the filename
                number = int(latest_filename.split("_")[1].split(".")[0])
                new_filename = f"summary_{number + 1:02d}.csv"
        self._save_summary(summary_df, filename=new_filename)
        if hp_df_save_dict:
            for key, value in hp_df_save_dict.items():
                if not value:
                    continue
                hid_filename = new_filename.replace(".csv", "_" + key + "_hparams.csv")
                self._save_summary(value, filename=hid_filename)

    def _update_dictionary_recursively(self, dict1: dict, dict2: dict):
        """Recursively update dict1 with dict2."""
        for key, value in dict2.items():
            if isinstance(value, dict) and key in dict1:
                self._update_dictionary_recursively(dict1[key], value)
            else:
                if key in ["window_sizes", "quantiles"] and key in dict1:
                    # If the key is "window_sizes" or "quantiles", append the new values to the existing list
                    dict1[key] = list(set(dict1[key] + value))
                else:
                    dict1[key] = value
        return dict1

    def _get_hyperparameters(self, method, hparam_id, resolve=True, include_hparams: list = []) -> dict:
        """Get the hyperparameters of the method."""
        complete_results = self._load_results()
        method_results = complete_results[method]
        if hparam_id not in method_results:
            raise ValueError(f"Hyperparameter ID {hparam_id} not found for method {method}.")
        hparams: DictConfig = method_results[hparam_id].get("hparams", {})
        if not hparams:
            raise ValueError(f"Hyperparameters not found for method {method} and hyperparameter ID {hparam_id}.")

        if resolve:
            hparams = OmegaConf.to_container(hparams, resolve=True)
        if include_hparams:
            hparams = {key: value for key, value in hparams.items() if key in include_hparams}
        return hparams

    def _remove_hparam_ids_from_results(self, methods: list, hparam_ids: list) -> None:
        """Remove a hyperparameter ID for a method from the results."""
        methods, hparam_ids = ensure_list(methods, hparam_ids)
        assert len(methods) == len(hparam_ids), "Methods and hyperparameter IDs must have the same length."
        combined_results: dict = self._load_results()
        for method, hparam_id in zip(methods, hparam_ids):
            if method not in combined_results:
                continue
            method_results: dict = combined_results[method]
            if hparam_id not in method_results:
                continue
            method_results.pop(hparam_id)
        # Save the combined results
        self._save_results(combined_results)
        # Recreate the complete DataFrame
        complete_df: pd.DataFrame = self._create_complete_df(combined_results)
        self._save_dataframe(complete_df, filename="complete_results")

    def _clean_results(self, clean_dict: dict) -> None:
        """Clean the results of a method by removing specified hyperparameter IDs.

        Args:
            clean_dict (dict): A dictionary with the following structure:
                {method_name: {keyword: keyword, value: value}}, where key is in "del", "keep", "keep_best", "del_worst" and value is a list of hyperparameter IDs or a integer.
        """
        assert isinstance(clean_dict, dict), "clean_dict must be a dictionary."
        methods_to_clean = []
        hparam_ids_to_clean = []
        for method in clean_dict:
            if not clean_dict[method] or clean_dict[method] is None:
                continue
            if not isinstance(clean_dict[method], dict):
                continue
            if "keyword" not in clean_dict[method] or "value" not in clean_dict[method]:
                continue
            keyword = clean_dict["keyword"]
            value = clean_dict["value"]
            assert keyword in ["del", "keep", "keep_best", "del_worst"], (
                f"Invalid keyword: {keyword}. Must be one of ['del', 'keep', 'keep_best', 'del_worst']."
            )
            assert isinstance(value, list) or isinstance(value, int), "value must be a list or an integer."
            if isinstance(value, int):
                value = [value]
            if keyword == "del":
                hparam_ids_to_clean.extend(value)
                methods_to_clean.extend([method for _ in range(len(value))])
            elif keyword == "del_worst":
                value = value[0]

        raise NotImplementedError

    def _get_differences_dict(self, dict1, dict2):
        """Get the differences between two dictionaries."""
        differences = {}
        for key in dict1.keys():
            if key not in dict2:
                differences[key] = dict1[key]
            elif isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
                nested_differences = self._get_differences_dict(dict1[key], dict2[key])
                if nested_differences:
                    differences[key] = nested_differences
            elif dict1[key] != dict2[key]:
                differences[key] = (dict1[key], dict2[key])
        return differences

    def generate_uncertainty_plots(self, **kwargs) -> None:
        """Generate uncertainty plots for methods and parameters defined in the configuration optionally overwritten by kwargs.

        Args:
            kwargs (dict): Additional arguments to control the plot generation. Overrides the config file.
        """
        # Update and unpack the config
        cfg = self.cfg.copy()
        OmegaConf.set_struct(cfg, False)
        cfg.update(kwargs)
        cfg.update(cfg.uncertainty_plots)
        OmegaConf.set_struct(cfg, True)
        if not cfg.create_plots:
            return
        cfg.metric_columns = OmegaConf.to_container(cfg.metric_columns, resolve=True)
        cfg.filter_values = OmegaConf.to_container(cfg.filter_values, resolve=True)
        # Load the results
        results = self._load_results()
        df = self._load_dataframe()

        methods = list(results.keys())
        if cfg.filter_values.Method is not None:
            methods = [method for method in methods if method in cfg.filter_values.Method]
        # Option to exclude all windows
        if cfg.exclude_all:
            df = df[~df["Window"].astype(str).str.contains("all", case=False, na=False)]

        for method_name in methods:
            if method_name not in results:
                continue

            # Get the results for the method
            method_results = results[method_name]
            # Filter the DataFrame for the method
            df_method = self._filter_dataframe(df=df[df["Method"].isin([method_name])], cfg=cfg)
            # Get available tasks
            tasks = [key for key in df_method["Task"].unique()]
            tasks = list(set(tasks))
            # Extract the values of the first row
            best_values = {}
            for task in tasks:
                best_values[task] = dict(df_method[df_method["Task"] == task].iloc[0])

            # Extract the uncertainty scores for each task, normalizing them to the treshold
            uncertainty_scores_by_task = {}

            for task in tasks:
                # Get the results for the task and the best values
                uncertainty_scores_by_task[task] = self._extract_uncertainty_scores(
                    task_results=method_results[best_values[task]["HID"]][task],
                    best_values=best_values[task],
                    show_scores=cfg.show,
                    combine_scores=cfg.combine_scores,
                )

            # Create individual plots for each method-task combination
            for task in tasks:
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.set_title(
                    f"Normalized Scores for Method: {cfg.method_name_mapping[method_name]}, Threshold Type: {best_values[task]['Threshold']}, Task: {task}",
                    fontsize=cfg.font_sizes.title,
                )
                ax.set_xlabel("Steps", fontsize=cfg.font_sizes.label)
                ax.set_ylabel("Uncertainty Score", fontsize=cfg.font_sizes.label)

                uncertainty_data = uncertainty_scores_by_task[task]
                for key, values in uncertainty_data.items():
                    if isinstance(values, (list, np.ndarray)):
                        max_length = len(values)
                        ax.plot(range(max_length), values, label=key)
                    elif isinstance(values, (int, float)):
                        ax.axhline(y=values, linestyle="--", label=key)

                ax.legend(fontsize=cfg.font_sizes.legend)
                # ax.set_ybound(lower=0.0, upper=3.0)
                ax.grid(True, linestyle="--", alpha=0.7)

                # Adjust layout and save the figure
                plt.tight_layout()
                save_dir = os.path.join(self.results_dir, "uncertainty_plots", method_name)
                os.makedirs(save_dir, exist_ok=True)
                plot_filename = os.path.join(save_dir, f"{task}_scores.pdf")
                fig.savefig(plot_filename, dpi=300, format="pdf")
                plt.close(fig)

    def _filter_dataframe(self, df, cfg):
        """Filter the DataFrame based on the configuration."""
        metric_columns = cfg.metric_columns
        metric_columns = {col: metric_columns[col] for col in metric_columns if col in df.columns}
        # Remove specified columns
        if cfg.filter_columns:
            df = df.drop(columns=[col for col in cfg.filter_columns if col in df.columns])
            metric_columns = {col: metric_columns[col] for col in metric_columns if col not in cfg.filter_columns}
        cfg.metric_columns = metric_columns

        filter_values = cfg.filter_values
        for column, values in filter_values.items():
            if column not in df.columns or not values:
                continue
            # Handle "best" keyword
            if "best" in values:
                best_position = values.index("best")
                if best_position != 0:
                    values = values[:best_position]
                    if isinstance(values[0], str) and values[0].startswith("!"):
                        values = [value.replace("!", "") for value in values]
                        df = df[~df[column].isin(values)]
                    else:
                        df = df[df[column].isin(values)]

                if not metric_columns:
                    continue

                # Group by specified columns and compute averages
                group_by_columns = [col for col in df.columns if col in cfg.groups_for_best or col == column]
                agg_dict = {col: "mean" for col in metric_columns.keys()}  # Compute the average of metric columns

                grouped_df = (
                    df.groupby(group_by_columns, as_index=False)
                    .agg(agg_dict)
                    .round(3)
                    .sort_values(by=list(metric_columns.keys()), ascending=list(metric_columns.values()))
                    .reset_index(drop=True)
                )

                # Identify non-grouped columns
                non_grouped_columns = [col for col in group_by_columns if col != column]

                # Iterate over all unique combinations of non-grouped columns
                filtered_dfs = []
                for combination in grouped_df[non_grouped_columns].drop_duplicates().to_dict(orient="records"):
                    # Filter the grouped DataFrame for the current combination
                    combination_filter = (grouped_df[list(combination)] == pd.Series(combination)).all(axis=1)
                    filtered_combination_df = grouped_df[combination_filter]

                    # Select the best value for the current combination
                    best_value = filtered_combination_df.iloc[0][column]
                    filtered_df = df[
                        (df[list(combination)] == pd.Series(combination)).all(axis=1) & df[column].isin([best_value])
                    ]
                    filtered_dfs.append(filtered_df)

                # Concatenate all filtered DataFrames
                if filtered_dfs:
                    df = pd.concat(filtered_dfs, ignore_index=True)

                if cfg.drop_after_best:
                    # Remove the column after processing "best"
                    df = df.drop(columns=[column])
            else:
                # Handle negated values (!value)
                if isinstance(values[0], str) and values[0].startswith("!"):
                    values = [value.replace("!", "") for value in values]
                    df = df[~df[column].isin(values)]
                else:
                    df = df[df[column].isin(values)]
        return df

    def _sort_and_reorder_dataframe(self, df, cfg):
        """Sort and reorder the DataFrame columns."""
        if cfg.column_order:
            column_order = [col for col in cfg.column_order if col in df.columns]
            column_order += [col for col in df.columns if col not in column_order]
            df = df[column_order]
        if cfg.sorting:
            sort_by = list(cfg.sorting.keys())
            sort_by = [col for col in sort_by if col in df.columns]
            orders = [cfg.sorting[col] for col in sort_by]
            df = df.sort_values(by=sort_by, ascending=orders)
        return df

    def _average_dataframe(self, df, cfg):
        """Average the DataFrame over specified columns."""
        if not cfg.average_columns:
            return df
        average_columns = [col for col in cfg.average_columns if col in df.columns]
        remaining_columns = [
            col for col in df.columns if col not in average_columns and col not in cfg.metric_columns.keys()
        ]
        agg_dict = {col: "mean" for col in cfg.metric_columns.keys()}
        return df.groupby(remaining_columns, as_index=False).agg(agg_dict).round(3)

    def _get_best_hyperparameter_id(self, reduced_df: pd.DataFrame, cfg) -> list:
        """
        Calculate the best hyperparameter ID based on metrics.

        Args:
            reduced_df (pd.DataFrame): The DataFrame filtered for a specific method.
            cfg: The configuration object.

        Returns:
            list: The best hyperparameter ID(s).
        """
        metric_columns = cfg.metric_columns
        remaining_columns = [col for col in reduced_df.columns if col not in metric_columns.keys() and col != "Task"]
        agg_dict = {col: "mean" for col in metric_columns.keys()}
        reduced_df = reduced_df.groupby(remaining_columns, as_index=False).agg(agg_dict)
        return reduced_df["HID"].iloc[0]

    def _create_hyperparameter_dataframe(self, method: str, hparam_ids: list, cfg) -> tuple:
        """
        Create a DataFrame and dictionary for hyperparameters.

        Args:
            method (str): The method name.
            hparam_ids (list): List of hyperparameter IDs.
            cfg: The configuration object.

        Returns:
            tuple: A tuple containing the hyperparameter DataFrame and dictionary.
        """
        hp_df = pd.DataFrame()
        hp_dict = {}

        for hparam_id in hparam_ids:
            hp = self._get_hyperparameters(method, hparam_id, include_hparams=cfg.include_hparams)
            hp_dict[hparam_id] = hp
            hp["HID"] = hparam_id
            hp["Method"] = method
            hp_df_new = pd.json_normalize(hp)

            # Drop columns with list values or None
            for col in hp_df_new.columns:
                value = hp_df_new[col].iloc[0]
                if isinstance(value, list) or value is None:
                    hp_df_new = hp_df_new.drop(columns=col)

            hp_df = pd.concat([hp_df, hp_df_new], ignore_index=True)

        return hp_df, hp_dict

    def _get_hyperparameter_differences(self, hp_df: pd.DataFrame) -> pd.DataFrame:
        """
        Identify differences between hyperparameters.

        Args:
            hp_df (pd.DataFrame): The hyperparameter DataFrame.

        Returns:
            pd.DataFrame: A DataFrame containing only the differing hyperparameters.
        """
        hp_df_diff = hp_df.copy()
        for col in hp_df_diff.columns:
            if len(hp_df_diff[col].unique()) == 1:
                hp_df_diff = hp_df_diff.drop(columns=col)
        check_cols = [col for col in hp_df_diff.columns if col not in ["HID", "index", "Method"]]
        return hp_df_diff.drop_duplicates(subset=check_cols, keep="first")

    def _output_hyperparameters(
        self, method: str, best_hparam_id, hp_df: pd.DataFrame, hp_df_diff, hp_dict, hparam_ids, cfg
    ) -> pd.DataFrame:
        if cfg.print_hyperparameters:
            print_hparams = cfg.print_hparams
            if print_hparams.only_best:
                print(f"Best Hyperparameters for {method} with HID {best_hparam_id}:")
                print(OmegaConf.to_yaml(hp_dict[best_hparam_id]))
            elif print_hparams.only_differences:
                print(f"Hyperparameter differences for {method}:")
                print(hp_df_diff)
            else:
                for hparam_id in hparam_ids:
                    print(f"Hyperparameters for {method} with HID {hparam_id}:")
                    print(OmegaConf.to_yaml(hp_dict[hparam_id]))

        if cfg.save_hyperparameters:
            save_hparams = cfg.save_hparams
            if save_hparams.only_best:
                return hp_df[hp_df["HID"].isin(best_hparam_id)]
            elif save_hparams.only_differences and not hp_df_diff.empty:
                return hp_df_diff
            else:
                return hp_df

    def _extract_uncertainty_scores(
        self,
        task_results: dict,
        best_values: dict,
        show_scores: dict,
        combine_scores: str = "median",
    ) -> dict:
        def convert_scores(scores_by_threshold: dict, max_episode_length: int) -> np.ndarray:
            # Transpose the combined scores to have a list of lists (steps as rows)
            scores = [[] for _ in range(max_episode_length)]
            for arr in scores_by_threshold:
                for i, value in enumerate(arr):
                    scores[i].append(value)

            # Apply the specified aggregation method (e.g., mean or median)
            aggregated_scores = []
            for i in range(len(scores)):
                if len(scores[i]) == 0:
                    continue
                if combine_scores == "mean":
                    new_score = np.mean(scores[i])
                elif combine_scores == "median":
                    new_score = np.median(scores[i])
                else:
                    raise ValueError(f"Unknown combine scores method: {combine_scores}")
                aggregated_scores.append(new_score)
            return np.array(aggregated_scores)

        best_window = best_values["Window"]
        if isinstance(best_window, str) and best_window.isdigit():
            best_window = int(best_window)
        data = {
            "successful_test_rollouts": task_results["successful_test_rollouts"],
            "ood_test_rollouts": task_results["ood_test_rollouts"],
            "id_test_rollouts": task_results["id_test_rollouts"],
            "max_episode_length": task_results["max_episode_length"],
            "scores_by_threshold": task_results["test_scores_by_threshold"][best_values["Threshold"]][
                best_values["Quantile"]
            ][best_window],
        }
        data_new = {}
        for key in show_scores.test.keys():
            if not show_scores.test[key]:
                continue
            mask = np.ones(len(data["successful_test_rollouts"]), dtype=bool)
            mask = mask & data["successful_test_rollouts"] if "success" in key else mask
            mask = mask & data["id_test_rollouts"] if "id_" in key else mask
            mask = mask & ~data["successful_test_rollouts"] if "fail" in key else mask
            mask = mask & data["ood_test_rollouts"] if "ood_" in key else mask

            if sum(mask) == 0:
                continue

            scores_by_threshold = [score for i, score in enumerate(data["scores_by_threshold"]) if mask[i]]

            data_new["test_" + key] = convert_scores(
                scores_by_threshold,
                data["max_episode_length"],
            )
        if show_scores.calibration and "calibration_scores_by_threshold" in data:
            scores_by_threshold = task_results["calibration_scores_by_threshold"][best_values["Threshold"]][
                best_values["Quantile"]
            ][best_window]
            data_new["calibration"] = convert_scores(
                scores_by_threshold,
                data["max_episode_length"],
            )

        # data_new["threshold"] = data["calibration_thresholds"] if not normalize else 1
        if show_scores.threshold:
            data_new["threshold"] = 1

        return data_new

    def plot_quantile_threshold_impact(self, **kwargs):
        """Plot the impact of quantiles and threshold styles on the results by plotting a metric for the threshold styles, averaged over the tasks for the best window size, over the quantiles."""
        # Update and unpack the config
        cfg = self.cfg.copy()
        OmegaConf.set_struct(cfg, False)
        cfg.update(kwargs)
        cfg.update(cfg.quantile_impact)
        OmegaConf.set_struct(cfg, True)

        if not cfg.create_plots:
            return

        # Load the DataFrame
        df = self._load_dataframe()
        # Remove the "all" window size if exclude_all is set
        if cfg.exclude_all:
            df = df[~df["Window"].astype(str).str.contains("all", case=False, na=False)]

        # Remove unneeded columns
        keep_columns = ["Quantile", "Method", "Threshold", "Window", "Task", "HID"] + list(
            set(cfg.metrics_to_plot + list(cfg.metric_columns.keys()))
        )

        keep_columns = [col for col in keep_columns if col in df.columns]
        df = df[keep_columns]
        cfg.filter_columns = []

        # Filter the DataFrame based on the configuration
        df = self._filter_dataframe(df=df, cfg=cfg)
        # Average over tasks
        df = self._average_dataframe(df=df, cfg=cfg)

        # Ensure the metric column exists in the DataFrame
        metric_columns = cfg.metrics_to_plot
        if any(col not in df.columns for col in metric_columns):
            raise ValueError(f"Not all metric columns to plot '{metric_columns}' found in the DataFrame.")

        # Group by quantiles and calculate the average metric
        df = (
            df.groupby(["Quantile", "Method", "Threshold"], as_index=False)
            .agg({k: "mean" for k in metric_columns})
            .sort_values(by="Quantile")
        )
        # Dataframe should only contain the quantile, threshold, method, and the chosen metric columns
        df = df[["Quantile", "Method", "Threshold"] + metric_columns]

        # Determine whether to create subplots or a single plot
        num_thresholds = len(df["Threshold"].unique())
        create_subplots = num_thresholds * len(metric_columns) > 6

        # Plot the quantile impact for each method
        for method in df["Method"].unique():
            method_df = df[df["Method"] == method]
            save_dir = os.path.join(self.results_dir, "quantile_plots", method)
            os.makedirs(save_dir, exist_ok=True)

            if create_subplots:
                # Create individual plots for each metric
                for metric_column in metric_columns:
                    plt.figure(figsize=(10, 6))
                    for threshold in method_df["Threshold"].unique():
                        threshold_df = method_df[method_df["Threshold"] == threshold]
                        plt.plot(
                            threshold_df["Quantile"],
                            threshold_df[metric_column],
                            marker="o",
                            label=f"Threshold: {threshold}",
                        )
                    plt.title(f"Impact of Quantiles on {metric_column} for {cfg.method_name_mapping[method]}", fontsize=cfg.font_sizes.title)
                    plt.xlabel("Quantile", fontsize=cfg.font_sizes.label)
                    plt.ylabel(metric_column, fontsize=cfg.font_sizes.label)
                    plt.grid(True, linestyle="--", alpha=0.7)
                    plt.legend(fontsize=cfg.font_sizes.legend)

                    # Save the plot
                    plot_filename = os.path.join(save_dir, f"{metric_column}.pdf")
                    plt.tight_layout()
                    plt.savefig(plot_filename, dpi=300, format="pdf")
                    plt.close()
            else:
                # Create a single plot with multiple lines
                plt.figure(figsize=(10, 6))
                for metric_column in metric_columns:
                    for threshold in method_df["Threshold"].unique():
                        threshold_df = method_df[method_df["Threshold"] == threshold]
                        plt.plot(
                            threshold_df["Quantile"],
                            threshold_df[metric_column],
                            marker="o",
                            label=f"{metric_column} (Threshold: {threshold})",
                        )
                plt.title(f"Impact of Quantiles for {cfg.method_name_mapping[method]}", fontsize=cfg.font_sizes.title)
                plt.xlabel("Quantile", fontsize=cfg.font_sizes.label)
                plt.ylabel("Metrics", fontsize=cfg.font_sizes.label)
                plt.grid(True, linestyle="--", alpha=0.7)
                plt.legend(fontsize=cfg.font_sizes.legend)

                # Save the plot
                filename = ""
                for metric_column in metric_columns:
                    filename += f"{metric_column}_"
                filename = filename[:-1]  # Remove the last underscore
                plot_filename = os.path.join(save_dir, f"{filename}.pdf")
                plt.tight_layout()
                plt.savefig(plot_filename, dpi=300, format="pdf")
                plt.close()

    def plot_window_impact(self, **kwargs):
        """Plot the impact of window sizes on the results by plotting a metric, averaged over the tasks for the best or averaged quantiles, over the window sizes."""
        # Update and unpack the config
        cfg = self.cfg.copy()
        OmegaConf.set_struct(cfg, False)
        cfg.update(kwargs)
        cfg.update(cfg.window_impact)
        cfg.exclude_all = True  # Always exclude all
        OmegaConf.set_struct(cfg, True)

        if not cfg.create_plots:
            return

        # Load the DataFrame
        df = self._load_dataframe()
        # Remove the "all" window size if exclude_all is set
        if cfg.exclude_all:
            df = df[~df["Window"].astype(str).str.contains("all", case=False, na=False)]
        # Remove the metrics in the metric_columns that are not metric_to_plot
        filter_columns = [col for col in cfg.metric_columns if col != cfg.metric_to_plot]
        if cfg.metric_to_plot in cfg.filter_columns:
            cfg.filter_columns.remove(cfg.metric_to_plot)
        cfg.filter_columns = list(set(cfg.filter_columns + filter_columns))

        # Filter the DataFrame based on the configuration
        df = self._filter_dataframe(df=df, cfg=cfg)
        # Average over tasks
        df = self._average_dataframe(df=df, cfg=cfg)

        # Ensure the metric column exists in the DataFrame
        metric_column = cfg.metric_to_plot
        if metric_column not in df.columns:
            raise ValueError(f"Metric column '{metric_column}' not found in the DataFrame.")

        # Dataframe should only contain the quantile, method, and the chosen metric columns
        df = df[["Window", "Method", metric_column]]

        # Group by quantiles and calculate the average metric
        window_impact_df = (
            df.groupby(["Window", "Method"], as_index=False).agg({metric_column: "mean"}).sort_values(by="Window")
        )

        # Plot the quantile impact for each method
        for method in window_impact_df["Method"].unique():
            method_df = window_impact_df[window_impact_df["Method"] == method]
            windows = list(method_df["Window"])
            values = list(method_df[metric_column])
            # Convert the window sizes to float
            windows_new = []
            values_new = []
            for window, value in zip(windows, values):
                if isinstance(window, str) and window.isdigit():
                    windows_new.append(float(window))
                    values_new.append(value)
                elif isinstance(window, str) and "/" in window:
                    windows_new.append((float(window.split("/")[0]) + float(window.split("/")[1])) / 2)
                    values_new.append(value)
                elif isinstance(window, (int, float)):
                    windows_new.append(float(window))
                    values_new.append(value)
                else:
                    continue
            # Calculate bar width based on the spacing between windows
            # if len(windows_new) > 1:
            #     bar_width = min(
            #         abs(windows_new[i + 1] - windows_new[i]) for i in range(len(windows_new) - 1)
            #     ) * 0.8  # Adjust factor for spacing
            # else:
            #     bar_width = 0.5  # Default width for a single bar

            plt.figure(figsize=(10, 6))
            plt.bar(
                windows_new,
                values_new,
                # width=bar_width,
                alpha=0.8,
                label=f"Average {metric_column} for {method}",
            )
            plt.title(f"Impact of Window Size on {metric_column} for {cfg.method_name_mapping[method]}", fontsize=cfg.font_sizes.title)
            plt.xlabel("Window Size", fontsize=cfg.font_sizes.label)
            plt.ylabel(metric_column, fontsize=cfg.font_sizes.label)
            plt.ylim(bottom=0.0, top=1.0)
            plt.grid(True, linestyle="--", alpha=0.7)
            plt.xticks(windows_new)  # Ensure x-axis ticks match the bar positions
            plt.legend(fontsize=cfg.font_sizes.legend)

            # Save the plot
            save_dir = os.path.join(self.results_dir, "window_plots", method)
            os.makedirs(save_dir, exist_ok=True)
            plot_filename = os.path.join(save_dir, f"{metric_column}.pdf")
            plt.tight_layout()
            plt.savefig(plot_filename, dpi=300, format="pdf")
            plt.close()

    def plot_rollout_type_stats(self, **kwargs):
        """Plot the average scores by rollout type for each method."""

        # Update and unpack the config
        cfg = self.cfg.copy()
        OmegaConf.set_struct(cfg, False)
        cfg.update(kwargs)
        cfg.update(cfg.rollout_type_stats)
        OmegaConf.set_struct(cfg, True)
        # Set drop_after_best to True
        cfg.drop_after_best = False

        if not cfg.create_plots:
            return

        # Load the DataFrame
        df = self._load_dataframe()
        # Remove the "all" window size if exclude_all is set
        if cfg.exclude_all:
            df = df[~df["Window"].astype(str).str.contains("all", case=False, na=False)]

        # Filter and process the DataFrame
        df = self._filter_dataframe(df=df, cfg=cfg)
        df = self._average_dataframe(df=df, cfg=cfg)
        df = self._sort_and_reorder_dataframe(df=df, cfg=cfg)

        # Load the results
        results = self._load_results()

        def extract_scores(scores: list, mask: np.ndarray) -> list:
            """Extract scores based on the mask and convert to numpy array."""
            if sum(mask) == 0:
                return np.array([])
            return np.hstack([score for i, score in enumerate(scores) if mask[i]])

        for method in df["Method"].unique():
            save_dir = os.path.join(self.results_dir, "rollout_type_stats", method)
            os.makedirs(save_dir, exist_ok=True)

            # Get the data for the method (first row)
            first_row = df[df["Method"] == method].iloc[0]
            results_data = results[method][first_row["HID"]]
            task_keys = [key for key in results_data.keys() if key != "hparams"]
            task_keys = (
                [key for key in task_keys if key in cfg.filter_values.Task]
                if cfg.filter_values.Task is not None and cfg.filter_values.Task
                else task_keys
            )
            # print(f"Method: {method}, Tasks: {task_keys}")
            window = first_row["Window"]
            if isinstance(window, str) and window.isdigit():
                window = int(window)

            # Initialize arrays for scores and rollout masks
            test_scores_by_threshold = []
            successful_test_rollouts = []
            ood_test_rollouts = []
            id_test_rollouts = []

            # Extract data for all tasks
            for task in task_keys:
                task_data = results_data[task]
                test_scores_by_threshold.extend(
                    task_data["test_scores_by_threshold"][first_row["Threshold"]][float(first_row["Quantile"])][window]
                )
                successful_test_rollouts.extend(task_data["successful_test_rollouts"])
                ood_test_rollouts.extend(task_data["ood_test_rollouts"])
                id_test_rollouts.extend(task_data["id_test_rollouts"])

            # Convert to numpy arrays and stack
            successful_test_rollouts = np.hstack(successful_test_rollouts)
            ood_test_rollouts = np.hstack(ood_test_rollouts)
            id_test_rollouts = np.hstack(id_test_rollouts)
            # test_scores_by_threshold = np.hstack(test_scores_by_threshold)

            # Extract scores for each rollout type
            rollout_masks = {
                "Success ID": id_test_rollouts & successful_test_rollouts,
                "Success OOD": ood_test_rollouts & successful_test_rollouts,
                "Fail ID": id_test_rollouts & ~successful_test_rollouts,
                "Fail OOD": ood_test_rollouts & ~successful_test_rollouts,
            }
            data = {key: extract_scores(test_scores_by_threshold, mask) for key, mask in rollout_masks.items()}

            ylabel = "Scores by Threshold"

            plotname = "violin_plot"

            # Filter outliers for each rollout type
            if cfg.filter_outliers:
                data = {key: filter_outliers(data[key]) for key in data.keys() if len(data[key]) > 0}

            if cfg.normalize_by_mean:
                # Normalize the scores by the mean of the "OOD F" scores
                ood_f_mean = np.mean(data["Fail OOD"]) if len(data["Fail OOD"]) > 0 else np.mean(data["Fail ID"])
                for key in data.keys():
                    if len(data[key]) > 0:
                        data[key] = data[key] / ood_f_mean

                ylabel = "Scores/Threshold (normalized)"
                plotname += "_norm"

            # Create the violin plot
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.set_title(f"{cfg.method_name_mapping[method]}", fontsize=cfg.font_sizes.title)
            ax.set_xlabel("Rollout Type", fontsize=cfg.font_sizes.label)
            ax.set_ylabel(ylabel, fontsize=cfg.font_sizes.label)
            ax.grid(True, linestyle="--", alpha=0.7)

            # Define quantiles for lower and upper limits
            min_quantile = 0.05
            max_quantile = 0.95

            # Compute min and max quantiles across all data keys
            min_q = min([np.quantile(data[key], min_quantile) for key in data.keys() if len(data[key]) > 0])
            max_q = max([np.quantile(data[key], max_quantile) for key in data.keys() if len(data[key]) > 0])

            # Set y-limits dynamically
            ax.set_ylim(
                bottom=min(0.0, min_q - 0.1),
                top=max(2.0, max_q + 0.1)
            )

            # Extract plots info
            info = cfg.annotations
            quantiles = []
            if info.quantiles.show:
                for quantile in info.quantiles.show:
                    quantiles.append(1 - quantile)
                    quantiles.append(quantile)

            colors = ["green", "red", "blue", "grey"]  # Define 
            # Create violin plots
            for i, (rollout_type, color) in enumerate(zip(data.keys(), colors)):
                if rollout_type not in data or len(data[rollout_type]) == 0:
                    continue
                scores = data[rollout_type]
                violin_parts = ax.violinplot(
                    scores,
                    positions=[i],
                    showmeans=info.mean.show,
                    showmedians=info.median.show,
                    showextrema=False,
                    quantiles=quantiles,
                )

                for pc in violin_parts['bodies']:
                    pc.set_facecolor(color)
                    pc.set_edgecolor(color)
                    pc.set_alpha(0.5)  # Adjust transparency if needed

                if "cmeans" in violin_parts:
                    violin_parts["cmeans"].set_color(color)  # Mean line
                if "cquantiles" in violin_parts:
                    violin_parts["cquantiles"].set_color(color)  # Quantile lines
                if "cbars" in violin_parts:
                    violin_parts["cbars"].set_color(color)  # Vertical bars

                # Add scatter points with jitter
                # jitter = np.random.uniform(-0.1, 0.1, size=len(scores))
                # ax.scatter(i + jitter, scores, alpha=0.5, color="black", s=5)

                if info.mean.text:
                    # Add mean value as text
                    ax.text(
                        i,
                        np.mean(scores),
                        f"Mean: {np.mean(scores):.2f}",
                        horizontalalignment="center",
                        verticalalignment="bottom",
                        fontsize=cfg.font_sizes.annotation,
                    )
                if info.median.text:
                    # Add median value as text
                    ax.text(
                        i,
                        np.median(scores),
                        f"Median: {np.median(scores):.2f}",
                        horizontalalignment="center",
                        verticalalignment="bottom",
                        fontsize=cfg.font_sizes.annotation,
                    )

            ax.set_xticks(range(len(data.keys())))
            ax.tick_params(axis='y', labelsize=cfg.font_sizes.ticks)
            ax.set_xticklabels(list(data.keys()), fontsize=cfg.font_sizes.label)

            # Adjust layout and save the figure
            plt.tight_layout()
            plot_filename = os.path.join(save_dir, plotname + ".pdf")
            fig.savefig(plot_filename, dpi=300, format="pdf")
            plt.close(fig)


# Function to filter outliers based on the IQR method
def filter_outliers(data, threshold=3):
    """Remove outliers from the data using the IQR method."""
    q1 = np.percentile(data, 10)  # First quartile (25th percentile)
    q3 = np.percentile(data, 98)  # Third quartile (75th percentile)
    iqr = q3 - q1  # Interquartile range
    # lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    upper_bound = max(upper_bound, 5.0)  # Set minimum upper bound to 5.0
    # print(f"Upper bound: {upper_bound}")
    return [x for x in data if 0.0 <= x <= upper_bound]
