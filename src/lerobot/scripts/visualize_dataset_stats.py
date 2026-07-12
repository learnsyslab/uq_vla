"""
Usage examples:

Visualize dataset statistics for selected LIBERO tasks:

```
python -m lerobot.scripts.visualize_dataset_stats \
  --repo-id HuggingFaceVLA/libero \
  --libero-tasks '{"libero_spatial":[],"libero_object":[],"libero_goal":[],"libero_10":[0,4,7,8,9]}'
```
"""


import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import filter_libero_episodes, patch_dataset_episode_boundaries


def compute_feature_histograms_for_episodes(
    dataset: LeRobotDataset,
    episode_ids: list[int],
    features: dict[str, dict[str, Any]] | None = None,
    bins: int = 100,
) -> dict[str, dict[str, np.ndarray]]:
    """
    Aggregate per-dimension histograms for numeric features (e.g., states/actions) over a subset of episodes.

    Args:
        dataset: A LeRobotDataset instance.
        episode_ids: Episode indices to include.
        features: Optional subset of dataset.meta.features to histogram. If None,
            all non-string/non-image/non-video features are used.
        bins: Number of histogram bins.
    """
    # Pick numeric features only by default
    if features is None:
        features = {
            k: spec for k, spec in dataset.meta.features.items()
            if spec.get("dtype") not in {"string", "image", "video"}
        }

    hf = dataset.hf_dataset

    # Compute min/max for each feature dim across selected episodes
    feature_mins, feature_maxs = {}, {}
    feature_dims = {}
    for ep_id in episode_ids:
        episode_rows = hf[dataset.meta.episodes["dataset_from_index"][ep_id]:dataset.meta.episodes["dataset_to_index"][ep_id]]
        for feature_key in features:
            feature_vals = episode_rows[feature_key]
            feature_vals = torch.stack(feature_vals).cpu().numpy()
            if feature_vals.ndim == 1:
                feature_vals = feature_vals[:, None]
            feature_dims[feature_key] = feature_vals.shape[1]
            cur_min = np.nanmin(feature_vals, axis=0)
            cur_max = np.nanmax(feature_vals, axis=0)
            if feature_key not in feature_mins:
                feature_mins[feature_key] = cur_min
                feature_maxs[feature_key] = cur_max
            else:
                feature_mins[feature_key] = np.minimum(feature_mins[feature_key], cur_min)
                feature_maxs[feature_key] = np.maximum(feature_maxs[feature_key], cur_max)

    # Prepare fixed bin edges per feature dim so we can aggregate counts
    result = {}
    for featue_key, dims in feature_dims.items():
        bin_edges = np.empty((dims, bins + 1))
        for d in range(dims):
            min, max = feature_mins[featue_key][d], feature_maxs[featue_key][d]
            if max == min:
                min, max = min - 1, max + 1
            bin_edges[d] = np.linspace(min, max, bins + 1)
        dim_names = dataset.meta.names.get(featue_key)
        if not (isinstance(dim_names, list) and len(dim_names) == dims):
            dim_names = [f"dim_{i}" for i in range(dims)]
        result[featue_key] = {
            "counts": np.zeros((dims, bins)),
            "bin_edges": bin_edges,
            "dim_names": dim_names,
        }

    # Accumulate hist counts per episode using fixed edges
    total_counts_per_feature = {}  # Track total counts per feature for normalization
    for ep_id in episode_ids:
        episode_rows = hf[dataset.meta.episodes["dataset_from_index"][ep_id]:dataset.meta.episodes["dataset_to_index"][ep_id]]
        for feature_key, out in result.items():
            feature_vals = episode_rows[feature_key]
            feature_vals = torch.stack(feature_vals).cpu().numpy()
            if feature_vals.ndim == 1:
                feature_vals = feature_vals[:, None]
            # Iterate per-dimension with fixed edges
            for d in range(feature_vals.shape[1]):
                counts, _ = np.histogram(feature_vals[:, d], bins=out["bin_edges"][d])
                out["counts"][d] += counts

                # Track total counts for normalization
                if feature_key not in total_counts_per_feature:
                    total_counts_per_feature[feature_key] = np.zeros(feature_vals.shape[1])
                total_counts_per_feature[feature_key][d] += counts.sum()

    # Convert absolute counts to fractions
    for feature_key, out in result.items():
        if feature_key in total_counts_per_feature:
            for d in range(len(total_counts_per_feature[feature_key])):
                total = total_counts_per_feature[feature_key][d]
                if total > 0:
                    out["counts"][d] = out["counts"][d] / total

    return result


def plot_feature_histograms(
    histograms: dict[str, Any],
    output_dir: Path,
    feature_keys: list[str] | None = None,
):
    """
    Plot histograms visualizing dataset statistics.
    """
    feature_keys = feature_keys or list(histograms.keys())
    for feature_key in feature_keys:
        bin_edges = histograms[feature_key]["bin_edges"]
        feature_dim = bin_edges.shape[0]
        ncols = min(3, feature_dim)
        nrows = (feature_dim + ncols - 1) // ncols
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(5 * ncols, 3.2 * nrows)
        )
        axes = np.atleast_1d(axes).ravel()
        for i in range(feature_dim):
            ax = axes[i]
            centers = 0.5 * (bin_edges[i, 1:] + bin_edges[i, :-1])
            fractions = histograms[feature_key]["counts"][i].copy()

            if feature_key == "action" and i != 6:
                # Find and exclude the largest bin
                max_bin_idx = np.argmax(fractions)
                excluded_value = fractions[max_bin_idx]
                fractions[max_bin_idx] = 0

                ax.text(
                    0.05, 0.95, f'excluded bin: {excluded_value:.3f}',
                    transform=ax.transAxes, verticalalignment='top',
                    bbox={"boxstyle": 'round', "facecolor": 'white', "alpha": 0.8}
                )

            ax.bar(centers, fractions, width=np.diff(bin_edges[i]), align="center")
            ax.set_title(f"{feature_key} | {histograms[feature_key]['dim_names'][i]}")
            ax.set_xlabel("value")
            ax.set_ylabel("fraction")
        for j in range(feature_dim, len(axes)):
            axes[j].axis("off")
        fig.tight_layout()
        if output_dir:
            fig.savefig(output_dir / f"{feature_key}.png", dpi=150)
            plt.close(fig)
        else:
            plt.show()


def main():
    parser = argparse.ArgumentParser(description="Histogram visualizer for LeRobotDataset episode subsets.")
    parser.add_argument("--repo-id", type=str, required=True, help="Name of hugging face repository containing a LeRobotDataset dataset (e.g. `lerobot/pusht`).")
    parser.add_argument("--root", type=str, default=None, help="Root directory for the dataset stored locally (e.g. `--root data`). By default, the dataset will be loaded from hugging face cache folder, or downloaded from the hub if available.")
    parser.add_argument("--libero-tasks", type=json.loads, default=None, help="Optionally choose LIBERO tasks to filter dataset episodes.")
    parser.add_argument("--features", nargs="+", default=None, help="Feature keys to include, e.g. --features actions robot_state")
    parser.add_argument("--hbins", type=int, default=100, help="Number of histogram bins.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory path to save a picture of the histogram.")

    args = parser.parse_args()

    dataset = LeRobotDataset(
        repo_id=args.repo_id,
        root=args.root,
    )
    dataset = patch_dataset_episode_boundaries(dataset=dataset)

    features = None
    if args.features:
        features = {
            k: dataset.meta.features[k]
            for k in args.features
            if k in dataset.meta.features and dataset.meta.features[k].get("dtype") not in {"string", "image", "video"}
        }

    all_episode_ids = list(dataset.meta.episodes["episode_index"])
    dataset_name = dataset.repo_id.rsplit("/", 1)[1]
    if dataset_name == "libero" and args.libero_tasks is not None:
        episode_ids_to_use = filter_libero_episodes(
            dataset=dataset,
            tasks_to_use=args.libero_tasks,
        )
    else:
        episode_ids_to_use = all_episode_ids

    histograms = compute_feature_histograms_for_episodes(
        dataset=dataset,
        episode_ids=episode_ids_to_use,
        features=features,
        bins=args.hbins,
    )

    if args.output_dir is None:
        now = dt.datetime.now()
        run_dir = f"{now:%Y-%m-%d}/{now:%H-%M-%S}_{dataset_name}"
        output_dir = Path("outputs/visualize_dataset_stats") / run_dir
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_feature_histograms(
        histograms=histograms,
        feature_keys=list(features.keys()) if features else None,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
