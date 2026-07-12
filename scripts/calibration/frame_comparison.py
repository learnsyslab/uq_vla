"""Compare calibration correlations across scored episode frames.

For each frame cache, this plots the mean and variance across seeds of the
per-seed mean round-wise Pearson and Spearman correlations for one method.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.calibration import calibration_summary


def frame_cache_name(frame: int) -> str:
    return "calibration_comparison" if frame == 0 else f"calibration_comparison_frame{frame}"


def per_seed_frame_values(
    run_dirs: list[Path],
    frame: int,
    method: str,
    env: str,
    max_round: int | None,
) -> tuple[list[float], list[float], list[int]]:
    pearson_values: list[float] = []
    spearman_values: list[float] = []
    round_counts: list[int] = []
    cache_dir_name = frame_cache_name(frame)

    for run_dir in run_dirs:
        corrs = calibration_summary.correlations_for_seed(
            run_dir,
            max_round=max_round,
            cache_dir_name=cache_dir_name,
            env=env,
        )
        pearson = corrs.get(method, {}).get("pearson", [])
        spearman = corrs.get(method, {}).get("spearman", [])
        round_counts.append(min(len(pearson), len(spearman)))
        if pearson:
            pearson_values.append(float(np.mean(pearson)))
        if spearman:
            spearman_values.append(float(np.mean(spearman)))

    return pearson_values, spearman_values, round_counts


def plot_frame_comparison(
    frames: list[int],
    pearson_by_frame: dict[int, list[float]],
    spearman_by_frame: dict[int, list[float]],
    round_counts_by_frame: dict[int, list[int]],
    method: str,
    output: Path,
) -> None:
    labels = [str(frame) for frame in frames]
    xs = np.arange(len(frames))

    fig, axes = plt.subplots(1, 2, figsize=(max(9, len(frames) * 1.3), 4), sharey=True)
    fig.suptitle(f"Frame Comparison: {method}", fontsize=13, fontweight="bold")

    for ax, values_by_frame, title, ylabel in [
        (axes[0], spearman_by_frame, "Spearman", "- Spearman rho"),
        (axes[1], pearson_by_frame, "Pearson", "- Pearson r"),
    ]:
        means = [-float(np.mean(values_by_frame[frame])) if values_by_frame[frame] else np.nan for frame in frames]
        variances = [float(np.var(values_by_frame[frame])) if values_by_frame[frame] else np.nan for frame in frames]
        stds = [float(np.sqrt(var)) if np.isfinite(var) else np.nan for var in variances]

        ax.errorbar(xs, means, yerr=stds, marker="o", linewidth=2, capsize=4, color="tab:blue")
        ax.set_title(title)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels)
        ax.set_xlabel("Frame index")
        ax.set_ylabel(ylabel)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.grid(True, axis="y", alpha=0.3)

        for x, mean, variance, frame in zip(xs, means, variances, frames):
            counts = round_counts_by_frame[frame]
            count_label = ",".join(str(c) for c in counts)
            if np.isfinite(mean):
                ax.text(
                    x,
                    mean + (0.04 if mean >= 0 else -0.08),
                    f"mean={mean:.2f}\nvar={variance:.3f}\nn={count_label}",
                    ha="center",
                    va="bottom" if mean >= 0 else "top",
                    fontsize=7,
                )

    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.savefig(output.with_suffix(".pdf"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dirs", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--frames", nargs="+", type=int, default=[0, 1, 2, 4, 6, 8, 10])
    parser.add_argument("--method", default="vfd")
    parser.add_argument("--env", choices=["libero", "metaworld"], default="metaworld")
    parser.add_argument("--max_round", type=int, default=None, help="Only include rounds < max_round.")
    args = parser.parse_args()

    pearson_by_frame: dict[int, list[float]] = {}
    spearman_by_frame: dict[int, list[float]] = {}
    round_counts_by_frame: dict[int, list[int]] = {}

    for frame in args.frames:
        pearson, spearman, round_counts = per_seed_frame_values(
            args.run_dirs,
            frame,
            args.method,
            args.env,
            args.max_round,
        )
        pearson_by_frame[frame] = pearson
        spearman_by_frame[frame] = spearman
        round_counts_by_frame[frame] = round_counts

    plot_frame_comparison(
        args.frames,
        pearson_by_frame,
        spearman_by_frame,
        round_counts_by_frame,
        args.method,
        args.output,
    )


if __name__ == "__main__":
    main()
