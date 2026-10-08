#!/usr/bin/env python3
"""Aggregate language-variation calibration results across seeds.

Reads per-seed outputs produced by `scripts/calibration/language_variation.py`
and creates combined plots / JSON summaries.

Example:
    source .env
    PYTHONPATH=src python scripts/calibration/language_variation_aggregate.py \
        --run_dirs \
          outputs/calibration/smolvla/random_s01 \
          outputs/calibration/smolvla/random_s23 \
          outputs/calibration/smolvla/random_s45 \
        --round 14 \
        --input_root outputs/calibration/smolvla/language_variation \
        --output_root plots/language_variation
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "plots"))
from paper_style import page_style  # noqa: E402
import numpy as np
from scipy import stats as scipy_stats

METHODS = [
    "action_l2",
    "ace",
    "decu",
    "ensemble_terminal_variance",
    "vlm_token_entropy",
    "vlm_perplexity",
    "inter_vel_diff",
    # "inter_vel_diff_2way",
]
METHOD_COLORS = {
    "inter_vel_diff": "#1f77b4",
    "inter_vel_diff_2way": "#17becf",
    "action_l2": "tab:orange",
    "ace": "tab:green",
    "decu": "tab:red",
    "ensemble_terminal_variance": "tab:pink",
    "vlm_token_entropy": "tab:gray",
    "vlm_perplexity": "tab:brown",
}
METHOD_LABELS = {
    "inter_vel_diff": "VFD (ours)",
    "inter_vel_diff_2way": "VFD-2way (ours)",
    "action_l2": "Action-L2",
    "ace": "ACE",
    "decu": "DECU",
    "ensemble_terminal_variance": "GU",
    "vlm_token_entropy": "Entropy",
    "vlm_perplexity": "Perplexity",
}
NUM_TASKS = 10
NUM_PROMPTS = 5


def calibration_json_path(input_root: Path, run_dir: Path, round_idx: int) -> Path:
    return input_root / run_dir.name / f"round_{round_idx:03d}_calibration.json"


def load_seed_calibration(path: Path) -> dict | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    return json.loads(path.read_text())


def _cache_dirs(run_dir: Path, round_idx: int) -> list[Path]:
    rel_round = Path(f"round_{round_idx:03d}") / "language_variation"
    dirs = [run_dir / rel_round]


    unique_dirs = []
    seen = set()
    for path in dirs:
        key = path.resolve() if path.exists() else path
        if key in seen:
            continue
        seen.add(key)
        unique_dirs.append(path)
    return unique_dirs


def load_cached_round(run_dir: Path, round_idx: int) -> dict[int, dict]:
    results: dict[int, dict] = {}
    for cache_dir in _cache_dirs(run_dir, round_idx):
        if not cache_dir.exists():
            continue
        for path in sorted(cache_dir.glob("task_*.json")):
            task_id = int(path.stem.split("_")[1])
            results.setdefault(task_id, json.loads(path.read_text()))
    return results


def _normalize_methods(data: dict) -> dict[str, dict]:
    if "methods" in data:
        return data["methods"]
    return {
        "inter_vel_diff": {
            "per_task": data.get("per_task", []),
            "mean_spearman_r": data.get("mean_spearman_r", float("nan")),
            "std_spearman_r": data.get("std_spearman_r", float("nan")),
        }
    }


def aggregate_calibration(seed_data: list[tuple[str, dict]]) -> dict:
    run_names = [run_name for run_name, _ in seed_data]
    method_names = sorted({method for _, data in seed_data for method in _normalize_methods(data)})
    methods_summary: dict[str, dict] = {}

    for method in method_names:
        per_task_values: dict[int, list[float]] = {}
        per_task_by_seed: dict[int, dict[str, float]] = {}
        seed_means: dict[str, float] = {}

        for run_name, data in seed_data:
            method_data = _normalize_methods(data).get(method)
            if method_data is None:
                continue
            finite_seed_values: list[float] = []
            for item in method_data.get("per_task", []):
                task_id = int(item["task_id"])
                r = float(item["spearman_r"])
                if math.isnan(r):
                    continue
                finite_seed_values.append(r)
                per_task_values.setdefault(task_id, []).append(r)
                per_task_by_seed.setdefault(task_id, {})[run_name] = r
            val = method_data.get("mean_spearman_r")
            if val is not None and not math.isnan(val):
                seed_means[run_name] = float(val)
            elif finite_seed_values:
                seed_means[run_name] = float(np.mean(finite_seed_values))

        per_task = []
        for task_id in sorted(per_task_values):
            vals = per_task_values[task_id]
            per_task.append(
                {
                    "task_id": task_id,
                    "n_seeds": len(vals),
                    "mean_spearman_r": float(np.mean(vals)),
                    "std_spearman_r": float(np.std(vals)),
                    "seed_values": {
                        run_name: per_task_by_seed[task_id][run_name]
                        for run_name in sorted(per_task_by_seed[task_id])
                    },
                }
            )

        mean_vals = list(seed_means.values())
        methods_summary[method] = {
            "seed_mean_spearman_r": seed_means,
            "mean_spearman_r": float(np.mean(mean_vals)) if mean_vals else float("nan"),
            "std_spearman_r": float(np.std(mean_vals)) if mean_vals else float("nan"),
            "per_task": per_task,
        }

    return {
        "run_names": run_names,
        "n_seeds": len(seed_data),
        "methods": methods_summary,
    }


def aggregate_prompt_results(seed_results: list[tuple[str, dict[int, dict]]]) -> dict:
    uncertainty_range_by_method: dict[str, dict[str, float]] = {}
    for _, results in seed_results:
        for result in results.values():
            for method, values in (result.get("uncertainties_by_method") or {}).items():
                finite_values = [float(v) for v in values if np.isfinite(float(v))]
                if finite_values:
                    method_range = uncertainty_range_by_method.setdefault(
                        method,
                        {"min": float("inf"), "max": float("-inf")},
                    )
                    method_range["min"] = min(method_range["min"], min(finite_values))
                    method_range["max"] = max(method_range["max"], max(finite_values))

    per_task: dict[int, dict] = {}
    for task_id in range(NUM_TASKS):
        task_summary: dict[str, object] = {
            "task_id": task_id,
            "success_mean": [],
            "success_std": [],
            "success_seed_values": [],
            "uncertainties_by_method": {},
        }
        success_by_prompt: list[list[float]] = [[] for _ in range(NUM_PROMPTS)]
        unc_by_method: dict[str, list[list[float]]] = {
            method: [[] for _ in range(NUM_PROMPTS)] for method in METHODS
        }

        for _, results in seed_results:
            result = results.get(task_id)
            if not result:
                continue
            success_rates = result.get("success_rates") or []
            for prompt_idx, value in enumerate(success_rates[:NUM_PROMPTS]):
                success_by_prompt[prompt_idx].append(float(value))

            for method, values in (result.get("uncertainties_by_method") or {}).items():
                method_range = uncertainty_range_by_method.get(method)
                if not method_range:
                    continue
                denom = method_range["max"] - method_range["min"]
                for prompt_idx, value in enumerate(values[:NUM_PROMPTS]):
                    unc_by_method.setdefault(method, [[] for _ in range(NUM_PROMPTS)])
                    if denom <= 0:
                        unc_by_method[method][prompt_idx].append(0.0)
                    else:
                        unc_by_method[method][prompt_idx].append(
                            (float(value) - method_range["min"]) / denom
                        )

        task_summary["success_mean"] = [
            float(np.mean(values)) if values else float("nan") for values in success_by_prompt
        ]
        task_summary["success_std"] = [
            float(np.std(values)) if values else float("nan") for values in success_by_prompt
        ]
        task_summary["success_seed_values"] = success_by_prompt

        for method in METHODS:
            task_summary["uncertainties_by_method"][method] = {
                "mean": [
                    float(np.mean(values)) if values else float("nan")
                    for values in unc_by_method.get(method, [[] for _ in range(NUM_PROMPTS)])
                ],
                "std": [
                    float(np.std(values)) if values else float("nan")
                    for values in unc_by_method.get(method, [[] for _ in range(NUM_PROMPTS)])
                ],
                "seed_values": unc_by_method.get(method, [[] for _ in range(NUM_PROMPTS)]),
            }

        per_task[task_id] = task_summary

    return {
        "run_names": [run_name for run_name, _ in seed_results],
        "n_seeds": len(seed_results),
        "normalization": {
            "mode": "per_method_global_min_max_across_seeds_tasks_prompts",
            "uncertainty_range_by_method": uncertainty_range_by_method,
        },
        "per_task": [per_task[task_id] for task_id in range(NUM_TASKS)],
    }


def _spearman_or_nan(xs: list[float], ys: list[float]) -> float:
    x_arr = np.asarray(xs, dtype=float)
    y_arr = np.asarray(ys, dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    if len(x_arr) < 2 or len(np.unique(x_arr)) < 2 or len(np.unique(y_arr)) < 2:
        return float("nan")
    r, _ = scipy_stats.spearmanr(x_arr, y_arr)
    return float(r) if np.isfinite(r) else float("nan")


def add_across_task_calibration(summary: dict, prompt_summary: dict) -> None:
    """Add one Spearman value per method using task-level aggregate points.

    The per-seed calibration JSONs store Spearman per task across prompt variants.
    Those values are often undefined when a task has constant success across the
    five prompts. For the aggregate plot, also compute a single Spearman across
    the ten tasks by averaging prompts within each task first.
    """

    by_method: dict[str, dict] = {}
    for method in METHODS:
        task_uncertainties: list[float] = []
        task_success_rates: list[float] = []
        prompt_uncertainties: list[float] = []
        prompt_success_rates: list[float] = []

        for task in prompt_summary.get("per_task", []):
            successes = np.asarray(task.get("success_mean", []), dtype=float)
            method_unc = (task.get("uncertainties_by_method") or {}).get(method, {})
            uncertainties = np.asarray(method_unc.get("mean", []), dtype=float)
            n = min(len(successes), len(uncertainties))
            if n == 0:
                continue
            successes = successes[:n]
            uncertainties = uncertainties[:n]
            mask = np.isfinite(successes) & np.isfinite(uncertainties)
            if not np.any(mask):
                continue

            task_success_rates.append(float(np.mean(successes[mask])))
            task_uncertainties.append(float(np.mean(uncertainties[mask])))
            prompt_success_rates.extend(float(x) for x in successes[mask])
            prompt_uncertainties.extend(float(x) for x in uncertainties[mask])

        by_method[method] = {
            "across_task_spearman_r": _spearman_or_nan(task_uncertainties, task_success_rates),
            "across_task_n": len(task_success_rates),
            "task_prompt_spearman_r": _spearman_or_nan(prompt_uncertainties, prompt_success_rates),
            "task_prompt_n": len(prompt_success_rates),
        }

    summary["across_task_calibration"] = by_method
    for method, values in by_method.items():
        if method in summary.get("methods", {}):
            summary["methods"][method].update(values)


def _format_rho(value: float | None) -> str:
    if value is None:
        return "nan"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "nan"
    return f"{value:.2f}" if np.isfinite(value) else "nan"


def _mean_finite(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if len(arr) else float("nan")


def plot_aggregate_success_rates(summary: dict, round_idx: int, title: str, output_path: Path) -> None:
    labels = [f"P{i + 1}" for i in range(NUM_PROMPTS)]
    prompt_colors = ["#2196F3", "#64B5F6", "#FFAB40", "#FF7043", "#E53935"]
    method_offsets = {
        method: offset for method, offset in zip(METHODS, np.linspace(-0.22, 0.22, len(METHODS)))
    }
    method_markers = {
        "inter_vel_diff": "o",
        "inter_vel_diff_2way": "D",
        "action_l2": "s",
        "ace": "^",
        "decu": "v",
        "ensemble_terminal_variance": "P",
        "vlm_token_entropy": "X",
        "vlm_perplexity": "*",
    }

    fig, axes = plt.subplots(2, 5, figsize=(20, 8), sharey=True)
    fig.suptitle(
        f"{title} — Round {round_idx}: aggregated success rate and normalized uncertainty by prompt variant",
        fontsize=13,
    )

    task_summaries = {int(item["task_id"]): item for item in summary["per_task"]}
    for task_id, ax in enumerate(axes.flat):
        task = task_summaries.get(task_id)
        if task is None:
            ax.set_visible(False)
            continue

        means = np.asarray(task["success_mean"], dtype=float)
        stds = np.asarray(task["success_std"], dtype=float)
        finite_sort_values = np.nan_to_num(means, nan=-np.inf)
        prompt_order = np.argsort(-finite_sort_values)
        means = means[prompt_order]
        stds = stds[prompt_order]
        x = np.arange(NUM_PROMPTS)
        bars = ax.bar(labels, means, yerr=stds, capsize=3, color=prompt_colors, width=0.6)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"Task {task_id}", fontsize=10)
        ax.set_ylabel("Success Rate / Uncertainty" if task_id % 5 == 0 else "")
        if np.isfinite(means[0]):
            ax.axhline(means[0], color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.grid(True, axis="y", alpha=0.3)
        for bar, rate in zip(bars, means):
            if np.isfinite(rate):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f"{rate:.0%}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

        for method in METHODS:
            values = (task["uncertainties_by_method"] or {}).get(method)
            if not values:
                continue
            unc_means = np.asarray(values["mean"], dtype=float)[prompt_order]
            unc_stds = np.asarray(values["std"], dtype=float)[prompt_order]
            ax.errorbar(
                x + method_offsets[method],
                unc_means,
                yerr=unc_stds,
                fmt=method_markers.get(method, "o"),
                color=METHOD_COLORS[method],
                markerfacecolor=METHOD_COLORS[method],
                markeredgecolor="white",
                markeredgewidth=0.5,
                markersize=5,
                capsize=2,
                linestyle="none",
                alpha=0.9,
                zorder=4,
            )

    handles = [
        plt.Line2D([0], [0], marker="s", linestyle="none", color=color, markersize=8, label=label)
        for color, label in zip(prompt_colors, labels)
    ]
    handles.extend(
        plt.Line2D(
            [0],
            [0],
            marker=method_markers.get(method, "o"),
            linestyle="none",
            markerfacecolor=METHOD_COLORS[method],
            markeredgecolor="white",
            color=METHOD_COLORS[method],
            markersize=7,
            label=f"{method} uncertainty",
        )
        for method in METHODS
    )
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, 0.025))
    fig.text(
        0.5,
        0.01,
        "Bars/dots show mean across seeds; error bars show std across seeds. "
        "Uncertainties are min-max normalized per method.",
        ha="center",
        fontsize=9,
        style="italic",
    )
    plt.tight_layout(rect=[0, 0.09, 1, 1])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_prompt_summary(summary: dict, round_idx: int, title: str, output_path: Path) -> None:
    labels = [f"P{i + 1}" for i in range(NUM_PROMPTS)]
    # Type comes from the paper style: the figure is built at the 0.5\textwidth its wrapfigure
    # occupies, so LaTeX scales it by 1. The old 30 pt labels in a 9 in canvas came out at
    # ~9.2 pt on the page once shrunk, which is what the bundle gives directly.
    page_style("language_variation")
    label_fontsize = tick_fontsize = legend_fontsize = None
    method_markers = {
        "inter_vel_diff": "o",
        "inter_vel_diff_2way": "D",
        "action_l2": "s",
        "ace": "^",
        "decu": "v",
        "ensemble_terminal_variance": "P",
        "vlm_token_entropy": "X",
        "vlm_perplexity": "*",
    }

    success_by_prompt = [[] for _ in range(NUM_PROMPTS)]
    unc_by_method: dict[str, list[list[float]]] = {
        method: [[] for _ in range(NUM_PROMPTS)] for method in METHODS
    }

    for task in summary["per_task"]:
        success_means = np.asarray(task["success_mean"], dtype=float)
        prompt_order = np.argsort(-np.nan_to_num(success_means, nan=-np.inf))
        ordered_success = success_means[prompt_order]
        for prompt_idx, value in enumerate(ordered_success):
            if np.isfinite(value):
                success_by_prompt[prompt_idx].append(float(value))

        for method in METHODS:
            values = (task.get("uncertainties_by_method") or {}).get(method)
            if not values:
                continue
            unc_means = np.asarray(values.get("mean", []), dtype=float)
            if len(unc_means) < NUM_PROMPTS:
                continue
            ordered_unc = unc_means[prompt_order]
            for prompt_idx, value in enumerate(ordered_unc):
                if np.isfinite(value):
                    unc_by_method[method][prompt_idx].append(float(value))

    fig, ax = plt.subplots()
    x = np.arange(NUM_PROMPTS)
    success_means = [_mean_finite(values) for values in success_by_prompt]
    success_stds = [
        float(np.std(np.asarray(values, dtype=float))) if values else float("nan")
        for values in success_by_prompt
    ]
    ax.bar(
        x,
        success_means,
        width=0.55,
        color="lightgray",
        edgecolor="gray",
        linewidth=0.8,
        alpha=0.5,
        label="Success Rate",
        zorder=1,
    )

    for method in METHODS:
        means = [_mean_finite(values) for values in unc_by_method[method]]
        if not any(np.isfinite(means)):
            continue
        method_scale = np.nanmax(means)
        if np.isfinite(method_scale) and method_scale > 0:
            means = [value / method_scale for value in means]
        ax.plot(
            x,
            means,
            marker=method_markers.get(method, "o"),
            linewidth=1.8,
            color=METHOD_COLORS[method],
            markerfacecolor=METHOD_COLORS[method],
            markeredgecolor="white",
            markeredgewidth=0.4,
            markersize=4.5,
            label=METHOD_LABELS.get(method, method),
            alpha=0.95,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Prompt")
    ax.set_ylabel("Success Rate / Uncertainty")
    ax.yaxis.set_label_coords(-0.1, 0.4)
    # ax.set_title(f"{title} — Round {round_idx}: mean across tasks")
    
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="lower right", bbox_to_anchor=(1.01, 0.0))
    # fig.text(
    #     0.5,
    #     0.01,
    #     "Each task is sorted by descending prompt success before averaging; uncertainty lines are normalized to max mean 1.",
    #     ha="center",
    #     fontsize=8,
    #     style="italic",
    # )
    # No tight_layout: the paper style turns constrained layout on and the two conflict.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    # also save as PDF
    plt.savefig(output_path.with_suffix(".pdf"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_aggregate_bar(summary: dict, round_idx: int, title: str, output_path: Path) -> None:
    methods = summary["methods"]
    if not methods:
        raise ValueError("No per-task calibration values available to plot.")

    fig, ax = plt.subplots(figsize=(12, 4.5))
    method_names = list(methods)
    task_ids = sorted({item["task_id"] for method in methods.values() for item in method["per_task"]})
    x = np.arange(len(task_ids))
    width = 0.22
    colors = METHOD_COLORS

    for idx, method in enumerate(method_names):
        per_task = {item["task_id"]: item for item in methods[method]["per_task"]}
        means = [per_task.get(task_id, {}).get("mean_spearman_r", np.nan) for task_id in task_ids]
        stds = [per_task.get(task_id, {}).get("std_spearman_r", 0.0) for task_id in task_ids]
        bar_mean = _mean_finite(means)
        ax.bar(
            x + (idx - (len(method_names) - 1) / 2) * width,
            means,
            yerr=stds,
            capsize=4,
            width=width,
            color=colors.get(method),
            alpha=0.9,
            label=(
                f"{method} "
                f"(rho_tasks={_format_rho(methods[method].get('across_task_spearman_r'))}, "
                f"bar mean={_format_rho(bar_mean)})"
            ),
        )

    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels([f"T{t}" for t in task_ids])
    ax.set_ylabel("Spearman r (uncertainty vs. success rate)")
    ax.set_ylim(-1.1, 1.1)
    rho_parts = [
        f"{method}: {_format_rho(methods[method].get('across_task_spearman_r'))}"
        for method in method_names
    ]
    ax.set_title(
        f"{title} — Round {round_idx}: aggregated calibration per task\n"
        "Bars: within-task prompt Spearman. Across-task Spearman rho: " + ", ".join(rho_parts),
    )
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="lower left", title="method")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_seed_summary(summary: dict, round_idx: int, title: str, output_path: Path) -> None:
    methods = summary["methods"]
    if not methods:
        raise ValueError("No seed-level mean calibration values available to plot.")

    method_names = list(methods)
    run_names = summary["run_names"]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(run_names))
    width = 0.22
    colors = METHOD_COLORS

    for idx, method in enumerate(method_names):
        seed_means = methods[method]["seed_mean_spearman_r"]
        values = [seed_means.get(name, np.nan) for name in run_names]
        ax.bar(
            x + (idx - (len(method_names) - 1) / 2) * width,
            values,
            width=width,
            color=colors.get(method),
            alpha=0.85,
            label=f"{method} ({methods[method]['mean_spearman_r']:.2f})",
        )
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(run_names, rotation=20)
    ax.set_ylabel("Mean Spearman r across tasks")
    ax.set_ylim(-1.1, 1.1)
    ax.set_title(f"{title} — Round {round_idx}: seed-level mean calibration")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="lower left", title="method (mean r)")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dirs", nargs="+", required=True, type=Path)
    parser.add_argument("--round", type=int, required=True, dest="round_idx")
    parser.add_argument("--input_root", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--title", type=str, default=None)
    args = parser.parse_args()

    seed_data: list[tuple[str, dict]] = []
    missing: list[str] = []
    for run_dir in args.run_dirs:
        path = calibration_json_path(args.input_root, run_dir, args.round_idx)
        data = load_seed_calibration(path)
        if data is None:
            missing.append(str(path))
            continue
        seed_data.append((run_dir.name, data))

    if not seed_data:
        raise FileNotFoundError(
            "No per-seed language-variation calibration JSONs found.\n"
            + "\n".join(missing)
        )

    title = args.title or args.output_root.parent.name

    args.output_root.mkdir(parents=True, exist_ok=True)
    summary = aggregate_calibration(seed_data)

    cached_seed_results = [
        (run_dir.name, load_cached_round(run_dir, args.round_idx))
        for run_dir in args.run_dirs
    ]
    cached_seed_results = [(name, results) for name, results in cached_seed_results if results]
    if cached_seed_results:
        prompt_summary = aggregate_prompt_results(cached_seed_results)
        add_across_task_calibration(summary, prompt_summary)
        prompt_summary_path = args.output_root / f"round_{args.round_idx:03d}_success_rates_aggregated.json"
        prompt_summary_path.write_text(json.dumps(prompt_summary, indent=2))
        plot_prompt_summary(
            summary=prompt_summary,
            round_idx=args.round_idx,
            title=title,
            output_path=args.output_root / f"round_{args.round_idx:03d}_success_rates_summary.png",
        )
        plot_aggregate_success_rates(
            summary=prompt_summary,
            round_idx=args.round_idx,
            title=title,
            output_path=args.output_root / f"round_{args.round_idx:03d}_success_rates.png",
        )

    summary_path = args.output_root / f"round_{args.round_idx:03d}_calibration_aggregated.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    plot_aggregate_bar(
        summary=summary,
        round_idx=args.round_idx,
        title=title,
        output_path=args.output_root / f"round_{args.round_idx:03d}_calibration_aggregated.png",
    )
    plot_seed_summary(
        summary=summary,
        round_idx=args.round_idx,
        title=title,
        output_path=args.output_root / f"round_{args.round_idx:03d}_calibration_seed_means.png",
    )

    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
