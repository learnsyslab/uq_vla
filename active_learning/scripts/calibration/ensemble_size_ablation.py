"""
Ablate ensemble size for inter_vel_diff_2way calibration.

Computes task-level correlations between success rate and uncertainty for
different total ensemble sizes. Ensemble size K means member_00 is used as the
sampler/reference model and members 01..K-1 are used as scorer models.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import matplotlib.pyplot as plt
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "plots"))
from paper_style import page_style, tex  # noqa: E402
import numpy as np
import torch
from scipy import stats as scipy_stats

from lerobot.uncertainty.uncertainty_scoring.ensemble_utils.factory import build_ensemble_models

from scripts.calibration import calibration_comparison as cc

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

METHOD = "inter_vel_diff_2way"
CACHE_DIR_NAME = "ensemble_size_ablation"


def cache_name(ensemble_size: int) -> str:
    return f"{METHOD}_ens{ensemble_size}"


def load_member_paths(round_dir: Path, ensemble_size: int) -> list[Path]:
    manifest_path = round_dir / "training_manifest.json"
    paths: list[Path] = []
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        by_index = {
            int(member["member_index"]): Path(member["final_model_path"])
            for member in manifest.get("member_training_runs", [])
            if member.get("final_model_path")
        }
        paths = [by_index[i] for i in range(ensemble_size) if i in by_index]

    if len(paths) < ensemble_size:
        paths = [
            round_dir / "training" / f"member_{i:02d}" / "checkpoints" / "last" / "pretrained_model"
            for i in range(ensemble_size)
        ]

    missing = [str(path) for path in paths if not path.exists()]
    if len(paths) < ensemble_size or missing:
        raise FileNotFoundError(
            f"Need members 00..{ensemble_size - 1:02d} for {round_dir}; missing: {missing or paths}"
        )
    return paths


def load_or_compute_uncertainties(
    *,
    round_dir: Path,
    ensemble_size: int,
    policy,
    adapter,
    preprocessor,
    task_list: list[tuple[str, int]],
    env: str,
    frame_index: int,
    num_action_samples: int,
    inference_batch_size: int,
    dataset_root: str | None,
) -> dict[int, dict[int, float]]:
    method_cache_name = cache_name(ensemble_size)
    cached = cc.load_cached(round_dir, method_cache_name, cache_dir_name=CACHE_DIR_NAME)
    if cached is not None:
        logger.info("    ens%d: loaded from %s cache", ensemble_size, CACHE_DIR_NAME)
        return cached

    if ensemble_size == 2:
        cached = cc.load_cached(round_dir, METHOD, cache_dir_name="calibration_comparison")
        if cached is not None:
            logger.info("    ens2: reused calibration_comparison cache")
            cc.save_cache(round_dir, method_cache_name, cached, cache_dir_name=CACHE_DIR_NAME)
            return cached

    member_paths = load_member_paths(round_dir, ensemble_size)
    scorer_paths = [str(path) for path in member_paths[1:ensemble_size]]
    logger.info("    ens%d: building scorer models %s", ensemble_size, scorer_paths)
    ensemble_models = build_ensemble_models(scorer_paths, policy.config)
    sampler = cc.build_ensemble_sampler(adapter, ensemble_models, METHOD, num_action_samples)
    try:
        episode_uncertainties = cc.compute_episode_uncertainties(
            sampler=sampler,
            policy_config=policy.config,
            pretrained_path=str(member_paths[0]),
            device=str(next(policy.parameters()).device),
            env=env,
            task_list=task_list,
            frame_index=frame_index,
            inference_batch_size=inference_batch_size,
            round_idx=int(round_dir.name.split("_")[1]),
            method=method_cache_name,
            dataset_root=dataset_root,
        )
    finally:
        del sampler
        del ensemble_models
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    cc.save_cache(round_dir, method_cache_name, episode_uncertainties, cache_dir_name=CACHE_DIR_NAME)
    return episode_uncertainties


def load_cached_uncertainties(round_dir: Path, ensemble_size: int) -> dict[int, dict[int, float]] | None:
    cached = cc.load_cached(round_dir, cache_name(ensemble_size), cache_dir_name=CACHE_DIR_NAME)
    if cached is not None:
        return cached
    if ensemble_size == 2:
        return cc.load_cached(round_dir, METHOD, cache_dir_name="calibration_comparison")
    return None


def process_run(
    *,
    run_dir: Path,
    rounds: list[int],
    ensemble_sizes: list[int],
    env: str,
    device: str,
    frame_index: int,
    num_action_samples: int,
    inference_batch_size: int,
    aggregate_only: bool = False,
) -> dict[int, dict[int, dict]]:
    dataset_root = cc._load_dataset_root(run_dir)
    task_list = cc._load_task_list(run_dir, env)
    results: dict[int, dict[int, dict]] = {}

    policy = None
    adapter = None
    preprocessor = None

    for round_idx in rounds:
        round_dir = run_dir / f"round_{round_idx:03d}"
        if not round_dir.exists():
            logger.warning("%s is missing; skipping.", round_dir)
            continue

        success_rates = cc.load_success_rates(round_dir, env=env, task_list=task_list)
        if success_rates is None:
            logger.warning("%s has no success rates; skipping.", round_dir)
            continue

        results[round_idx] = {}
        cached_by_size = {
            ensemble_size: load_cached_uncertainties(round_dir, ensemble_size)
            for ensemble_size in ensemble_sizes
        }
        missing_sizes = [
            ensemble_size for ensemble_size, cached in cached_by_size.items()
            if cached is None
        ]
        if aggregate_only and missing_sizes:
            raise FileNotFoundError(
                f"{round_dir} is missing cached uncertainties for ensemble sizes {missing_sizes}"
            )

        member_paths = None
        if missing_sizes:
            member_paths = load_member_paths(round_dir, max(ensemble_sizes))
            if policy is None:
                policy, adapter = cc.load_adapter(str(member_paths[0]), device)
            else:
                adapter = cc.reload_adapter(policy, str(member_paths[0]), device)
            preprocessor, _ = cc.make_pre_post_processors(
                policy_cfg=policy.config,
                pretrained_path=str(member_paths[0]),
            )

        for ensemble_size in ensemble_sizes:
            if ensemble_size < 2:
                raise ValueError("ensemble sizes must be >= 2")
            if member_paths is not None and ensemble_size > len(member_paths):
                logger.warning("%s has only %d members; skipping ens%d", round_dir, len(member_paths), ensemble_size)
                continue
            episode_uncertainties = cached_by_size.get(ensemble_size)
            if episode_uncertainties is None:
                episode_uncertainties = load_or_compute_uncertainties(
                    round_dir=round_dir,
                    ensemble_size=ensemble_size,
                    policy=policy,
                    adapter=adapter,
                    preprocessor=preprocessor,
                    task_list=task_list,
                    env=env,
                    frame_index=frame_index,
                    num_action_samples=num_action_samples,
                    inference_batch_size=inference_batch_size,
                    dataset_root=dataset_root,
                )
            task_uncertainties = {
                int(task_id): float(np.mean(list(uncs.values())))
                for task_id, uncs in episode_uncertainties.items()
                if uncs
            }
            results[round_idx][ensemble_size] = {
                "task_uncertainties": task_uncertainties,
                "task_success_rates": success_rates,
            }

    return results


def correlations_by_round(
    per_run_results: list[dict[int, dict[int, dict]]],
    rounds: list[int],
    ensemble_sizes: list[int],
) -> dict[str, dict[int, dict[int, list[float]]]]:
    output = {
        "pearson": {size: {round_idx: [] for round_idx in rounds} for size in ensemble_sizes},
        "spearman": {size: {round_idx: [] for round_idx in rounds} for size in ensemble_sizes},
    }
    for run in per_run_results:
        for round_idx in rounds:
            for ensemble_size in ensemble_sizes:
                data = run.get(round_idx, {}).get(ensemble_size)
                if not data:
                    continue
                task_ids = sorted(set(data["task_uncertainties"]) & set(data["task_success_rates"]))
                if len(task_ids) < 3:
                    continue
                uncs = [data["task_uncertainties"][task_id] for task_id in task_ids]
                srs = [data["task_success_rates"][task_id] for task_id in task_ids]
                if len(set(uncs)) <= 1 or len(set(srs)) <= 1:
                    continue
                pearson = float(scipy_stats.pearsonr(uncs, srs).statistic)
                spearman = float(scipy_stats.spearmanr(uncs, srs).statistic)
                if not math.isnan(pearson):
                    output["pearson"][ensemble_size][round_idx].append(pearson)
                if not math.isnan(spearman):
                    output["spearman"][ensemble_size][round_idx].append(spearman)
    return output


# Demonstrations queried per active fine-tuning round (`iteration.episodes_per_round` in the
# run config). The x-axis reports collected episodes rather than the round index, matching
# scripts/plots/success_rate_paper.py.
EPISODES_PER_ROUND = 5


# How each correlation is written on an axis. `plot_metric` used to hardcode the Spearman
# label, so pearson_across_rounds.pdf claimed to be Spearman; keep the two in one place.
TICK_STEP_DEMOS = 10   # x-axis tick spacing, in demonstrations

METRIC_LABELS = {
    "Spearman rho": "$-$Spearman $\\rho$",
    "Pearson r": "$-$Pearson $r$",
}


def _draw_metric(ax, values, rounds, colors, legend: bool) -> None:
    """One ensemble-size panel: -correlation against demonstrations collected."""
    for ensemble_size in sorted(values):
        valid_rounds, means, stds = [], [], []
        for round_idx in rounds:
            vals = values[ensemble_size].get(round_idx, [])
            if not vals:
                continue
            valid_rounds.append(round_idx + 1)
            means.append(-float(np.mean(vals)))
            stds.append(float(np.std(vals)))
        if not valid_rounds:
            continue
        color = colors.get(ensemble_size)
        ax.plot(valid_rounds, means, marker="o", markersize=3, linewidth=1.4,
                label=f"$M={ensemble_size}$", color=color)
        ax.fill_between(
            valid_rounds,
            [m - sd for m, sd in zip(means, stds, strict=True)],
            [m + sd for m, sd in zip(means, stds, strict=True)],
            color=color, alpha=0.18,
        )
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel(tex("# Demonstrations"))
    ax.set_ylim(0, 1)
    # A tick every TICK_STEP_DEMOS demonstrations, i.e. every other round at 5 per round.
    tick_values = [t for t in sorted({r + 1 for r in rounds})
                   if (t * EPISODES_PER_ROUND) % TICK_STEP_DEMOS == 0]
    ax.set_xticks(tick_values)
    ax.set_xticklabels([str(t * EPISODES_PER_ROUND) for t in tick_values])
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.grid(True, alpha=0.3, axis="y")
    if legend:
        ax.legend(loc="lower left")


def plot_metrics_side_by_side(
    *, correlations: dict, rounds: list[int], output_path: Path
) -> None:
    """Spearman and Pearson as two panels of one figure (the paper's Fig. ensemble_size_ablation).

    The two share a y-axis: both are a negative correlation on [0, 1], so one set of tick labels
    is enough and the saved width goes to the panels. Which correlation a panel shows is in its
    title rather than a y-label, for the same reason.
    """
    colors = {2: "tab:blue", 3: "tab:orange", 4: "tab:green"}
    page_style("ensemble_ablation_2panel")
    fig, axes = plt.subplots(1, 2, sharey=True)
    for ax, key in zip(axes, ("spearman", "pearson")):
        _draw_metric(ax, correlations[key], rounds, colors, legend=(key == "spearman"))
        ax.set_title(METRIC_LABELS["Spearman rho" if key == "spearman" else "Pearson r"])
    # No tight_layout: the paper style turns constrained layout on and the two conflict.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_metric(
    *,
    values: dict[int, dict[int, list[float]]],
    rounds: list[int],
    output_path: Path,
    metric_name: str,
) -> None:
    colors = {2: "tab:blue", 3: "tab:orange", 4: "tab:green"}
    # Type comes from the paper style now (see scripts/plots/paper_style.py): the figure is
    # built at the 0.4\textwidth its wrapfigure occupies, so LaTeX scales it by 1.
    page_style("ensemble_ablation")
    label_fontsize = tick_fontsize = legend_fontsize = None
    fig, ax = plt.subplots()
    for ensemble_size in sorted(values):
        valid_rounds, means, stds = [], [], []
        for round_idx in rounds:
            vals = values[ensemble_size].get(round_idx, [])
            if not vals:
                continue
            valid_rounds.append(round_idx + 1)
            means.append(-float(np.mean(vals)))
            stds.append(float(np.std(vals)))
        if not valid_rounds:
            continue
        color = colors.get(ensemble_size)
        # ax.errorbar(
        #     valid_rounds,
        #     means,
        #     yerr=stds,
        #     marker="o",
        #     linewidth=2,
        #     elinewidth=1.2,
        #     capsize=3,
        #     capthick=1.2,
        #     label=f"$M={ensemble_size}$",
        #     color=color,
        # )
        # Plot instead
        ax.plot(
            valid_rounds,
            means,
            marker="o",
            linewidth=2,
            label=f"$M={ensemble_size}$",
            color=color,
        )
        ax.fill_between(
            valid_rounds,
            [mean - std for mean, std in zip(means, stds, strict=True)],
            [mean + std for mean, std in zip(means, stds, strict=True)],
            color=color,
            alpha=0.18,
        )
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    # ax.set_title(f"{metric_name} for inter_vel_diff_2way by ensemble size", fontweight="bold")
    ax.set_xlabel(tex("# Demonstrations"))
    # ax.set_ylabel(metric_name)
    ax.set_ylabel(METRIC_LABELS.get(metric_name, metric_name))
    ax.set_ylim(0, 1)
    # Ticks stay on round positions; the labels report how many demonstrations had been
    # collected by then (EPISODES_PER_ROUND per round, 5 in every run the paper uses).
    tick_values = [t for t in (2, 4, 6, 8, 10, 12, 14) if t in {r + 1 for r in rounds}]
    ax.set_xticks(tick_values)
    ax.set_xticklabels([str(t * EPISODES_PER_ROUND) for t in tick_values])
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(loc="lower left")
    # No tight_layout: the paper style turns constrained layout on and the two conflict.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def correlations_by_seed_summary(
    per_run_results: list[dict[int, dict[int, dict]]],
    rounds: list[int],
    ensemble_sizes: list[int],
) -> dict[str, dict[int, list[float]]]:
    seed_means = {
        "pearson": {size: [] for size in ensemble_sizes},
        "spearman": {size: [] for size in ensemble_sizes},
    }
    for run in per_run_results:
        run_corrs = correlations_by_round([run], rounds, ensemble_sizes)
        for metric in seed_means:
            for ensemble_size in ensemble_sizes:
                vals = [
                    vals[0]
                    for vals in run_corrs[metric][ensemble_size].values()
                    if vals
                ]
                if vals:
                    seed_means[metric][ensemble_size].append(float(np.mean(vals)))
    return seed_means


def plot_calibration_summary(
    *,
    seed_means: dict[str, dict[int, list[float]]],
    output_path: Path,
) -> None:
    ensemble_sizes = [
        size
        for size in sorted(seed_means["spearman"])
        if seed_means["spearman"][size] or seed_means["pearson"][size]
    ]
    if not ensemble_sizes:
        logger.warning("No ensemble-size summary values available; skipping %s", output_path)
        return

    colors = {2: "tab:blue", 3: "tab:orange", 4: "tab:green"}
    fig, axes = plt.subplots(1, 2, figsize=(4 + len(ensemble_sizes) * 1.2, 4), sharey=False)
    for ax, metric, title, ylabel in [
        (axes[0], "spearman", "Spearman", "- Spearman $\\rho$"),
        (axes[1], "pearson", "Pearson", "- Pearson $r$"),
    ]:
        xs = np.arange(len(ensemble_sizes))
        means = [
            -float(np.mean(seed_means[metric][size]))
            if seed_means[metric][size]
            else np.nan
            for size in ensemble_sizes
        ]
        stds = [
            float(np.std(seed_means[metric][size]))
            if seed_means[metric][size]
            else 0.0
            for size in ensemble_sizes
        ]
        ax.bar(
            xs,
            means,
            yerr=stds,
            color=[colors.get(size, None) for size in ensemble_sizes],
            capsize=5,
            width=0.6,
            zorder=3,
        )
        ax.set_xticks(xs)
        ax.set_xticklabels([f"K={size}" for size in ensemble_sizes], fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.grid(True, alpha=0.3, axis="y", zorder=0)
        ax.set_ylim(0, 0.8)
        for x, mean, std in zip(xs, means, stds, strict=True):
            if np.isnan(mean):
                continue
            ax.text(x, mean + std + 0.02, f"{mean:.2f}", ha="center", va="bottom", fontsize=8)

    # This scratch bar chart still uses tight_layout, but the paper style may have switched
    # constrained layout on globally, and the two conflict.
    if not plt.rcParams["figure.constrained_layout.use"]:
        fig.tight_layout(rect=[0, 0.04, 1, 1])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def write_summary_json(
    *,
    correlations: dict[str, dict[int, dict[int, list[float]]]],
    output_path: Path,
) -> None:
    payload = {}
    for metric, by_size in correlations.items():
        payload[metric] = {}
        for ensemble_size, by_round in by_size.items():
            payload[metric][str(ensemble_size)] = {}
            for round_idx, vals in by_round.items():
                payload[metric][str(ensemble_size)][str(round_idx)] = {
                    "values": vals,
                    "mean": float(np.mean(vals)) if vals else None,
                    "std": float(np.std(vals)) if vals else None,
                    "n": len(vals),
                }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dirs", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--rounds", nargs="+", type=int, required=True)
    parser.add_argument("--ensemble_sizes", nargs="+", type=int, default=[2, 3, 4])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--frame_index", type=int, default=0)
    parser.add_argument("--num_action_samples", type=int, default=5)
    parser.add_argument("--inference_batch_size", type=int, default=16)
    parser.add_argument("--compute_only", action="store_true")
    parser.add_argument("--aggregate_only", action="store_true")
    args = parser.parse_args()
    if args.compute_only and args.aggregate_only:
        raise ValueError("--compute_only and --aggregate_only are mutually exclusive")

    rounds = sorted(dict.fromkeys(args.rounds))
    ensemble_sizes = sorted(dict.fromkeys(args.ensemble_sizes))
    per_run_results = []
    for run_dir in args.run_dirs:
        logger.info("Processing %s", run_dir)
        per_run_results.append(
            process_run(
                run_dir=run_dir,
                rounds=rounds,
                ensemble_sizes=ensemble_sizes,
                env="libero",
                device=args.device,
                frame_index=args.frame_index,
                num_action_samples=args.num_action_samples,
                inference_batch_size=args.inference_batch_size,
                aggregate_only=args.aggregate_only,
            )
        )

    if args.compute_only:
        logger.info("Compute-only mode: caches are ready; skipping aggregate plots.")
        return

    correlations = correlations_by_round(per_run_results, rounds, ensemble_sizes)
    aggregated_dir = args.output / "aggregated"
    write_summary_json(correlations=correlations, output_path=aggregated_dir / "correlations.json")
    plot_metric(
        values=correlations["pearson"],
        rounds=rounds,
        output_path=aggregated_dir / "pearson_across_rounds.png",
        metric_name="Pearson r",
    )
    plot_metric(
        values=correlations["spearman"],
        rounds=rounds,
        output_path=aggregated_dir / "spearman_across_rounds.png",
        metric_name="Spearman rho",
    )
    plot_metrics_side_by_side(
        correlations=correlations,
        rounds=rounds,
        output_path=aggregated_dir / "spearman_pearson_across_rounds.png",
    )
    seed_means = correlations_by_seed_summary(per_run_results, rounds, ensemble_sizes)
    (aggregated_dir / "seed_means.json").write_text(json.dumps(seed_means, indent=2))
    plot_calibration_summary(
        seed_means=seed_means,
        output_path=args.output / "calibration_summary.png",
    )
    logger.info("Done.")


if __name__ == "__main__":
    main()
