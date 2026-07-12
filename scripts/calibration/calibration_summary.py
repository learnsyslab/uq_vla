"""
Generate a calibration summary bar plot for a single experiment group.

For each method, computes:
  - Per-seed mean of per-round task-level Pearson / Spearman (uncertainty vs. success rate)
  - Mean ± std across seeds

Bars show −r so that "up = well calibrated" (uncertainty anti-correlates with success).

Usage:
    PYTHONPATH=src python scripts/calibration/calibration_summary.py \
        --run_dirs outputs/iterative_fine_tuning/uniform_leak3_weighted_pools_lr_schedule_history05_steps2000_s01 \
                   outputs/iterative_fine_tuning/uniform_leak3_weighted_pools_lr_schedule_history05_steps2000_s23 \
                   outputs/iterative_fine_tuning/uniform_leak3_weighted_pools_lr_schedule_history05_steps2000_s45 \
        --output plots/calibration_comparison/uniform_leak3_weighted_pools_lr_schedule_history05_steps2000/calibration_summary.png
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as scipy_stats

METHODS = [
    "action_l2",
    "ace",
    "decu",
    "ensemble_terminal_variance",
    # "ensemble_terminal_variance_b1",
    "vfd",
    # "vfd_laplace",
    # "vfd_oneway"
    "vlm_token_entropy",
    "vlm_perplexity",
]

LIBERO_TASKS = [("libero_10", i) for i in range(10)]
METAWORLD_TASKS = (
    [("hard", i) for i in range(5)] + [("very_hard", i) for i in range(5)]
)

METHOD_LABELS = {
    # "vfd_oneway": "VD-l (ours)",
    "vfd": "VFD (ours)",
    "action_l2": "Action-L2",
    "ace": "ACE",
    "decu": "DECU",
    "ensemble_terminal_variance": "GU",
    "ensemble_terminal_variance_b1": "GU-B1",
    "vfd_laplace": "VFD-Laplace",
    "vlm_token_entropy": "VLM-Entropy",
    "vlm_perplexity": "VLM-Perplexity",
}

METHOD_COLORS = {
    "vfd_oneway": "tab:purple",
    "vfd": "tab:blue",
    "action_l2": "tab:orange",
    "ace": "tab:green",
    "decu": "tab:red",
    "ensemble_terminal_variance": "tab:pink",
    "ensemble_terminal_variance_b1": "tab:olive",
    "vfd_laplace": "tab:brown",
    "vlm_token_entropy": "tab:gray",
    "vlm_perplexity": "tab:cyan",
}

YLIM = [0, 0.8]

def _load_task_list(run_dir: Path, env: str) -> list[tuple[str, int]]:
    if env == "libero":
        return LIBERO_TASKS

    config_path = run_dir / "iterative_fine_tuning_config.json"
    if not config_path.exists():
        return METAWORLD_TASKS

    dataset_cfg = json.loads(config_path.read_text()).get("dataset", {})
    metaworld_tasks = dataset_cfg.get("metaworld_tasks") or {}
    task_list = [
        (task_group, int(task_id))
        for task_group, task_ids in metaworld_tasks.items()
        for task_id in task_ids
    ]
    return task_list or METAWORLD_TASKS


def load_success_rates(
    round_dir: Path,
    env: str = "libero",
    task_list: list[tuple[str, int]] | None = None,
) -> dict[int, float] | None:
    eval_path = round_dir / "round_evaluation.json"
    if not eval_path.exists():
        return None
    data = json.loads(eval_path.read_text())
    for member_result in data.get("member_results", []):
        if member_result.get("member_index", 0) != 0:
            continue
        # Patch stale paths caused by folder rename leak3fixed → leak3
        result_path = Path(str(member_result["result_path"]).replace("leak3fixed", "leak3"))
        if not result_path.exists():
            return None
        member_eval = json.loads(result_path.read_text())
        if env == "libero":
            return {task["task_id"]: float(task["success_rate"]) for task in member_eval.get("per_task", [])}

        task_list = task_list or METAWORLD_TASKS
        global_id_map = {(task_group, task_id): gid for gid, (task_group, task_id) in enumerate(task_list)}
        result = {}
        for task in member_eval.get("per_task", []):
            key = (task["task_group"], task["task_id"])
            if key in global_id_map:
                result[global_id_map[key]] = float(task["success_rate"])
        return result
    return None


def load_episode_uncertainties(
    round_dir: Path,
    method: str,
    cache_dir_name: str = "calibration_comparison",
) -> dict[int, dict[int, float]] | None:
    cache = round_dir / cache_dir_name / f"{method}_episode_uncertainties.json"
    if not cache.exists():
        cache = round_dir / "methods_comparison" / f"{method}_episode_uncertainties.json"
    if not cache.exists():
        return None
    raw = json.loads(cache.read_text())
    return {int(t): {int(ep): v for ep, v in eps.items()} for t, eps in raw.items()}


def correlations_for_seed(
    run_dir: Path,
    max_round: int | None = None,
    cache_dir_name: str = "calibration_comparison",
    env: str = "libero",
) -> dict[str, dict[str, list[float]]]:
    """Return {method: {"pearson": [...per round...], "spearman": [...per round...]}}."""
    result: dict[str, dict[str, list[float]]] = {m: {"pearson": [], "spearman": []} for m in METHODS}
    task_list = _load_task_list(run_dir, env)

    round_dirs = sorted(run_dir.glob("round_*"), key=lambda p: int(p.name.split("_")[1]))
    if max_round is not None:
        round_dirs = [d for d in round_dirs if int(d.name.split("_")[1]) < max_round]
    for round_dir in round_dirs:
        success_rates = load_success_rates(round_dir, env=env, task_list=task_list)
        if success_rates is None:
            continue
        for method in METHODS:
            ep_uncs = load_episode_uncertainties(round_dir, method, cache_dir_name=cache_dir_name)
            if ep_uncs is None:
                continue
            # Task-level: mean uncertainty per task vs success rate
            task_ids = sorted(set(ep_uncs) & set(success_rates))
            if len(task_ids) < 3:
                continue
            uncs = [float(np.mean(list(ep_uncs[t].values()))) for t in task_ids]
            srs = [success_rates[t] for t in task_ids]
            pearson_r, _ = scipy_stats.pearsonr(uncs, srs)
            spearman_r, _ = scipy_stats.spearmanr(uncs, srs)
            result[method]["pearson"].append(float(pearson_r))
            result[method]["spearman"].append(float(spearman_r))

    return result


def main():
    global METHODS

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run_dirs", nargs="+", required=True, type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max_round", type=int, default=None, help="Only include rounds < max_round")
    parser.add_argument("--env", choices=["libero", "metaworld"], default="libero")
    parser.add_argument("--methods", nargs="+", default=None, choices=METHODS)
    parser.add_argument(
        "--cache_dir_name",
        type=str,
        default="calibration_comparison",
        help="Round-local uncertainty cache directory to read.",
    )
    args = parser.parse_args()
    if args.methods is not None:
        METHODS = args.methods

    # Per-seed mean across rounds
    seed_means: dict[str, dict[str, list[float]]] = {
        m: {"pearson": [], "spearman": []} for m in METHODS
    }
    for run_dir in args.run_dirs:
        corrs = correlations_for_seed(
            run_dir,
            max_round=args.max_round,
            cache_dir_name=args.cache_dir_name,
            env=args.env,
        )
        for method in METHODS:
            if corrs[method]["pearson"]:
                seed_means[method]["pearson"].append(float(np.mean(corrs[method]["pearson"])))
            if corrs[method]["spearman"]:
                seed_means[method]["spearman"].append(float(np.mean(corrs[method]["spearman"])))

    # Build bar data (flip sign: −r so up = good calibration)
    methods_present = [m for m in METHODS if seed_means[m]["pearson"]]
    n = len(methods_present)

    pearson_means = [-float(np.mean(seed_means[m]["pearson"])) for m in methods_present]
    pearson_stds  = [float(np.std(seed_means[m]["pearson"]))  for m in methods_present]
    spearman_means = [-float(np.mean(seed_means[m]["spearman"])) for m in methods_present]
    spearman_stds  = [float(np.std(seed_means[m]["spearman"]))  for m in methods_present]

    fig, axes = plt.subplots(1, 2, figsize=(4 + n * 1.2, 4), sharey=False)
    # fig.suptitle("Calibration summary (−r: uncertainty vs. success rate)", fontsize=11, fontweight="bold")

    for ax, means, stds, title, ylabel in [
        (axes[0], spearman_means, spearman_stds, "Spearman", "- Spearman $\\rho$"),
        (axes[1], pearson_means,  pearson_stds,  "Pearson", "- Pearson $r$"),
    ]:
        xs = np.arange(n)
        colors = [METHOD_COLORS[m] for m in methods_present]
        bars = ax.bar(xs, means, yerr=stds, color=colors, capsize=5, width=0.6, zorder=3)
        ax.set_xticks(xs)
        ax.set_xticklabels([METHOD_LABELS[m] for m in methods_present], fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.grid(True, alpha=0.3, axis="y", zorder=0)
        ax.set_ylim(YLIM)
        # Annotate values
        for x, mean, std in zip(xs, means, stds):
            ax.text(x, mean + std + 0.02, f"{mean:.2f}", ha="center", va="bottom", fontsize=8)

    n_seeds = len(args.run_dirs)
    # fig.text(0.5, 0.01, f"Mean ± std across {n_seeds} seeds, averaged over all rounds", ha="center", fontsize=8, color="gray")
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    # Also save as pdf
    plt.savefig(args.output.with_suffix(".pdf"), dpi=150, bbox_inches="tight")
    print(f"Saved: {args.output}")

    label_w = max(len(METHOD_LABELS.get(m, m)) for m in methods_present)
    print(f"\n{'Method':<{label_w}}  {'Spearman':>14}  {'Pearson':>14}")
    print("-" * (label_w + 32))
    for m, sm, ss, pm, ps in zip(methods_present, spearman_means, spearman_stds, pearson_means, pearson_stds):
        label = METHOD_LABELS.get(m, m)
        print(f"{label:<{label_w}}  {sm:+.2f} ± {ss:.2f}  {pm:+.2f} ± {ps:.2f}")


if __name__ == "__main__":
    main()
