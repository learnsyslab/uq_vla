"""Shared helpers of the calibration scripts: per-task success rates, cached per-episode uncertainties
and the task-level correlation between the two.

Calibration protocol (tab:calibration_summary / tab:calibration_full): in every round of a calibration
run, correlate each LIBERO-10 task's mean frame-0 uncertainty over its demonstrations with that task's
success rate (member 0), across the 10 tasks. Per seed pair, average the per-round correlations;
report mean +- population std over the seed pairs.
"""

import json
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats

# The uncertainty methods of the calibration table, in column order.
METHODS = [
    "action_l2",
    "ace",
    "decu",
    "ensemble_terminal_variance",
    "inter_vel_diff_2way",
    "vlm_token_entropy",
    "vlm_perplexity",
]

METHOD_LABELS = {
    "inter_vel_diff_2way": "VFD (ours)",
    "action_l2": "Action-L2",
    "ace": "ACE",
    "decu": "DECU",
    "ensemble_terminal_variance": "GU",
    "inter_vel_diff_2way_laplace": "VFD-Laplace",
    "vlm_token_entropy": "Entropy",
    "vlm_perplexity": "Perplexity",
}


def _member_result_path(round_dir: Path, recorded: str) -> Path:
    """A member evaluation's result file, rebased onto `round_dir` if the run directory moved."""
    result_path = Path(recorded)
    if not result_path.exists():
        parts = result_path.parts
        if round_dir.name in parts:
            result_path = round_dir.joinpath(*parts[parts.index(round_dir.name) + 1:])
    return result_path


def load_success_rates(round_dir: Path) -> dict[int, float] | None:
    """{task_id: success rate} of ensemble member 0 in this round, or None if not evaluated."""
    eval_path = round_dir / "round_evaluation.json"
    if not eval_path.exists():
        return None
    data = json.loads(eval_path.read_text())
    for member_result in data.get("member_results", []):
        if member_result.get("member_index", 0) != 0:
            continue
        result_path = _member_result_path(round_dir, member_result["result_path"])
        if not result_path.exists():
            return None
        member_eval = json.loads(result_path.read_text())
        return {task["task_id"]: float(task["success_rate"]) for task in member_eval.get("per_task", [])}
    return None


def load_episode_uncertainties(
    round_dir: Path,
    method: str,
    cache_dir_name: str = "calibration_comparison",
) -> dict[int, dict[int, float]] | None:
    """{task_id: {episode_id: uncertainty}} as cached by calibration_comparison.py."""
    cache = round_dir / cache_dir_name / f"{method}_episode_uncertainties.json"
    if not cache.exists():
        return None
    raw = json.loads(cache.read_text())
    return {int(t): {int(ep): v for ep, v in eps.items()} for t, eps in raw.items()}


def correlations_for_seed(
    run_dir: Path,
    max_round: int | None = None,
    cache_dir_name: str = "calibration_comparison",
    methods: list[str] | None = None,
) -> dict[str, dict[str, list[float]]]:
    """{method: {"pearson": [...per round...], "spearman": [...per round...]}} for rounds < max_round."""
    methods = methods or METHODS
    result: dict[str, dict[str, list[float]]] = {m: {"pearson": [], "spearman": []} for m in methods}
    round_dirs = sorted(run_dir.glob("round_*"), key=lambda p: int(p.name.split("_")[1]))
    if max_round is not None:
        round_dirs = [d for d in round_dirs if int(d.name.split("_")[1]) < max_round]
    for round_dir in round_dirs:
        success_rates = load_success_rates(round_dir)
        if success_rates is None:
            continue
        for method in methods:
            ep_uncs = load_episode_uncertainties(round_dir, method, cache_dir_name=cache_dir_name)
            if ep_uncs is None:
                continue
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
