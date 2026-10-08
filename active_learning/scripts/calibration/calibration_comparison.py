"""
Compare uncertainty metrics using checkpoints from
iterative fine-tuning experiments.

For each available round, loads the two ensemble member checkpoints, computes
per-episode uncertainties for each method on all LIBERO-10 candidate episodes,
then plots calibration (uncertainty vs. success rate) at task level and
episode level.

Results are cached per round so re-running skips already-computed rounds.

Usage:
    PYTHONPATH=src python scripts/calibration/calibration_comparison.py \
        --run_dirs outputs/calibration/smolvla/random_s01 \
        --rounds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 \
        --output plots/calibration_comparison/smolvla
"""

import argparse
import contextlib
import json
import logging
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats as scipy_stats
from tqdm import tqdm

from lerobot.configs.default import DatasetConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.utils import filter_libero_episodes
from safetensors.torch import load_model as load_model_as_safetensor

from lerobot.policies.factory import (
    get_policy_class,
    make_flow_matching_adapter_from_policy,
    make_pre_post_processors,
)
from lerobot.uncertainty.uncertainty_samplers.configuration_uncertainty_sampler import (
    ACESamplerConfig,
    CrossBayesianSamplerConfig,
    DECUSamplerConfig,
    LaplaceConfig,
    ScoringMetricConfig,
)
from lerobot.uncertainty.uncertainty_samplers.cross_bayesian_sampler import CrossBayesianSampler
from lerobot.uncertainty.uncertainty_samplers.decu_sampler import DECUSampler
from lerobot.uncertainty.uncertainty_samplers.entropy_sampler import ACESampler
from lerobot.uncertainty.uncertainty_scoring.ensemble_utils.factory import build_ensemble_models
from lerobot.uncertainty.uncertainty_scoring.laplace_utils.posterior_builder import (
    get_laplace_posterior,
    make_laplace_path,
    make_laplace_wrapper,
)
from lerobot.uncertainty.uncertainty_scoring.scorer_artifacts import ScorerArtifacts
from lerobot.uncertainty.uncertainty_samplers.vlm_token_sampler import (
    VLMTokenSampler,
    VLM_ENTROPY_METHOD,
    VLM_PERPLEXITY_METHOD,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LAPLACE_METHOD = "inter_vel_diff_laplace"
LAPLACE_2WAY_METHOD = "inter_vel_diff_2way_laplace"
TERMINAL_VARIANCE_LAPLACE_METHOD = "terminal_variance_laplace"
DECU_METHOD = "decu"
ENSEMBLE_TV_METHOD = "ensemble_terminal_variance"
ENSEMBLE_TV_B1_METHOD = "ensemble_terminal_variance_b1"
METHODS = [
    "inter_vel_diff",
    "inter_vel_diff_2way",
    "action_l2",
    # "terminal_variance",
    "ace",
    # LAPLACE_METHOD,
    # LAPLACE_2WAY_METHOD,
    # TERMINAL_VARIANCE_LAPLACE_METHOD,
    DECU_METHOD,
    ENSEMBLE_TV_METHOD,
    # ENSEMBLE_TV_B1_METHOD,
    VLM_ENTROPY_METHOD,
    VLM_PERPLEXITY_METHOD,
]
METHOD_CHOICES = sorted(
    set(
        METHODS
        + [
            LAPLACE_METHOD,
            LAPLACE_2WAY_METHOD,
            TERMINAL_VARIANCE_LAPLACE_METHOD,
        ]
    )
)
LAPLACE_CONFIG = LaplaceConfig(scopes=["action_out_proj"], calib_fraction=1.0, batch_size=4)

# (task_group, local_task_id) pairs in global-id order
LIBERO_TASKS = [("libero_10", i) for i in range(10)]
LIBERO_REPO_ID = "HuggingFaceVLA/libero"


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

def load_adapter(model_path: str, device: str):
    # Read the policy type from the checkpoint config so this works for any
    # flow-matching policy (smolvla, xvla, ...), not just smolvla.
    from lerobot.configs.policies import PreTrainedConfig

    policy_type = PreTrainedConfig.from_pretrained(model_path).type
    policy_cls = get_policy_class(policy_type)
    policy = policy_cls.from_pretrained(pretrained_name_or_path=model_path)
    policy.to(device)
    policy.eval()
    return policy, make_flow_matching_adapter_from_policy(policy=policy)


def reload_adapter(policy, model_path: str, device: str):
    """Reload weights in-place — skips model construction, ~10× faster than from_pretrained."""
    load_model_as_safetensor(policy, str(Path(model_path) / "model.safetensors"))
    policy.to(device)
    policy.eval()
    return make_flow_matching_adapter_from_policy(policy=policy)


def build_ensemble_sampler(
    adapter,
    ensemble_models,
    metric: str,
    num_action_samples: int,
    velocity_eval_times: tuple[float, ...] | None = None,
):
    scorer_artifacts = ScorerArtifacts(ensemble_models=ensemble_models)
    config = CrossBayesianSamplerConfig(
        scorer_type="ensemble",
        num_action_samples=num_action_samples,
        scoring_metric=ScoringMetricConfig(metric_type=metric, velocity_eval_times=velocity_eval_times),
    )
    return CrossBayesianSampler(
        config=config,
        sampler_model=adapter,
        scorer_artifacts=scorer_artifacts,
    )


def build_ensemble_tv_sampler(adapter, ensemble_models, num_action_samples: int):
    scorer_artifacts = ScorerArtifacts(ensemble_models=ensemble_models)
    config = CrossBayesianSamplerConfig(
        scorer_type="ensemble",
        num_action_samples=num_action_samples,
        scoring_metric=ScoringMetricConfig(metric_type="ensemble_terminal_variance"),
    )
    return CrossBayesianSampler(config=config, sampler_model=adapter, scorer_artifacts=scorer_artifacts)


def build_decu_sampler(adapter, ensemble_models, num_action_samples: int, branching_time: float = 0.995):
    """
    Build a DECUSampler. The ensemble used for the PaiDE score is composed of the
    primary `adapter` model plus all `ensemble_models`, giving M >= 2 members.
    """
    full_ensemble = [adapter, *ensemble_models]
    scorer_artifacts = ScorerArtifacts(ensemble_models=full_ensemble)
    config = DECUSamplerConfig(
        num_action_samples=num_action_samples,
        scoring_metric=ScoringMetricConfig(metric_type="decu", branching_time=branching_time),
        trunk_model_index=0,
    )
    return DECUSampler(
        config=config,
        sampler_model=adapter,
        scorer_artifacts=scorer_artifacts,
    )


def build_laplace_sampler(adapter, laplace_posterior, num_action_samples: int, metric_type: str = "inter_vel_diff"):
    scorer_artifacts = ScorerArtifacts(laplace_posterior=laplace_posterior)
    config = CrossBayesianSamplerConfig(
        scorer_type="laplace",
        num_action_samples=num_action_samples,
        scoring_metric=ScoringMetricConfig(metric_type=metric_type),
        laplace_config=LAPLACE_CONFIG,
    )
    return CrossBayesianSampler(
        config=config,
        sampler_model=adapter,
        scorer_artifacts=scorer_artifacts,
    )




# ---------------------------------------------------------------------------
# Dataset / observation helpers
# ---------------------------------------------------------------------------

def get_episode_frame_indices(dataset, episode_ids: list[int], frame_index: int = 0) -> list[int]:
    episodes = dataset.meta.episodes
    all_episode_indices = list(episodes["episode_index"])
    ep_id_to_pos = {ep_id: pos for pos, ep_id in enumerate(all_episode_indices)}
    frame_indices = []
    for ep_id in episode_ids:
        pos = ep_id_to_pos[ep_id]
        start = int(episodes["dataset_from_index"][pos])
        end = int(episodes["dataset_to_index"][pos])
        frame_indices.append(min(start + frame_index, end - 1))
    return frame_indices


def prepare_observation(sample: dict[str, Any], device: str) -> dict[str, torch.Tensor]:
    obs = {}
    for key, value in sample.items():
        if key == "task" and isinstance(value, (str, list, tuple)):
            # Keep the raw instruction as a per-sample list. SmolVLA/X-VLA tokenise it in their
            # preprocessor and ignore this key; FastWAM reads it to look up its precomputed text
            # context and fails with "Either `prompt` or both `context/context_mask`" without it.
            # Mirrors iterative_fine_tuning.selection._prepare_observation_for_sampler.
            obs[key] = [value] if isinstance(value, str) else [str(v) for v in value]
            continue
        if not isinstance(value, torch.Tensor):
            continue
        if "language.tokens" in key or "language.attention_mask" in key:
            # Already has a leading dim of 1 from tokenisation
            obs[key] = value.to(device) if (value.dim() == 2 and value.shape[0] == 1) else value.unsqueeze(0).to(device)
        elif "images" in key and "_is_pad" not in key and "_padding_mask" not in key:
            obs[key] = value.unsqueeze(0).to(device)
        elif "state" in key and "_is_pad" not in key:
            obs[key] = value.unsqueeze(0).to(device)
        else:
            obs[key] = value.unsqueeze(0).to(device) if value.dim() == 0 else value.unsqueeze(0).to(device)
    return obs


def collate_observations(observations: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Stack a list of batch-1 observations into a single batched observation.

    Tensors are concatenated on the batch dim; list-valued entries (the raw `task` strings that
    FastWAM consumes) are concatenated as lists so they stay aligned with the batch.
    """
    out: dict[str, Any] = {}
    for key in observations[0]:
        values = [obs[key] for obs in observations]
        if isinstance(values[0], torch.Tensor):
            out[key] = torch.cat(values, dim=0)
        else:
            out[key] = [item for v in values for item in (v if isinstance(v, (list, tuple)) else [v])]
    return out


# ---------------------------------------------------------------------------
# Uncertainty computation
# ---------------------------------------------------------------------------

def compute_episode_uncertainties(
    sampler,
    policy_config,
    pretrained_path: str,
    device: str,
    env: str = "libero",
    task_list: list[tuple[str, int]] | None = None,
    frame_index: int = 0,
    inference_batch_size: int = 8,
    round_idx: int | None = None,
    method: str | None = None,
    dataset_root: str | None = None,
) -> dict[int, dict[int, float]]:
    """Returns {global_task_id: {episode_id: uncertainty}} for all candidate tasks."""
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=policy_config,
        pretrained_path=pretrained_path,
    )
    results: dict[int, dict[int, float]] = {}

    prefix = ""
    if round_idx is not None and method is not None:
        prefix = f"[round {round_idx} | {method}] "

    task_list = task_list or LIBERO_TASKS
    repo_id = LIBERO_REPO_ID

    for global_task_id, (task_group, local_task_id) in enumerate(task_list):
        dataset_cfg = DatasetConfig(repo_id=repo_id, root=dataset_root, libero_tasks={task_group: [local_task_id]})

        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
            ds = make_dataset(dataset_cfg=dataset_cfg, policy_cfg=policy_config)
            episode_ids = filter_libero_episodes(ds, tasks_to_use={task_group: [local_task_id]})
            frame_indices = get_episode_frame_indices(ds, episode_ids, frame_index=frame_index)
        pairs = list(zip(episode_ids, frame_indices))

        task_results: dict[int, float] = {}
        for start in tqdm(
            range(0, len(pairs), inference_batch_size),
            desc=f"{prefix}task {global_task_id}/{len(task_list) - 1} ({task_group}:{local_task_id})",
            leave=True,
        ):
            batch_pairs = pairs[start:start + inference_batch_size]
            batch_obs = [prepare_observation(preprocessor(ds[fi]), device) for _, fi in batch_pairs]
            batched = collate_observations(batch_obs)
            with torch.no_grad():
                _, uncertainties = sampler.conditional_sample_with_uncertainty_batch(observation=batched)
            for (ep_id, _), unc in zip(batch_pairs, uncertainties.tolist()):
                task_results[ep_id] = float(unc)

        results[global_task_id] = task_results

    return results


# ---------------------------------------------------------------------------
# Loading success rates
# ---------------------------------------------------------------------------

def load_success_rates(
    round_dir: Path,
    env: str = "libero",
    task_list: list[tuple[str, int]] | None = None,
) -> dict[int, float] | None:
    eval_path = round_dir / "round_evaluation.json"
    if not eval_path.exists() or eval_path.stat().st_size == 0:
        return None
    data = json.loads(eval_path.read_text())

    for member_result in data.get("member_results", []):
        if member_result.get("member_index", 0) != 0:
            continue
        member_eval_path = Path(member_result["result_path"])
        if not member_eval_path.exists():
            # Stored path may use a stale run name; reconstruct relative to round_dir.
            member_idx = member_result.get("member_index", 0)
            fallback = round_dir / "training" / f"member_{member_idx:02d}" / "member_evaluation.json"
            if fallback.exists():
                member_eval_path = fallback
            else:
                return None
        member_eval = json.loads(member_eval_path.read_text())
        return {task["task_id"]: float(task["success_rate"]) for task in member_eval.get("per_task", [])}

    return None


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def scatter_calibration(
    ax: plt.Axes,
    uncertainties: list[float],
    success_rates: list[float],
    labels: list[str],
    title: str,
    xlabel: str,
):
    if len(uncertainties) >= 3:
        r, _ = scipy_stats.pearsonr(uncertainties, success_rates)
        r_str = f"r={r:.2f}"
    else:
        r_str = "n<3"

    colors = plt.get_cmap("tab10")(np.linspace(0, 1, max(len(uncertainties), 1)))
    for unc, sr, label, color in zip(uncertainties, success_rates, labels, colors):
        ax.scatter(unc, sr, color=color, s=60, zorder=3)
        ax.annotate(label, (unc, sr), textcoords="offset points", xytext=(4, 4), fontsize=7, color=color)

    ax.set_title(f"{title}  ({r_str})", fontsize=9)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel("Success rate", fontsize=8)
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
    ax.axhline(1, color="gray", linewidth=0.5, linestyle=":")
    ax.grid(True, alpha=0.3)


def plot_round_comparison(
    round_results: dict[str, dict],
    round_idx: int,
    output_path: Path,
):
    fig, axes = plt.subplots(2, len(METHODS), figsize=(6 * len(METHODS), 8))
    if len(METHODS) == 1:
        axes = axes.reshape(2, 1)
    fig.suptitle(f"Method Comparison — Round {round_idx}", fontsize=13, fontweight="bold")

    for col, method in enumerate(METHODS):
        if method not in round_results:
            axes[0][col].set_visible(False)
            axes[1][col].set_visible(False)
            continue

        data = round_results[method]
        task_ids_with_sr = sorted(
            t for t in data["task_uncertainties"] if t in data["task_success_rates"]
        )

        # Task-level
        scatter_calibration(
            axes[0][col],
            uncertainties=[data["task_uncertainties"][t] for t in task_ids_with_sr],
            success_rates=[data["task_success_rates"][t] for t in task_ids_with_sr],
            labels=[str(t) for t in task_ids_with_sr],
            title=f"{method} — task-level",
            xlabel="Mean episode uncertainty",
        )

        # Episode-level (task success rate as proxy per episode)
        ep_uncs, ep_srs, ep_labels = [], [], []
        for t in task_ids_with_sr:
            sr = data["task_success_rates"][t]
            for ep_id, unc in data["episode_uncertainties"][t].items():
                ep_uncs.append(unc)
                ep_srs.append(sr)
                ep_labels.append(f"T{t}")

        scatter_calibration(
            axes[1][col],
            uncertainties=ep_uncs,
            success_rates=ep_srs,
            labels=ep_labels,
            title=f"{method} — episode-level",
            xlabel="Episode uncertainty",
        )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved: {output_path}")
    plt.close()


def plot_correlation_across_rounds(
    seed_rs: dict[int, dict[str, list[float]]],
    output_path: Path,
    metric_name: str,
    ylabel: str,
):
    """Plot mean ± std of per-seed correlation values across rounds.

    seed_rs: {round_idx: {method: [corr_seed0, corr_seed1, ...]}}
    """
    rounds = sorted(seed_rs)
    method_colors = {
        "inter_vel_diff": "tab:blue",
        "inter_vel_diff_2way": "tab:purple",
        "action_l2": "tab:orange",
        "terminal_variance": "tab:cyan",
        "ace": "tab:green",
        LAPLACE_METHOD: "tab:red",
        LAPLACE_2WAY_METHOD: "tab:brown",
        TERMINAL_VARIANCE_LAPLACE_METHOD: "tab:brown",
        DECU_METHOD: "tab:cyan",
        ENSEMBLE_TV_METHOD: "tab:pink",
        ENSEMBLE_TV_B1_METHOD: "tab:olive",
        VLM_ENTROPY_METHOD: "tab:gray",
        VLM_PERPLEXITY_METHOD: "tab:brown",
    }

    fig, ax = plt.subplots(figsize=(max(8, len(rounds) * 0.9), 4))

    for method in METHODS:
        valid_rounds, means, stds = [], [], []
        for r in rounds:
            rs = seed_rs[r].get(method, [])
            if len(rs) == 0:
                continue
            valid_rounds.append(r)
            means.append(float(np.mean(rs)))
            stds.append(float(np.std(rs)))
        if valid_rounds:
            color = method_colors.get(method)
            ax.errorbar(
                valid_rounds,
                means,
                yerr=stds,
                marker="o",
                label=method,
                color=color,
                linewidth=2,
                elinewidth=1.2,
                capsize=3,
                capthick=1.2,
            )
            ax.fill_between(
                valid_rounds,
                [m - s for m, s in zip(means, stds)],
                [m + s for m, s in zip(means, stds)],
                color=color, alpha=0.2,
            )

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.axhline(0.5, color="gray", linewidth=0.5, linestyle=":")
    ax.axhline(-0.5, color="gray", linewidth=0.5, linestyle=":")
    ax.set_title(
        f"{metric_name} (uncertainty vs. task success rate) — method comparison",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Round")
    ax.set_ylabel(ylabel)
    ax.set_ylim(-1.05, 0.0)
    ax.set_xticks(rounds)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved: {output_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def cache_path(
    round_dir: Path,
    method: str,
    cache_dir_name: str = "calibration_comparison",
) -> Path:
    return round_dir / cache_dir_name / f"{method}_episode_uncertainties.json"


def load_member_model_paths(round_dir: Path) -> tuple[Path, Path]:
    manifest_path = round_dir / "training_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        final_model_paths = [Path(path) for path in manifest.get("final_model_paths", [])]
        if len(final_model_paths) >= 2:
            return final_model_paths[0], final_model_paths[1]

    return (
        round_dir / "training" / "member_00" / "checkpoints" / "last" / "pretrained_model",
        round_dir / "training" / "member_01" / "checkpoints" / "last" / "pretrained_model",
    )


def cache_paths(
    round_dir: Path,
    method: str,
    cache_dir_name: str = "calibration_comparison",
) -> list[Path]:
    paths = [cache_path(round_dir, method, cache_dir_name=cache_dir_name)]
    if method == LAPLACE_METHOD:
        # Preserve compatibility with previously generated Laplace caches.
        paths.append(cache_path(round_dir, "laplace", cache_dir_name=cache_dir_name))
    return paths


def _load_or_fit_laplace_posterior(
    round_dir: Path,
    policy,
    preprocessor,
    member_00_path: Path,
    env: str = "libero",
    dataset_root: str | None = None,
):
    """Load or fit the Laplace posterior for the given round. Shared by both Laplace methods."""
    laplace_wrapper = make_laplace_wrapper(policy=policy, scopes=LAPLACE_CONFIG.scopes)
    laplace_path = make_laplace_path(
        laplace_wrapper=laplace_wrapper,
        pretrained_path=member_00_path,
        calib_fraction=LAPLACE_CONFIG.calib_fraction,
    )
    if not laplace_path.exists():
        return None, laplace_path
    from lerobot.configs.default import DatasetConfig as _DC
    repo_id = LIBERO_REPO_ID
    manifest = json.loads((round_dir / "training_manifest.json").read_text())
    laplace_posterior = get_laplace_posterior(
        policy=policy,
        preprocessor=preprocessor,
        laplace_config=LAPLACE_CONFIG,
        dataset_cfg=_DC(repo_id=repo_id, root=dataset_root),
        episode_ids=manifest["all_training_episode_ids"],
        pretrained_path=member_00_path,
    )
    return laplace_posterior, laplace_path


def load_cached(
    round_dir: Path,
    method: str,
    cache_dir_name: str = "calibration_comparison",
) -> dict[int, dict[int, float]] | None:
    for path in cache_paths(round_dir, method, cache_dir_name=cache_dir_name):
        if not path.exists():
            continue
        raw = json.loads(path.read_text())
        return {int(t): {int(ep): v for ep, v in eps.items()} for t, eps in raw.items()}
    return None


def save_cache(
    round_dir: Path,
    method: str,
    data: dict[int, dict[int, float]],
    cache_dir_name: str = "calibration_comparison",
):
    path = cache_path(round_dir, method, cache_dir_name=cache_dir_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(
        {str(t): {str(ep): v for ep, v in eps.items()} for t, eps in data.items()},
        indent=2,
    ))


# ---------------------------------------------------------------------------
# Per-run processing
# ---------------------------------------------------------------------------

def _load_dataset_root(run_dir: Path) -> str | None:
    """Load dataset.root from the saved run config JSON, so the dataset is found in the right place."""
    config_path = run_dir / "iterative_fine_tuning_config.json"
    if config_path.exists():
        return json.loads(config_path.read_text()).get("dataset", {}).get("root")
    return None


def _load_task_list(run_dir: Path, env: str) -> list[tuple[str, int]]:
    """The evaluated task list (the ten LIBERO-10 tasks)."""
    return LIBERO_TASKS


def process_run(
    run_dir: Path,
    output_dir: Path,
    device: str,
    num_action_samples: int,        # Generated samples
    inference_batch_size: int,      # Computation throughput
    rounds: list[int] | None = None,
    env: str = "libero",
    frame_index: int = 0,
    cache_dir_name: str = "calibration_comparison",
) -> dict[int, dict[str, dict]]:
    all_results: dict[int, dict[str, dict]] = {}

    dataset_root = _load_dataset_root(run_dir)
    task_list = _load_task_list(run_dir, env)

    # Kept alive across rounds to avoid re-building model architecture
    policy = None
    ensemble_models = None

    round_dirs = sorted(run_dir.glob("round_*"), key=lambda p: int(p.name.split("_")[1]))
    if rounds is not None:
        round_dirs = [d for d in round_dirs if int(d.name.split("_")[1]) in set(rounds)]
    for round_dir in round_dirs:
        round_idx = int(round_dir.name.split("_")[1])

        member_00_path, member_01_path = load_member_model_paths(round_dir)

        success_rates = load_success_rates(round_dir, env=env, task_list=task_list)
        if success_rates is None:
            logger.info(f"  Round {round_idx}: no evaluation results, skipping.")
            continue

        logger.info(f"  Round {round_idx}: processing...")
        round_results: dict[str, dict] = {}

        _non_ensemble = {LAPLACE_METHOD, LAPLACE_2WAY_METHOD, TERMINAL_VARIANCE_LAPLACE_METHOD, "ace", DECU_METHOD, ENSEMBLE_TV_METHOD, ENSEMBLE_TV_B1_METHOD, VLM_ENTROPY_METHOD, VLM_PERPLEXITY_METHOD}
        # CrossBayesianSampler-based ensemble methods (do not include DECU/ensemble_tv here, they have their own builders).
        ensemble_methods = [m for m in METHODS if m not in _non_ensemble and load_cached(round_dir, m, cache_dir_name=cache_dir_name) is None]
        # DECU and ensemble_tv also need the auxiliary ensemble member loaded.
        decu_pending = DECU_METHOD in METHODS and load_cached(round_dir, DECU_METHOD, cache_dir_name=cache_dir_name) is None
        ensemble_tv_pending = ENSEMBLE_TV_METHOD in METHODS and load_cached(round_dir, ENSEMBLE_TV_METHOD, cache_dir_name=cache_dir_name) is None
        ensemble_tv_b1_pending = ENSEMBLE_TV_B1_METHOD in METHODS and load_cached(round_dir, ENSEMBLE_TV_B1_METHOD, cache_dir_name=cache_dir_name) is None
        vlm_methods_pending = [m for m in (VLM_ENTROPY_METHOD, VLM_PERPLEXITY_METHOD) if m in METHODS and load_cached(round_dir, m, cache_dir_name=cache_dir_name) is None]
        needs_any_model = ensemble_methods or decu_pending or ensemble_tv_pending or ensemble_tv_b1_pending or vlm_methods_pending or any(
            m in METHODS and load_cached(round_dir, m, cache_dir_name=cache_dir_name) is None for m in (LAPLACE_METHOD, LAPLACE_2WAY_METHOD, TERMINAL_VARIANCE_LAPLACE_METHOD, "ace")
        )

        # Model checkpoints are often pruned after training. That is fine as long as
        # every requested method is already cached; only require the checkpoints when
        # we actually need to run inference for this round.
        if needs_any_model and (not member_00_path.exists() or not member_01_path.exists()):
            logger.info(
                f"  Round {round_idx}: missing checkpoints and not all methods cached, skipping."
            )
            continue

        if needs_any_model:
            if policy is None:
                policy, adapter = load_adapter(str(member_00_path), device)
            else:
                logger.info("    Reloading weights in-place (skipping model construction)...")
                adapter = reload_adapter(policy, str(member_00_path), device)
            preprocessor, _ = make_pre_post_processors(
                policy_cfg=policy.config, pretrained_path=str(member_00_path)
            )

        if ensemble_methods or decu_pending or ensemble_tv_pending or ensemble_tv_b1_pending:
            if ensemble_models is None:
                ensemble_models = build_ensemble_models([str(member_01_path)], policy.config)
            else:
                from safetensors.torch import load_file as _load_safetensors
                _sd = _load_safetensors(str(member_01_path / "model.safetensors"))
                # The checkpoint holds the *policy* state dict; the adapter wraps the inner
                # model, which the policy stores under `model` (SmolVLA, X-VLA) or
                # `flow_matching` (Push-T FlowMatchingPolicy). Strip that one prefix.
                _prefixes = ("model.", "flow_matching.")
                _sd = {
                    next((k[len(p):] for p in _prefixes if k.startswith(p)), k): v
                    for k, v in _sd.items()
                }
                # strict=False: some policies (e.g. X-VLA) tie weights (shared.weight
                # <-> encoder.embed_tokens.weight) that safetensors deduplicates on
                # save. The tied key is absent from the file but updated in-place via
                # its partner when loaded. No-op effect for untied policies (SmolVLA).
                missing, unexpected = ensemble_models[0].model.load_state_dict(_sd, strict=False)
                if unexpected:
                    raise RuntimeError(f"Unexpected keys reloading ensemble member: {unexpected[:5]}")
                ensemble_models[0].model.to(device).eval()

        for method in METHODS:
            cached = load_cached(round_dir, method, cache_dir_name=cache_dir_name)
            if cached is not None:
                logger.info(f"    {method}: loaded from cache.")
                episode_uncertainties = cached
            elif method in {LAPLACE_METHOD, LAPLACE_2WAY_METHOD, TERMINAL_VARIANCE_LAPLACE_METHOD}:
                laplace_posterior, laplace_path = _load_or_fit_laplace_posterior(
                    round_dir=round_dir,
                    policy=policy,
                    preprocessor=preprocessor,
                    member_00_path=member_00_path,
                    env=env,
                    dataset_root=dataset_root,
                )
                if laplace_posterior is None:
                    logger.warning(
                        f"    {method}: posterior not found at {laplace_path}. "
                        "Run fit_laplace.py first. Skipping."
                    )
                    continue
                logger.info(f"    {method}: running inference...")
                if method == TERMINAL_VARIANCE_LAPLACE_METHOD:
                    laplace_metric = "terminal_variance"
                elif method == LAPLACE_2WAY_METHOD:
                    laplace_metric = "inter_vel_diff_2way"
                else:
                    laplace_metric = "inter_vel_diff"
                sampler = build_laplace_sampler(adapter, laplace_posterior, num_action_samples, laplace_metric)
                episode_uncertainties = compute_episode_uncertainties(
                    sampler=sampler,
                    policy_config=policy.config,
                    pretrained_path=str(member_00_path),
                    device=device,
                    env=env,
                    task_list=task_list,
                    frame_index=frame_index,
                    inference_batch_size=inference_batch_size,
                    round_idx=round_idx,
                    method=method,
                    dataset_root=dataset_root,
                )
                save_cache(round_dir, method, episode_uncertainties, cache_dir_name=cache_dir_name)
            elif method == ENSEMBLE_TV_METHOD:
                logger.info(f"    {method}: running inference...")
                sampler = build_ensemble_tv_sampler(adapter, ensemble_models, num_action_samples)
                episode_uncertainties = compute_episode_uncertainties(
                    sampler=sampler,
                    policy_config=policy.config,
                    pretrained_path=str(member_00_path),
                    device=device,
                    env=env,
                    task_list=task_list,
                    frame_index=frame_index,
                    inference_batch_size=inference_batch_size,
                    round_idx=round_idx,
                    method=method,
                    dataset_root=dataset_root,
                )
                save_cache(round_dir, method, episode_uncertainties, cache_dir_name=cache_dir_name)
            elif method == ENSEMBLE_TV_B1_METHOD:
                logger.info(f"    {method}: running inference...")
                sampler = build_ensemble_tv_sampler(adapter, ensemble_models, num_action_samples=1)
                episode_uncertainties = compute_episode_uncertainties(
                    sampler=sampler,
                    policy_config=policy.config,
                    pretrained_path=str(member_00_path),
                    device=device,
                    env=env,
                    task_list=task_list,
                    frame_index=frame_index,
                    inference_batch_size=inference_batch_size,
                    round_idx=round_idx,
                    method=method,
                    dataset_root=dataset_root,
                )
                save_cache(round_dir, method, episode_uncertainties, cache_dir_name=cache_dir_name)
            elif method == DECU_METHOD:
                logger.info(f"    {method}: running inference...")
                sampler = build_decu_sampler(adapter, ensemble_models, num_action_samples)
                episode_uncertainties = compute_episode_uncertainties(
                    sampler=sampler,
                    policy_config=policy.config,
                    pretrained_path=str(member_00_path),
                    device=device,
                    env=env,
                    task_list=task_list,
                    frame_index=frame_index,
                    inference_batch_size=inference_batch_size,
                    round_idx=round_idx,
                    method=method,
                    dataset_root=dataset_root,
                )
                save_cache(round_dir, method, episode_uncertainties, cache_dir_name=cache_dir_name)
            elif method == "ace":
                logger.info(f"    {method}: running inference...")
                sampler = ACESampler(ACESamplerConfig(), adapter)
                episode_uncertainties = compute_episode_uncertainties(
                    sampler=sampler,
                    policy_config=policy.config,
                    pretrained_path=str(member_00_path),
                    device=device,
                    env=env,
                    task_list=task_list,
                    frame_index=frame_index,
                    inference_batch_size=inference_batch_size,
                    round_idx=round_idx,
                    method=method,
                    dataset_root=dataset_root,
                )
                save_cache(round_dir, method, episode_uncertainties, cache_dir_name=cache_dir_name)
            elif method in (VLM_ENTROPY_METHOD, VLM_PERPLEXITY_METHOD):
                logger.info(f"    {method}: running inference...")
                sampler = VLMTokenSampler(adapter, method)
                episode_uncertainties = compute_episode_uncertainties(
                    sampler=sampler,
                    policy_config=policy.config,
                    pretrained_path=str(member_00_path),
                    device=device,
                    env=env,
                    task_list=task_list,
                    frame_index=frame_index,
                    inference_batch_size=inference_batch_size,
                    round_idx=round_idx,
                    method=method,
                    dataset_root=dataset_root,
                )
                save_cache(round_dir, method, episode_uncertainties, cache_dir_name=cache_dir_name)
            else:
                logger.info(f"    {method}: running inference...")
                sampler = build_ensemble_sampler(adapter, ensemble_models, method, num_action_samples)
                episode_uncertainties = compute_episode_uncertainties(
                    sampler=sampler,
                    policy_config=policy.config,
                    pretrained_path=str(member_00_path),
                    device=device,
                    env=env,
                    task_list=task_list,
                    frame_index=frame_index,
                    inference_batch_size=inference_batch_size,
                    round_idx=round_idx,
                    method=method,
                    dataset_root=dataset_root,
                )
                save_cache(round_dir, method, episode_uncertainties, cache_dir_name=cache_dir_name)

            task_uncertainties = {
                task_id: float(np.mean(list(eps.values())))
                for task_id, eps in episode_uncertainties.items()
            }
            round_results[method] = {
                "task_uncertainties": task_uncertainties,
                "task_success_rates": success_rates,
                "episode_uncertainties": episode_uncertainties,
            }

        all_results[round_idx] = round_results
        plot_round_comparison(
            round_results=round_results,
            round_idx=round_idx,
            output_path=output_dir / run_dir.name / f"round_{round_idx:03d}_comparison.png",
        )

    return all_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global METHODS

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run_dirs", nargs="+", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("plots/calibration_comparison"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_action_samples", type=int, default=5)
    parser.add_argument("--inference_batch_size", type=int, default=64,
                        help="Number of episodes to process in a single forward pass.")
    parser.add_argument("--rounds", nargs="+", type=int, default=None,
                        help="Only process these round indices (e.g. --rounds 0 1 2). Default: all rounds.")
    parser.add_argument("--frame_index", type=int, default=0,
                        help="Frame offset inside each episode to score. Default: 0.")
    parser.add_argument("--methods", nargs="+", default=None,
                        choices=METHOD_CHOICES,
                        help="Only process these uncertainty methods. Default: all configured methods.")
    args = parser.parse_args()

    if args.methods is not None:
        METHODS = args.methods

    for rd in args.run_dirs:
        if not rd.exists():
            raise FileNotFoundError(f"Run directory not found: {rd}")

    per_run_results: list[dict[int, dict[str, dict]]] = []
    for run_dir in args.run_dirs:
        logger.info(f"Processing {run_dir.name}...")
        per_run_results.append(
            process_run(
                run_dir,
                args.output,
                args.device,
                args.num_action_samples,
                args.inference_batch_size,
                args.rounds,
                "libero",
                frame_index=args.frame_index,
                cache_dir_name=(
                    "calibration_comparison"
                    if args.frame_index == 0
                    else f"calibration_comparison_frame{args.frame_index}"
                ),
            )
        )

    # Aggregate across seeds
    all_rounds = sorted({r for run in per_run_results for r in run})

    # For the correlation plots: per-seed Pearson / Spearman values
    pearson_rs: dict[int, dict[str, list[float]]] = {}
    spearman_rs: dict[int, dict[str, list[float]]] = {}
    # For the round scatter plots: mean task uncertainties / success rates across seeds
    aggregated: dict[int, dict[str, dict]] = {}

    for round_idx in all_rounds:
        pearson_rs[round_idx] = {}
        spearman_rs[round_idx] = {}
        aggregated[round_idx] = {}
        for method in METHODS:
            task_uncs_all: dict[int, list[float]] = {}
            task_srs_all: dict[int, list[float]] = {}
            ep_uncs_all: dict[int, dict[int, list[float]]] = {}
            pearson_this_method: list[float] = []
            spearman_this_method: list[float] = []

            for run in per_run_results:
                if round_idx not in run or method not in run[round_idx]:
                    continue
                data = run[round_idx][method]
                for tid, unc in data["task_uncertainties"].items():
                    task_uncs_all.setdefault(tid, []).append(unc)
                for tid, sr in data["task_success_rates"].items():
                    task_srs_all.setdefault(tid, []).append(sr)
                for tid, eps in data["episode_uncertainties"].items():
                    ep_uncs_all.setdefault(tid, {})
                    for ep_id, unc in eps.items():
                        ep_uncs_all[tid].setdefault(ep_id, []).append(unc)
                # Pearson / Spearman for this seed
                task_ids = sorted(set(data["task_uncertainties"]) & set(data["task_success_rates"]))
                if len(task_ids) >= 3:
                    uncs = [data["task_uncertainties"][t] for t in task_ids]
                    srs = [data["task_success_rates"][t] for t in task_ids]
                    pearson_r, _ = scipy_stats.pearsonr(uncs, srs)
                    spearman_r, _ = scipy_stats.spearmanr(uncs, srs)
                    pearson_this_method.append(float(pearson_r))
                    spearman_this_method.append(float(spearman_r))

            if not task_uncs_all:
                continue
            pearson_rs[round_idx][method] = pearson_this_method
            spearman_rs[round_idx][method] = spearman_this_method
            aggregated[round_idx][method] = {
                "task_uncertainties": {t: float(np.mean(v)) for t, v in task_uncs_all.items()},
                "task_success_rates": {t: float(np.mean(v)) for t, v in task_srs_all.items()},
                "episode_uncertainties": {
                    t: {ep: float(np.mean(v)) for ep, v in eps.items()}
                    for t, eps in ep_uncs_all.items()
                },
            }

    for round_idx, round_results in aggregated.items():
        plot_round_comparison(
            round_results=round_results,
            round_idx=round_idx,
            output_path=args.output / "aggregated" / f"round_{round_idx:03d}_comparison.png",
        )

    plot_correlation_across_rounds(
        seed_rs=pearson_rs,
        output_path=args.output / "aggregated" / "pearson_across_rounds.png",
        metric_name="Pearson r",
        ylabel="Pearson r",
    )
    plot_correlation_across_rounds(
        seed_rs=spearman_rs,
        output_path=args.output / "aggregated" / "spearman_across_rounds.png",
        metric_name="Spearman rho",
        ylabel="Spearman rho",
    )
    logger.info("Done.")


if __name__ == "__main__":
    main()
