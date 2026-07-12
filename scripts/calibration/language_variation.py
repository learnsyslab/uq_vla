"""
Language variation calibration analysis.

For a given run and round, evaluates each LIBERO-10 task under 5 prompt variants
(prompt 1 = original instruction, prompts 2-5 with increasing paraphrase distortion).

For each (task, prompt) pair computes:
  - success_rate: actual LIBERO rollouts (10 per default)
  - mean_uncertainty for multiple methods on dataset frames:
      - vfd_oneway
      - action_l2
      - ace

Calibration analysis: Spearman correlation between uncertainty rank and success
rate rank across the 5 prompt variants, per task and per uncertainty method. A
well-calibrated model should assign higher uncertainty to prompts that yield
lower success rates.

Results are cached per task under round_dir/language_variation/.

Usage:
    python scripts/calibration/language_variation.py \
        --run_dir /cluster/scratch/.../uniform_leak3_history05_steps2000_s01 \
        --round 19 \
        --output plots/language_variation \
        --n_rollouts 10
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
from contextlib import nullcontext
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
from lerobot.envs.configs import LiberoEnv as LiberoEnvCfg
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import preprocess_observation as preprocess_env_obs
from lerobot.policies.factory import (
    get_policy_class,
    make_flow_matching_adapter_from_policy,
    make_pre_post_processors,
)
from lerobot.uncertainty.uncertainty_samplers.configuration_uncertainty_sampler import (
    ACESamplerConfig,
    CrossBayesianSamplerConfig,
    DECUSamplerConfig,
    ScoringMetricConfig,
)
from lerobot.uncertainty.uncertainty_samplers.entropy_sampler import ACESampler
from lerobot.uncertainty.uncertainty_samplers.cross_bayesian_sampler import CrossBayesianSampler
from lerobot.uncertainty.uncertainty_samplers.decu_sampler import DECUSampler
from lerobot.uncertainty.uncertainty_scoring.ensemble_utils.factory import build_ensemble_models
from lerobot.uncertainty.uncertainty_scoring.scorer_artifacts import ScorerArtifacts
from lerobot.uncertainty.uncertainty_samplers.vlm_token_sampler import (
    VLMTokenSampler,
    VLM_ENTROPY_METHOD,
    VLM_PERPLEXITY_METHOD,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TASK_GROUP = "libero_10"
NUM_TASKS = 10
NUM_PROMPTS = 5
METHODS = [
    "vfd_oneway",
    "vfd",
    "action_l2",
    "ace",
    "decu",
    "ensemble_terminal_variance",
    VLM_ENTROPY_METHOD,
    VLM_PERPLEXITY_METHOD,
]
NUM_ACTION_SAMPLES = 5
PROMPTS_FILE = Path(__file__).parent / "language_variation_prompts.json"
METHOD_COLORS = {
    "vfd_oneway": "#1f77b4",
    "vfd": "#17becf",
    "action_l2": "tab:orange",
    "ace": "tab:green",
    "decu": "tab:cyan",
    "ensemble_terminal_variance": "tab:pink",
    VLM_ENTROPY_METHOD: "tab:gray",
    VLM_PERPLEXITY_METHOD: "tab:brown",
}


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

def load_prompts(path: Path = PROMPTS_FILE) -> dict[int, list[str]]:
    data = json.loads(path.read_text())
    return {int(k): v for k, v in data.items()}


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def _cache_path(round_dir: Path, task_id: int) -> Path:
    return round_dir / "language_variation" / f"task_{task_id:02d}.json"


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


def load_cached(round_dir: Path, task_id: int) -> dict | None:
    path = _cache_path(round_dir, task_id)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def save_cached(round_dir: Path, task_id: int, data: dict) -> None:
    path = _cache_path(round_dir, task_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


# ---------------------------------------------------------------------------
# Dataset / observation helpers (mirrored from calibration_comparison.py)
# ---------------------------------------------------------------------------

def get_initial_frame_indices(dataset, episode_ids: list[int]) -> list[int]:
    episodes = dataset.meta.episodes
    ep_id_to_pos = {ep_id: pos for pos, ep_id in enumerate(episodes["episode_index"])}
    return [int(episodes["dataset_from_index"][ep_id_to_pos[ep_id]]) for ep_id in episode_ids]


def prepare_observation(sample: dict[str, Any], device: str) -> dict[str, torch.Tensor]:
    obs = {}
    for key, value in sample.items():
        if not isinstance(value, torch.Tensor):
            continue
        if "language.tokens" in key or "language.attention_mask" in key:
            obs[key] = value.to(device) if (value.dim() == 2 and value.shape[0] == 1) else value.unsqueeze(0).to(device)
        elif "images" in key and "_is_pad" not in key and "_padding_mask" not in key:
            obs[key] = value.unsqueeze(0).to(device)
        elif "state" in key and "_is_pad" not in key:
            obs[key] = value.unsqueeze(0).to(device)
        else:
            obs[key] = value.unsqueeze(0).to(device) if value.dim() == 0 else value.unsqueeze(0).to(device)
    return obs


def collate_observations(observations: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {key: torch.cat([obs[key] for obs in observations], dim=0) for key in observations[0]}


def get_lang_tokens_for_prompt(preprocessor, raw_sample: dict, prompt: str, device: str) -> dict[str, torch.Tensor]:
    """Tokenize a custom prompt by overriding the task in a raw dataset sample."""
    # Strip any pre-tokenised language so the preprocessor re-tokenises from the string.
    sample = {k: v for k, v in raw_sample.items() if "language" not in k}
    sample["task"] = prompt
    obs = prepare_observation(preprocessor(sample), device)
    return {k: v for k, v in obs.items() if "language" in k}


# ---------------------------------------------------------------------------
# Uncertainty computation
# ---------------------------------------------------------------------------

def compute_prompt_uncertainties(
    sampler,
    policy_config,
    preprocessor,
    prompts: list[str],
    task_id: int,
    device: str,
    inference_batch_size: int = 8,
    method: str = "vfd_oneway",
) -> list[float]:
    """Return mean uncertainty for each prompt across dataset episodes."""
    dataset_cfg = DatasetConfig(
        repo_id="HuggingFaceVLA/libero",
        libero_tasks={TASK_GROUP: [task_id]},
    )
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        ds = make_dataset(dataset_cfg=dataset_cfg, policy_cfg=policy_config)
        episode_ids = filter_libero_episodes(ds, tasks_to_use={TASK_GROUP: [task_id]})
        frame_indices = get_initial_frame_indices(ds, episode_ids)
    pairs = list(zip(episode_ids, frame_indices))

    # Pre-fetch visual/state observations without language (will be swapped per prompt).
    base_obs = []
    for _, fi in pairs:
        obs = prepare_observation(preprocessor(ds[fi]), device)
        base_obs.append({k: v for k, v in obs.items() if "language" not in k})

    template_raw = dict(ds[frame_indices[0]])

    prompt_uncertainties: list[float] = []
    for prompt_idx, prompt in enumerate(prompts):
        lang_tokens = get_lang_tokens_for_prompt(preprocessor, template_raw, prompt, device)
        ep_uncertainties: list[float] = []
        for start in range(0, len(pairs), inference_batch_size):
            n = min(inference_batch_size, len(pairs) - start)
            batch_obs = [{**base_obs[start + i], **lang_tokens} for i in range(n)]
            batched = collate_observations(batch_obs)
            with torch.no_grad():
                _, uncs = sampler.conditional_sample_with_uncertainty_batch(observation=batched)
            ep_uncertainties.extend(uncs.tolist())
        prompt_uncertainties.append(float(np.mean(ep_uncertainties)))
        logger.info(
            f"    T{task_id} P{prompt_idx + 1} [{method}]: "
            f"unc={prompt_uncertainties[-1]:.4f}"
        )

    return prompt_uncertainties


# ---------------------------------------------------------------------------
# Success rate computation (rollouts)
# ---------------------------------------------------------------------------

def _extract_successes(info: dict, batch_size: int) -> list[bool]:
    for src in (info.get("final_info", {}), info):
        if isinstance(src, dict) and "is_success" in src:
            raw = src["is_success"]
            if isinstance(raw, np.ndarray):
                return raw.astype(bool).tolist()
            if isinstance(raw, list):
                return [bool(x) for x in raw]
        if isinstance(src, list):
            return [bool((x or {}).get("is_success", False)) for x in src]
    return [False] * batch_size


def evaluate_success_rate(
    policy,
    preprocessor,
    postprocessor,
    env_preprocessor,
    env_postprocessor,
    task_id: int,
    prompt: str,
    n_rollouts: int,
    eval_batch_size: int,
    max_steps: int,
    device_type: str,
    use_amp: bool,
) -> float:
    env_cfg = LiberoEnvCfg(task=TASK_GROUP, task_ids=[task_id])
    batch_size = min(eval_batch_size, n_rollouts)
    envs = make_env(env_cfg, n_envs=batch_size, use_async_envs=False)
    env = envs[TASK_GROUP][task_id]
    successes_total = 0
    try:
        for batch_start in range(0, n_rollouts, batch_size):
            remaining = n_rollouts - batch_start
            seeds = list(range(batch_start, batch_start + batch_size))
            if hasattr(env, "envs"):
                for i, sub_env in enumerate(env.envs):
                    base = getattr(sub_env, "unwrapped", sub_env)
                    init_states = getattr(base, "_init_states", None)
                    if init_states is not None:
                        base._init_state_id = (batch_start + i) % len(init_states)
            policy.reset()
            obs, _ = env.reset(seed=seeds)
            done = np.zeros(batch_size, dtype=bool)
            successes = np.zeros(batch_size, dtype=bool)
            task_batch = [prompt] * batch_size
            step = 0
            while not np.all(done) and step < max_steps:
                observation = preprocess_env_obs(obs)
                observation["task"] = task_batch
                observation = env_preprocessor(observation)
                observation = preprocessor(observation)
                ctx = torch.autocast(device_type=device_type) if use_amp and device_type in {"cuda", "cpu"} else nullcontext()
                with torch.inference_mode(), ctx:
                    action = policy.select_action(observation)
                action = postprocessor(action)
                if isinstance(action, torch.Tensor):
                    action = action.detach().cpu().numpy()
                if action.ndim == 1:
                    action = action[None, :]
                obs, _, terminated, truncated, info = env.step(action)
                batch_succ = np.array(_extract_successes(info, batch_size), dtype=bool)
                successes[~done] |= batch_succ[~done]
                done |= np.asarray(terminated, dtype=bool) | np.asarray(truncated, dtype=bool)
                step += 1
            successes_total += int(successes[:remaining].sum())
    finally:
        env.close()
    return successes_total / n_rollouts


# ---------------------------------------------------------------------------
# Calibration analysis
# ---------------------------------------------------------------------------

def compute_calibration(all_results: dict[int, dict]) -> dict:
    """Spearman r (uncertainty vs. success rate) across prompts, per task and method."""
    methods_summary: dict[str, dict] = {}
    for method in METHODS:
        per_task = []
        for task_id, result in sorted(all_results.items()):
            sr = result.get("success_rates")
            unc_by_method = result.get("uncertainties_by_method") or {}
            unc = unc_by_method.get(method)
            if sr is None or unc is None or len(sr) < 3 or len(unc) < 3:
                continue
            r, p = scipy_stats.spearmanr(unc, sr)
            per_task.append({"task_id": task_id, "spearman_r": float(r), "p_value": float(p)})
        rs = [x["spearman_r"] for x in per_task if np.isfinite(x["spearman_r"])]
        methods_summary[method] = {
            "per_task": per_task,
            "mean_spearman_r": float(np.mean(rs)) if rs else float("nan"),
            "std_spearman_r": float(np.std(rs)) if rs else float("nan"),
        }

    return {
        "methods": methods_summary,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _normalized_uncertainty_series(all_results: dict[int, dict]) -> dict[str, dict[int, dict[str, list[float]]]]:
    """Normalize per-prompt uncertainty means/stds to [0, 1] per method."""
    max_by_method: dict[str, float] = {}
    for result in all_results.values():
        unc_by_method = result.get("uncertainties_by_method") or {}
        for method, values in unc_by_method.items():
            finite_values = [float(v) for v in values if np.isfinite(float(v))]
            if finite_values:
                max_by_method[method] = max(max_by_method.get(method, 0.0), max(finite_values))

    normalized: dict[str, dict[int, dict[str, list[float]]]] = {}
    for task_id, result in all_results.items():
        unc_by_method = result.get("uncertainties_by_method") or {}
        unc_std_by_method = result.get("uncertainty_stds_by_method") or {}
        for method, values in unc_by_method.items():
            denom = max_by_method.get(method, 0.0)
            if denom <= 0:
                continue
            normalized.setdefault(method, {})[task_id] = {
                "means": [float(v) / denom for v in values],
                "stds": [float(v) / denom for v in unc_std_by_method.get(method, [])],
            }
    return normalized


def plot_success_rates(all_results: dict[int, dict], round_idx: int, run_name: str, output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 5, figsize=(20, 8), sharey=True)
    fig.suptitle(
        f"{run_name} — Round {round_idx}: success rate and normalized uncertainty by prompt variant",
        fontsize=13,
    )
    colors = ["#2196F3", "#64B5F6", "#FFAB40", "#FF7043", "#E53935"]
    labels = [f"P{i + 1}" for i in range(NUM_PROMPTS)]
    normalized_uncertainties = _normalized_uncertainty_series(all_results)
    method_offsets = {
        method: offset
        for method, offset in zip(METHODS, np.linspace(-0.22, 0.22, len(METHODS)))
    }
    method_markers = {
        "vfd_oneway": "o",
        "vfd": "D",
        "action_l2": "s",
        "ace": "^",
        "decu": "v",
        "ensemble_terminal_variance": "P",
        VLM_ENTROPY_METHOD: "X",
        VLM_PERPLEXITY_METHOD: "*",
    }

    for task_id, ax in enumerate(axes.flat):
        result = all_results.get(task_id)
        if result is None or result.get("success_rates") is None:
            ax.set_visible(False)
            continue
        rates = result["success_rates"]
        x = np.arange(len(labels))
        bars = ax.bar(labels, rates, color=colors, width=0.6)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"Task {task_id}", fontsize=10)
        ax.set_ylabel("Rate / normalized uncertainty" if task_id % 5 == 0 else "")
        ax.axhline(rates[0], color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.grid(True, axis="y", alpha=0.3)
        for bar, rate in zip(bars, rates):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{rate:.0%}", ha="center", va="bottom", fontsize=7)

        for method in METHODS:
            values = normalized_uncertainties.get(method, {}).get(task_id)
            if not values:
                continue
            means = values["means"]
            stds = values.get("stds") or []
            offset_x = x + method_offsets[method]
            yerr = stds if len(stds) == len(means) else None
            ax.errorbar(
                offset_x,
                means,
                yerr=yerr,
                fmt=method_markers.get(method, "o"),
                color=METHOD_COLORS[method],
                markerfacecolor=METHOD_COLORS[method],
                markeredgecolor="white",
                markeredgewidth=0.5,
                markersize=5,
                capsize=2 if yerr is not None else 0,
                linestyle="none",
                alpha=0.9,
                zorder=4,
            )

    handles = [
        plt.Line2D([0], [0], marker="s", linestyle="none", color=color, markersize=8, label=label)
        for color, label in zip(colors, labels)
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
    fig.text(0.5, 0.01, "P1 = original prompt  ·  P2–P5 = increasing paraphrase distortion  ·  dashed = P1 baseline",
             ha="center", fontsize=9, style="italic")
    plt.tight_layout(rect=[0, 0.09, 1, 1])
    out = output_dir / f"round_{round_idx:03d}_success_rates.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    logger.info(f"Saved: {out}")
    plt.close()


def plot_calibration(all_results: dict[int, dict], calibration: dict, round_idx: int, run_name: str, output_dir: Path) -> None:
    labels = [f"P{i + 1}" for i in range(NUM_PROMPTS)]
    prompt_colors = ["#2196F3", "#64B5F6", "#FFAB40", "#FF7043", "#E53935"]

    # Per-method scatter: uncertainty vs success rate
    fig, axes = plt.subplots(len(METHODS), 5, figsize=(20, 4 * len(METHODS)))
    if len(METHODS) == 1:
        axes = np.array([axes])
    method_summaries = calibration["methods"]
    fig.suptitle(
        f"{run_name} — Round {round_idx}: uncertainty vs. success rate by method and task",
        fontsize=12,
    )

    for row, method in enumerate(METHODS):
        calib_by_task = {x["task_id"]: x for x in method_summaries[method]["per_task"]}
        for task_id, ax in enumerate(axes[row]):
            result = all_results.get(task_id)
            unc_by_method = (result or {}).get("uncertainties_by_method") or {}
            if result is None or result.get("success_rates") is None or unc_by_method.get(method) is None:
                ax.set_visible(False)
                continue
            sr = result["success_rates"]
            unc = unc_by_method[method]
            for u, s, label, c in zip(unc, sr, labels, prompt_colors):
                ax.scatter(u, s, color=c, s=55, zorder=3)
                ax.annotate(label, (u, s), textcoords="offset points", xytext=(4, 4), fontsize=7)
            ax.set_xlabel("Mean uncertainty", fontsize=8)
            ax.set_ylabel(f"{method}\nSuccess rate" if task_id % 5 == 0 else "", fontsize=8)
            r_info = calib_by_task.get(task_id)
            r_str = f"r={r_info['spearman_r']:.2f}" if r_info else ""
            ax.set_title(f"Task {task_id}  {r_str}", fontsize=9)
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = output_dir / f"round_{round_idx:03d}_calibration_scatter.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    logger.info(f"Saved: {out}")
    plt.close()

    # Grouped summary bar: Spearman r per task and method
    fig2, ax2 = plt.subplots(figsize=(12, 4.5))
    width = 0.22
    task_ids = list(range(NUM_TASKS))
    x = np.arange(NUM_TASKS)
    for idx, method in enumerate(METHODS):
        calib_by_task = {
            item["task_id"]: item["spearman_r"]
            for item in method_summaries[method]["per_task"]
        }
        vals = [calib_by_task.get(task_id, np.nan) for task_id in task_ids]
        ax2.bar(
            x + (idx - (len(METHODS) - 1) / 2) * width,
            vals,
            width=width,
            label=f"{method} ({method_summaries[method]['mean_spearman_r']:.2f})",
            color=METHOD_COLORS[method],
            alpha=0.85,
        )
    ax2.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"T{task_id}" for task_id in task_ids])
    ax2.set_ylabel("Spearman r (uncertainty vs. success rate)")
    ax2.set_ylim(-1.1, 1.1)
    ax2.set_title(
        f"{run_name} — Round {round_idx}: calibration comparison by task\n"
        "Negative r = higher uncertainty -> lower success (desired)"
    )
    ax2.legend(title="method (mean r)")
    ax2.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    out2 = output_dir / f"round_{round_idx:03d}_calibration_spearman.png"
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    logger.info(f"Saved: {out2}")
    plt.close()

    # Method-level summary across tasks
    fig3, ax3 = plt.subplots(figsize=(7, 4))
    means = [method_summaries[m]["mean_spearman_r"] for m in METHODS]
    stds = [method_summaries[m]["std_spearman_r"] for m in METHODS]
    ax3.bar(METHODS, means, yerr=stds, capsize=4, color=[METHOD_COLORS[m] for m in METHODS], alpha=0.9)
    ax3.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax3.set_ylim(-1.1, 1.1)
    ax3.set_ylabel("Mean Spearman r across tasks")
    ax3.set_title(f"{run_name} — Round {round_idx}: method comparison summary")
    ax3.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    out3 = output_dir / f"round_{round_idx:03d}_calibration_method_summary.png"
    plt.savefig(out3, dpi=150, bbox_inches="tight")
    logger.info(f"Saved: {out3}")
    plt.close()


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def evaluate_round(
    run_dir: Path,
    round_idx: int,
    output_dir: Path,
    n_rollouts: int,
    eval_batch_size: int,
    max_steps: int,
    inference_batch_size: int,
    device: str,
    prompts: dict[int, list[str]],
    force: bool,
) -> None:
    round_dir = run_dir / f"round_{round_idx:03d}"
    if not round_dir.exists():
        raise FileNotFoundError(f"Round directory not found: {round_dir}")

    member_00, member_01 = load_member_model_paths(round_dir)
    for p in (member_00, member_01):
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {p}")

    # Load policy (for rollouts)
    policy_cls = get_policy_class("smolvla")
    policy = policy_cls.from_pretrained(pretrained_name_or_path=str(member_00))
    policy.to(device)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(member_00),
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    device_type = torch.device(device).type
    use_amp = bool(getattr(policy.config, "use_amp", False))

    # Load ensemble / samplers for uncertainty computation
    logger.info("Loading uncertainty samplers...")
    adapter = make_flow_matching_adapter_from_policy(policy=policy)
    ensemble_models = build_ensemble_models([str(member_01)], policy.config)
    scorer_artifacts = ScorerArtifacts(ensemble_models=ensemble_models)
    samplers = {
        method: CrossBayesianSampler(
            config=CrossBayesianSamplerConfig(
                scorer_type="ensemble",
                num_action_samples=NUM_ACTION_SAMPLES,
                scoring_metric=ScoringMetricConfig(metric_type=method),
            ),
            sampler_model=adapter,
            scorer_artifacts=scorer_artifacts,
        )
        for method in ("vfd_oneway", "vfd", "action_l2", "ensemble_terminal_variance")
    }
    samplers["ace"] = ACESampler(ACESamplerConfig(), adapter)
    decu_artifacts = ScorerArtifacts(ensemble_models=[adapter, ensemble_models[0]])
    samplers["decu"] = DECUSampler(
        config=DECUSamplerConfig(
            num_action_samples=NUM_ACTION_SAMPLES,
            scoring_metric=ScoringMetricConfig(metric_type="decu", branching_time=0.995),
            trunk_model_index=0,
        ),
        sampler_model=adapter,
        scorer_artifacts=decu_artifacts,
    )
    samplers[VLM_ENTROPY_METHOD] = VLMTokenSampler(adapter, VLM_ENTROPY_METHOD)
    samplers[VLM_PERPLEXITY_METHOD] = VLMTokenSampler(adapter, VLM_PERPLEXITY_METHOD)

    all_results: dict[int, dict] = {}
    out_dir = output_dir / run_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    for task_id in tqdm(range(NUM_TASKS), desc="tasks"):
        task_prompts = prompts[task_id]

        cached = load_cached(round_dir, task_id)
        has_sr = cached is not None and cached.get("success_rates") is not None
        if cached is not None and "uncertainties_by_method" not in cached and cached.get("uncertainties") is not None:
            cached["uncertainties_by_method"] = {"vfd_oneway": cached["uncertainties"]}
        unc_by_method = (cached or {}).get("uncertainties_by_method") or {}
        has_all_unc = all(method in unc_by_method for method in METHODS)

        if has_sr and has_all_unc and not force:
            logger.info(f"Task {task_id}: fully cached, skipping.")
            all_results[task_id] = cached
            continue

        result = cached or {"prompts": task_prompts, "n_rollouts": n_rollouts}

        # Success rates
        if not has_sr or force:
            logger.info(f"Task {task_id}: computing success rates...")
            env_cfg = LiberoEnvCfg(task=TASK_GROUP, task_ids=[task_id])
            env_pre, env_post = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy.config)
            success_rates = []
            for prompt_idx, prompt in enumerate(task_prompts):
                sr = evaluate_success_rate(
                    policy=policy,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    env_preprocessor=env_pre,
                    env_postprocessor=env_post,
                    task_id=task_id,
                    prompt=prompt,
                    n_rollouts=n_rollouts,
                    eval_batch_size=eval_batch_size,
                    max_steps=max_steps,
                    device_type=device_type,
                    use_amp=use_amp,
                )
                success_rates.append(sr)
                logger.info(f"  T{task_id} P{prompt_idx + 1}: sr={sr:.2f}")
            result["success_rates"] = success_rates

        # Uncertainties
        result.setdefault("uncertainties_by_method", {})
        if force:
            result["uncertainties_by_method"] = {}
        for method in METHODS:
            if method in result["uncertainties_by_method"] and not force:
                continue
            logger.info(f"Task {task_id}: computing uncertainties for {method}...")
            uncertainties = compute_prompt_uncertainties(
                sampler=samplers[method],
                policy_config=policy.config,
                preprocessor=preprocessor,
                prompts=task_prompts,
                task_id=task_id,
                device=device,
                inference_batch_size=inference_batch_size,
                method=method,
            )
            result["uncertainties_by_method"][method] = uncertainties
        # Preserve legacy single-method field for compatibility with existing outputs.
        if "vfd_oneway" in result["uncertainties_by_method"]:
            result["uncertainties"] = result["uncertainties_by_method"]["vfd_oneway"]

        save_cached(round_dir, task_id, result)
        all_results[task_id] = result

    calibration = compute_calibration(all_results)
    calib_out = out_dir / f"round_{round_idx:03d}_calibration.json"
    calib_out.write_text(json.dumps(calibration, indent=2))
    summary_str = ", ".join(
        f"{method}={calibration['methods'][method]['mean_spearman_r']:.3f}"
        for method in METHODS
    )
    logger.info(f"Calibration mean Spearman r: {summary_str}")

    plot_success_rates(all_results, round_idx, run_dir.name, out_dir)
    plot_calibration(all_results, calibration, round_idx, run_dir.name, out_dir)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--round", type=int, required=True, dest="round_idx")
    parser.add_argument("--output", type=Path, default=Path("plots/language_variation"))
    parser.add_argument("--n_rollouts", type=int, default=10)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=600)
    parser.add_argument("--inference_batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--prompts_file", type=Path, default=PROMPTS_FILE)
    parser.add_argument("--force", action="store_true", help="Recompute even if cached.")
    args = parser.parse_args()

    if not args.run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {args.run_dir}")

    prompts = load_prompts(args.prompts_file)
    evaluate_round(
        run_dir=args.run_dir,
        round_idx=args.round_idx,
        output_dir=args.output,
        n_rollouts=args.n_rollouts,
        eval_batch_size=args.eval_batch_size,
        max_steps=args.max_steps,
        inference_batch_size=args.inference_batch_size,
        device=args.device,
        prompts=prompts,
        force=args.force,
    )
    logger.info("Done.")


if __name__ == "__main__":
    main()
