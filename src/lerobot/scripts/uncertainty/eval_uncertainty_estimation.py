#!/usr/bin/env python
"""
Evaluate a Flow Matching policy's uncertainty estimates for in-distribution and out-of-distribution scenarios.

Runs rollouts with multiple uncertainty estimation methods, records per-step uncertainty scores,
saves rollout videos, and generates comparison plots for each method.

Usage example:

You want to plot uncertainty estimation scores for the Composed Action Sequence Likelihood method for 10 episodes:

```
python src/lerobot/scripts/eval_uncertainty_estimation.py \
    --policy.path=outputs/train/flow_matching_pusht/checkpoints/last/pretrained_model \
    --policy.device=cuda \
    --env.type=pusht \
    --eval_uncert_est.n_episodes=10 \
    --eval_uncert_est.uncert_est_methods='["likelihood", "composed_sequence"]'
```
"""
import logging
import time
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from tqdm import trange

from lerobot.configs import parser
from lerobot.configs.eval_uncertainty_estimation import EvalUncertaintyEstimationPipelineConfig
from lerobot.envs.factory import build_env_for_domain
from lerobot.envs.utils import add_envs_task, preprocess_observation
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import get_device_from_parameters
from lerobot.processor import PolicyAction, PolicyProcessorPipeline
from lerobot.uncertainty.uncertainty_samplers.configuration_uncertainty_sampler import (
    UncertaintySamplerConfig,
)
from lerobot.uncertainty.uncertainty_scoring.scorer_artifacts import (
    ScorerArtifacts,
    build_scorer_artifacts_for_uncertainty_sampler,
)
from lerobot.utils.io_utils import get_task_dir, get_task_group_dir, save_episode_video
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import get_safe_torch_device, init_logging


def plot_uncertainties(
    uncert_est_method: str,
    scoring_metric: Optional[str],
    task_group: str,
    task_id: Optional[int],
    uncertainty_buckets: Dict[str, list[np.ndarray]],
    color_map: Dict[str, str],
    output_dir: Path,
):
    """
    Plot per-episode uncertainty trajectories and mean ± std bands.

    Generates two plots per task:
      - With individual episode curves.
      - With only aggregate mean ± std per bucket.

    Args:
        uncert_est_method: Name of the uncertainty estimation method.
        scoring_metric: Metric used for uncertainty scoring.
        task_group: Task group identifier.
        task_id: Task ID within the task group.
        uncertainty_buckets: Mapping from label (e.g., "ID Success") to episode uncertainty arrays.
        color_map: Mapping from bucket label to plot color.
        output_dir: Directory to save the plots.
    """
    # Compute mean ± std across episodes in each bucket
    all_uncertainties = [u for _, eps in uncertainty_buckets.items() for u in eps]
    max_len = max(len(u) for u in all_uncertainties) if all_uncertainties else 0

    def compute_uncert_stats(arrays: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        if not arrays:
            return np.full(max_len, np.nan), np.full(max_len, np.nan)
        means, stds = [], []
        for t in range(max_len):
            vals = np.array([a[t] for a in arrays if len(a) > t and np.isfinite(a[t])])
            means.append(vals.mean() if vals.size else np.nan)
            stds.append(vals.std() if vals.size else np.nan)
        return np.array(means), np.array(stds)

    for show_individual in (True, False):
        plt.figure()

        # Plot every episode as a faint line
        if show_individual:
            for label, uncertainties_per_episode in uncertainty_buckets.items():
                for ep_idx, uncertainties in enumerate(uncertainties_per_episode):
                    uncertainties = np.where(np.isneginf(uncertainties), np.nan, uncertainties)
                    plt.plot(
                        np.arange(len(uncertainties)),
                        uncertainties,
                        color=color_map[label],
                        alpha=0.3,
                        label=label if ep_idx == 0 else None,   # only first gets legend
                    )

        for label, uncertainties_per_episode in uncertainty_buckets.items():
            mean, std = compute_uncert_stats([np.where(np.isneginf(u), np.nan, u) for u in uncertainties_per_episode])
            plt.plot(
                np.arange(max_len),
                mean,
                color=color_map[label],
                label=None if show_individual else label
            )
            plt.fill_between(
                np.arange(max_len), mean - std, mean + std, color=color_map[label], alpha=0.5
            )

        # Presentation: fontsize=18, labelpad=4
        plt.xlabel("Rollout Step")

        if scoring_metric == "mode_distance":
            y_label = "Mode Distance Score"
        elif scoring_metric == "terminal_vel_norm":
            y_label = "Terminal Velocity Norm Score"
        elif scoring_metric == "vfd_oneway":
            y_label = "Velocity Diff. Score"
        elif scoring_metric == "likelihood":
            y_label = "Neg. Log-Likelihood"
        else:
            y_label = "Uncertainty Score"
        # Presentation: fontsize=18, labelpad=4
        plt.ylabel(ylabel=y_label)
        task_title = get_task_title(task_group=task_group, task_id=task_id)
        plt.title(f"{uncert_est_method.replace('_', ' ').title()} | {task_title}")
        plt.legend()
        # Presentation
        # plt.xticks(fontsize=16)
        # plt.yticks(fontsize=16)
        # plt.legend(fontsize=15)
        output_dir.mkdir(parents=True, exist_ok=True)
        # Presentation dpi=600
        prefix = "uncertainty_scores"
        filename = f"{prefix}_{uncert_est_method}_{task_group.lower()}"
        if "libero" in task_group and task_id is not None:
            filename += f"_task{task_id:02d}"
        if show_individual:
            filename += "_individual"
        plt.savefig(
            output_dir / (filename + ".png"),
            dpi=300, bbox_inches="tight"
        )
        plt.close()


def rollout(
    env: gym.Env,
    policy: PreTrainedPolicy,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run a single rollout of a policy in a gymnasium environment.

    Args:
        env: A single gymnasium environment to run the rollout in.
        policy: The pretrained policy to evaluate.
        preprocessor: A pipeline that processes raw environment observations into the format
            expected by the policy.
        postprocessor: A pipeline that processes the raw policy actions into the format expected
            by the environment.
        seed: Optional random seed for reproducibility of environment and policy behavior.

    Returns:
        Dictionary with the following entries:
            - "ep_uncertainties" (list[np.ndarray]): Per-step uncertainty scores recorded whenever the
                policy generates a new action sequence.
            - "ep_frames" (dict[str, list[np.ndarray]] or list[np.ndarray]): Rendered frames collected
                from each environment camera (if available), or a flat list of frames otherwise.
            - "outcome" (str): Either "success" or "failure".
    """
    assert isinstance(policy, nn.Module), "Policy must be a PyTorch nn module."

    # Check device is available
    device = get_device_from_parameters(policy)

    # Initialize random number generator to deterministically select actions
    generator = torch.Generator(device=device).manual_seed(seed) if seed is not None else None

    ep_uncertainties = []
    if hasattr(env, "camera_names") and env.camera_names is not None:
        ep_frames: Dict[str, list[np.ndarray]] = {
            cam: [] for cam in env.camera_names
        }
    else:
        ep_frames: list[np.ndarray] = []

    start_time = time.time()

    # Reset the policy and environment
    policy.reset()
    observation, _ = env.reset(seed=seed)

    if hasattr(env, "camera_names") and env.camera_names is not None:
        for camera in env.camera_names:
            ep_frames[camera].append(env.unwrapped.render(camera_name=camera))
    else:
        ep_frames.append(env.render())

    if env.spec is None:
        max_episode_steps = env._max_episode_steps
    else:
        max_episode_steps = env.spec.max_episode_steps

    progbar = trange(
        max_episode_steps,
        desc=f"Running rollout with at most {max_episode_steps} steps."
    )
    for _ in progbar:
        # Numpy array to tensor and changing dictionary keys to LeRobot policy format.
        observation = preprocess_observation(observation)

        # Infer "task" from attributes of environments.
        observation = add_envs_task(env, observation)
        observation = preprocessor(observation)

        # Decide whether a new sequence will be generated
        new_action_gen = len(policy._queues["action"]) == 0

        with torch.no_grad():
            action = policy.select_action(observation, generator)
        action = postprocessor(action)

        if new_action_gen:
            uncertainty = policy.uncertainty_sampler.uncertainty
            ep_uncertainties.append(uncertainty)

        # Apply the next action
        observation, _, terminated, truncated, info = env.step(action[0].to("cpu").numpy())
        if hasattr(env, "camera_names") and env.camera_names is not None:
            for camera in env.camera_names:
                ep_frames[camera].append(env.unwrapped.render(camera_name=camera))
        else:
            ep_frames.append(env.render())

        if info is not None and "is_success" in info:
            outcome = "success" if bool(info["is_success"]) else "failure"

        # Stop early if environment terminates
        done = terminated or truncated
        if done:
            break

    logging.info(f"Finished episode in {time.time() - start_time:.1f}s")

    info = {
        "ep_uncertainties": ep_uncertainties,
        "ep_frames": ep_frames,
        "outcome": outcome,
    }

    return info


def get_task_title(
    task_group: str,
    task_id: Optional[int],
) -> str:
    """
    Return a human-readable title for a given task.
    """
    if "libero" in task_group and task_id is not None:
        return f"{task_group.title()} | Task {task_id:02d}"
    elif "pusht" in task_group:
        return "PushT"
    return task_group.title()


def compose_uncertainty_buckets(
    uncertainty_buckets_for_method: Dict[Tuple[str, str], List[np.ndarray]],
    domains: List[str],
    collapse_success_failure: bool,
) -> tuple[Dict[str, List[np.ndarray]], Dict[str, str]]:
    if collapse_success_failure:
        # Merge success and failure scores per domain
        uncertainty_buckets: Dict[str, List[np.ndarray]] = {}
        for dom in domains:
            label = "In-Distribution" if dom.lower() == "id" else "Out-of-Distribution"
            merged_uncertainties: List[np.ndarray] = []
            for outcome in ("success", "failure"):
                merged_uncertainties.extend(uncertainty_buckets_for_method[(dom, outcome)])
            uncertainty_buckets[label] = merged_uncertainties
        color_map = {
            "In-Distribution": "C0",
            "Out-of-Distribution": "C1",
        }
    else:
        # Keep all uncertainty buckets: Requested domains ID/OoD × success/failure
        uncertainty_buckets = {}
        for dom in domains:
            for outcome in ("success", "failure"):
                label = f"{'ID' if dom == 'id' else 'OOD'} {outcome.title()}"
                uncertainty_buckets[label] = uncertainty_buckets_for_method[(dom, outcome)]
        color_map = {
            "ID Success": "C0",
            "ID Failure": "C3",
            "OOD Success": "C2",
            "OOD Failure": "C1",
        }
    return uncertainty_buckets, color_map


def evaluate_methods_on_task_env(
    cfg: EvalUncertaintyEstimationPipelineConfig,
    policy: PreTrainedPolicy,
    preprocessor,
    postprocessor,
    env: gym.Env,
    domain: str,
    task_group: str,
    task_id: int,
    episode_idx: int,
    uncertainty_config_by_method: Dict[str, UncertaintySamplerConfig],
    scorer_artifacts_by_method: Dict[str, ScorerArtifacts],
    scoring_metric_by_method: Dict[str, Optional[str]],
    uncertainty_scores: Dict[Tuple[str, int], Dict[str, Dict[Tuple[str, str], List[np.ndarray]]]],
    group_uncertainty_scores: Dict[str, Dict[str, Dict[Tuple[str, str], List[np.ndarray]]]],
    plot_group: bool,
    output_root: Path,
) -> None:
    """
    Run all uncertainty methods on a single (task_group, task_id, env) combination and plot the resulting
    uncertainty scores.
    """
    # Flatten dirs once
    task_dir = get_task_dir(out_root=output_root, task_group=task_group, task_id=task_id)
    task_group_dir = get_task_group_dir(out_root=output_root, task_group=task_group)
    fps = env.metadata.get("render_fps", 30)

    for method in cfg.eval_uncert_est.uncert_est_methods:
        # Initialize sampler for this method
        policy.init_uncertainty_sampler(
            config=uncertainty_config_by_method[method],
            scorer_artifacts=scorer_artifacts_by_method[method],
        )

        # Rollout
        ep_info = rollout(
            env=env,
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
        )

        # Update per-task and group uncertainty scores
        uncertainty_scores[(task_group, task_id)][method][(domain, ep_info["outcome"])].append(
            ep_info["ep_uncertainties"]
        )
        group_uncertainty_scores[task_group][method][(domain, ep_info["outcome"])].append(
            ep_info["ep_uncertainties"]
        )

        # Save episode video
        save_episode_video(
            ep_frames=ep_info["ep_frames"],
            out_root=task_dir / method / f"{domain}_{ep_info['outcome']}",
            episode_idx=episode_idx,
            fps=fps,
        )

        # Plot per-task uncertainty scores
        task_buckets, task_colors = compose_uncertainty_buckets(
            uncertainty_buckets_for_method=uncertainty_scores[(task_group, task_id)][method],
            domains=cfg.eval_uncert_est.domains,
            collapse_success_failure=cfg.eval_uncert_est.collapse_success_failure,
        )
        plot_uncertainties(
            uncert_est_method=method,
            scoring_metric=getattr(scoring_metric_by_method[method], "name", None),
            task_group=task_group,
            task_id=task_id,
            uncertainty_buckets=task_buckets,
            color_map=task_colors,
            output_dir=task_dir / method,
        )

        # Plot group-aggregated uncertainty scores
        if plot_group:
            group_buckets, group_colors = compose_uncertainty_buckets(
                uncertainty_buckets_for_method=group_uncertainty_scores[task_group][method],
                domains=cfg.eval_uncert_est.domains,
                collapse_success_failure=cfg.eval_uncert_est.collapse_success_failure,
            )
            plot_uncertainties(
                uncert_est_method=method,
                scoring_metric=getattr(scoring_metric_by_method[method], "name", None),
                task_group=task_group,
                task_id=None,
                uncertainty_buckets=group_buckets,
                color_map=group_colors,
                output_dir=task_group_dir / method,
            )


@parser.wrap()
def main(cfg: EvalUncertaintyEstimationPipelineConfig):
    """
    Run evaluation of uncertainty estimation methods on a policy that generates actions
    using a flow matching model.

    For each specified domain (e.g., ID/OOD), task, and uncertainty estimation method,
    this function runs multiple rollouts, records per-step uncertainty scores, and generates
    comparison plots of the collected scores.

    Args:
        cfg: Experiment configuration specifying the policy, environment(s),
            uncertainty estimation methods, number of episodes, and output settings.
    """
    # Set global seed
    set_seed(cfg.seed)

    # Check device is available
    device = get_safe_torch_device(cfg.policy.device, log=True)

    allowed_policies = {"flow_matching", "smolvla"}
    if cfg.policy.type not in allowed_policies:
        raise ValueError(
            f"eval_uncertainty_estimation.py only supports policy types {allowed_policies}, "
            f"but got '{cfg.policy.type}'."
        )

    logging.info("Making environment.")
    # Build evaluation environments for each domain (ID/OOD)
    envs_by_domain: Dict[str, Dict[str, Dict[int, gym.Env]]] = {
        domain: build_env_for_domain(cfg.env, domain)
        for domain in cfg.eval_uncert_est.domains
    }

    # Collect task IDs per group across domains
    task_ids_by_group: Dict[str, set[int]] = defaultdict(set)
    for envs in envs_by_domain.values():
        for tg, group in envs.items():
            task_ids_by_group[tg].update(group.keys())
    # Only plot group-level if a group actually has more than one task
    plot_group: Dict[str, bool] = {tg: (len(ids) > 1) for tg, ids in task_ids_by_group.items()}

    logging.info("Loading policy")
    policy: PreTrainedPolicy = make_policy(
        cfg.policy,
        env_cfg=cfg.env,
    ).to(device)
    policy.eval()

    # Build preprocessing/postprocessing pipelines for observations/actions
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        # The inference device is automatically set to match the detected hardware, overriding any previous device settings from training to ensure compatibility.
        preprocessor_overrides={"device_processor": {"device": str(policy.config.device)}},
    )

    # Prepare configs and scorer artifacts for each uncertainty estimation method
    scorer_artifacts_by_method: Dict[str, ScorerArtifacts] = {}
    uncertainty_config_by_method: Dict[str, UncertaintySamplerConfig] = {}
    scoring_metric_by_method: Dict[str, Optional[str]] = {}
    for uncert_est_method in cfg.eval_uncert_est.uncert_est_methods:
        uncertainty_config = deepcopy(cfg.uncertainty_sampler)
        uncertainty_config.type = uncert_est_method
        uncertainty_config_by_method[uncert_est_method] = uncertainty_config
        scorer_artifacts_by_method[uncert_est_method] = build_scorer_artifacts_for_uncertainty_sampler(
            uncertainty_sampler_cfg=uncertainty_config,
            policy=policy,
            preprocessor=preprocessor,
            dataset_cfg=cfg.dataset,
        )
        scoring_metric_by_method[uncert_est_method] = getattr(uncertainty_config.active_config, "scoring_metric", None)

    # Storage for uncertainty scores (per task) and group-aggregated scores
    uncertainty_scores: Dict[Tuple[str, int], Dict[str, Dict[Tuple[str, str], List[np.ndarray]]]] = defaultdict(
        lambda: defaultdict(lambda: {(d, o): [] for d in cfg.eval_uncert_est.domains for o in ("success", "failure")})
    )
    group_uncertainty_scores: Dict[str, Dict[str, Dict[Tuple[str, str], List[np.ndarray]]]] = defaultdict(
        lambda: defaultdict(lambda: {(d, o): [] for d in cfg.eval_uncert_est.domains for o in ("success", "failure")})
    )

    # Roll out each episode across all domains, tasks, and methods
    n_episodes = cfg.eval_uncert_est.n_episodes
    progbar = trange(
        n_episodes,
        desc=f"Evaluating uncertainty estimation for {n_episodes} episodes."
    )
    for episode in progbar:
        for domain, envs in envs_by_domain.items():
            # Flatten envs into list of (task_group, task_id, env)
            tasks = [(tg, tid, env) for tg, group in envs.items() for tid, env in group.items()]
            for task_group, task_id, env in tasks:
                logging.info(f"Evaluating {task_group} | Task ID {task_id}")
                # Evaluate uncertainty methods on this task instance
                evaluate_methods_on_task_env(
                    cfg=cfg,
                    policy=policy,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    env=env,
                    domain=domain,
                    task_group=task_group,
                    task_id=task_id,
                    episode_idx=episode,
                    uncertainty_config_by_method=uncertainty_config_by_method,
                    scorer_artifacts_by_method=scorer_artifacts_by_method,
                    scoring_metric_by_method=scoring_metric_by_method,
                    uncertainty_scores=uncertainty_scores,
                    group_uncertainty_scores=group_uncertainty_scores,
                    plot_group=plot_group[task_group],
                    output_root=cfg.output_dir,
                )


if __name__ == "__main__":
    init_logging()
    main()
