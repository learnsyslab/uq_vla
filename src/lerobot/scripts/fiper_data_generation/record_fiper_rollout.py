import logging
import random
import time
from itertools import cycle
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from tqdm import trange

from lerobot.configs import parser
from lerobot.configs.fiper_rollout_recording import FiperRolloutRecordingPipelineConfig
from lerobot.envs.factory import build_env_for_domain, make_env_pre_post_processors
from lerobot.envs.utils import add_envs_task, preprocess_observation
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import get_device_from_parameters
from lerobot.processor import PolicyAction, PolicyProcessorPipeline
from lerobot.utils.io_utils import get_task_dir, save_episode_video
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import get_safe_torch_device, init_logging


def rollout(
    env: gym.Env,
    policy: PreTrainedPolicy,
    env_preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    env_postprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    seed: int | None = None,
) -> tuple[dict[str, bool], list[np.ndarray]]:
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
        - Dictionary with the following entries:
            - "successful": True if the rollout was successful.
        - ep_frames: Rendered frames collected from each environment camera (if available), or a flat list of frames otherwise.
    """
    assert isinstance(policy, nn.Module), "Policy must be a PyTorch nn module."

    # Check device is available
    device = get_device_from_parameters(policy)

    # Initialize random number generator to deterministically select actions
    generator = torch.Generator(device=device).manual_seed(seed) if seed is not None else None

    if hasattr(env, "camera_names") and env.camera_names is not None:
        ep_frames: dict[str, list[np.ndarray]] = {
            cam: [] for cam in env.camera_names
        }
    else:
        ep_frames: list[np.ndarray] = []

    success = False

    start_time = time.time()

    # Reset the policy and environment
    policy.reset()
    observation, _ = env.reset(seed=seed)

    if hasattr(env, "camera_names") and env.camera_names is not None:
        for camera in env.camera_names:
            ep_frames[camera].append(env.unwrapped.render(camera_name=camera))
    else:
        ep_frames.append(env.render())

    max_episode_steps = env._max_episode_steps if env.spec is None else env.spec.max_episode_steps

    progbar = trange(
        max_episode_steps,
        desc=f"Running rollout with at most {max_episode_steps} steps."
    )
    for _ in progbar:
        # Numpy array to tensor and changing dictionary keys to LeRobot policy format.
        observation = preprocess_observation(observation)

        # Infer "task" from attributes of environments.
        observation = add_envs_task(env, observation)

        # Apply environment-specific preprocessing (e.g., LiberoProcessorStep for LIBERO)
        observation = env_preprocessor(observation)

        observation = preprocessor(observation)
        with torch.no_grad():
            action = policy.select_action(observation, generator)
        action = postprocessor(action)

        action_transition = {"action": action}
        action_transition = env_postprocessor(action_transition)
        action = action_transition["action"]

        # Convert to CPU / numpy.
        action_numpy: np.ndarray = action.to("cpu").numpy()
        assert action_numpy.ndim == 2, "Action dimensions should be (batch, action_dim)"

        # Apply the next action
        observation, _, terminated, truncated, info = env.step(action_numpy[0])
        if hasattr(env, "camera_names") and env.camera_names is not None:
            for camera in env.camera_names:
                ep_frames[camera].append(env.unwrapped.render(camera_name=camera))
        else:
            ep_frames.append(env.render())

        if info is not None and "is_success" in info:
            success = bool(info["is_success"])

        # Stop early if environment terminates
        done = terminated or truncated
        if done:
            break

    logging.info(f"Finished episode in {time.time() - start_time:.1f}s")

    info = {
        "successful": success,
    }

    return info, ep_frames

def choose_seed(rng: random.Random | None = None) -> int:
    """
    Return a fresh 32-bit random seed using a random number generator.
    """
    return rng.randrange(2**31 - 1)

def prepare_run_metadata(cfg: FiperRolloutRecordingPipelineConfig) -> dict[str, Any]:
    """
    Prepare metadata dictionary for this data recording run.
    """
    if cfg.policy.type == "flow_matching":
        horizon = cfg.policy.horizon
    elif cfg.policy.type == "smolvla":
        horizon = cfg.policy.chunk_size
    else:
        raise ValueError(
            f"Horizon cannot be determined for policy type '{cfg.policy.type}'. "
            "Expected a policy of type 'flow_matching' or 'smolvla'."
        )
    run_metadata = {
        "metadata": True,
        "task": cfg.env.task,
        "action_prediction_horizon": horizon,
        "action_execution_horizon": cfg.policy.n_action_steps,
        "action_batch_size": cfg.fiper_rollout_recorder.num_uncertainty_sequences,
    }

    return run_metadata

@parser.wrap()
def main(cfg: FiperRolloutRecordingPipelineConfig):
    # Set global seed
    if cfg.seed is not None:
        set_seed(cfg.seed)

    # Check device is available
    device = get_safe_torch_device(cfg.policy.device, log=True)

    # Random number generator to choose seeds for the single runs
    rng = random.Random(cfg.seed)

    allowed_policies = {"flow_matching", "smolvla"}
    if cfg.policy.type not in allowed_policies:
        raise ValueError(
            f"eval_uncertainty_estimation.py only supports policy types {allowed_policies}, "
            f"but got '{cfg.policy.type}'."
        )

    run_metadata = prepare_run_metadata(cfg)

    logging.info("Making environment.")
    # Build evaluation environments for each domain (ID/OOD)
    envs_by_domain: dict[str, dict[str, dict[int, gym.Env]]] = {
        domain: build_env_for_domain(cfg.env, domain)
        for domain in cfg.domains
    }

    logging.info("Loading policy.")
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

    # Create environment-specific preprocessor and postprocessor (e.g., for LIBERO environments)
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=cfg.env, policy_cfg=cfg.policy)

    policy.init_fiper_rollout_recorder(
        config=cfg.fiper_rollout_recorder,
    )

    # Alternate ID/OOD for test set, calibration set only contains ID rollouts
    test_domain_cycle = cycle(envs_by_domain.keys())

    # Counters for collected rollouts in calibration and test set
    n_collected_calib_eps: int = 0
    n_collected_test_eps: int = 0

    # Alternate between collecting calibration and test episodes until quotas are met
    total_num_eps = cfg.n_calib_episodes + cfg.n_test_episodes
    progbar = trange(
        total_num_eps,
        desc=f"Recording FIPER data for {total_num_eps} episodes."
    )
    for _ in progbar:
        # Decide whether to collect a calibration or test episode next
        collect_calib = n_collected_calib_eps < cfg.n_calib_episodes
        collect_test = n_collected_test_eps < cfg.n_test_episodes

        if collect_calib and collect_test:
            # Prioritize collection of calibration set
            target_split = "calibration" if 2 * n_collected_calib_eps <= n_collected_test_eps else "test"
        elif collect_calib:
            target_split = "calibration"
        elif collect_test:
            target_split = "test"
        else:
            break

        # Data recording for calibration set
        if target_split == "calibration":
            # Flatten envs into list of (task_group, task_id, env)
            tasks = [(tg, tid, env) for tg, group in envs_by_domain["id"].items() for tid, env in group.items()]
            for task_group, task_id, env in tasks:
                attempt = 0
                while True:
                    attempt += 1
                    # Reset the FIPER data recorder before a new episode
                    policy.fiper_rollout_recorder.reset()

                    seed = choose_seed(rng)
                    info, frames = rollout(
                        env=env,
                        policy=policy,
                        env_postprocessor=env_postprocessor,
                        env_preprocessor=env_preprocessor,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        seed=seed,
                    )

                    max_attempts = cfg.max_calibration_attempts_per_task
                    reached_max_attempts = max_attempts is not None and attempt >= max_attempts

                    # Accept successful ID rollouts for calibration. For smoke tests,
                    # optionally save the last failed attempt to keep collection finite.
                    if info["successful"] or reached_max_attempts:
                        if reached_max_attempts and not info["successful"]:
                            logging.warning(
                                "Calibration episode for task %s/%s was not successful after %s attempts; "
                                "saving failed rollout for smoke-test continuity.",
                                task_group,
                                task_id,
                                attempt,
                            )
                        task_dir = get_task_dir(out_root=cfg.output_dir, task_group=task_group, task_id=task_id)
                        calib_dir = task_dir / "rollouts" / "calibration"
                        if cfg.save_videos:
                            save_episode_video(
                                ep_frames=frames,
                                out_root=calib_dir / "videos",
                                episode_idx=n_collected_calib_eps + 1,
                                fps=env.metadata["render_fps"],
                            )

                        ep_metadata = {
                            **run_metadata,
                            **info,
                            "task_id": task_id,
                            "episode": n_collected_calib_eps,
                            "rollout_type": "calibration",
                            "rollout_subtype": "ca",
                        }

                        policy.fiper_rollout_recorder.save_data(
                            output_dir=calib_dir,
                            episode_metadata=ep_metadata
                        )
                        break
                    else:
                        logging.info("Calibration episode was not successful; retrying...")

            n_collected_calib_eps += 1
        # Data recording for test set
        elif target_split == "test":
            domain = next(test_domain_cycle)
            logging.info(f"Evaluating {domain} environments for test set.")
            tasks = [(tg, tid, env) for tg, group in envs_by_domain[domain].items() for tid, env in group.items()]
            for task_group, task_id, env in tasks:
                # Reset the FIPER data recorder before a new episode
                policy.fiper_rollout_recorder.reset()

                seed = choose_seed(rng)
                rollout_info, ep_frames = rollout(
                    env=env,
                    policy=policy,
                    env_postprocessor=env_postprocessor,
                    env_preprocessor=env_preprocessor,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    seed=seed
                )

                task_dir = get_task_dir(out_root=cfg.output_dir, task_group=task_group, task_id=task_id)
                test_dir = task_dir / "rollouts" / "test"
                if cfg.save_videos:
                    save_episode_video(
                        ep_frames=ep_frames,
                        out_root=test_dir / "videos",
                        episode_idx=n_collected_test_eps + 1,
                        fps=env.metadata["render_fps"],
                    )

                ep_metadata = {
                    **run_metadata,
                    **rollout_info,
                    "task_id": task_id,
                    "episode": n_collected_test_eps,
                    "rollout_type": "test",
                    "rollout_subtype": domain,
                }

                policy.fiper_rollout_recorder.save_data(
                    output_dir=test_dir,
                    episode_metadata=ep_metadata,
                )

            n_collected_test_eps += 1

if __name__ == "__main__":
    init_logging()
    main()
