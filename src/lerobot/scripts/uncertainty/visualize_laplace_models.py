#!/usr/bin/env python
import logging
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import gymnasium as gym
import numpy as np
import torch
from tqdm import trange

from lerobot.configs import parser
from lerobot.configs.visualize_laplace import VisualizeLaplacePipelineConfig
from lerobot.envs.factory import make_single_env
from lerobot.envs.utils import preprocess_observation
from lerobot.policies.factory import make_policy
from lerobot.uncertainty.uncertainty_scoring.laplace_utils.posterior_builder import (
    get_laplace_posterior,
    make_laplace_path,
)
from lerobot.utils.io_utils import write_video
from lerobot.utils.live_window import LiveWindow
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import get_safe_torch_device, init_logging


def capture_pusht_state(env: gym.Env) -> Dict[str, np.ndarray]:
    """Get state of Push-T env as [agent_x, agent_y, block_x, block_y, block_angle]
    as well as other info."""
    last_action = env.unwrapped._last_action if env.unwrapped._last_action is not None else None
    return {
        "state": np.array([
            *env.unwrapped.agent.position,
            *env.unwrapped.block.position,
            env.unwrapped.block.angle,
        ], dtype=np.float64),
        "agent_velocity": np.array([*env.unwrapped.agent.velocity]),
        "last_action": last_action
    }

def restore_pusht_state(env: gym.Env, info: Dict[str, np.ndarray]) -> np.ndarray:
    """Place PushT into the given state."""
    state = info["state"]
    env.unwrapped.agent.position = list(state[:2])
    env.unwrapped.block.position = list(state[2:4])
    env.unwrapped.block.angle = state[4]
    agent_velocity = list(info["agent_velocity"])
    env.unwrapped.agent.velocity = agent_velocity
    last_action = info["last_action"]
    env.unwrapped._last_action = last_action
    env.unwrapped.space.step(0.001)

    return env.get_obs()

def rollout(
    cfg: VisualizeLaplacePipelineConfig,
    env: gym.Env,
    policy,
    output_dir: Path,
    seed: int,
    env_states: Optional[List[Any]] = None
) -> list[Any]:
    device = get_safe_torch_device(cfg.policy.device, log=True)

    observation, _ = env.reset(seed=seed)
    env.unwrapped._last_action = None

    # Callback for visualization.
    def render_frame(env: gym.Env) -> np.ndarray:
        rgb_frame = env.render()
        video_frames.append(rgb_frame)

        # Live visualization
        if cfg.show:
            live_view.enqueue_frame(rgb_frame[..., ::-1])

        return rgb_frame

    # Cache frames for creating video
    video_frames: list[np.ndarray] = []
    # Chache environment states
    state_history: list[Any] = []
    # Setup for live visualization
    if env_states is None:
        if cfg.show:
            live_view = LiveWindow("Live Visualization")
        render_frame(env)

    # Prepare visualisers
    visualizers = make_flow_matching_visualizers(
        vis_cfg=cfg.vis,
        model_cfg=policy.config,
        velocity_model=policy.flow_matching.unet,
        output_root=output_dir,
        unnormalize_outputs=policy.unnormalize_outputs,
    )

    # Roll through one episode
    max_episode_steps = env.spec.max_episode_steps
    max_vis_steps = (max_episode_steps if cfg.vis.max_steps is None
                    else min(max_episode_steps, cfg.vis.max_steps))

    start_time = time.time() 

    progbar = trange(
        max_vis_steps,
        desc=f"Running rollout with at most {max_vis_steps} steps"
    )
    for step_idx in progbar:
        if env_states is None:
            state_history.append(capture_pusht_state(env))

        if env_states is not None:
            if step_idx >= len(env_states):
                break
            if step_idx > 0:
                observation = restore_pusht_state(env, env_states[step_idx])

        # Numpy array to tensor and changing dictionary keys to LeRobot policy format.
        observation = preprocess_observation(observation)
        observation = {
            key: observation[key].to(device, non_blocking=device.type == "cuda") for key in observation
        }

        # Decide whether a new sequence will be generated
        new_action_gen = len(policy._queues["action"]) == 0

        with torch.no_grad():
            action = policy.select_action(observation)

        if new_action_gen and (cfg.vis.start_step is None or step_idx >= cfg.vis.start_step):
            # Stack the history of observations
            batch = {
                k: torch.stack(list(policy._queues[k]), dim=1)
                for k in policy._queues
                if k != "action"
            }

            # build global-conditioning with the policy's helper
            global_cond = policy.flow_matching.prepare_global_conditioning(batch)

            for visualizer in visualizers:
                visualizer.visualize(global_cond=global_cond, env=env)

        # Apply the next action
        observation, _, terminated, _, _ = env.step(action[0].cpu().numpy())
        if env_states is None:
            render_frame(env)

        # Stop early if environment terminates
        if terminated:
            break

    logging.info(f"Finished in {time.time() - start_time:.1f}s")

    # Close the live visualization
    if cfg.show:
        live_view.close()

    # Save the buffered video
    if env_states is None:
        write_video(
            str(output_dir / "rollout.mp4"),
            np.stack(video_frames, axis=0),
            fps=env.metadata["render_fps"]
        )

    return state_history

@parser.wrap()
def main(cfg: VisualizeLaplacePipelineConfig):
    # Set global seed
    if cfg.seed is not None:
        set_seed(cfg.seed)

    logging.info("Loading policy")
    if cfg.policy.type != "flow_matching":
        raise ValueError(
            f"visualize_laplace.py only supports Flow Matching policies, "
            f"but got policy type '{cfg.policy.type}'."
        )
    device = get_safe_torch_device(cfg.policy.device, log=True)
    policy = make_policy(cfg.policy, env_cfg=cfg.env).to(device)
    policy.eval()

    logging.info("Creating environment")
    env = make_single_env(cfg.env)
    # ------------------------------------------
    flow_matching_model = policy.flow_matching
    # Get path to save or load the Laplace posterior
    laplace_output_path = make_laplace_path(
        repo_id=cfg.dataset.repo_id,
        scope=cfg.uncertainty_sampler.active_config.laplace_scope,
        calib_fraction=cfg.uncertainty_sampler.active_config.calib_fraction,
    )

    # Create the Laplace calibration data loader if Laplace posterior if not stored
    # on disk
    if not laplace_output_path.exists():
        laplace_calib_loader = create_laplace_flow_matching_calib_loader(
            cfg=cfg,
            policy=policy,
        )
    else:
        laplace_calib_loader = None

    # Get the fitted Laplace posterior
    laplace_posterior = get_laplace_posterior(
        cfg=cfg.uncertainty_sampler.active_config,
        flow_matching_model=flow_matching_model,
        laplace_calib_loader=laplace_calib_loader,
        laplace_path=laplace_output_path,
    )

    laplace_flow_matching_models = []
    for k in range(cfg.n_laplace_models):
        laplace_flow_matching_models.append(draw_laplace_flow_matching_model(
            laplace_posterior=laplace_posterior,
            flow_matching_model=flow_matching_model,
        ))
    # ------------------------------------------
    seed = cfg.seed if cfg.seed is not None else random.randrange(2**32)
    state_history = rollout(
        cfg=cfg,
        env=env,
        policy=policy,
        seed=seed,
        output_dir=cfg.output_dir,
    )

    for k, laplace_flow_matching_model in enumerate(laplace_flow_matching_models):
        policy.reset()
        # Replace flow matching MAP velocity model with a Laplace approximation
        policy.flow_matching = laplace_flow_matching_model
        output_dir = cfg.output_dir / f"laplace_model_{k+1}"

        rollout(
            cfg=cfg,
            env=env,
            policy=policy,
            output_dir=output_dir,
            seed=seed,
            env_states=state_history,
        )


if __name__ == "__main__":
    init_logging()
    main()
