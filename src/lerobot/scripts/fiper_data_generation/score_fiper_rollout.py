import logging
import pickle
import random
from itertools import zip_longest
from pathlib import Path
from typing import NamedTuple

import torch
from tqdm import tqdm

from lerobot.configs import parser
from lerobot.configs.fiper_rollout_scoring import FiperRolloutScoringPipelineConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.fiper_data_generator.fiper_rollout_scorer import FiperRolloutScorer
from lerobot.policies.factory import (
    make_flow_matching_adapter_from_policy,
    make_policy,
    make_pre_post_processors,
)
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.uncertainty.uncertainty_scoring.scorer_artifacts import build_scorer_artifacts_for_fiper_scorer
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import get_safe_torch_device, init_logging


def get_episode_idx(path: Path) -> int:
    """
    Extract the numeric episode index from filenames like:
    'episode_f_0146(_task03).pkl' or 'episode_s_0008(_task01).pkl'.
    """
    stem = path.stem
    idx_str = stem.split("_")[2]
    return int(idx_str)

class TaskRollouts(NamedTuple):
    name: str
    calib_files: list[Path]
    test_files: list[Path]

def build_task_rollouts(
    input_root: Path,
    start_ep: int | None,
    end_ep: int | None,
    task_ids: list[int] | None = None,
) -> list[TaskRollouts]:
    """
    Discover task folders under `input_root`, collect calibration and test rollouts
    for each task, and return them sorted.
    """

    def in_range(p: Path) -> bool:
        idx = get_episode_idx(p)
        if start_ep is not None and idx < start_ep:
            return False
        if end_ep is not None and idx > end_ep:
            return False
        return True

    allowed_task_names = None
    if task_ids is not None:
        allowed_task_names = {f"task{task_id:02d}" for task_id in task_ids}

    task_dirs = sorted(
        [
            d for d in input_root.iterdir()
            if d.is_dir()
            and d.name.startswith("task")
            and (allowed_task_names is None or d.name in allowed_task_names)
        ]
    )

    tasks: list[TaskRollouts] = []

    for task_dir in task_dirs:
        rollouts_dir = task_dir / "rollouts"
        calib_dir = rollouts_dir / "calibration"
        test_dir = rollouts_dir / "test"

        calib_files = []
        test_files = []

        if calib_dir.is_dir():
            calib_files = [p for p in calib_dir.glob("*.pkl") if in_range(p)]
            calib_files.sort(key=get_episode_idx)

        if test_dir.is_dir():
            test_files = [p for p in test_dir.glob("*.pkl") if in_range(p)]
            test_files.sort(key=get_episode_idx)

        # If a task has no files at all, we skip it
        if not calib_files and not test_files:
            logging.warning(f"No calibration/test files found for task '{task_dir.name}'. Skipping.")
            continue

        tasks.append(TaskRollouts(name=task_dir.name, calib_files=calib_files, test_files=test_files))

    return tasks

def iter_round_robin_tasks(tasks: list[TaskRollouts]):
    if not tasks:
        return

    max_len = max(max(len(t.calib_files), len(t.test_files)) for t in tasks)

    for ep_idx in range(max_len):
        for t in tasks:
            if ep_idx < len(t.calib_files):
                yield t.name, "calibration", t.calib_files[ep_idx]
            if ep_idx < len(t.test_files):
                yield t.name, "test", t.test_files[ep_idx]

@parser.wrap()
def main(cfg: FiperRolloutScoringPipelineConfig):
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

    logging.info("Loading policy.")
    ds_meta = LeRobotDatasetMetadata(
        cfg.dataset.repo_id, root=cfg.dataset.root, revision=cfg.dataset.revision
    )
    policy: PreTrainedPolicy = make_policy(
        cfg=cfg.policy,
        ds_meta=ds_meta,
    ).to(device)
    policy.eval()

    # Build preprocessing/postprocessing pipelines for observations/actions
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        # The inference device is automatically set to match the detected hardware, overriding any previous device settings from training to ensure compatibility.
        preprocessor_overrides={"device_processor": {"device": str(policy.config.device)}},
    )

    scorer_artifacts = build_scorer_artifacts_for_fiper_scorer(
        fiper_data_scorer_cfg=cfg.fiper_rollout_scorer,
        policy=policy,
        preprocessor=preprocessor,
        dataset_cfg=cfg.dataset,
    )

    flow_matching_adapter = make_flow_matching_adapter_from_policy(policy)

    fiper_rollout_scorer = FiperRolloutScorer(
        config=cfg.fiper_rollout_scorer,
        flow_matching_adapter=flow_matching_adapter,
        scorer_artifacts=scorer_artifacts,
    )

    input_dir = Path(cfg.input_dir)

    start_ep = cfg.start_episode
    end_ep = cfg.end_episode

    tasks = build_task_rollouts(
        input_root=input_dir,
        start_ep=start_ep,
        end_ep=end_ep,
        task_ids=cfg.env.task_ids,
    )

    if not tasks:
        raise FileNotFoundError(
            f"No calibration or test files found in any task subdirectory of {input_dir}."
        )

    output_root = Path(cfg.output_dir)

    # Compute total number of files for the progress bar
    total_num_files = sum(len(t.calib_files) + len(t.test_files) for t in tasks)

    for task_name, split, pkl_path in tqdm(
        iter_round_robin_tasks(tasks),
        total=total_num_files,
        desc="Scoring rollouts",
        unit="file",
    ):
        logging.info(f"Scoring [task={task_name} | split={split}] {pkl_path.name}")

        with pkl_path.open("rb") as f:
            data = pickle.load(f)

        episode_metadata = data.get("metadata")
        rollout_data = data.get("rollout")

        # Reset state per episode/file
        fiper_rollout_scorer.reset()

        with torch.no_grad():
            fiper_rollout_scorer.score_rollout_data(rollout_data=rollout_data)

        # NEW: mirror task structure in the output directory
        task_output_root = output_root / task_name
        task_output_rollouts_dir = task_output_root / "rollouts"
        output_calib_dir = task_output_rollouts_dir / "calibration"
        output_test_dir = task_output_rollouts_dir / "test"

        # Make sure directories exist
        output_calib_dir.mkdir(parents=True, exist_ok=True)
        output_test_dir.mkdir(parents=True, exist_ok=True)

        # Save scored result into task- and split-specific folder
        output_dir_for_split = output_calib_dir if split == "calibration" else output_test_dir
        fiper_rollout_scorer.save_data(
            output_dir=output_dir_for_split,
            episode_metadata=episode_metadata,
        )

    logging.info("Done scoring all calibration/test rollouts for all tasks.")


if __name__ == "__main__":
    init_logging()
    main()
