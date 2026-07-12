"""
Fit a Laplace approximation for each round of an iterative fine-tuning run,
using the exact episode IDs that were used to train that round.

The posterior is saved alongside the member_00 checkpoint as
    round_N/training/member_00/checkpoints/last/pretrained_model/laplace_<scope>_frac100pct.bin

Skips rounds where the posterior file already exists.

Usage:
    python scripts/calibration/fit_laplace.py \
        --run_dir ${STORAGE_ROOT}/outputs/iterative_fine_tuning/task_weighted_leak3_s01 \
        --device cuda \
        --scope action_out_proj
"""

import argparse
import json
import logging
from pathlib import Path

from lerobot.configs.default import DatasetConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.uncertainty.uncertainty_samplers.configuration_uncertainty_sampler import LaplaceConfig
from lerobot.uncertainty.uncertainty_scoring.laplace_utils.posterior_builder import (
    get_laplace_posterior,
    make_laplace_path,
    make_laplace_wrapper,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LIBERO_DATASET_CFG = DatasetConfig(repo_id="HuggingFaceVLA/libero")


def load_member_model_path(round_dir: Path, member_index: int) -> Path:
    manifest_path = round_dir / "training_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        final_model_paths = [Path(path) for path in manifest.get("final_model_paths", [])]
        if len(final_model_paths) > member_index:
            return final_model_paths[member_index]

    return (
        round_dir
        / "training"
        / f"member_{member_index:02d}"
        / "checkpoints"
        / "last"
        / "pretrained_model"
    )


def fit_round(
    round_dir: Path,
    device: str,
    laplace_config: LaplaceConfig,
) -> bool:
    """Fit and save Laplace posterior for one round. Returns True if fitted, False if skipped."""
    member_00_path = load_member_model_path(round_dir, 0)
    manifest_path = round_dir / "training_manifest.json"

    if not member_00_path.exists():
        logger.info(f"  {round_dir.name}: no member_00 checkpoint, skipping.")
        return False
    if not manifest_path.exists():
        logger.info(f"  {round_dir.name}: no training_manifest.json, skipping.")
        return False

    manifest = json.loads(manifest_path.read_text())
    episode_ids: list[int] = manifest["all_training_episode_ids"]
    logger.info(f"  {round_dir.name}: {len(episode_ids)} training episodes.")

    policy_cls = get_policy_class("smolvla")
    logger.info(f"  {round_dir.name}: stage 1/4 - loading policy from {member_00_path}")
    policy = policy_cls.from_pretrained(pretrained_name_or_path=str(member_00_path))
    logger.info(f"  {round_dir.name}: stage 1/4 complete - policy loaded")
    policy.to(device)
    policy.eval()

    logger.info(f"  {round_dir.name}: stage 2/4 - building preprocessor")
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(member_00_path),
    )
    logger.info(f"  {round_dir.name}: stage 2/4 complete - preprocessor ready")

    # Check if posterior already exists
    laplace_wrapper = make_laplace_wrapper(policy=policy, scopes=laplace_config.scopes)
    laplace_path = make_laplace_path(
        laplace_wrapper=laplace_wrapper,
        pretrained_path=member_00_path,
        calib_fraction=laplace_config.calib_fraction,
    )
    if laplace_path.exists():
        logger.info(f"  {round_dir.name}: posterior already exists at {laplace_path}, skipping.")
        return False

    logger.info(f"  {round_dir.name}: stage 3/4 - creating calibration dataset/loader")
    logger.info(f"  {round_dir.name}: stage 4/4 - fitting Laplace posterior")
    get_laplace_posterior(
        policy=policy,
        preprocessor=preprocessor,
        laplace_config=laplace_config,
        dataset_cfg=LIBERO_DATASET_CFG,
        episode_ids=episode_ids,
        pretrained_path=member_00_path,
    )
    logger.info(f"  {round_dir.name}: stage 4/4 complete - saved to {laplace_path}.")
    return True


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--scope", type=str, default="action_out_proj",
        choices=["action_out_proj", "action_time_embed", "expert_last",
                 "state_proj", "vlm_connector", "vision_last"],
        help="Which submodule to place the Laplace posterior on.",
    )
    parser.add_argument("--calib_fraction", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--rounds",
        nargs="+",
        type=int,
        default=None,
        help="Only fit these round indices. Default: all rounds.",
    )
    parser.add_argument(
        "--max_round",
        type=int,
        default=None,
        help="Only fit rounds with index < max_round.",
    )
    args = parser.parse_args()

    if not args.run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {args.run_dir}")

    laplace_config = LaplaceConfig(
        scopes=[args.scope],
        calib_fraction=args.calib_fraction,
        batch_size=args.batch_size,
    )

    round_dirs = sorted(args.run_dir.glob("round_*"), key=lambda p: int(p.name.split("_")[1]))
    if args.rounds is not None:
        keep_rounds = set(args.rounds)
        round_dirs = [d for d in round_dirs if int(d.name.split("_")[1]) in keep_rounds]
    if args.max_round is not None:
        round_dirs = [d for d in round_dirs if int(d.name.split("_")[1]) < args.max_round]
    logger.info(f"Found {len(round_dirs)} rounds in {args.run_dir.name}.")

    fitted, skipped = 0, 0
    for round_dir in round_dirs:
        if fit_round(round_dir, args.device, laplace_config):
            fitted += 1
        else:
            skipped += 1

    logger.info(f"Done. Fitted: {fitted}, skipped: {skipped}.")


if __name__ == "__main__":
    main()
