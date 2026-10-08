"""Worker entrypoint for training one ensemble member on a dedicated device."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from lerobot.utils.import_utils import register_third_party_plugins

from .config import IterativeFineTuningConfig
from .layout import RunLayout
from .manifests import TrainedEnsembleMember, load_json, save_dataclass_json
from .training import MemberTrainingRequest, _run_single_training_round, make_training_accelerator

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request_path", required=True, type=Path)
    return parser.parse_args()


def _load_request(path: Path) -> MemberTrainingRequest:
    return MemberTrainingRequest(**load_json(path))


def _load_config(path: str) -> IterativeFineTuningConfig:
    cfg = IterativeFineTuningConfig.from_pretrained(path, cli_args=[])
    cfg.validate()
    return cfg


def _round_layout_from_request(request: MemberTrainingRequest):
    round_dir = Path(request.round_dir)
    run_layout = RunLayout(root=round_dir.parent)
    return run_layout.round_layout(request.round_index)


def _configure_devices(cfg: IterativeFineTuningConfig, request: MemberTrainingRequest) -> None:
    cfg.policy.device = request.device
    cfg.selection.device = request.device


def run_worker(request_path: Path) -> TrainedEnsembleMember:
    request = _load_request(request_path)
    if request.result_path is None:
        raise ValueError("Member training request is missing result_path.")

    cfg = _load_config(request.config_snapshot_path)
    _configure_devices(cfg, request)
    round_layout = _round_layout_from_request(request)

    # Under `accelerate launch` (multi-GPU member, see training.member_worker_command) every rank runs
    # this function; the Accelerator is built here so only the main process writes the result file.
    accelerator = make_training_accelerator(cfg)
    trained_member = _run_single_training_round(
        cfg,
        round_index=request.round_index,
        member_index=request.member_index,
        model_reference=request.model_reference,
        history_episode_ids=request.history_episode_ids,
        selected_episode_ids=request.selected_episode_ids,
        round_layout=round_layout,
        accelerator=accelerator,
    )
    # `_run_single_training_round` already synchronised the ranks and called `end_training()`, which
    # tears the process group down -- no collective calls after this point.
    if accelerator.is_main_process:
        save_dataclass_json(trained_member, Path(request.result_path))
    return trained_member


def main() -> None:
    register_third_party_plugins()
    args = _parse_args()
    trained_member = run_worker(args.request_path)
    logger.info(
        "Finished ensemble member %s -> %s",
        trained_member.member_index,
        trained_member.final_model_path,
    )


if __name__ == "__main__":
    main()
