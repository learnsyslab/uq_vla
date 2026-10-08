"""Filesystem layout helpers for iterative fine-tuning runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from lerobot.utils.constants import CHECKPOINTS_DIR, LAST_CHECKPOINT_LINK, PRETRAINED_MODEL_DIR

from .constants import (
    CANDIDATE_SCORES_NAME,
    INITIAL_EVALUATION_DIRNAME,
    INITIAL_EVALUATION_NAME,
    ITERATIVE_CONFIG_NAME,
    MEMBER_EVALUATION_NAME,
    ROUND_EVALUATION_NAME,
    ROUND_DIR_TEMPLATE,
    RUN_EVALUATION_NAME,
    RUN_STATE_NAME,
    SELECTION_MANIFEST_NAME,
    TRAINING_DIRNAME,
    TRAINING_MANIFEST_NAME,
)


@dataclass(frozen=True)
class RoundLayout:
    round_dir: Path
    training_dir: Path
    candidate_scores_path: Path
    selection_manifest_path: Path
    training_manifest_path: Path

    @property
    def round_evaluation_path(self) -> Path:
        return self.round_dir / ROUND_EVALUATION_NAME

    def member_training_dir(self, member_index: int) -> Path:
        return self.training_dir / f"member_{member_index:02d}"

    def member_evaluation_path(self, member_index: int) -> Path:
        return self.member_training_dir(member_index) / MEMBER_EVALUATION_NAME

    @property
    def last_checkpoint_dir(self) -> Path:
        return self.training_dir / CHECKPOINTS_DIR / LAST_CHECKPOINT_LINK

    @property
    def final_model_path(self) -> Path:
        return self.last_checkpoint_dir / PRETRAINED_MODEL_DIR


@dataclass(frozen=True)
class RunLayout:
    root: Path

    @property
    def initial_evaluation_dir(self) -> Path:
        return self.root / INITIAL_EVALUATION_DIRNAME

    @property
    def initial_evaluation_path(self) -> Path:
        return self.initial_evaluation_dir / INITIAL_EVALUATION_NAME

    def initial_member_dir(self, member_index: int) -> Path:
        return self.initial_evaluation_dir / f"member_{member_index:02d}"

    def initial_member_evaluation_path(self, member_index: int) -> Path:
        return self.initial_member_dir(member_index) / MEMBER_EVALUATION_NAME

    @property
    def config_snapshot_path(self) -> Path:
        return self.root / ITERATIVE_CONFIG_NAME

    @property
    def run_evaluation_path(self) -> Path:
        return self.root / RUN_EVALUATION_NAME

    @property
    def run_state_path(self) -> Path:
        return self.root / RUN_STATE_NAME

    def round_layout(self, round_index: int) -> RoundLayout:
        round_dir = self.root / ROUND_DIR_TEMPLATE.format(round_index=round_index)
        training_dir = round_dir / TRAINING_DIRNAME
        return RoundLayout(
            round_dir=round_dir,
            training_dir=training_dir,
            candidate_scores_path=round_dir / CANDIDATE_SCORES_NAME,
            selection_manifest_path=round_dir / SELECTION_MANIFEST_NAME,
            training_manifest_path=round_dir / TRAINING_MANIFEST_NAME,
        )
