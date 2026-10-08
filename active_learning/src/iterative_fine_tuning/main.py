"""Main entrypoint for iterative active-learning fine-tuning."""

import logging
from pathlib import Path

import draccus

from lerobot.configs import parser
from lerobot.utils.import_utils import register_third_party_plugins

from .config import IterativeFineTuningConfig
from .constants import STAGE_SELECT, STAGE_TRAIN
from .layout import RunLayout
from .manifests import (
    RoundRecord,
    RunState,
    load_episode_ids_from_manifest,
    load_run_state,
    load_selection_manifest,
    save_dataclass_json,
    total_timing_entry,
)
from .selection import run_selection_round
from .training import run_training_round

logger = logging.getLogger(__name__)


def _save_config_snapshot(cfg: IterativeFineTuningConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle, draccus.config_type("json"):
        draccus.dump(cfg, handle, indent=2)


def _load_or_create_state(cfg: IterativeFineTuningConfig, layout: RunLayout) -> RunState:
    if layout.run_state_path.exists():
        if not cfg.execution.resume and not cfg.execution.overwrite_existing_round:
            raise FileExistsError(
                f"Run state already exists at {layout.run_state_path}. "
                "Set execution.resume=true or execution.overwrite_existing_round=true."
            )
        return load_run_state(layout.run_state_path)

    bootstrap_episode_ids: list[int] = []
    for manifest_path in cfg.bootstrap_selection_paths():
        bootstrap_episode_ids.extend(load_episode_ids_from_manifest(manifest_path))

    state = RunState(
        run_name=cfg.paths.run_name,
        run_dir=str(layout.root),
        bootstrap_episode_ids=[int(ep) for ep in bootstrap_episode_ids],
    )
    save_dataclass_json(state, layout.run_state_path)
    return state


def _determine_round_start_models(
    cfg: IterativeFineTuningConfig,
    state: RunState,
) -> list[str]:
    latest_trained_models = state.latest_trained_model_paths()
    if latest_trained_models and cfg.iteration.warm_start_from_previous_round:
        return latest_trained_models
    return cfg.resolved_ensemble_model_paths()


def _selection_sampler_model_path(model_paths: list[str]) -> str:
    if not model_paths:
        raise ValueError("No model paths are available for selection.")
    return model_paths[0]


def _training_outputs_exist(model_paths: list[str]) -> bool:
    return bool(model_paths) and all(Path(path).exists() for path in model_paths)


def _run_single(cfg: IterativeFineTuningConfig) -> None:
    layout = RunLayout(root=cfg.run_dir())
    layout.root.mkdir(parents=True, exist_ok=True)
    # Freeze config to disk so evaluate_run.py can reconstruct settings without the original file.
    _save_config_snapshot(cfg, layout.config_snapshot_path)

    # Load existing run state (for resumption) or create a fresh one.
    state = _load_or_create_state(cfg, layout)

    for round_index in range(cfg.iteration.total_rounds):
        round_layout = layout.round_layout(round_index)
        round_layout.round_dir.mkdir(parents=True, exist_ok=True)

        # Resume a partially completed round or start a new one.
        round_record = state.get_round(round_index)
        if round_record is None:
            round_record = RoundRecord(
                round_index=round_index,
                # Warm-start from the previous round's fine-tuned models if available.
                start_model_paths=_determine_round_start_models(cfg, state),
            )
            state.upsert_round(round_record)

        # All episodes selected in prior rounds, used to build the cumulative training set.
        previous_selected_episode_ids = state.cumulative_selected_episode_ids(
            include_round=round_index - 1
        )
        start_model_paths = round_record.start_model_paths or _determine_round_start_models(cfg, state)
        round_record.start_model_paths = list(start_model_paths)

        if STAGE_SELECT in cfg.execution.stages:
            should_run_selection = (
                cfg.execution.overwrite_existing_round
                or round_record.selection_manifest_path is None
                or not Path(round_record.selection_manifest_path).exists()
            )
            if should_run_selection:
                logger.info(
                    "Selecting episodes for round %s using %s ensemble members",
                    round_index,
                    len(start_model_paths),
                )
                # Score all candidate episodes by epistemic uncertainty and pick the top-K.
                selection_manifest = run_selection_round(
                    cfg,
                    round_index=round_index,
                    sampler_model_reference=_selection_sampler_model_path(start_model_paths),
                    scorer_model_references=start_model_paths,
                    # Keep prior selections eligible so repeated picks can upweight them in
                    # training -- the uncertainty is rescored every round, so a repeat is a
                    # real signal. A predefined ranking is fixed, though: leaving prior picks
                    # eligible would re-pop the same top entries every round and the run would
                    # never advance past the first `episodes_per_round` episodes.
                    excluded_episode_ids=(
                        set(previous_selected_episode_ids)
                        if cfg.selection.strategy == "predefined_ranking"
                        else set()
                    ),
                    round_layout=round_layout,
                    # Ordered acquisition history; AMF needs it to represent tasks by the
                    # demonstrations already collected for them.
                    acquired_episode_ids=previous_selected_episode_ids,
                )
                round_record.mark_selected(
                    round_layout.selection_manifest_path,
                    selection_manifest.episode_ids,
                    timing=total_timing_entry(selection_manifest.timings),
                )
                state.upsert_round(round_record)
                save_dataclass_json(state, layout.run_state_path)

                # Release the selection models' GPU memory back to the driver so
                # the member-training child processes can use the same devices.
                # The parent process otherwise keeps the freed tensors cached in
                # its allocator for the rest of the round.
                import gc

                import torch

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                # Selection already done for this round; load the cached manifest.
                selection_manifest = load_selection_manifest(Path(round_record.selection_manifest_path))
        else:
            if round_record.selection_manifest_path is None:
                raise FileNotFoundError(
                    f"No selection manifest recorded for round {round_index}; "
                    "cannot continue without the select stage."
                )
            selection_manifest = load_selection_manifest(Path(round_record.selection_manifest_path))

        if STAGE_TRAIN in cfg.execution.stages:
            should_run_training = (
                cfg.execution.overwrite_existing_round
                or round_record.training_manifest_path is None
                or not Path(round_record.training_manifest_path).exists()
            )
            if should_run_training:
                logger.info(
                    "Training round %s with %s new episodes across %s ensemble members",
                    round_index,
                    len(selection_manifest.episode_ids),
                    len(start_model_paths),
                )
                # Accumulate all selected episodes across rounds; repeated selections increase their weight.
                cumulative_selected_episode_ids = list(previous_selected_episode_ids)
                cumulative_selected_episode_ids.extend(selection_manifest.episode_ids)
                # Fine-tune each ensemble member on the cumulative selected + history replay set.
                training_manifest = run_training_round(
                    cfg,
                    round_index=round_index,
                    model_references=start_model_paths,
                    selected_episode_ids=cumulative_selected_episode_ids,
                    round_layout=round_layout,
                )
                round_record.mark_trained(
                    round_layout.training_manifest_path,
                    [Path(path) for path in training_manifest.final_model_paths],
                    timing=total_timing_entry(training_manifest.timings),
                )
                state.upsert_round(round_record)
                save_dataclass_json(state, layout.run_state_path)

    logger.info("Iterative fine-tuning complete. Run directory: %s", layout.root)


@parser.wrap()
def run(cfg: IterativeFineTuningConfig):
    cfg.validate()
    _run_single(cfg)


def main():
    register_third_party_plugins()
    run()


if __name__ == "__main__":
    main()
