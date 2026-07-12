"""Main entrypoint for iterative active-learning fine-tuning."""

import copy
import logging
from pathlib import Path

import draccus

from lerobot.configs import parser
from lerobot.utils.import_utils import register_third_party_plugins

from .config import IterativeFineTuningConfig
from .constants import ITERATIVE_CONFIG_NAME, MULTI_SEED_RUN_NAME, STAGE_SELECT, STAGE_TRAIN
from .layout import RunLayout
from .manifests import (
    MultiSeedChildRun,
    MultiSeedRunManifest,
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
                    # Keep prior selections eligible so repeated picks can upweight them in training.
                    excluded_episode_ids=set(),
                    round_layout=round_layout,
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


def _selection_seed_for_child_run(cfg: IterativeFineTuningConfig, seed: int) -> int:
    return int(seed) + int(cfg.multi_seed.selection_seed_offset)


def _child_run_name(cfg: IterativeFineTuningConfig, *, seed: int, run_name: str | None = None) -> str:
    if run_name is not None:
        return str(run_name)
    return cfg.multi_seed.run_name_template.format(seed=int(seed))


def _build_child_cfg(
    cfg: IterativeFineTuningConfig,
    *,
    parent_dir: Path,
    run_spec,
) -> IterativeFineTuningConfig:
    child_cfg = copy.deepcopy(cfg)
    child_cfg.paths.run_root = str(parent_dir)
    child_cfg.paths.run_name = _child_run_name(
        cfg,
        seed=int(run_spec.seed),
        run_name=run_spec.run_name,
    )
    child_cfg.job_name = f"{cfg.job_name or cfg.paths.run_name}_seed_{int(run_spec.seed)}"
    child_cfg.seed = int(run_spec.seed)
    child_cfg.selection.seed = (
        int(run_spec.selection_seed)
        if run_spec.selection_seed is not None
        else _selection_seed_for_child_run(cfg, int(run_spec.seed))
    )
    if run_spec.pretrained_path is not None:
        child_cfg.policy.pretrained_path = run_spec.pretrained_path
    if run_spec.ensemble_model_paths:
        child_cfg.selection.ensemble_model_paths = list(run_spec.ensemble_model_paths)
    child_cfg.multi_seed.seeds = []
    child_cfg.multi_seed.runs = []
    return child_cfg


def _multi_seed_manifest_path(parent_dir: Path) -> Path:
    return parent_dir / MULTI_SEED_RUN_NAME


def _completed_round_indices(state: RunState | None) -> list[int]:
    if state is None:
        return []
    return sorted(record.round_index for record in state.rounds if record.final_model_paths)


def _run_multi_seed(cfg: IterativeFineTuningConfig) -> None:
    parent_dir = cfg.run_dir()
    parent_dir.mkdir(parents=True, exist_ok=True)
    _save_config_snapshot(cfg, parent_dir / ITERATIVE_CONFIG_NAME)
    manifest = MultiSeedRunManifest(
        run_name=cfg.paths.run_name,
        run_dir=str(parent_dir),
        seeds=[int(run_spec.seed) for run_spec in cfg.multi_seed.iter_run_specs()],
    )
    save_dataclass_json(manifest, _multi_seed_manifest_path(parent_dir))

    for run_spec in cfg.multi_seed.iter_run_specs():
        child_cfg = _build_child_cfg(cfg, parent_dir=parent_dir, run_spec=run_spec)
        child_cfg.validate()
        child_layout = RunLayout(root=child_cfg.run_dir())
        logger.info(
            "Starting multi-seed child run %s for parent run %s",
            child_layout.root.name,
            parent_dir.name,
        )
        _run_single(child_cfg)

        child_state = (
            load_run_state(child_layout.run_state_path)
            if child_layout.run_state_path.exists()
            else None
        )
        manifest.child_runs.append(
            MultiSeedChildRun(
                seed=int(run_spec.seed),
                selection_seed=child_cfg.selection.seed,
                run_name=child_cfg.paths.run_name,
                run_dir=str(child_layout.root),
                run_state_path=str(child_layout.run_state_path),
                run_evaluation_path=str(child_layout.run_evaluation_path),
                pretrained_path=str(child_cfg.policy.pretrained_path)
                if child_cfg.policy is not None and child_cfg.policy.pretrained_path is not None
                else None,
                ensemble_model_paths=list(child_cfg.selection.ensemble_model_paths),
                latest_model_paths=child_state.latest_trained_model_paths() or []
                if child_state is not None
                else [],
                completed_round_indices=_completed_round_indices(child_state),
            )
        )
        save_dataclass_json(manifest, _multi_seed_manifest_path(parent_dir))

    logger.info(
        "Multi-seed iterative fine-tuning complete. Parent run directory: %s",
        parent_dir,
    )


@parser.wrap()
def run(cfg: IterativeFineTuningConfig):
    cfg.validate()
    if cfg.multi_seed.iter_run_specs():
        _run_multi_seed(cfg)
        return
    _run_single(cfg)


def main():
    register_third_party_plugins()
    run()


if __name__ == "__main__":
    main()
