"""Train additional members for an existing iterative fine-tuning run.

This utility reuses the already selected episodes in an existing active-learning
run and appends new member checkpoints to each round. It does not run selection.
"""

from __future__ import annotations

import argparse
import fcntl
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import draccus

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from iterative_fine_tuning.config import IterativeFineTuningConfig
from iterative_fine_tuning.constants import ITERATIVE_CONFIG_NAME
from iterative_fine_tuning.layout import RunLayout
from iterative_fine_tuning.manifests import (
    TrainingManifest,
    TrainedEnsembleMember,
    load_run_state,
    load_selection_manifest,
    load_training_manifest,
    save_dataclass_json,
)
from iterative_fine_tuning.training import (
    MemberTrainingRequest,
    _member_log_path,
    _member_request_path,
    _member_result_path,
    _normalize_worker_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-run-dir", required=True, type=Path)
    parser.add_argument("--checkpoint-run-dir", required=True, type=Path)
    parser.add_argument("--pretrain-root", required=True, type=Path)
    parser.add_argument("--pretrained-seeds", nargs="+", required=True, type=int)
    parser.add_argument("--member-indices", nargs="+", default=[2, 3], type=int)
    parser.add_argument("--devices", nargs="+", default=["cuda:0", "cuda:1"])
    parser.add_argument("--rounds", nargs="+", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    lengths = {len(args.pretrained_seeds), len(args.member_indices), len(args.devices)}
    if len(lengths) != 1:
        raise ValueError(
            "--pretrained-seeds, --member-indices, and --devices must have the same length; "
            f"got {len(args.pretrained_seeds)}, {len(args.member_indices)}, {len(args.devices)}"
        )
    return args


def load_cfg(
    metadata_run_dir: Path,
    checkpoint_run_dir: Path,
    reference_model_paths: list[str],
) -> IterativeFineTuningConfig:
    cfg_path = metadata_run_dir / ITERATIVE_CONFIG_NAME
    cfg = IterativeFineTuningConfig.from_pretrained(cfg_path, cli_args=[])
    cfg.paths.run_root = str(metadata_run_dir.parent)
    cfg.paths.run_name = checkpoint_run_dir.name
    cfg.paths.checkpoint_root = str(checkpoint_run_dir.parent)
    cfg.policy.pretrained_path = reference_model_paths[0]
    cfg.selection.ensemble_model_paths = reference_model_paths
    cfg.validate()
    return cfg


def write_config_snapshot(cfg: IterativeFineTuningConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle, draccus.config_type("json"):
        draccus.dump(cfg, handle, indent=2)


def pretrain_model_path(pretrain_root: Path, seed: int) -> str:
    checkpoint_root = pretrain_root / f"seed_{seed}" / "checkpoints"
    candidates = [
        checkpoint_root / "last" / "pretrained_model",
        checkpoint_root / "030000" / "pretrained_model",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(
        f"Could not find pretrained model for seed {seed}; tried: "
        + ", ".join(str(path) for path in candidates)
    )


def available_rounds(layout: RunLayout, requested: list[int] | None) -> list[int]:
    if requested is not None:
        return requested
    rounds = []
    for path in sorted(layout.root.glob("round_*")):
        try:
            rounds.append(int(path.name.split("_")[1]))
        except (IndexError, ValueError):
            continue
    return rounds


def cumulative_selected_episode_ids(layout: RunLayout, rounds: list[int], current_round: int) -> list[int]:
    run_state_path = layout.run_state_path
    selected: list[int] = []
    if run_state_path.exists():
        selected.extend(load_run_state(run_state_path).bootstrap_episode_ids)
    for round_index in rounds:
        if round_index > current_round:
            break
        manifest = load_selection_manifest(layout.round_layout(round_index).selection_manifest_path)
        selected.extend(manifest.episode_ids)
    return [int(ep) for ep in selected]


def member_final_exists(member: TrainedEnsembleMember) -> bool:
    return bool(member.final_model_path) and Path(member.final_model_path).exists()


def existing_member_by_index(manifest: TrainingManifest, member_index: int) -> TrainedEnsembleMember | None:
    for member in manifest.member_training_runs:
        if member.member_index == member_index:
            return member
    return None


def merge_members(manifest: TrainingManifest, new_members: list[TrainedEnsembleMember]) -> TrainingManifest:
    by_index = {member.member_index: member for member in manifest.member_training_runs}
    for member in new_members:
        by_index[member.member_index] = member
    merged = [by_index[index] for index in sorted(by_index)]
    manifest.member_training_runs = merged
    manifest.start_model_paths = [member.start_model_path for member in merged]
    manifest.final_model_paths = [member.final_model_path for member in merged]
    return manifest


def update_run_state(layout: RunLayout, round_index: int, manifest: TrainingManifest) -> None:
    if not layout.run_state_path.exists():
        return
    state = load_run_state(layout.run_state_path)
    record = state.get_round(round_index)
    if record is None:
        return
    record.start_model_paths = list(manifest.start_model_paths)
    record.final_model_paths = list(manifest.final_model_paths)
    state.upsert_round(record)
    save_dataclass_json(state, layout.run_state_path)


@contextmanager
def run_manifest_lock(layout: RunLayout):
    lock_path = layout.root / ".extra_members_manifest.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def launch_round_members(
    *,
    config_snapshot_path: Path,
    layout: RunLayout,
    round_index: int,
    member_indices: list[int],
    model_references: list[str],
    devices: list[str],
    history_episode_ids: list[int],
    selected_episode_ids: list[int],
) -> list[TrainedEnsembleMember]:
    round_layout = layout.round_layout(round_index)
    launches = []
    for member_index, model_reference, member_device in zip(
        member_indices, model_references, devices, strict=True
    ):
        training_dir = round_layout.member_training_dir(member_index)
        training_dir.mkdir(parents=True, exist_ok=True)
        request_path = _member_request_path(training_dir)
        result_path = _member_result_path(training_dir)
        log_path = _member_log_path(training_dir)

        worker_device, visible_device = _normalize_worker_device(member_device)
        request = MemberTrainingRequest(
            config_snapshot_path=str(config_snapshot_path),
            round_dir=str(round_layout.round_dir),
            round_index=round_index,
            member_index=member_index,
            model_reference=model_reference,
            history_episode_ids=sorted(set(int(ep) for ep in history_episode_ids)),
            selected_episode_ids=[int(ep) for ep in selected_episode_ids],
            device=worker_device,
            result_path=str(result_path),
        )
        save_dataclass_json(request, request_path)

        env = os.environ.copy()
        env["PYTHONPATH"] = f"{PROJECT_ROOT / 'src'}{os.pathsep}{env.get('PYTHONPATH', '')}".rstrip(os.pathsep)
        if visible_device is not None:
            env["CUDA_VISIBLE_DEVICES"] = visible_device

        log_handle = open(log_path, "w", encoding="utf-8")
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "iterative_fine_tuning.train_worker",
                "--request_path",
                str(request_path),
            ],
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        launches.append((member_index, process, log_handle, result_path, log_path))

    trained_members: list[TrainedEnsembleMember] = []
    failures = []
    for member_index, process, log_handle, result_path, log_path in launches:
        return_code = process.wait()
        log_handle.close()
        if return_code != 0 or not result_path.exists():
            failures.append((member_index, return_code, log_path))
            continue
        from iterative_fine_tuning.manifests import load_trained_ensemble_member

        trained_members.append(load_trained_ensemble_member(result_path))

    if failures:
        messages = []
        for member_index, return_code, log_path in failures:
            tail = log_path.read_text(errors="replace").splitlines()[-80:] if log_path.exists() else []
            messages.append(
                f"member_{member_index:02d} failed with code {return_code}; log={log_path}\n"
                + "\n".join(tail)
            )
        raise RuntimeError("\n\n".join(messages))

    return sorted(trained_members, key=lambda member: member.member_index)


def main() -> None:
    args = parse_args()
    metadata_run_dir = args.metadata_run_dir.resolve()
    checkpoint_run_dir = args.checkpoint_run_dir.resolve()
    layout = RunLayout(root=metadata_run_dir)
    current_models = [
        pretrain_model_path(args.pretrain_root, seed)
        for seed in args.pretrained_seeds
    ]
    config_reference_models = list(current_models)
    if len(config_reference_models) < 2:
        used_seeds = set(args.pretrained_seeds)
        for seed in range(6):
            if seed in used_seeds:
                continue
            config_reference_models.append(pretrain_model_path(args.pretrain_root, seed))
            break
    config_reference_models = list(dict.fromkeys(config_reference_models))
    cfg = load_cfg(metadata_run_dir, checkpoint_run_dir, config_reference_models)
    config_snapshot_path = metadata_run_dir / "extra_members_config.json"
    write_config_snapshot(cfg, config_snapshot_path)

    rounds = available_rounds(layout, args.rounds)
    if not rounds:
        raise ValueError(f"No rounds found in {metadata_run_dir}")

    for round_index in rounds:
        round_layout = layout.round_layout(round_index)
        training_manifest = load_training_manifest(round_layout.training_manifest_path)

        to_train_indices: list[int] = []
        to_train_models: list[str] = []
        to_train_devices: list[str] = []
        for member_index, model_reference, device in zip(
            args.member_indices, current_models, args.devices, strict=True
        ):
            existing = existing_member_by_index(training_manifest, member_index)
            if existing is not None and member_final_exists(existing) and not args.overwrite:
                print(
                    f"round {round_index:03d} member_{member_index:02d}: exists, skipping",
                    flush=True,
                )
                model_reference = existing.final_model_path
            else:
                to_train_indices.append(member_index)
                to_train_models.append(model_reference)
                to_train_devices.append(device)
            current_models[args.member_indices.index(member_index)] = model_reference

        if to_train_indices:
            print(
                f"round {round_index:03d}: training members {to_train_indices} from {to_train_models}",
                flush=True,
            )
            selected_episode_ids = cumulative_selected_episode_ids(layout, rounds, round_index)
            trained_members = launch_round_members(
                config_snapshot_path=config_snapshot_path,
                layout=layout,
                round_index=round_index,
                member_indices=to_train_indices,
                model_references=to_train_models,
                devices=to_train_devices,
                history_episode_ids=training_manifest.history_episode_ids,
                selected_episode_ids=selected_episode_ids,
            )
            with run_manifest_lock(layout):
                # Reload under the lock so parallel one-member jobs do not clobber
                # each other's additions to the shared round manifest.
                training_manifest = load_training_manifest(round_layout.training_manifest_path)
                training_manifest = merge_members(training_manifest, trained_members)
                save_dataclass_json(training_manifest, round_layout.training_manifest_path)
                update_run_state(layout, round_index, training_manifest)
            for trained in trained_members:
                idx = args.member_indices.index(trained.member_index)
                current_models[idx] = trained.final_model_path
        else:
            print(f"round {round_index:03d}: nothing to train", flush=True)

        time.sleep(0.1)


if __name__ == "__main__":
    main()
