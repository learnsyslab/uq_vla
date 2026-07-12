#!/usr/bin/env python3
"""Aggregate run-level evaluation summaries across multi-seed child runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from lerobot.utils.import_utils import register_third_party_plugins

from .config import IterativeFineTuningConfig
from .constants import ITERATIVE_CONFIG_NAME, MULTI_SEED_EVALUATION_NAME, MULTI_SEED_RUN_NAME
from .env import resolve_local_path
from .manifests import load_json, save_dataclass_json, utc_timestamp


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--config-path", type=Path, default=None)
    return parser.parse_args()


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _std(values: list[float]) -> float:
    return float(np.std(values)) if values else 0.0


def _load_multi_seed_cfg(config_path: Path) -> IterativeFineTuningConfig:
    cfg = IterativeFineTuningConfig.from_pretrained(str(config_path), cli_args=[])
    cfg.validate()
    if not cfg.multi_seed.iter_run_specs():
        raise ValueError("The provided config does not define any multi-seed runs.")
    return cfg


def _resolve_parent_run_dir(args: argparse.Namespace) -> tuple[Path, dict]:
    cfg: IterativeFineTuningConfig | None = None
    if args.config_path is not None:
        config_path = resolve_local_path(args.config_path, base_dir=Path.cwd(), must_exist=True)
        cfg = _load_multi_seed_cfg(config_path)
        run_dir = cfg.run_dir()
    elif args.run_dir is not None:
        run_dir = resolve_local_path(args.run_dir, base_dir=Path.cwd(), must_exist=True)
    else:
        raise ValueError("Either --run-dir or --config-path must be provided.")

    manifest_path = run_dir / MULTI_SEED_RUN_NAME
    if manifest_path.exists():
        return run_dir, load_json(manifest_path)

    if cfg is None:
        raise FileNotFoundError(
            f"Could not find {MULTI_SEED_RUN_NAME} in {run_dir}. "
            "Provide --config-path so child run locations can be derived."
        )

    child_runs = []
    for run_spec in cfg.multi_seed.iter_run_specs():
        seed = int(run_spec.seed)
        run_name = (
            run_spec.run_name
            if run_spec.run_name is not None
            else cfg.multi_seed.run_name_template.format(seed=seed)
        )
        selection_seed = (
            int(run_spec.selection_seed)
            if run_spec.selection_seed is not None
            else seed + int(cfg.multi_seed.selection_seed_offset)
        )
        child_runs.append(
            {
                "seed": seed,
                "selection_seed": selection_seed,
                "run_name": run_name,
                "run_dir": str(run_dir / run_name),
                "run_evaluation_path": str(run_dir / run_name / "run_evaluation.json"),
            }
        )

    seeds = [int(run_spec.seed) for run_spec in cfg.multi_seed.iter_run_specs()]
    return run_dir, {
        "created_at": utc_timestamp(),
        "run_name": cfg.paths.run_name,
        "run_dir": str(run_dir),
        "seeds": seeds,
        "child_runs": child_runs,
    }


def main() -> None:
    import lerobot.policies  # noqa: F401 - registers built-in policy config choices.

    register_third_party_plugins()
    args = _parse_args()
    parent_run_dir, manifest = _resolve_parent_run_dir(args)

    evaluated_child_runs = []
    per_round: dict[int, list[dict]] = {}
    for child in manifest.get("child_runs", []):
        run_dir = resolve_local_path(child["run_dir"], base_dir=Path.cwd())
        run_evaluation_path = resolve_local_path(
            child.get("run_evaluation_path") or (run_dir / "run_evaluation.json"),
            base_dir=Path.cwd(),
        )
        if not run_evaluation_path.exists():
            continue

        run_evaluation = load_json(run_evaluation_path)
        evaluated_child_runs.append(
            {
                "seed": int(child["seed"]),
                "selection_seed": child.get("selection_seed"),
                "run_dir": str(run_dir),
                "result_path": str(run_evaluation_path),
                "num_rounds_evaluated": int(run_evaluation.get("num_rounds_evaluated", 0)),
            }
        )
        for round_summary in run_evaluation.get("rounds", []):
            round_index = int(round_summary["round_index"])
            per_round.setdefault(round_index, []).append(
                {
                    "seed": int(child["seed"]),
                    "selection_seed": child.get("selection_seed"),
                    "run_dir": str(run_dir),
                    "result_path": str(round_summary["result_path"]),
                    "mean_member_macro_success_rate": float(
                        round_summary["mean_member_macro_success_rate"]
                    ),
                    "mean_member_pooled_success_rate": float(
                        round_summary["mean_member_pooled_success_rate"]
                    ),
                    "mean_member_macro_avg_reward": float(
                        round_summary["mean_member_macro_avg_reward"]
                    ),
                    "mean_member_macro_avg_steps": float(
                        round_summary["mean_member_macro_avg_steps"]
                    ),
                    "best_member_index": round_summary.get("best_member_index"),
                    "best_member_macro_success_rate": round_summary.get(
                        "best_member_macro_success_rate"
                    ),
                }
            )

    summary = {
        "created_at": utc_timestamp(),
        "run_name": manifest.get("run_name", parent_run_dir.name),
        "run_dir": str(parent_run_dir),
        "config_snapshot_path": str(parent_run_dir / ITERATIVE_CONFIG_NAME),
        "num_seed_runs_configured": len(manifest.get("seeds", [])),
        "num_seed_runs_evaluated": len(evaluated_child_runs),
        "seeds": list(manifest.get("seeds", [])),
        "evaluated_child_runs": evaluated_child_runs,
        "rounds": [],
    }

    for round_index in sorted(per_round):
        round_entries = per_round[round_index]
        summary["rounds"].append(
            {
                "round_index": round_index,
                "num_seed_runs": len(round_entries),
                "mean_run_macro_success_rate": _mean(
                    [entry["mean_member_macro_success_rate"] for entry in round_entries]
                ),
                "std_run_macro_success_rate": _std(
                    [entry["mean_member_macro_success_rate"] for entry in round_entries]
                ),
                "mean_run_pooled_success_rate": _mean(
                    [entry["mean_member_pooled_success_rate"] for entry in round_entries]
                ),
                "std_run_pooled_success_rate": _std(
                    [entry["mean_member_pooled_success_rate"] for entry in round_entries]
                ),
                "mean_run_macro_avg_reward": _mean(
                    [entry["mean_member_macro_avg_reward"] for entry in round_entries]
                ),
                "std_run_macro_avg_reward": _std(
                    [entry["mean_member_macro_avg_reward"] for entry in round_entries]
                ),
                "mean_run_macro_avg_steps": _mean(
                    [entry["mean_member_macro_avg_steps"] for entry in round_entries]
                ),
                "std_run_macro_avg_steps": _std(
                    [entry["mean_member_macro_avg_steps"] for entry in round_entries]
                ),
                "seed_results": round_entries,
            }
        )

    output_path = parent_run_dir / MULTI_SEED_EVALUATION_NAME
    save_dataclass_json(summary, output_path)
    print(output_path)


if __name__ == "__main__":
    main()
