#!/usr/bin/env python3
"""Run one child seed specification from a multi-seed iterative fine-tuning config."""

from __future__ import annotations

import argparse
from pathlib import Path

from lerobot.utils.import_utils import register_third_party_plugins

from .config import IterativeFineTuningConfig
from .env import resolve_local_path
from .main import _build_child_cfg, _run_single


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-path", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    return parser.parse_args()


def _load_cfg(config_path: Path) -> IterativeFineTuningConfig:
    cfg = IterativeFineTuningConfig.from_pretrained(str(config_path), cli_args=[])
    cfg.validate()
    if not cfg.multi_seed.iter_run_specs():
        raise ValueError("The provided config does not define any multi-seed child runs.")
    return cfg


def main() -> None:
    register_third_party_plugins()
    args = _parse_args()

    config_path = resolve_local_path(args.config_path, base_dir=Path.cwd(), must_exist=True)
    parent_cfg = _load_cfg(config_path)
    run_specs = parent_cfg.multi_seed.iter_run_specs()
    matching_specs = [run_spec for run_spec in run_specs if int(run_spec.seed) == int(args.seed)]
    if not matching_specs:
        available = ", ".join(str(int(run_spec.seed)) for run_spec in run_specs)
        raise ValueError(
            f"Seed {args.seed} is not configured in {config_path}. Available seeds: {available}."
        )

    child_cfg = _build_child_cfg(
        parent_cfg,
        parent_dir=parent_cfg.run_dir(),
        run_spec=matching_specs[0],
    )
    child_cfg.validate()
    _run_single(child_cfg)


if __name__ == "__main__":
    main()
