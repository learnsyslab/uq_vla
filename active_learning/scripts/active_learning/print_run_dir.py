#!/usr/bin/env python
"""Print the run directory an iterative-fine-tuning config resolves to for one seed pair.

    PYTHONPATH=src python scripts/active_learning/print_run_dir.py <config.yaml> <seed_pair> [overrides...]

Extra arguments are the same draccus overrides `iterative_fine_tuning.main` takes (e.g.
--paths.run_name=...), so the printed directory is exactly the one the run writes to.
"""
import contextlib
import io
import sys

import draccus

import lerobot.envs  # noqa: F401  (registers the env config classes)
import lerobot.policies  # noqa: F401  (registers the policy config classes)
from iterative_fine_tuning.config import IterativeFineTuningConfig


def main() -> None:
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    config_path, seed_pair, *overrides = sys.argv[1:]
    sys.argv = sys.argv[:1]  # validate() inspects sys.argv for --policy.path
    with contextlib.redirect_stdout(io.StringIO()):
        cfg = draccus.parse(IterativeFineTuningConfig, config_path=config_path,
                            args=[f"--seed_pair={seed_pair}", *overrides])
        cfg.validate()
    print(cfg.run_dir())


if __name__ == "__main__":
    main()
