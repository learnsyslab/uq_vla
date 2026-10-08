#!/usr/bin/env python
"""Break exact ties in DECU episode uncertainties so a degenerate round yields a ranking.

WHY. DECU saturates at log 2 on X-VLA and FastWAM, and the cached scores were written at
bfloat16 precision, where the quantum near 0.69 is ~0.0039. Every episode of a round therefore
collapses onto the SAME stored value, every task mean is identical, and scipy returns nan for
both correlations. `np.mean` over the rounds then propagates that nan and the whole cell prints
NA -- even when some rounds of the run do carry a real signal.

WHAT THIS DOES. For rounds whose task means are exactly constant, add uniform noise of
+-`--scale` (default 1e-5) to each episode score. That is more than two orders of magnitude
below the bf16 quantum, so it can never reorder two genuinely different stored values -- it only
randomises within a tie group. Rounds that already have distinct task means are left alone.

WHAT THE NUMBERS THEN MEAN. Inside a fully tied round the resulting ranking is pure noise, so
its correlation is an unbiased draw around zero. That is the honest reading: DECU carries no
information there. It is NOT evidence of a weak-but-real correlation.

ORIGINALS ARE KEPT. The untouched file is copied to <name>.orig.json before anything is
written, and a file that already has one is skipped -- so this is idempotent and runs once.
Restore with --restore. The noise is seeded from the cache's path, so re-running reproduces the
same values bit for bit.

    python scripts/calibration/jitter_degenerate_decu.py --dry-run
    python scripts/calibration/jitter_degenerate_decu.py
    python scripts/calibration/jitter_degenerate_decu.py --restore
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

IFT = Path("outputs/calibration")
# Only the two policies whose DECU saturates; SmolVLA's DECU scores are not degenerate.
DEFAULT_RUNS = [IFT / policy / f"random_{s}" for policy in ("xvla", "fastwam") for s in ("s01", "s23", "s45")]


def rng_for(path: Path) -> np.random.Generator:
    """Deterministic per-file generator -- the same cache always gets the same noise."""
    digest = hashlib.sha256(str(path).encode()).digest()[:8]
    return np.random.default_rng(int.from_bytes(digest, "big"))


def task_means_constant(data: dict) -> bool:
    means = {float(np.mean(list(eps.values()))) for eps in data.values()}
    return len(means) == 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dirs", nargs="+", type=Path, default=DEFAULT_RUNS)
    ap.add_argument("--method", default="decu")
    ap.add_argument("--scale", type=float, default=1e-5,
                    help="Uniform noise half-width (default 1e-5, vs a ~4e-3 bf16 quantum).")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--restore", action="store_true", help="Put the .orig.json files back.")
    a = ap.parse_args()

    name = f"{a.method}_episode_uncertainties.json"
    touched = skipped = intact = 0
    for run in a.run_dirs:
        for cache in sorted(run.glob(f"round_*/calibration_comparison/{name}")):
            orig = cache.with_suffix(".orig.json")
            if a.restore:
                if orig.exists():
                    shutil.copy2(orig, cache)
                    touched += 1
                continue
            if orig.exists():           # already jittered on an earlier run
                skipped += 1
                continue
            data = json.loads(cache.read_text())
            if not task_means_constant(data):
                intact += 1
                continue
            rng = rng_for(cache.relative_to(Path.cwd()) if cache.is_absolute() else cache)
            jittered = {
                t: {ep: float(v + rng.uniform(-a.scale, a.scale)) for ep, v in eps.items()}
                for t, eps in data.items()
            }
            touched += 1
            if a.dry_run:
                continue
            shutil.copy2(cache, orig)   # keep the original BEFORE writing
            cache.write_text(json.dumps(jittered))

    verb = "restored" if a.restore else ("would jitter" if a.dry_run else "jittered")
    print(f"{verb}: {touched}   already done: {skipped}   left alone (not degenerate): {intact}")


if __name__ == "__main__":
    main()
