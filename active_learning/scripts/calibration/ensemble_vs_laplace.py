#!/usr/bin/env python
"""fig:ensemble_vs_laplace: calibration of the two-member ensemble vs. a last-layer Laplace
approximation of member 0 over the first five rounds of the SmolVLA calibration runs, as task-level
-Spearman and -Pearson (two panels).

Both use VFD (inter_vel_diff_2way); the Laplace variant samples the second model from the posterior
fitted by fit_laplace.py instead of using the second ensemble member. Reads the per-round uncertainty
caches of calibration_comparison.py and writes
<output>/ensemble_vs_laplace_correlations.json and <output>/ensemble_vs_laplace.{pdf,png}.

    PYTHONPATH=src python scripts/calibration/ensemble_vs_laplace.py
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "plots"))
import calibration_utils as cu  # noqa: E402
from paper_style import page_style, tex  # noqa: E402

METHODS = {"Ensemble": "inter_vel_diff_2way", "Laplace": "inter_vel_diff_2way_laplace"}
N_ROUNDS = 5
EPISODES_PER_ROUND = 5
TICK_STEP_DEMOS = 10   # x-axis tick spacing, in demonstrations
# Panel titles carry the metric, so neither panel needs a y-label: both are a negative
# correlation on [0, 1] and share the axis.
PANELS = [("spearman", r"$-$Spearman $\rho$"), ("pearson", r"$-$Pearson $r$")]


def correlations(run_dirs: list[Path]) -> dict:
    """{metric: {label: {round: {values, mean, std, n, inverted_mean}}}}, std over seed pairs (ddof=0)."""
    per_run = [cu.correlations_for_seed(r, max_round=N_ROUNDS, methods=list(METHODS.values())) for r in run_dirs]
    out = {}
    for metric, _ in PANELS:
        out[metric] = {}
        for label, method in METHODS.items():
            out[metric][label] = {}
            for rnd in range(N_ROUNDS):
                vals = [c[method][metric][rnd] for c in per_run if len(c[method][metric]) > rnd]
                if not vals:
                    continue
                a = np.array(vals)
                out[metric][label][str(rnd)] = {"values": vals, "mean": float(a.mean()), "std": float(a.std()),
                                                "n": len(vals), "inverted_mean": float(-a.mean())}
    return out


def plot(data: dict, out: Path) -> None:
    page_style("ensemble_vs_laplace_2panel")
    fig, axes = plt.subplots(1, 2, sharey=True)
    for ax, (metric, title) in zip(axes, PANELS):
        for label, color, marker in [("Ensemble", "tab:blue", "o"), ("Laplace", "tab:orange", "s")]:
            rounds = sorted(int(r) for r in data[metric][label])
            means = [-float(data[metric][label][str(r)]["mean"]) for r in rounds]
            stds = [float(data[metric][label][str(r)]["std"]) for r in rounds]
            x = [r + 1 for r in rounds]
            ax.plot(x, means, marker=marker, markersize=3, linewidth=1.4, label=label, color=color)
            ax.fill_between(x, [m - s for m, s in zip(means, stds)], [m + s for m, s in zip(means, stds)],
                            color=color, alpha=0.18)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        # Ticks stay on round positions; labels report demonstrations collected (5 per round).
        ax.set_xlabel(tex("# Demonstrations"))
        ax.set_title(title)
        ax.set_ylim(0, 1)
        ticks = [r for r in x if (r * EPISODES_PER_ROUND) % TICK_STEP_DEMOS == 0]
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(r * EPISODES_PER_ROUND) for r in ticks])
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.grid(True, alpha=0.3, axis="y")
    axes[0].legend(loc="lower right")
    # No tight_layout: the paper style turns constrained layout on and the two conflict.
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}.pdf")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dirs", nargs="+", type=Path,
                    default=[Path(f"outputs/calibration/smolvla/random_{s}") for s in ("s01", "s23", "s45")])
    ap.add_argument("--output", type=Path, default=Path("plots/ensemble_vs_laplace"))
    a = ap.parse_args()
    data = correlations(a.run_dirs)
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / "ensemble_vs_laplace_correlations.json").write_text(json.dumps(data, indent=2))
    plot(data, a.output / "ensemble_vs_laplace")


if __name__ == "__main__":
    main()
