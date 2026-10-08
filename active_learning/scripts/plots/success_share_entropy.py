#!/usr/bin/env python
"""Paper Fig. fig:success_and_uncertainty_share -- the effect of the task-sampling temperature.

Two panels, both from the LIBERO/SmolVLA tau sweep:

  Overall success        success rate vs demonstrations, for tau = 0 and 2.5, each with
                         uncertainty-guided ("top-k") and uniform selection of the initial
                         observation WITHIN a sampled task.
  Task concentration     share of the round's total task uncertainty sitting in the single most
                         uncertain task. 1/K = 0.1 would mean uncertainty spread evenly over the
                         K = 10 tasks, so a falling curve means the budget is reaching the tasks
                         that need it.

The exploration-vs-success view lives in two other figures only: small in the
main text (--standalone-right here, exploration_vs_success_small.pdf) and big in the appendix
(pareto_and_selection.py, exploration_exploitation_entropy.pdf). It used to be a third panel
here as well.

The task-sampling distribution is NOT a softmax: selection.py builds it as
`w_k proportional to max(u_k, 0) ** tau`, uniform at tau = 0 (see _sample_task_counts there).
This script reproduces that exactly, so the entropy is the one the sampler actually had, and
tau = 0 sits at entropy 1.0 by construction.

Uncertainties come from each round's candidate_scores.json, which carries all K tasks -- the
selection manifest lists only the tasks that were sampled, so it cannot give the share.

    PYTHONPATH=src python scripts/plots/success_share_entropy.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_style import page_style, tex  # noqa: E402

BASE = Path("outputs/active_learning/smolvla")
SEEDS = ("s01", "s23", "s45")
EPISODES_PER_ROUND = 5
FINAL_ROUND = 15
# (tau label, run-stem infix). The sweep the paper reports.
TAUS = (("0", "0"), ("1", "1"), ("1.5", "1.5"), ("2", "2"), ("2.5", "2.5"))
# Both panels: tau x {uncertainty-guided, uniform} episode selection within the sampled task.
LEFT = (
    ("0", "lr", r"$\tau=0$", "-"),
    ("0", "uniform_episode", r"$\tau=0$ uniform", ":"),
    ("2.5", "lr", r"$\tau=2.5$", "-"),
    ("2.5", "uniform_episode", r"$\tau=2.5$ uniform", ":"),
)
# tau = 2.5 is the paper's SAVE w/ VFD setting, so it carries VFD's colour from every other
# figure; tau = 0 is uniform task sampling, i.e. Random's regime, so it carries Random's grey.
# The temperatures in between get distinct hues rather than a blue ramp: light blues are hard
# to tell apart. Olive, red and purple are tab10 colours no method uses
# in the other AL figures, and gray -> olive -> red -> purple -> blue still steps through
# the hue circle in tau order.
TAU_COLORS = {
    "0": "tab:gray",
    "1": "tab:olive",
    "1.5": "tab:red",
    "2": "tab:purple",
    "2.5": "tab:blue",
    "3": "#08519c",
}


def tau_color(tau) -> str:
    return TAU_COLORS[str(tau)]


def run_stem(tau: str, variant: str) -> str:
    """Run name (without the seed pair) of a SAVE w/ VFD run of the SmolVLA temperature sweep.

    variant "lr" selects the most uncertain episode within each sampled task, "uniform_episode" a
    uniformly random one. The paper settings are the main runs (configs/active_learning/smolvla/):
    tau = 2.5 top-k is `vfd`, tau = 0 uniform is `random`; scripts/run_smolvla_sweep.sh writes
    the others.
    """
    if variant == "lr":
        return "vfd" if tau == "2.5" else f"vfd_t{tau}"
    if variant == "uniform_episode":
        return "random" if tau == "0" else f"vfd_uniform_t{tau}"
    raise ValueError(variant)


def run_dirs(infix: str, variant: str):
    return [BASE / f"{run_stem(infix, variant)}_{s}" for s in SEEDS]


def success_by_round(infix: str, variant: str) -> dict[int, np.ndarray]:
    """{round: per-seed success rate}. Round 0 is the pretrained ensemble."""
    out: dict[int, list[float]] = {}
    for run in run_dirs(infix, variant):
        init = run / "initial_evaluation" / "initial_evaluation.json"
        if init.exists():
            out.setdefault(0, []).append(
                float(json.loads(init.read_text())["mean_member_macro_success_rate"]))
        for rd in run.glob("round_*"):
            f = rd / "round_evaluation.json"
            if f.exists():
                r = int(rd.name.split("_")[1]) + 1
                out.setdefault(r, []).append(
                    float(json.loads(f.read_text())["mean_member_macro_success_rate"]))
    return {r: np.array(v, float) for r, v in sorted(out.items())}


def task_uncertainties(run: Path, round_idx: int) -> np.ndarray | None:
    f = run / f"round_{round_idx:03d}" / "candidate_scores.json"
    if not f.exists():
        return None
    u = np.array([e["uncertainty"] for e in json.loads(f.read_text())], float)
    # selection.py clamps non-finite and negative scores to 0 before weighting.
    return np.maximum(np.where(np.isfinite(u), u, 0.0), 0.0)


def top_task_share(infix: str, variant: str) -> dict[int, np.ndarray]:
    """{round: per-seed share of total task uncertainty held by the most uncertain task}."""
    out: dict[int, list[float]] = {}
    for run in run_dirs(infix, variant):
        for rd in run.glob("round_*"):
            u = task_uncertainties(run, int(rd.name.split("_")[1]))
            if u is None or u.sum() <= 0:
                continue
            out.setdefault(int(rd.name.split("_")[1]) + 1, []).append(float(u.max() / u.sum()))
    return {r: np.array(v, float) for r, v in sorted(out.items())}


def task_entropy(infix: str, variant: str, tau: float) -> dict[int, np.ndarray]:
    """{round: per-seed entropy of the task-sampling distribution, normalised by log K}."""
    out: dict[int, list[float]] = {}
    for run in run_dirs(infix, variant):
        for rd in run.glob("round_*"):
            u = task_uncertainties(run, int(rd.name.split("_")[1]))
            if u is None:
                continue
            k = len(u)
            if tau == 0.0:
                w = np.full(k, 1.0 / k)
            else:
                powered = np.power(u, tau)
                w = powered / powered.sum() if powered.sum() > 0 else np.full(k, 1.0 / k)
            nz = w[w > 0]
            out.setdefault(int(rd.name.split("_")[1]) + 1, []).append(
                float(-(nz * np.log(nz)).sum() / np.log(k)))
    # Round 0 = the pretrained policy, before any selection. Its task distribution is the one
    # that goes on to select round 1 (round_000/candidate_scores), so reuse that entry.
    if 1 in out:
        out[0] = list(out[1])
    return {r: np.array(v, float) for r, v in sorted(out.items())}


def band(ax, series, color, label, ls="-"):
    """mean +- std over seeds against demonstrations collected."""
    rounds = sorted(series)
    x = np.array(rounds) * EPISODES_PER_ROUND
    m = np.array([series[r].mean() for r in rounds])
    sd = np.array([series[r].std() for r in rounds])
    ax.plot(x, m, ls, color=color, linewidth=1.2, marker="o", markersize=2.6, label=label)
    ax.fill_between(x, m - sd, m + sd, color=color, alpha=0.15, linewidth=0)


BY_ROUND = (0, 5, 10, 15)
# Colour and marker encode tau, the same tau colours as the other tau figures so tau = 2.5 is
# SAVE w/ VFD's blue in both. tau = 0 is darkened from tab:gray so it stands off the grey
# connecting lines. The lines themselves carry no colour: each one is a round, and is labelled
# in the plot next to its left end instead of in the legend.
TAU_MARKERS = {"0": "s", "1": "^", "1.5": "v", "2": "D", "2.5": "o"}
TAU_MARKER_COLORS = dict(TAU_COLORS, **{"0": "0.3"})
ROUND_LINE_COLOR = "0.62"
ENTROPY_XMIN = 0.69


def draw_exploration_by_round(ax) -> None:
    """One grey line per round through the temperatures, tau encoded by colour and marker.

    Reads across the sweep at a fixed budget: where does each temperature stand after the same
    number of rounds? Round 0 is the pretrained policy -- its initial success against the
    entropy of the distribution that will select round 1 -- so its line is flat: the
    temperature reshapes the distribution but the policy is the same for every tau.
    """
    from matplotlib.lines import Line2D

    ent = {tau: task_entropy(infix, "lr", float(tau)) for tau, infix in TAUS}
    sr = {tau: success_by_round(infix, "lr") for tau, infix in TAUS}
    for r in BY_ROUND:
        taus = [t for t, _ in TAUS if r in ent[t] and r in sr[t]]
        if len(taus) < 2:
            continue
        xs = np.array([ent[t][r].mean() for t in taus]); ys = np.array([sr[t][r].mean() for t in taus])
        ax.plot(xs, ys, "-", color=ROUND_LINE_COLOR, linewidth=1.1, zorder=2)
        for t, x, y in zip(taus, xs, ys):
            if r > 0:
                # No spread bars on round 0: the policy is identical for every tau, so a
                # success bar says nothing about temperature.
                ax.errorbar(x, y, xerr=ent[t][r].std(), yerr=sr[t][r].std(), fmt="none",
                            ecolor=TAU_MARKER_COLORS[t], elinewidth=0.6, capsize=1.2,
                            capthick=0.6, zorder=3)
            ax.plot(x, y, TAU_MARKERS[t], color=TAU_MARKER_COLORS[t], markersize=5.6,
                    markeredgecolor="white", markeredgewidth=0.6, zorder=4)
        # Label the round next to the line's left end (tau = 2.5, the lowest entropy). Round 0
        # sits at the x-axis edge, so its label goes above the line instead.
        i = int(np.argmin(xs))
        if r == 0:
            ax.annotate(f"Round {r}", (xs[i], ys[i]), xytext=(2, 4), textcoords="offset points",
                        ha="left", va="bottom", color="0.3")
        else:
            ax.annotate(f"Round {r}", (xs[i], ys[i]), xytext=(-5, 0), textcoords="offset points",
                        ha="right", va="center", color="0.3")
    ax.set_xlim(left=ENTROPY_XMIN)
    ax.set_xlabel("Normalized task entropy")
    ax.set_ylabel("Success Rate")          # as in its row partner sr_over_rounds
    ax.grid(True, alpha=0.3, axis="y")
    handles = [Line2D([], [], marker=TAU_MARKERS[t], color=TAU_MARKER_COLORS[t],
                      markeredgecolor="white", markeredgewidth=0.6, markersize=5.6,
                      linestyle="none", label=rf"$\tau={t}$") for t, _ in TAUS]
    # Compact, with a little headroom above the data, so the legend clears the "Round 5" label
    # just below its lower-right corner (it covered the label's top at the 0.5 \linewidth size).
    ax.set_ylim(0.2, 0.71)   # top 0.71
    ax.legend(handles=handles, loc="upper left", handlelength=0.7, handletextpad=0.25,
              borderpad=0.2, labelspacing=0.15, borderaxespad=0.25)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", type=Path,
                    default=Path("plots/success_rate_comparison/success_share_entropy.pdf"))
    ap.add_argument("--standalone-right", type=Path, default=None,
                    help="Also write the exploration-vs-success panel alone, at wrapfigure "
                         "size and without a title, to this path.")
    args = ap.parse_args()

    if args.standalone_right is not None:
        page_style("exploration_standalone")
        fig, ax = plt.subplots()
        draw_exploration_by_round(ax)
        # exact size (not cropped): it shares a row with sr_over_rounds, see paper_style
        from paper_style import save_exact
        save_exact(fig, args.standalone_right)
        plt.close(fig)
        print(f"Saved: {args.standalone_right}")

    page_style("success_share")
    fig, axes = plt.subplots(1, 2)

    # --- left: overall success ------------------------------------------------------------
    ax = axes[0]
    for tau, variant, label, ls in LEFT:
        infix = dict(TAUS)[tau]
        band(ax, success_by_round(infix, variant), tau_color(tau), label, ls)
    ax.set_title("Overall success")
    ax.set_xlabel(tex("# Demonstrations"))
    ax.set_ylabel("Success rate")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(loc="lower right", ncol=1, handlelength=1.4, labelspacing=0.3)

    # --- middle: task concentration -------------------------------------------------------
    ax = axes[1]
    # Same four settings as the left panel. The uniform-episode runs log the
    # same per-task score (mean initial-frame VFD) in candidate_scores.json, so the share is
    # comparable across all four.
    for tau, variant, label, ls in LEFT:
        band(ax, top_task_share(dict(TAUS)[tau], variant), tau_color(tau), label, ls)
    ax.set_ylim(bottom=0.12)
    ax.set_title("Task concentration")
    ax.set_xlabel(tex("# Demonstrations"))
    ax.set_ylabel("Top-task uncertainty share")
    ax.grid(True, alpha=0.3, axis="y")
    # No legend: the four settings and their styles are the left panel's, and a second copy
    # of that four-row legend covered the tau = 0 curves here.


    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
