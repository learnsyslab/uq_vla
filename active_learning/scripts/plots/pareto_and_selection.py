#!/usr/bin/env python
"""Two appendix figures of the LIBERO/SmolVLA temperature sweep.

  fig:exploration-success-pareto  (pics/exploration_exploitation_entropy.pdf)
      Per-tau trajectory of normalised task-sampling entropy vs success rate over the 15
      rounds, the Pareto front of the round-15 points (maximise both), and a zoomed inset on
      that front.
  fig:selection_uncertainty       (pics/selection_uncertainty_heatmap.pdf)
      Per run, two task x round heatmaps averaged over seeds: selected demonstrations per task
      (left) and each task's share of the round's total uncertainty (right).

Both are built from round_evaluation.json, selection_manifest.json and
candidate_scores.json. Entropy, success and tau colours are imported from
success_share_entropy.py so the three tau figures agree.

Uncertainty share uses the score transform selection.py applies before weighting: scores are
clamped at 0, except ensemble_terminal_variance (GU), whose log-variances can be negative and
are shifted by U - min(U) + 0.01 * (max(U) - min(U)). That is what the sampler actually saw.

    PYTHONPATH=src python scripts/plots/pareto_and_selection.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_style import page_style, tex  # noqa: E402
import success_share_entropy as sse  # noqa: E402

BASE = sse.BASE
SEEDS = sse.SEEDS
FINAL_ROUND = sse.FINAL_ROUND
EPISODES_PER_ROUND = sse.EPISODES_PER_ROUND

# Heatmap rows: (label, run stem, scoring metric or None if the method does not select by a
# score). Order follows the comparison tables (success_rate_paper.TABLE_ROWS): Random,
# Diversity, AMF, GU, Action-L2, then the VFD temperature sweep. Stems for Random and AMF are
# the ones BENCHMARKS["libero"] uses. tau = 3 is left out, as in the other tau figures and the
# ablation table.
# Random does log per-task scores (mean initial-frame uncertainty) but samples uniformly and
# ignores them, so its right panel is the "no uncertainty scores" placeholder, like Diversity.
# AMF's share is of its own acquisition score (posterior variance reduction).
HEATMAP_RUNS = (
    [("Random", "random", None),
     ("Diversity", "diversity", None),
     ("AMF", "amf", "amf"),
     ("SAVE w/ GU\n$\\tau=1$", "gu", "ensemble_terminal_variance"),
     ("SAVE w/ Action-L2\n$\\tau=1.5$", "action_l2", "action_l2")]
    + [(f"SAVE w/ VFD\n$\\tau={t}$", sse.run_stem(t, "lr"), "inter_vel_diff") for t, _ in sse.TAUS]
)
N_TASKS = 10
# Near-black: tab:red, its old colour, is now tau = 1.5 (success_share_entropy.TAU_COLORS).
FRONT_COLOR = "0.1"


# --------------------------------------------------------------------------------------
# Pareto figure
# --------------------------------------------------------------------------------------
def pareto_front(points: dict[str, tuple[float, float]]) -> list[str]:
    """Keys of the non-dominated points when maximising both coordinates, by entropy."""
    front = [k for k, (x, y) in points.items()
             if not any((x2 >= x and y2 >= y) and (x2 > x or y2 > y)
                        for k2, (x2, y2) in points.items() if k2 != k)]
    return sorted(front, key=lambda k: points[k][0])


def draw_trajectories(ax, ent, sr, small: bool) -> dict[str, tuple[float, float]]:
    """All-round trajectories per tau; returns each tau's round-15 point."""
    finals = {}
    for tau, _ in sse.TAUS:
        rounds = [r for r in range(1, FINAL_ROUND + 1) if r in ent[tau] and r in sr[tau]]
        if len(rounds) < 2:
            continue
        c = sse.tau_color(tau)
        xs = np.array([ent[tau][r].mean() for r in rounds])
        ys = np.array([sr[tau][r].mean() for r in rounds])
        ax.plot(xs, ys, "-o", color=c, linewidth=1.1, markersize=2.2 if small else 2.6, zorder=2)
        ax.plot(xs[0], ys[0], "o", color=c, markerfacecolor="white", markersize=5.5,
                markeredgewidth=1.2, zorder=4)
        ax.plot(xs[-1], ys[-1], "D", color=c, markersize=4.5, zorder=4)
        finals[tau] = (xs[-1], ys[-1])
    front = pareto_front(finals)
    ax.plot([finals[t][0] for t in front], [finals[t][1] for t in front], "-",
            color=FRONT_COLOR, linewidth=1.6, zorder=3)
    return finals


def plot_pareto(out: Path) -> None:
    ent = {tau: sse.task_entropy(k, "lr", float(tau)) for tau, k in sse.TAUS}
    sr = {tau: sse.success_by_round(k, "lr") for tau, k in sse.TAUS}

    page_style("pareto")
    fig = plt.figure()
    fig.set_layout_engine(None)          # manual placement: the inset sits outside the axes
    ax = fig.add_axes([0.09, 0.14, 0.56, 0.80])
    finals = draw_trajectories(ax, ent, sr, small=False)
    ax.set_xlabel("Normalized task entropy")
    ax.set_ylabel("Success rate")
    ax.grid(True, alpha=0.3, axis="y")

    # Zoom on the round-15 points, where the front lives.
    fx = np.array([p[0] for p in finals.values()]); fy = np.array([p[1] for p in finals.values()])
    pad_x = 0.25 * (fx.max() - fx.min()) + 0.01; pad_y = 0.25 * (fy.max() - fy.min()) + 0.01
    x0, x1 = fx.min() - pad_x, min(fx.max() + pad_x, 1.005)
    y0, y1 = fy.min() - pad_y, fy.max() + pad_y
    ins = fig.add_axes([0.73, 0.24, 0.25, 0.56])
    draw_trajectories(ins, ent, sr, small=True)
    ins.set_xlim(x0, x1); ins.set_ylim(y0, y1)
    ins.tick_params(labelsize=6)
    ins.grid(True, alpha=0.3, axis="y")
    for sp in ins.spines.values():
        sp.set_linestyle("--"); sp.set_edgecolor("0.35")
    ax.indicate_inset_zoom(ins, edgecolor="0.35", linestyle="--", alpha=1.0)

    handles = [Line2D([], [], color=sse.tau_color(t), marker="o", markersize=2.6,
                      linewidth=1.1, label=rf"$\tau={t}$") for t, _ in sse.TAUS]
    handles += [Line2D([], [], color=FRONT_COLOR, linewidth=1.6, label="Pareto front"),
                Line2D([], [], marker="o", color="0.3", markerfacecolor="white", markersize=5,
                       markeredgewidth=1.0, linestyle="none", label="Round 1"),
                Line2D([], [], marker="D", color="0.3", markersize=4.2, linestyle="none",
                       label=f"Round {FINAL_ROUND}")]
    ax.legend(handles=handles, loc="upper left", handlelength=1.4, handletextpad=0.4,
              borderpad=0.3, labelspacing=0.25)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# --------------------------------------------------------------------------------------
# Selection / uncertainty-share heatmaps
# --------------------------------------------------------------------------------------
def selection_counts(stem: str) -> np.ndarray:
    """(task, round) mean over seeds of demonstrations selected, from the manifests."""
    per_seed = []
    for s in SEEDS:
        m = np.full((N_TASKS, FINAL_ROUND), np.nan)
        for rd in (BASE / f"{stem}_{s}").glob("round_*"):
            r = int(rd.name.split("_")[1])
            f = rd / "selection_manifest.json"
            if r >= FINAL_ROUND or not f.exists():
                continue
            m[:, r] = 0.0
            for e in json.loads(f.read_text())["selected_episodes"]:
                m[int(e["task_id"]), r] += 1
        per_seed.append(m)
    return np.nanmean(np.stack(per_seed), axis=0)


def uncertainty_share(stem: str, metric: str) -> np.ndarray:
    """(task, round) mean over seeds of each task's share of the round's total score."""
    per_seed = []
    for s in SEEDS:
        m = np.full((N_TASKS, FINAL_ROUND), np.nan)
        for rd in (BASE / f"{stem}_{s}").glob("round_*"):
            r = int(rd.name.split("_")[1])
            f = rd / "candidate_scores.json"
            if r >= FINAL_ROUND or not f.exists():
                continue
            d = json.loads(f.read_text())
            u = np.full(N_TASKS, np.nan)
            for e in d:
                u[int(e["task_id"])] = e["uncertainty"]
            u = np.where(np.isfinite(u), u, np.nan)
            if metric == "ensemble_terminal_variance":      # selection.py's shift
                lo, hi = np.nanmin(u), np.nanmax(u)
                u = u - lo + 0.01 * (hi - lo)
            else:
                u = np.maximum(u, 0.0)
            if np.nansum(u) > 0:
                m[:, r] = u / np.nansum(u)
        per_seed.append(m)
    return np.nanmean(np.stack(per_seed), axis=0)


def plot_selection(out: Path) -> None:
    rows = [(lab, selection_counts(stem), uncertainty_share(stem, met) if met else None)
            for lab, stem, met in HEATMAP_RUNS]
    sel_max = np.ceil(max(np.nanmax(c) for _, c, _ in rows))
    share_max = np.ceil(max(np.nanmax(u) for _, _, u in rows if u is not None) / 0.05) * 0.05
    # The original used seaborn's rocket_r / mako; seaborn is not in the env, so these are the
    # closest matplotlib maps (cream -> dark purple, dark navy -> teal -> light).
    sel_cmap = plt.get_cmap("magma_r")
    share_cmap = plt.get_cmap("YlGnBu_r")

    page_style("selection_heatmap")
    fig, axes = plt.subplots(len(rows), 2, sharex=True)
    x_edges = np.arange(FINAL_ROUND + 1) + 0.5          # rounds 1..15 as cell centres
    y_edges = np.arange(N_TASKS + 1) + 0.5              # tasks 1..10
    mesh = dict(edgecolors="white", linewidth=0.25, rasterized=False)
    for i, (lab, counts, share) in enumerate(rows):
        a, b = axes[i]
        im_sel = a.pcolormesh(x_edges, y_edges, counts, cmap=sel_cmap, vmin=0, vmax=sel_max,
                              **mesh)
        if share is not None:
            # Rounds without scores (AMF logs none in its first two rounds, in every seed)
            # are NaN and show this grey, the same as the no-score placeholder.
            b.set_facecolor("0.93")
            im_share = b.pcolormesh(x_edges, y_edges, share, cmap=share_cmap, vmin=0,
                                    vmax=share_max, **mesh)
        else:
            b.set_facecolor("0.93")
            b.text(0.5, 0.5, "No uncertainty scores", ha="center", va="center",
                   transform=b.transAxes, color="0.35")
            b.set_xlim(x_edges[0], x_edges[-1]); b.set_ylim(y_edges[0], y_edges[-1])
        for ax in (a, b):
            ax.invert_yaxis()
            ax.set_yticks([1, 5, 10])
            ax.tick_params(length=0, pad=1)
            for sp in ax.spines.values():
                sp.set_visible(False)
        b.set_yticklabels([])
        a.set_ylabel(lab, rotation=0, ha="right", va="center", labelpad=6)
    axes[0, 0].set_title("Selection")
    axes[0, 1].set_title("Uncertainty share")
    ticks = list(range(1, FINAL_ROUND + 1, 2))
    for ax in axes[-1]:
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks])
        ax.set_xlabel("Round $r$")
    fig.colorbar(im_sel, ax=axes[:, 0], location="bottom", shrink=0.9, aspect=30,
                 label="Selected demonstrations")
    fig.colorbar(im_share, ax=axes[:, 1], location="bottom", shrink=0.9, aspect=30,
                 label="Uncertainty share")

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=Path("plots/success_rate_comparison"))
    a = ap.parse_args()
    plot_pareto(a.out_dir / "exploration_exploitation_entropy.pdf")
    plot_selection(a.out_dir / "selection_uncertainty_heatmap.pdf")


if __name__ == "__main__":
    main()
