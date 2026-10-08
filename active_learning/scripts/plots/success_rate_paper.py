#!/usr/bin/env python3
"""Paper success-rate figure: per-round mean +- std across seeds for 5 AL methods.

Recovered from scripts/paper_plots/libero_leak3_paper.py, which produced the published
LIBERO figure and was deleted in 9421dcc8 ("prune unused scripts"). Restored here and
generalised over benchmarks so Push-T and X-VLA get pixel-identical styling; the LIBERO
defaults reproduce the original figure exactly.

Legend shows AUC (trapezoidal, normalised to [0,1]).
Text box shows first round each method surpasses success rate thresholds.
Also prints a LaTeX tabular with per-round success rates.

Usage:
    PYTHONPATH=src python scripts/plots/success_rate_paper.py --benchmark libero
    PYTHONPATH=src python scripts/plots/success_rate_paper.py --panels
    PYTHONPATH=src python scripts/plots/success_rate_paper.py --tables

--panels draws one learning-curve panel per benchmark into a single figure (the paper's
fig:al_success_rate_comparison); --tables prints the cross-benchmark table, rows = methods,
columns = benchmark x {AULC, Last-3 SR}. Both take --benchmarks to choose the set.
--compact draws one benchmark at wrapfigure size (default LIBERO/SmolVLA).

The x-axis counts demonstrations collected, not rounds: see EPISODES_PER_ROUND.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

IFT = Path("outputs")

# The six acquisition rules and the run names they are written under (configs/active_learning/<benchmark>/).
METHODS = [
    ("random", "Random"),
    ("diversity", "Diversity"),
    ("gu", "GU"),
    ("action_l2", "Action-L2"),
    ("amf", "AMF"),
    ("vfd", "VFD (ours)"),
]
LIBERO_SEEDS = ["s01", "s23", "s45"]
PUSHT_SEEDS = ["s01", "s23", "s45", "s67", "s89", "s1011", "s1213", "s1415", "s1617", "s1819"]

# Per-benchmark: run root, seed pairs, and (run-name stem, legend-label) pairs. Run directories are
# <base>/<stem>_<seed pair>. A seed pair a method lacks loads as NaN and every aggregate ignores it.
BENCHMARKS = {
    "libero": dict(
        base=IFT / "active_learning" / "smolvla",
        seeds=LIBERO_SEEDS,
        experiments=METHODS,
        out=Path("plots/success_rate_comparison/libero/libero_all_rounds.pdf"),
        final_round=15, tick_step=1, ylim=(0.2, 0.7),
        benchmark="LIBERO", policy="SmolVLA", tick_step_episodes=3,
        benchmark_tex=r"\liberoten",
    ),
    "pusht": dict(
        base=IFT / "active_learning" / "pusht",
        seeds=PUSHT_SEEDS,
        experiments=[m for m in METHODS if m[0] != "amf"],
        out=Path("plots/success_rate_comparison/pusht/pusht_all_rounds.pdf"),
        final_round=30, tick_step=2, ylim=(0.15, 0.65), share_init=True,
        benchmark="Push-T", policy="FM-Policy", tick_step_episodes=5,
        benchmark_tex=r"\pusht",
    ),
    "xvla": dict(
        base=IFT / "active_learning" / "xvla",
        seeds=LIBERO_SEEDS,
        experiments=METHODS,
        out=Path("plots/success_rate_comparison/xvla/xvla_all_rounds.pdf"),
        final_round=15, tick_step=1, ylim=(0.1, 0.6), max_round=15,
        benchmark="LIBERO", policy="X-VLA", tick_step_episodes=3,
        benchmark_tex=r"\liberoten",
    ),
    "fastwam": dict(
        base=IFT / "active_learning" / "fastwam",
        seeds=LIBERO_SEEDS,
        experiments=METHODS,
        out=Path("plots/success_rate_comparison/fastwam/fastwam_all_rounds.pdf"),
        # The paper reports FastWAM over its first 10 rounds.
        final_round=10, tick_step=1, ylim=(0.15, 0.8),
        benchmark="LIBERO", policy="FastWAM", tick_step_episodes=2,
        benchmark_tex=r"\liberoten",
    ),
}

PANELS_OUT = Path("plots/success_rate_comparison/al_success_rate_comparison.pdf")
COMPACT_OUT = Path("plots/success_rate_comparison/libero/libero_success_rates_small.pdf")
# One per benchmark; `key` is the BENCHMARKS key.
class _BarsOut:
    def format(self, key, suffix=""):
        return Path(f"plots/success_rate_comparison/{key}/{key}_threshold_bars{suffix}.pdf")
BARS_OUT = _BarsOut()

# Demonstrations queried per active fine-tuning round. Every simulation run in the paper uses
# 5 (`iteration.episodes_per_round` in each run's iterative_fine_tuning_config.json, verified
# for LIBERO/SmolVLA, Push-T and X-VLA). The x-axis reports collected episodes
# rather than rounds because that, not the round index, is the quantity being economised.
EPISODES_PER_ROUND = 5

# Populated from the selected benchmark in main().
BASE = None
SEEDS = []
EXPERIMENTS = []
MAX_ROUND = None
THRESHOLDS = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65]
MATCHED_SUCCESS_ROUNDS = [13, 14, 15]

# (0-indexed round, display round label, n_demonstrations)
LATEX_ROWS = [(5, 5, 25), (10, 10, 50), (15, 15, 75)]

# Column order and LaTeX header strings for the table
LATEX_HEADERS = {
    "Random": "Random",
    "Diversity": "Diversity",
    "Action-L2": r"\method w/ Action-L2",
    "GU": r"\method w/ GU",
    "AMF": "AMF",
    "VFD (ours)": r"\textbf{\method w/ VFD}",
}
# Fixed column order; only the labels a benchmark actually has are kept (AMF is LIBERO-only,
# so referencing it unconditionally used to raise KeyError on Push-T and X-VLA).
LATEX_ORDER = ["Random", "Diversity", "AMF", "Action-L2", "GU", "VFD (ours)"]
LATEX_COLUMNS = []

# What a method is CALLED in a figure legend, as opposed to the short key everything is looked
# up by. The keys are the labels in BENCHMARKS[...]["experiments"]; only the legend text carries the \method prefix, spelled out
# because matplotlib has no macros. The three uncertainty-guided acquisition rules are named
# "SAVE w/ X" here and as `\method w/ X` in LATEX_HEADERS/ROW_HEADERS, so figures and tables
# agree; "(ours)" belongs to the calibration tables, where VFD is compared as an uncertainty
# estimator rather than as an acquisition rule. Keep keys and display names apart: renaming
# the keys would mean renaming every experiments list, COLORS, MARKERS, LATEX_HEADERS,
# ROW_HEADERS and TABLE_ROWS too.
DISPLAY = {
    "Random": "Random",
    "Diversity": "Diversity",
    "AMF": "AMF",
    "Action-L2": "SAVE w/ Action-L2",
    "GU": "SAVE w/ GU",
    "VFD (ours)": "SAVE w/ VFD",
}

_CANONICAL = {v: k for k, v in DISPLAY.items()}


def canonical(label: str) -> str:
    """Map a display name ("SAVE w/ VFD") back to the internal key ("VFD (ours)").

    COLORS, MARKERS, TABLE_ROWS and the experiments lists are all keyed by the internal name,
    so anything that accepts hand-written labels runs them through this first. That way either
    spelling works and renaming DISPLAY cannot break a caller.
    """
    return _CANONICAL.get(label, label)


COLORS = {
    "Random": "tab:gray",
    "Diversity": "tab:brown",
    "Action-L2": "tab:orange",
    "GU": "tab:pink",
    "AMF": "tab:green",
    "VFD (ours)": "tab:blue",
}

MARKERS = {
    "VFD (ours)": "o",   # circle
    "Action-L2":  "s",   # square
    "GU":         "P",   # filled cross
    "Random":     "*",   # 5-star
    "Diversity":  "X",   # filled diagonal cross
    "AMF":        "^",   # triangle
}


def sample_std(v) -> float:
    """Sample std (ddof=1) over seeds -- the convention every table in the paper uses.

    The tables used to disagree: the cross-benchmark one was ddof=1 while Table 2 and the
    ablation table were ddof=0, so the same run's spread differed by sqrt(n/(n-1)) = 1.22 at
    three seeds depending on which table you read it in. With fewer than two finite values
    ddof=1 is undefined; report 0.0, which reads as "one run, no spread".

    Figures still shade a POPULATION std band (draw_curves, rounds_to_target) -- see
    paper_plots/README.md.
    """
    a = np.asarray(v, dtype=float)
    return float(np.nanstd(a, ddof=1)) if int(np.isfinite(a).sum()) > 1 else 0.0


def load_success_rates(exp_name: str) -> tuple[list[int], np.ndarray]:
    """Return (rounds, arr) where arr is (n_seeds, n_rounds), round 0 = initial eval."""
    seed_series = []
    for seed in SEEDS:
        seed_dir = BASE / f"{exp_name}_{seed}"
        values: dict[int, float] = {}

        init = seed_dir / "initial_evaluation" / "initial_evaluation.json"
        if init.exists():
            d = json.loads(init.read_text())
            values[0] = float(d["mean_member_macro_success_rate"])

        for round_dir in sorted(seed_dir.glob("round_*"), key=lambda p: int(p.name.split("_")[1])):
            r = int(round_dir.name.split("_")[1]) + 1  # 1-indexed: round_000 → 1
            path = round_dir / "round_evaluation.json"
            if path.exists():
                d = json.loads(path.read_text())
                values[r] = float(d["mean_member_macro_success_rate"])

        seed_series.append(values)

    all_rounds = sorted({r for s in seed_series for r in s})
    if MAX_ROUND is not None:
        all_rounds = [r for r in all_rounds if r <= MAX_ROUND]
    arr = np.array([[s.get(r, float("nan")) for r in all_rounds] for s in seed_series], dtype=float)
    return all_rounds, arr


def compute_auc(x: np.ndarray, means: np.ndarray) -> float:
    """Trapezoidal AUC of improvement over initial success rate, normalised by x-range."""
    valid = np.isfinite(means)
    if valid.sum() < 2:
        return float("nan")
    xv, yv = x[valid].astype(float), means[valid]
    baseline = yv[0]
    raw = np.trapz(yv - baseline, xv)
    return raw / (xv[-1] - xv[0])


def first_round_above(x: np.ndarray, arr: np.ndarray, threshold: float) -> str:
    """Return 'mean±std' of per-seed crossing rounds, but only if the cross-seed mean
    itself surpasses the threshold. Returns '—' otherwise."""
    means = np.nanmean(arr, axis=0)
    if not np.any(np.isfinite(means) & (means >= threshold)):
        return "—"
    crossing_rounds = []
    for seed_vals in arr:
        idx = np.where(np.isfinite(seed_vals) & (seed_vals >= threshold))[0]
        if len(idx):
            crossing_rounds.append(float(x[idx[0]]))
    if not crossing_rounds:
        return "—"
    return f"{np.mean(crossing_rounds):.1f}±{sample_std(crossing_rounds):.1f}"


def print_latex_threshold_table(
    threshold_rows: dict[str, dict[float, str]],
    col_order: list[tuple[str, str]],
    data: dict[str, tuple[list[int], np.ndarray]],
    final_round_idx: int = 15,
) -> None:
    """Rows = thresholds, columns = methods. Cell = first round exceeding threshold (mean±std)."""
    col_labels = [lbl for lbl, _ in col_order]
    col_headers = [hdr for _, hdr in col_order]

    lines = [
        r"\begin{tabular}{l " + " ".join("c" * len(col_labels)) + "}",
        r"    \toprule",
        r"    Threshold & " + " & ".join(col_headers) + r" \\",
        r"    \midrule",
    ]

    for thr in THRESHOLDS:
        raw_means = []
        for lbl in col_labels:
            val = threshold_rows[lbl][thr]
            raw_means.append(float("inf") if val == "—" else float(val.split("±")[0]))

        best = min(raw_means)
        cells = []
        for lbl, m in zip(col_labels, raw_means):
            val = threshold_rows[lbl][thr]
            if val == "—":
                cells.append("—")
            else:
                mean_s, std_s = val.split("±")
                if m <= best + 0.05:
                    cells.append(f"$\\mathbf{{{mean_s}}}^{{\\pm {std_s}}}$")
                else:
                    cells.append(f"${mean_s}^{{\\pm {std_s}}}$")

        lines.append(f"    $\\geq \\qty{{{thr * 100:.0f}}}{{\\percent}}$ & " + " & ".join(cells) + r" \\")

    # Relative improvement of VFD over each method: mean(X_A / X_VFD) - 1 over valid rows
    vfd_label = "VFD (ours)"
    vfd_vals = {thr: (float(threshold_rows[vfd_label][thr].split("±")[0])
                      if threshold_rows[vfd_label][thr] != "—" else float("nan"))
                for thr in THRESHOLDS}

    rel_cells1, rel_cells2 = [], []
    for lbl in col_labels:
        if lbl == vfd_label:
            rel_cells1.append("—")
            rel_cells2.append("—")
            continue
        ratios = []
        for thr in THRESHOLDS:
            val_a = threshold_rows[lbl][thr]
            x = float(val_a.split("±")[0]) if val_a != "—" else float("nan")
            y = vfd_vals[thr]
            if np.isfinite(x) and np.isfinite(y) and x > 0 and y > 0:
                ratios.append((x, y))
        if ratios:
            xs, ys = zip(*ratios)
            rel_cells1.append(f"${1 - np.mean(ys) / np.mean(xs):+.2f}$")
            rel_cells2.append(f"${np.mean([x / y for x, y in ratios]) - 1:+.2f}$")
        else:
            rel_cells1.append("—")
            rel_cells2.append("—")

    lines.append(r"    \midrule")
    lines.append(r"    $1 - \bar{Y}/\bar{X}$ & " + " & ".join(rel_cells1) + r" \\")
    lines.append(r"    $\overline{X/Y} - 1$ & " + " & ".join(rel_cells2) + r" \\")
    lines.append(r"    \midrule")

    # Final success rate row
    final_means, final_stds = [], []
    for lbl in col_labels:
        rounds, arr = data[lbl]
        x = np.array(rounds)
        pos = np.where(x == final_round_idx)[0]
        if len(pos) == 0:
            final_means.append(float("nan"))
            final_stds.append(float("nan"))
        else:
            seed_vals = arr[:, pos[0]]
            final_means.append(float(np.nanmean(seed_vals)) * 100)
            final_stds.append(sample_std(seed_vals) * 100)

    best_final = max((m for m in final_means if np.isfinite(m)), default=float("nan"))
    final_cells = []
    for m, s in zip(final_means, final_stds):
        if not np.isfinite(m):
            final_cells.append("—")
        elif m >= best_final - 0.05:
            final_cells.append(f"$\\mathbf{{{m:.1f}}}^{{\\pm {s:.1f}}}$")
        else:
            final_cells.append(f"${m:.1f}^{{\\pm {s:.1f}}}$")

    lines.append(r"    Final SR & " + " & ".join(final_cells) + r" \\")
    lines += [r"    \bottomrule", r"\end{tabular}"]
    print("\n".join(lines))


def print_latex_table(data: dict[str, tuple[list[int], np.ndarray]]) -> None:
    col_labels = [label for label, _ in LATEX_COLUMNS]
    col_headers = [hdr for _, hdr in LATEX_COLUMNS]

    # Collect (mean, std) per row per column
    table: dict[int, dict[str, tuple[float, float]]] = {}
    for round_idx, _, _ in LATEX_ROWS:
        table[round_idx] = {}
        for label in col_labels:
            rounds, arr = data[label]
            x = np.array(rounds)
            pos = np.where(x == round_idx)[0]
            if len(pos) == 0:
                table[round_idx][label] = (float("nan"), float("nan"))
                continue
            seed_vals = arr[:, pos[0]]
            table[round_idx][label] = (float(np.nanmean(seed_vals)) * 100,
                                       sample_std(seed_vals) * 100)

    def fmt_cell(mean: float, std: float, bold: bool) -> str:
        if not np.isfinite(mean):
            return ""
        val = f"{mean:.1f}^{{\\pm {std:.1f}}}"
        return f"$\\mathbf{{{mean:.1f}}}^{{\\pm {std:.1f}}}$" if bold else f"${val}$"

    lines = [
        r"\begin{tabular}{l " + " ".join("c" * len(col_labels)) + "}",
        r"    \toprule",
        r"    Round $r$ & " + " & ".join(col_headers) + r" \\",
        r"    \midrule",
    ]

    for round_idx, round_label, n_dem in LATEX_ROWS:
        row_data = table[round_idx]
        means = [row_data[lbl][0] for lbl in col_labels]
        stds  = [row_data[lbl][1] for lbl in col_labels]
        best = max((m for m in means if np.isfinite(m)), default=float("nan"))
        cells = [fmt_cell(m, s, np.isfinite(m) and m >= best - 0.05)
                 for m, s in zip(means, stds)]
        row_label = f"\\num{{{round_label}}} ({n_dem} dem.)"
        lines.append(f"    {row_label} & " + " & ".join(cells) + r" \\")

    lines += [r"    \bottomrule", r"\end{tabular}"]
    print("\n".join(lines))


def mean_success_at_round(data: dict[str, tuple[list[int], np.ndarray]], label: str, round_idx: int) -> float:
    rounds, arr = data[label]
    x = np.array(rounds)
    pos = np.where(x == round_idx)[0]
    if len(pos) == 0:
        return float("nan")
    return float(np.nanmean(arr[:, pos[0]]))


def first_mean_round_at_or_above(
    data: dict[str, tuple[list[int], np.ndarray]],
    label: str,
    threshold: float,
) -> int | None:
    rounds, arr = data[label]
    x = np.array(rounds)
    means = np.nanmean(arr, axis=0)
    mask = np.isfinite(means) & (means >= threshold)
    if not np.any(mask):
        return None
    return int(x[np.where(mask)[0][0]])


def print_matched_success_savings(
    data: dict[str, tuple[list[int], np.ndarray]],
    col_order: list[tuple[str, str]],
    reference_label: str = "VFD (ours)",
    rounds_to_match: list[int] = MATCHED_SUCCESS_ROUNDS,
) -> None:
    """Compare each method to VFD at the method's own last-three success rates.

    For each method A and each paper round r in rounds_to_match, take A's mean success
    rate at r as the threshold. Find the first paper round where VFD's mean success
    rate reaches that threshold. Relative rounds are r_vfd / r_a - 1, so negative
    values mean VFD needed fewer rounds/samples.
    """
    labels = [label for label, _ in col_order if label != reference_label]

    print("Matched-success relative rounds vs VFD")
    print("round numbering: paper rounds 1..15")
    print("metric: r_vfd / r_A - 1 (negative = fewer VFD rounds/samples)")
    for label in labels:
        terms: list[float] = []
        inverse_terms: list[float] = []
        details = []
        for round_a in rounds_to_match:
            sr_a = mean_success_at_round(data, label, round_a)
            if not np.isfinite(sr_a):
                details.append(f"r{round_a}: missing")
                continue
            round_vfd = first_mean_round_at_or_above(data, reference_label, sr_a)
            if round_vfd is None:
                details.append(f"r{round_a}: SR={sr_a:.3f}, VFD never reaches")
                continue
            relative_rounds = round_vfd / round_a - 1
            inverse_relative_rounds = round_a / round_vfd - 1
            terms.append(relative_rounds)
            inverse_terms.append(inverse_relative_rounds)
            details.append(
                f"geq {sr_a * 100:.1f}% at A r{round_a}: VFD r{round_vfd}, "
                f"rel={relative_rounds:+.3f}, inv={inverse_relative_rounds:+.3f}"
            )

        if terms:
            print(
                f"geq matched {label}: avg r_vfd/r_A - 1 = {np.mean(terms):+.3f}; "
                f"avg r_A/r_vfd - 1 = {np.mean(inverse_terms):+.3f}; "
                + "; ".join(details)
            )
        else:
            print(
                f"geq matched {label}: avg r_vfd/r_A - 1 = —; "
                f"avg r_A/r_vfd - 1 = —; "
                + "; ".join(details)
            )

    print()
    lines = [
        r"\begin{tabular}{l c c c c}",
        r"    \toprule",
        r"    Method A & Target & $r_A$ & $r_{\mathrm{VFD}}$ & $r_{\mathrm{VFD}}/r_A - 1$ \\",
        r"    \midrule",
    ]
    for label in labels:
        row_savings = []
        for round_a in rounds_to_match:
            sr_a = mean_success_at_round(data, label, round_a)
            round_vfd = first_mean_round_at_or_above(data, reference_label, sr_a) if np.isfinite(sr_a) else None
            if round_vfd is None:
                continue
            row_savings.append(round_vfd / round_a - 1)
            lines.append(
                f"    {label} & $\\geq \\qty{{{sr_a * 100:.1f}}}{{\\percent}}$ & {round_a} & {round_vfd} & "
                f"${round_vfd / round_a - 1:+.2f}$ \\\\"
            )
        if row_savings:
            lines.append(
                f"    {label} mean & -- & -- & -- & "
                f"$\\mathbf{{{np.mean(row_savings):+.2f}}}$ \\\\"
            )
        lines.append(r"    \addlinespace")
    lines += [r"    \bottomrule", r"\end{tabular}"]
    print("\n".join(lines))


# --------------------------------------------------------------------------------------
# Data loading and curve drawing, shared by the single-benchmark and the panel figure
# --------------------------------------------------------------------------------------

from paper_style import (  # noqa: E402  (same directory)
    PAGE, TEXTWIDTH_IN, page_style, use_paper_style, tex,
)


def prepare_data(cfg) -> dict[str, tuple[list[int], np.ndarray]]:
    """Load every experiment of one benchmark as {label: (rounds, (n_seeds, n_rounds))}.

    Sets the module-level BASE/SEEDS/EXPERIMENTS/MAX_ROUND that load_success_rates() reads,
    so callers that iterate over benchmarks do not have to.
    """
    global BASE, SEEDS, EXPERIMENTS, MAX_ROUND
    BASE, SEEDS, EXPERIMENTS = cfg["base"], cfg["seeds"], cfg["experiments"]
    MAX_ROUND = cfg.get("max_round")

    data = {label: load_success_rates(exp_name) for exp_name, label in cfg["experiments"]}

    if cfg.get("share_init"):
        # Round 0 is the pretrained ensemble, before any selection has happened, so every
        # method in a seed group starts from the identical checkpoint and must share the
        # value. Evaluating the same ensemble on different machines can differ by a few points;
        # averaging the init over methods (per seed) removes that evaluation artefact from the
        # plot instead of showing it as a spurious method difference.
        init_stack = [arr[:, rounds.index(0)] for rounds, arr in data.values() if 0 in rounds]
        if init_stack:
            shared = np.nanmean(np.vstack(init_stack), axis=0)   # one value per seed
            for rounds, arr in data.values():
                if 0 not in rounds:
                    continue
                col = rounds.index(0)
                # Only where this method HAS that seed. Push-T Random ran five of the nine seed
                # groups; writing the shared init into the other four would give it a round 0 it
                # never evaluated and make the method look present at every seed.
                have = np.isfinite(arr[:, col])
                arr[have, col] = shared[have]
    return data


def draw_curves(ax, cfg, data, *, fontsize=12, legend=True, title=None,
                xlabel=True, ylabel=True, linewidth=3, markersize=8) -> None:
    """Draw one benchmark's mean +- std learning curves onto `ax`.

    Identical styling in the single-benchmark figure and in each panel of the combined one;
    only the sizes shrink for panels, so the two stay visually comparable.
    """
    rounds = None
    for label, (rounds, arr) in data.items():
        x = np.array(rounds)
        means = np.nanmean(arr, axis=0)
        stds = np.nanstd(arr, axis=0)
        color = COLORS[label]
        ax.plot(x, means, marker=MARKERS[label], markersize=markersize, linewidth=linewidth,
                markeredgecolor="white", markeredgewidth=0.5,
                label=DISPLAY.get(label, label), color=color)
        ax.fill_between(x, means - stds, means + stds, alpha=0.15, color=color)

    fs = {} if fontsize is None else {"fontsize": fontsize}
    if xlabel:
        ax.set_xlabel(tex("# Demonstrations"), **fs)
    if ylabel:
        ax.set_ylabel("Success Rate", **fs)
    if title:
        ax.set_title(title, **fs)
    ax.set_ylim(*cfg["ylim"])
    # Curves are indexed by round, but the axis is labelled in demonstrations collected:
    # round r cost r * EPISODES_PER_ROUND episodes, and round 0 (the pretrained ensemble,
    # before any selection) is 0. Ticks stay on round positions -- tick_step_episodes is how
    # many rounds apart they are -- so only the labels change.
    per_round = cfg.get("episodes_per_round", EPISODES_PER_ROUND)
    step = cfg.get("tick_step_episodes", cfg["tick_step"])
    shown = [rounds[0]] + [r for r in rounds[1:] if r % step == 0]
    ax.set_xticks(shown)
    ax.set_xticklabels([str(r * per_round) for r in shown], **fs)
    if fontsize is not None:
        ax.tick_params(axis="y", labelsize=fontsize)
    ax.grid(True, alpha=0.3, axis="y")
    if legend:
        ax.legend(fontsize=fontsize, loc="lower right")


def save_figure(fig, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print(f"Saved: {out}\n")


# Size and type scale of a wrapfigure-sized panel (~0.45\\textwidth). Everything drawn at this
# size goes through COMPACT_*, so such figures stay interchangeable on a page.
COMPACT_FIGSIZE = (3.4, 2.5)
COMPACT_FONTSIZE = 8
COMPACT_LINEWIDTH = 1.5
COMPACT_MARKERSIZE = 4


def apply_compact_style(ax, *, scale: float = 1.0) -> None:
    """Thin spines/ticks and tight label padding, for a figure that is small on the page.

    `scale` grows the tick marks, spines and padding together with the type; a figure drawn at
    larger fonts than COMPACT_FONTSIZE needs it, or its hairline ticks look detached from the
    labels.
    """
    ax.xaxis.labelpad = 2 * scale
    ax.yaxis.labelpad = 2 * scale
    ax.tick_params(axis="both", length=2.1 * scale, width=0.56 * scale, pad=1 * scale)
    for spine in ax.spines.values():
        spine.set_linewidth(0.56 * scale)


def compact_legend(ax, *, loc="lower right", ncol=2, fontsize=6.5, spacing: float = 1.0):
    # fontsize=None inherits rcParams["legend.fontsize"], which the paper style sets.
    """Legend inside the axes, in the canonical method order.

    A wrapped figure has no room for a legend underneath, and ordering by LATEX_ORDER rather
    than by plotting order keeps the entries in the same sequence as every other figure even
    when a run is missing.

    `spacing` multiplies every gap in the legend box at once -- between entries, around the
    frame, and between a marker and its text -- so a figure with larger type can open the
    legend up to match instead of tuning five parameters separately.
    """
    handles, labels = ax.get_legend_handles_labels()
    wanted = [DISPLAY.get(l, l) for l in LATEX_ORDER]
    order = [labels.index(l) for l in wanted if l in labels]
    legend = ax.legend([handles[i] for i in order], [labels[i] for i in order],
                       **({} if fontsize is None else {"fontsize": fontsize}),
                       loc=loc, ncol=ncol, frameon=True,
                       facecolor="white", edgecolor="0.6", framealpha=1.0,
                       handlelength=1.4 * spacing, labelspacing=0.25 * spacing,
                       columnspacing=1.0 * spacing, borderpad=0.3 * spacing,
                       handletextpad=0.4 * spacing)
    legend.get_frame().set_linewidth(0.56 * spacing)
    return legend


def plot_compact(key: str, out: Path) -> None:
    """One benchmark at wrapfigure size: ~0.45\\textwidth, so it must survive being small.

    Same data and colours as the full-size figure, but the type is sized for a two-column
    wrap rather than scaled down by LaTeX -- fonts stay legible instead of shrinking with the
    image.
    """
    cfg = BENCHMARKS[key]
    fig, ax = plt.subplots(figsize=COMPACT_FIGSIZE)
    draw_curves(ax, cfg, prepare_data(cfg), fontsize=COMPACT_FONTSIZE, legend=False,
                linewidth=COMPACT_LINEWIDTH, markersize=COMPACT_MARKERSIZE)
    apply_compact_style(ax)
    compact_legend(ax)
    fig.tight_layout()
    save_figure(fig, out)


# Bar groups, left to right. "gain" is measured against the benchmark's own base success rate
# (so it asks how much a method ADDED); "abs" is an absolute success rate, comparable across
# benchmarks but reached trivially early by one that starts high. The figure carries both and
# separates them with a dashed rule.
TARGETS = [("gain", 0.10), ("gain", 0.20), ("gain", 0.30), ("gain", 0.40),
           ("abs", 0.50), ("abs", 0.60)]


def base_success_rate(data) -> tuple[np.ndarray, int]:
    """Per-seed success rate before any selection, and the round it was read from.

    Round 0 is the pretrained ensemble, shared by every method of a benchmark, so averaging
    over methods there is a no-op that just tolerates the Push-T evaluation split (see
    `share_init`). FastWAM has no `initial_evaluation` on disk, so its earliest round is 1 and
    the methods have already diverged; the mean over methods is the closest available proxy and
    the caller says so. The returned round is 0 for a true base and >0 for that proxy.
    """
    first = min(r for rounds, _ in data.values() for r in rounds)
    stack = [arr[:, rounds.index(first)] for rounds, arr in data.values() if first in rounds]
    return np.nanmean(np.vstack(stack), axis=0), first


def rounds_to_target(rounds, arr, per_seed_target: np.ndarray, mean_target: float):
    """(mean, std) rounds for a method to reach its target success rate, or None.

    Same rule as the threshold table: a bar is drawn only if the CROSS-SEED MEAN curve clears
    the target, and its height is then the mean over the seeds that individually cleared it.
    Without the first test a single lucky seed would put up a bar for a method whose average
    never gets there; without the second the height would be pulled by seeds that never cross.

    The target is per seed because a gain-based one rides on that seed's own base.
    """
    x = np.array(rounds)
    post = x >= 1
    means = np.nanmean(arr, axis=0)
    if not np.any(post & np.isfinite(means) & (means >= mean_target)):
        return None
    per = []
    for seed_i, seed_vals in enumerate(arr):
        hit = np.where(post & np.isfinite(seed_vals) & (seed_vals >= per_seed_target[seed_i]))[0]
        if len(hit):
            per.append(float(x[hit[0]]))
    return (float(np.mean(per)), float(np.std(per))) if per else None


def target_levels(kind: str, value: float, base: np.ndarray) -> tuple[np.ndarray, float]:
    """(per-seed target, cross-seed-mean target) for one bar group."""
    if kind == "gain":
        return base + value, float(np.nanmean(base)) + value
    return np.full(base.shape, value), value


def parse_bar_targets(spec: str):
    """"+20,+30" or ">=50,>=60" -> the TARGETS subset, in the order given."""
    out = []
    for tok in (t.strip() for t in spec.split(",") if t.strip()):
        if tok.startswith("+"):
            out.append(("gain", round(float(tok[1:].rstrip("%")) / 100, 4)))
        else:
            out.append(("abs", round(float(tok.lstrip(">=").rstrip("%")) / 100, 4)))
    unknown = [t for t in out if t not in TARGETS]
    if unknown:
        raise SystemExit(f"--bar-targets: {unknown} not in {TARGETS}")
    return out


def plot_threshold_bars(key: str, out: Path, targets=None, ylim_top=None, legend_ncol=2) -> None:
    """Grouped bars: rounds needed to gain +10/+20/+30/+40 points over the base success rate.

    One group per gain, one bar per method, in the same colours and order as the learning-curve
    figures. A method that never reaches a gain simply has no bar there -- drawing a zero would
    read as "reached it immediately", which is the opposite.
    """
    cfg = BENCHMARKS[key]
    targets = targets or TARGETS
    data = prepare_data(cfg)
    base, base_round = base_success_rate(data)

    # Bars are reported in demonstrations, not rounds: that is the cost the acquisition rule is
    # economising, and it matches the "Collected episodes" axis of the learning curves. The
    # factor is per benchmark, so the numbers stay comparable across them.
    per_round = cfg.get("episodes_per_round", EPISODES_PER_ROUND)

    labels = [l for l in LATEX_ORDER if l in data]
    levels = [target_levels(kind, value, base) for kind, value in targets]
    heights = {l: [rounds_to_target(*data[l], per_seed, mean_t) for per_seed, mean_t in levels]
               for l in labels}

    page_style("bars_subset" if len(targets) < len(TARGETS) else "bars")
    fig, ax = plt.subplots()
    width = 0.8 / len(labels)
    for i, label in enumerate(labels):
        xs, ys, es = [], [], []
        for gi, cell in enumerate(heights[label]):
            if cell is None:
                continue
            xs.append(gi - 0.4 + width * (i + 0.5))
            ys.append(cell[0] * per_round); es.append(cell[1] * per_round)
        ax.bar(xs, ys, width=width * 0.92, yerr=es, capsize=2,
               color=COLORS[label], label=DISPLAY.get(label, label),
               error_kw=dict(elinewidth=0.8, capthick=0.8))

    # Dashed rule where the meaning of a group changes from "added over base" to "absolute".
    n_gain = sum(1 for kind, _ in targets if kind == "gain")
    if 0 < n_gain < len(targets):
        ax.axvline(n_gain - 0.5, color="0.55", linewidth=0.8, linestyle="--")
    ax.set_xticks(range(len(targets)))
    # Units live on the ticks, since the xlabel names the quantity only. Under usetex "\," is a
    # thin space and "%" MUST be escaped -- a bare one comments out the rest of the LaTeX line.
    # Without usetex both would render literally, hence the two spellings.
    usetex = plt.rcParams.get("text.usetex")
    thin, pct = (r"\,", r"\%") if usetex else (" ", "%")
    ax.set_xticklabels([(f"+{int(v * 100)}{thin}pp" if kind == "gain"
                         else f"$\\geq${thin}{int(v * 100)}{pct}")
                        for kind, v in targets])
    ax.set_xlabel("SR gain over base          Absolute success rate"
                  if n_gain and n_gain < len(targets) else
                  ("SR gain over base" if n_gain else "Absolute success rate"))
    ax.set_ylabel("Required demonstrations")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_axisbelow(True)
    # Headroom for the legend, sized from the tallest bar INCLUDING its error bar. The legend
    # sits top-left over the shortest group, but with few groups it spans most of the width, so
    # leaving it to autoscale buries the tall bars underneath it. `ylim_top` pins it instead,
    # for a figure that has to line up with a neighbour or sit on a round number.
    tallest = max((c[0] + c[1]) * per_round
                  for cells in heights.values() for c in cells if c is not None)
    ax.set_ylim(0, ylim_top if ylim_top is not None else tallest * 1.42)
    ax.legend(ncol=legend_ncol, loc="upper left", framealpha=0.9)
    # No tight_layout: the ICLR bundle turns constrained layout on and the two conflict.
    save_figure(fig, out)

    base_note = ("round 0" if base_round == 0 else
                 f"round {base_round} averaged over methods -- this benchmark has no "
                 f"initial_evaluation, so the true pre-selection base is unavailable")
    print(f"%   base success rate = {np.nanmean(base) * 100:.1f}% ({base_note})")
    for label in labels:
        missing = [(f"+{int(v * 100)}" if kind == "gain" else f">={int(v * 100)}%")
                   for (kind, v), c in zip(targets, heights[label]) if c is None]
        if missing:
            print(f"%   {DISPLAY.get(label, label)} never reaches {', '.join(missing)}")


# What each cfg["benchmark_tex"] macro expands to in main.tex (inside \texttt). The macros
# themselves are undefined in matplotlib's LaTeX, so the text is spelled out here.
# How far below its constrained-layout position the shared legend sits, in inches.
PANELS_LEGEND_DROP_IN = 0.07
PANEL_BENCHMARK_TT = {r"\pusht": "Push-T", r"\liberoten": "LIBERO-Long"}


def plot_panels(keys: list[str], out: Path, ncols: int = 2) -> None:
    """One figure, one panel per benchmark on a grid, with a shared legend underneath.

    The panels keep their own y-limits and x-ticks because the horizons (60-150 demonstrations)
    and success-rate ranges genuinely differ; sharing them would flatten every curve. That is
    also why every panel carries its own x-label rather than only the bottom row -- the columns
    are not a shared axis, and a label on the bottom row alone would suggest they are.
    """
    n = len(keys)
    ncols = max(1, min(ncols, n))
    nrows = -(-n // ncols)                       # ceil
    page_style("panels")
    fig, axes = plt.subplots(nrows, ncols)
    axes = np.atleast_1d(axes).ravel()

    for i, (key, ax) in enumerate(zip(keys, axes)):
        cfg = BENCHMARKS[key]
        draw_curves(
            ax, cfg, prepare_data(cfg),
            fontsize=None, legend=False, ylabel=(i % ncols == 0),
            # Benchmark in \texttt, spelled as the paper's macro for it sets it.
            title=rf"\texttt{{{PANEL_BENCHMARK_TT[cfg['benchmark_tex']]}}} ({cfg['policy']})",
            linewidth=1.4, markersize=3.5,
        )
        # 0.1 steps everywhere: at 9 pt ticks the auto locator picked 0.2 for FastWAM only.
        ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
    for ax in axes[n:]:                          # a grid wider than the benchmark count
        ax.axis("off")

    # One legend for the whole figure, in the fixed method order rather than whichever
    # panel happens to be first -- Push-T has no AMF run, so panel 0 alone would drop it.
    handles = {}
    for ax in axes[:n]:
        for h, l in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(l, h)
    ordered = [(handles[DISPLAY[l]], DISPLAY[l]) for l in LATEX_ORDER if DISPLAY.get(l) in handles]
    # Boxed, in the house style shared with compact_legend() and the bar figures.
    legend = fig.legend([h for h, _ in ordered], [l for _, l in ordered],
                        # Two rows: at 9 pt one row would not fit.
                        # "outside": constrained layout reserves the legend's height below
                        # the panels, so it cannot overlap the x-labels whatever its size.
                        loc="outside lower center", ncol=-(-len(ordered) // 2), frameon=True,
                        facecolor="white", edgecolor="0.6", framealpha=1.0)
    legend.get_frame().set_linewidth(0.8)
    # A little more air between the x-labels and the legend: let
    # constrained layout place everything once, freeze it, then lower only the legend. The
    # tight bbox at save time grows the page to include it.
    fig.canvas.draw()
    fig.set_layout_engine("none")
    top = legend.get_window_extent().transformed(fig.transFigure.inverted()).y1
    legend.set_loc("upper center")
    legend.set_bbox_to_anchor((0.5, top - PANELS_LEGEND_DROP_IN / fig.get_figheight()),
                              transform=fig.transFigure)

    # No tight_layout: the ICLR bundle turns constrained layout on and the two conflict.
    save_figure(fig, out)


# --------------------------------------------------------------------------------------
# Cross-benchmark summary tables
# --------------------------------------------------------------------------------------

# Benchmarks that become column GROUPS, in order.
TABLE_KEYS = ["pusht", "libero", "xvla", "fastwam"]
# Row order and the LaTeX name of each method (rows are methods now, one per line).
TABLE_ROWS = ["Random", "Diversity", "AMF", "GU", "Action-L2", "VFD (ours)"]
ROW_HEADERS = {
    "Random": "Random",
    "Diversity": "Diversity",
    "AMF": "AMF",
    "GU": r"\method w/ GU",
    "Action-L2": r"\method w/ Action-L2",
    "VFD (ours)": r"\textbf{\method w/ VFD}",
}
# (column label, per-seed metric). Two columns per benchmark group.
# Second column of the cross-benchmark table; --sr-metric switches it.
SR_METRICS = {"last3": ("Last-3 SR", "_per_seed_last3"), "final": ("Final SR", "_per_seed_final")}
TABLE_METRICS = [("AULC", "_per_seed_aulc"), SR_METRICS["last3"]]


def _per_seed_aulc(rounds: list[int], arr: np.ndarray) -> np.ndarray:
    """Mean success rate over the learning curve, per seed, with the init round excluded.

    Trapezoidal area over rounds >= 1 divided by the round span, i.e. the average height of
    the curve. Round 0 is the pretrained ensemble before any selection: including it would
    pull every method towards a shared constant and shrink the differences the table is
    meant to show. Normalising by the span keeps benchmarks with different horizons on the
    same scale, but note it does NOT make a 20-round and a 30-round number interchangeable.
    """
    idx = [i for i, r in enumerate(rounds) if r >= 1]
    x = np.array([rounds[i] for i in idx], dtype=float)
    out = []
    for seed_vals in arr[:, idx]:
        ok = np.isfinite(seed_vals)
        if ok.sum() < 2:
            out.append(float("nan")); continue
        xv, yv = x[ok], seed_vals[ok]
        out.append(float(np.trapz(yv, xv) / (xv[-1] - xv[0])) * 100)
    return np.array(out, dtype=float)


def _per_seed_final(rounds: list[int], arr: np.ndarray) -> np.ndarray:
    """Success rate at each seed's LAST evaluated round.

    Per seed rather than at a fixed round, because a partially synced benchmark has methods at
    different depths -- so "final" is not one round across a row. Where every method ran the
    same number of rounds this is simply the last round.
    """
    idx = [i for i, r in enumerate(rounds) if r >= 1]
    out = []
    for seed_vals in arr[:, idx]:
        fin = np.where(np.isfinite(seed_vals))[0]
        out.append(float(seed_vals[fin[-1]]) * 100 if len(fin) else float("nan"))
    return np.array(out, dtype=float)


def _per_seed_last3(rounds: list[int], arr: np.ndarray) -> np.ndarray:
    idx = [i for i, r in enumerate(rounds) if r >= 1][-3:]
    return np.nanmean(arr[:, idx], axis=1) * 100


def _last_round(rounds: list[int], arr: np.ndarray) -> int:
    """Highest round this method has in every seed it actually ran.

    All-NaN rows are seeds the method never ran, not seeds it ran and lost: Push-T Random has
    five seed groups where the other methods have nine. Requiring every ROW to be finite would
    read those absent seeds as a missing round and drop the method from the table entirely.
    """
    post = [i for i, r in enumerate(rounds) if r >= 1]
    present = np.isfinite(arr[:, post]).any(axis=1) if post else np.zeros(len(arr), bool)
    if not present.any():
        return 0
    sub = arr[present]
    ok = [r for i, r in enumerate(rounds) if r >= 1 and np.isfinite(sub[:, i]).all()]
    return max(ok) if ok else 0


def _table_cuts(data, forced: int | None = None) -> tuple[dict[str, int], dict[str, int]]:
    """Last round to read per method, plus each method's available depth.

    By default every method is read over ALL the rounds it has, even when a partially synced
    unfinished runs leave them at different depths. The columns are then not at a common horizon --
    an average curve height over 2 rounds is not the same measurement as one over 10, and the
    early rounds are the steep part of every curve -- so the table notes which benchmarks are
    uneven. Pass --horizon to truncate every method to one round instead, which reports only
    the methods that reach it.
    """
    depths = {label: _last_round(rounds, arr) for label, (rounds, arr) in data.items()}
    if forced is None:
        return dict(depths), depths
    return {label: (forced if d >= forced else 0) for label, d in depths.items()}, depths


def _collect(keys: list[str], horizon: int | None = None):
    """{(benchmark key, metric label): {method: (mean, std over seeds)}} + per-benchmark notes.

    Every benchmark is loaded once and both metrics are taken from the same arrays, so the
    AULC and Last-3 columns can never disagree about which runs they describe.
    """
    metrics = {name: globals()[fn] for name, fn in TABLE_METRICS}
    cells: dict[tuple[str, str], dict[str, tuple[float, float]]] = {}
    notes: list[str] = []
    for key in keys:
        data = prepare_data(BENCHMARKS[key])
        cuts, depths = _table_cuts(data, horizon)
        where = f"{BENCHMARKS[key]['benchmark']} ({BENCHMARKS[key]['policy']})"
        uneven = len(set(d for d in depths.values() if d > 0)) > 1
        if uneven or any(d == 0 for d in depths.values()):
            detail = ", ".join(f"{DISPLAY.get(lbl, lbl)} {depths[lbl]}"
                               for lbl in TABLE_ROWS if lbl in depths)
            if horizon is None:
                notes.append(f"% {where} is partially synced -- rounds per method: {detail}. "
                             f"Each method is read over all the rounds it has, so this "
                             f"benchmark's columns are NOT at a common horizon.")
            else:
                notes.append(f"% {where} is partially synced -- rounds per method: {detail}. "
                             f"All methods are truncated to round {horizon}; methods that do "
                             f"not reach it are '--'.")
        # Last-3 SR averages the final three rounds, so below 4 rounds it covers the whole
        # curve and coincides with AULC -- the two columns stop being different measurements.
        shallow = [DISPLAY.get(l, l) for l, c in cuts.items() if 0 < c < 4]
        if shallow:
            verb = "has" if len(shallow) == 1 else "have"
            notes.append(f"% WARNING {where}: {', '.join(shallow)} {verb} fewer than 4 rounds, "
                         f"so its Last-3 SR spans the whole curve and duplicates its AULC.")
        counts = {lbl: int(np.isfinite(a).any(axis=1).sum()) for lbl, (_, a) in data.items()}
        if len(set(counts.values())) > 1:
            detail = ", ".join(f"{DISPLAY.get(l, l)} {counts[l]}"
                               for l in TABLE_ROWS if l in counts)
            notes.append(f"% {where}: methods do not share a seed count -- {detail}. Means and "
                         f"spreads use whatever seeds each method has.")
        for metric_label, metric in metrics.items():
            col = {}
            for label, (rounds, arr) in data.items():
                cut = cuts[label]
                if cut == 0:
                    col[label] = (float("nan"), float("nan"))
                    continue
                idx = [i for i, r in enumerate(rounds) if r <= cut]
                vals = metric([rounds[i] for i in idx], arr[:, idx])
                col[label] = (float(np.nanmean(vals)), sample_std(vals))
            cells[(key, metric_label)] = col
    return cells, notes


def print_cross_benchmark_table(keys: list[str] | None = None, horizon: int | None = None) -> None:
    """One table: rows = methods, columns = benchmark x policy x {AULC, Last-3 SR}.

    Three header rows -- Benchmark, Policy, Metric. Benchmarks that share a name (every
    LIBERO-10 policy) merge into one spanning group in the first row. The best mean in each
    COLUMN is bold: a column is one policy and one metric, which is the only comparison that
    means anything, since AULC over 30 rounds and over 15 rounds are not on the same scale.

    Each method is read over all the rounds it has; see _table_cuts() for what that costs on a
    partially synced benchmark and how --horizon overrides it.
    """
    keys = keys or TABLE_KEYS
    cells, notes = _collect(keys, horizon)
    columns = [(k, m) for k in keys for m, _ in TABLE_METRICS]
    n_metrics = len(TABLE_METRICS)

    # Bold = best in the column, underline = second best, matching the failure-prediction
    # table (fiper_vfd/scripts/make_paper_table_3row.py). Ranks are compared at the DISPLAYED
    # precision, so two cells that print the same number are never marked differently, and a
    # tie shares its rank -- if two methods tie for best, both are bold and the next distinct
    # value is the one underlined.
    dec = 1
    bests, seconds = {}, {}
    for col in columns:
        vals = {m: v[0] for m, v in cells[col].items() if np.isfinite(v[0])}
        if not vals:
            bests[col], seconds[col] = set(), set()
            continue
        order = sorted(vals, key=lambda m: vals[m], reverse=True)
        top = round(vals[order[0]], dec)
        bests[col] = {m for m in vals if round(vals[m], dec) == top}
        rest = [m for m in order if m not in bests[col]]
        runner = round(vals[rest[0]], dec) if rest else None
        seconds[col] = ({m for m in vals if m not in bests[col] and round(vals[m], dec) == runner}
                        if rest else set())

    lines = [r"\begin{tabular}{l" + "c" * len(columns) + "}", r"    \toprule"]

    # Header 1: benchmark, merging adjacent policies of the same benchmark into one span.
    groups, spans = [], []
    for k in keys:
        name = BENCHMARKS[k]["benchmark_tex"]
        if groups and groups[-1] == name:
            spans[-1] += n_metrics
        else:
            groups.append(name); spans.append(n_metrics)
    cols_1 = " & ".join(rf"\multicolumn{{{w}}}{{c}}{{{g}}}" for g, w in zip(groups, spans))
    lines.append(f"    Benchmark & {cols_1} " + r"\\")
    # Rule only under a benchmark that spans more than one policy -- a single-policy group is
    # already delimited by the Policy row's rule directly underneath.
    at, rules = 2, []
    for w in spans:
        if w > n_metrics:
            rules.append(rf"\cmidrule(lr){{{at}-{at + w - 1}}}")
        at += w
    if rules:
        lines.append("    " + " ".join(rules))

    # Header 2: policy, one span per benchmark key.
    cols_2 = " & ".join(rf"\multicolumn{{{n_metrics}}}{{c}}{{{BENCHMARKS[k]['policy']}}}" for k in keys)
    lines.append(f"    Policy & {cols_2} " + r"\\")
    lines.append("    " + " ".join(
        rf"\cmidrule(lr){{{2 + i * n_metrics}-{1 + (i + 1) * n_metrics}}}" for i in range(len(keys))))

    lines.append("    Metric & " + " & ".join(m for _, m in columns) + r" \\")
    lines.append(r"    \midrule")

    for method in TABLE_ROWS:
        row = []
        for col in columns:
            v = cells[col].get(method)
            if v is None or not np.isfinite(v[0]):
                row.append("--")
                continue
            t = f"{v[0]:.{dec}f}"
            if method in bests[col]:
                t = r"\mathbf{" + t + "}"
            elif method in seconds[col]:
                t = r"\underline{" + t + "}"
            row.append(rf"${t}^{{\pm {v[1]:.{dec}f}}}$")
        lines.append(f"    {ROW_HEADERS[method]} & " + " & ".join(row) + r" \\")

    lines += [r"    \bottomrule", r"\end{tabular}"]
    print("\n".join(lines))
    sr_desc = {"Last-3 SR": "mean success rate over the final three rounds",
               "Final SR": "success rate at each seed's last evaluated round"}
    sr_name = TABLE_METRICS[1][0]
    print("% AULC: area under the learning curve, init round excluded. "
          f"{sr_name}: {sr_desc[sr_name]}.")
    print("% Both in percent, mean +- sample std over seeds.")
    for note in notes:
        print(note)


# --------------------------------------------------------------------------------------
# Per-tuning-parameter ablation table (tables/active_learning_ablation.tex)
# --------------------------------------------------------------------------------------
# One block per acquisition rule, one row per tuning parameter, columns = success rate after
# rounds 5/10/15. `star=True` marks the setting whose numbers feed the summary table.
# Everything is the LIBERO/SmolVLA sweep; run stems are relative to BENCHMARKS["libero"]["base"].
ABLATION_ROUNDS = (5, 10, 15)
# Optional trailing summary columns (bolded per block), e.g. ("Last-3 SR", "_per_seed_last3").
ABLATION_SUMMARY = ()
TAUS = ("0", "1", "1.5", "2", "2.5")


def _sweep_stem(method: str, tau: str, best: str) -> str:
    """Run name of one SmolVLA sweep setting: the main run at the paper setting, else
    `<method>_t<tau>` (written by scripts/run_smolvla_sweep.sh)."""
    return method if tau == best else f"{method}_t{tau}"


ABLATION_BLOCKS = [
    ("AMF", [(rf"$\sigma=10^{{{e}}}$", "amf" if e == "-2" else f"amf_sigma1e{e}", e == "-2", None)
             for e in ("-4", "-3", "-2", "-1")]),
    ("Action-L2", [(rf"$\tau = {t}$", _sweep_stem("action_l2", t, "1.5"), t == "1.5", None) for t in TAUS]),
    ("GU", [(rf"$\tau = {t}$", _sweep_stem("gu", t, "1"), t == "1", None) for t in TAUS]),
    ("VFD", [(rf"$\tau = {t}$", _sweep_stem("vfd", t, "2.5"), t == "2.5", None) for t in TAUS]),
]


def print_ablation_table() -> None:
    """The appendix's per-tuning-parameter table, replacing the hand-maintained version.

    The spread is the sample std (ddof=1) over the three seeds, matching every other table.
    NOTE this changes the published spreads, which were ddof=0: Diversity round 1 was
    29.6 +- 3.3 and is now 29.6 +- 4.0. The means are unaffected.
    """
    cfg = BENCHMARKS["libero"]
    global BASE, SEEDS, EXPERIMENTS, MAX_ROUND
    BASE, SEEDS, MAX_ROUND = cfg["base"], cfg["seeds"], cfg.get("max_round")

    def fmt(v, bold=False):
        """mean +- sample std (ddof=1), the convention shared by every table."""
        if not np.any(np.isfinite(v)):
            return "--"
        t = f"{np.nanmean(v):.1f}"
        if bold:
            t = r"\mathbf{" + t + "}"
        return f"${t}^{{\pm {sample_std(v):.1f}}}$"

    def row_values(stem):
        """(per-round seed arrays, per-summary-metric seed arrays) for one row."""
        rounds, arr = load_success_rates(stem)
        per_round = [arr[:, rounds.index(r)] * 100 if r in rounds else np.array([np.nan])
                     for r in ABLATION_ROUNDS]
        # Same AULC and Last-3 SR as the summary table, so a starred row's cells here equal
        # its cells there, spread included.
        summary = [globals()[fn](rounds, arr) for _, fn in ABLATION_SUMMARY]
        return per_round, summary

    def block_cells(rows):
        """Formatted cells for a whole block, bolding the best of each summary column.

        The comparison is within a block and only over its tuning parameters, so a single-row
        block (Diversity) gets no mark -- there is nothing to be best of. Ranks use the printed
        precision, so two rows showing the same number are never marked differently.
        """
        vals = [row_values(stem) for _, stem, _, _ in rows]
        best = []
        for j in range(len(ABLATION_SUMMARY)):
            means = [np.nanmean(v[1][j]) for v in vals]
            finite = [m for m in means if np.isfinite(m)]
            best.append(round(max(finite), 1) if finite and len(rows) > 1 else None)
        out = []
        for per_round, summary in vals:
            cells = [fmt(v) for v in per_round]
            for j, v in enumerate(summary):
                m = np.nanmean(v)
                cells.append(fmt(v, bold=best[j] is not None
                                  and np.isfinite(m) and round(m, 1) == best[j]))
            out.append(cells)
        return out

    n_cols = len(ABLATION_ROUNDS) + len(ABLATION_SUMMARY)
    print(r"\begin{tabular}{l l" + " c" * n_cols + "}")
    print(r"        \toprule")
    print("        Method & Hyperparameter & "
          + " & ".join([f"Round-{r} SR" for r in ABLATION_ROUNDS]
                       + [n for n, _ in ABLATION_SUMMARY]) + r" \\")
    for method, rows in ABLATION_BLOCKS:
        print(r"        \midrule")
        formatted = block_cells(rows)
        for i, ((param, stem, star, override), cells) in enumerate(zip(rows, formatted)):
            # A one-row block has no hyperparameter, so its star goes on the method name.
            single = len(rows) == 1
            name = (override or method) + ("*" if single and star else "")
            head = (f"\\multirow{{{len(rows)}}}{{*}}{{{name}}}" if not single and i == 0
                    else (name if single else ""))
            mark = "*" if star and not single else ""
            print(f"        {head} & {param}{mark} & " + " & ".join(cells) + r" \\")
    print(r"        \bottomrule")
    print(r"\end{tabular}")
    print("% Success rate in percent, mean +- sample std over 3 seeds. * marks the setting reported")
    print("% in the summary table (highest Last-3 SR). Summary columns, if any, are bolded per block.")


# --------------------------------------------------------------------------------------
# Sample-efficiency table, one per benchmark (tab:active_learning_comparison)
# --------------------------------------------------------------------------------------
EFFICIENCY_N_THRESHOLDS = 6
EFFICIENCY_STEP = 0.05


def rounds_to_threshold(rounds, arr, thr: float):
    """(mean, sample std) rounds until a method's success rate reaches `thr`, or None.

    Same rule as the bar figures and the old threshold table: only if the CROSS-SEED MEAN curve
    reaches the threshold, and then the mean over the seeds that individually reached it.
    """
    x = np.array(rounds)
    post = x >= 1
    means = np.nanmean(arr, axis=0)
    if not np.any(post & np.isfinite(means) & (means >= thr)):
        return None
    per = []
    for seed_vals in arr:
        hit = np.where(post & np.isfinite(seed_vals) & (seed_vals >= thr))[0]
        if len(hit):
            per.append(float(x[hit[0]]))
    return (float(np.mean(per)), sample_std(per)) if per else None


def efficiency_thresholds(data) -> list[float]:
    """Six consecutive multiples of 5 % ending at the highest one any method's mean reaches."""
    peak = max(float(np.nanmax(np.nanmean(arr, axis=0)[np.array(rounds) >= 1]))
               for rounds, arr in data.values())
    top = np.floor(peak / EFFICIENCY_STEP + 1e-9) * EFFICIENCY_STEP
    return [round(top - EFFICIENCY_STEP * i, 4) for i in range(EFFICIENCY_N_THRESHOLDS - 1, -1, -1)]


EFFICIENCY_CAPTION = (
    r"\textbf{Sample efficiency.} Number of demonstrations to reach certain success rates (SR), "
    r"area under the learning curve (AULC) ($\uparrow$), and final SR ($\uparrow$) for different "
    r"active fine-tuning strategies. We report the six highest SR thresholds that are multiples "
    r"of~\qty{5}{\percent} and reached by at least one method. Thresholds not reached within the "
    r'maximum demonstration budget are marked as "N/R".'
)


def efficiency_tabular(key: str) -> list[str]:
    """One benchmark's block: demonstrations to each threshold, then AULC and Last-3 SR.

    Threshold cells are DEMONSTRATIONS (rounds x the benchmark's episodes per round, 5 in every
    simulation benchmark), mean +- sample std over the seeds that reach the threshold; the
    scaling is applied per seed before averaging, so it is exact, not a rescaled rounded number.
    Bold is the best of the row (fewest demonstrations; highest AULC / Last-3 SR), compared at
    printed precision so equal-looking cells are marked alike. Only the methods this benchmark
    ran are columns, in the fixed LATEX_ORDER.
    """
    cfg = BENCHMARKS[key]
    per_round = cfg.get("episodes_per_round", EPISODES_PER_ROUND)
    data = prepare_data(cfg)
    labels = [l for l in LATEX_ORDER if l in data]
    thresholds = efficiency_thresholds(data)
    cuts, depths = _table_cuts(data)

    def mark(cells, better):
        """Bold the best finite mean in a row, ties at 1 decimal sharing it."""
        vals = {l: c[0] for l, c in cells.items() if c is not None}
        if not vals:
            return set()
        best = round(better(vals.values()), 1)
        return {l for l, v in vals.items() if round(v, 1) == best}

    def fmt(c, bold):
        if c is None:
            return "N/R"
        t = f"{c[0]:.1f}"
        if bold:
            t = r"\mathbf{" + t + "}"
        return f"${t}^{{\\pm {c[1]:.1f}}}$"

    rows = []
    for thr in thresholds:
        cells = {}
        for l in labels:
            c = rounds_to_threshold(*data[l], thr)
            cells[l] = None if c is None else (c[0] * per_round, c[1] * per_round)
        b = mark(cells, min)
        rows.append((rf"$\geq \qty{{{thr * 100:.0f}}}{{\percent}}$",
                     [fmt(cells[l], l in b) for l in labels]))
    summary = []
    for name, fn in (("AULC", _per_seed_aulc), ("Last-3 SR", _per_seed_last3)):
        cells = {}
        for l in labels:
            rounds, arr = data[l]
            idx = [i for i, r in enumerate(rounds) if r <= cuts[l]] if cuts[l] else []
            v = fn([rounds[i] for i in idx], arr[:, idx]) if idx else np.array([np.nan])
            cells[l] = (float(np.nanmean(v)), sample_std(v)) if np.any(np.isfinite(v)) else None
        b = mark(cells, max)
        summary.append((name, [fmt(cells[l], l in b) for l in labels]))

    n = len(labels) + 1
    out = [f"% {cfg['benchmark']} ({cfg['policy']}) -- demonstrations ({per_round} per round) "
           f"to reach a success rate, over {max(depths.values())} rounds."]
    if len(set(d for d in depths.values() if d > 0)) > 1:
        detail = ", ".join(f"{DISPLAY.get(l, l)} {depths[l]}" for l in labels)
        out.append(f"% partially synced -- rounds per method: {detail}; AULC and Last-3 SR use "
                   f"each method's own rounds.")
    out += [r"\begin{tabular}{l" + " c" * len(labels) + "}",
            r"    \toprule",
            rf"    \multicolumn{{{n}}}{{c}}{{{cfg['benchmark_tex']}: {cfg['policy']}}} \\",
            r"    \midrule",
            "    SR Threshold & " + " & ".join(LATEX_HEADERS[l] for l in labels) + r" \\",
            r"    \midrule"]
    out += [f"    {head} & " + " & ".join(cells) + r" \\" for head, cells in rows]
    out.append(r"    \midrule")
    out += [f"    {head} & " + " & ".join(cells) + r" \\" for head, cells in summary]
    out += [r"    \bottomrule", r"\end{tabular}"]
    return out


def print_efficiency_table(keys) -> None:
    """tab:active_learning_comparison: the whole table environment, one block per benchmark,
    in the layout of tables/active_learning_comparison.tex."""
    print(r"\begin{table}[tb!]")
    print(r"    \setlength{\tabcolsep}{5.5pt}")
    print(r"    \small")
    print(r"    \centering")
    print(r"    \caption{" + EFFICIENCY_CAPTION + "}")
    blocks = [efficiency_tabular(k) for k in keys]
    for i, block in enumerate(blocks):
        body = block
        if i < len(blocks) - 1:
            body = body[:-1] + [r"\end{tabular} \\", r"\vspace{2mm}"]
        print("\n".join(body))
    print(r"    \label{tab:active_learning_comparison}")
    print(r"\end{table}")


def main():
    import argparse

    global BASE, SEEDS, EXPERIMENTS, MAX_ROUND, MATCHED_SUCCESS_ROUNDS, LATEX_ROWS
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", choices=sorted(BENCHMARKS), default="libero")
    ap.add_argument("--output", type=Path, default=None, help="Override the output .pdf path.")
    ap.add_argument("--efficiency-table", action="store_true",
                    help="Print tab:active_learning_comparison (demonstrations to thresholds, "
                         "AULC, Last-3 SR; one block per benchmark in --benchmarks), and exit.")
    ap.add_argument("--ablation-table", action="store_true",
                    help="Print the per-tuning-parameter LIBERO table and exit.")
    ap.add_argument("--tables", action="store_true",
                    help="Print the cross-benchmark AULC / Last-3 table and exit.")
    ap.add_argument("--panels", action="store_true",
                    help="One figure with one learning-curve panel per benchmark, and exit.")
    ap.add_argument("--panel-cols", type=int, default=2,
                    help="Columns in the --panels grid (default 2, i.e. 2x2 for four "
                         "benchmarks).")
    ap.add_argument("--compact", action="store_true",
                    help="Wrapfigure-sized version of --benchmark's learning curves, and exit.")
    ap.add_argument("--bars", action="store_true",
                    help="One required-demonstrations bar figure per benchmark, and exit.")
    ap.add_argument("--bar-ylim-top", type=float, default=None,
                    help="Pin the bar figures' y-axis top instead of sizing it from the "
                         "tallest bar.")
    ap.add_argument("--bar-legend-ncol", type=int, default=2,
                    help="Columns in the bar figures' legend (default 2).")
    ap.add_argument("--bar-targets", default="",
                    help='Subset of the bar groups, e.g. "+20,+30" or ">=50,>=60"; empty = all. '
                         "Written to <key>_threshold_bars_<subset>.pdf so it sits beside the "
                         "full figure instead of overwriting it.")
    ap.add_argument("--benchmarks", nargs="+", choices=sorted(BENCHMARKS), default=None,
                    help=f"Benchmarks for --panels / --tables (default: {' '.join(TABLE_KEYS)}).")
    ap.add_argument("--sr-metric", choices=sorted(SR_METRICS), default="last3",
                    help="Second column of --tables: mean over the final three rounds (last3) "
                         "or the last round alone (final).")
    ap.add_argument("--horizon", type=int, default=None,
                    help="Truncate every method in --tables to this round, instead of reading "
                         "each over all the rounds it has.")
    args = ap.parse_args()
    keys = args.benchmarks or TABLE_KEYS
    TABLE_METRICS[1] = SR_METRICS[args.sr_metric]
    if args.efficiency_table:
        print_efficiency_table(keys)
        return
    if args.ablation_table:
        print_ablation_table()
        return
    if args.tables:
        print_cross_benchmark_table(keys, args.horizon)
        return
    if args.panels:
        plot_panels(keys, args.output or PANELS_OUT, ncols=args.panel_cols)
        return
    if args.compact:
        plot_compact(args.benchmark, args.output or COMPACT_OUT)
        return
    if args.bars:
        targets = parse_bar_targets(args.bar_targets) if args.bar_targets else None
        # A subset gets its own filename ("+20,+30" -> ..._2030.pdf) so the full six-group
        # figure is never silently replaced by a narrower one.
        suffix = ("_" + "".join(f"{int(v * 100)}" for _, v in targets)) if targets else ""
        for k in keys:
            print(f"% {BENCHMARKS[k]['benchmark']} ({BENCHMARKS[k]['policy']})")
            plot_threshold_bars(k, BARS_OUT.format(key=k, suffix=suffix), targets,
                                ylim_top=args.bar_ylim_top, legend_ncol=args.bar_legend_ncol)
        return
    cfg = BENCHMARKS[args.benchmark]

    BASE = cfg["base"]
    SEEDS = cfg["seeds"]
    EXPERIMENTS = cfg["experiments"]
    MAX_ROUND = cfg.get("max_round")
    final_round = cfg["final_round"]
    labels_present = {lbl for _, lbl in EXPERIMENTS}
    global LATEX_COLUMNS
    LATEX_COLUMNS = [(l, LATEX_HEADERS[l]) for l in LATEX_ORDER if l in labels_present]
    # The LaTeX helpers are written around the 15-round LIBERO run; retarget them so the
    # tables refer to this benchmark's horizon instead of silently reporting blanks.
    MATCHED_SUCCESS_ROUNDS = [final_round - 2, final_round - 1, final_round]
    LATEX_ROWS = [(r, r, r * 5) for r in (final_round // 3, 2 * final_round // 3, final_round)]

    data = prepare_data(cfg)
    threshold_rows = {
        label: {thr: first_round_above(np.array(rounds), arr, thr) for thr in THRESHOLDS}
        for label, (rounds, arr) in data.items()
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    draw_curves(ax, cfg, data)

    # Threshold reference lines
    # for thr in THRESHOLDS:
    #     ax.axhline(thr, color="black", linewidth=0.7, linestyle="--", alpha=0.4)

    # Threshold-crossing annotation
    col_w = 9
    header = f"{'':12}" + "".join(f">={thr:.0%}".rjust(col_w) for thr in THRESHOLDS)
    rows = [header, "─" * (12 + col_w * len(THRESHOLDS))]
    labels_ordered = [label for _, label in EXPERIMENTS]
    for label in labels_ordered:
        row = f"{label:<12}" + "".join(threshold_rows[label][thr].rjust(col_w) for thr in THRESHOLDS)
        rows.append(row)

    # Relative improvement rows
    vfd_label = "VFD (ours)"
    vfd_vals = {thr: (float(threshold_rows[vfd_label][thr].split("±")[0])
                      if threshold_rows[vfd_label][thr] != "—" else float("nan"))
                for thr in THRESHOLDS}
    rows.append("─" * (12 + col_w * len(THRESHOLDS)))
    for metric_label, fn in [
        ("1-Y/X", lambda xs, ys: 1 - np.mean(ys) / np.mean(xs)),
        ("X/Y-1",  lambda xs, ys: np.mean([x / y for x, y in zip(xs, ys)]) - 1),
    ]:
        rel_row = f"{metric_label:<12}"
        for label in labels_ordered:
            if label == vfd_label:
                rel_row += "—".rjust(col_w)
                continue
            pairs = []
            for thr in THRESHOLDS:
                val_a = threshold_rows[label][thr]
                x = float(val_a.split("±")[0]) if val_a != "—" else float("nan")
                y = vfd_vals[thr]
                if np.isfinite(x) and np.isfinite(y) and x > 0 and y > 0:
                    pairs.append((x, y))
            if pairs:
                xs, ys = zip(*pairs)
                rel_row += f"{fn(xs, ys):+.2f}".rjust(col_w)
            else:
                rel_row += "—".rjust(col_w)
        rows.append(rel_row)

    # Final success rate row
    rows.append("─" * (12 + col_w * len(THRESHOLDS)))
    final_row = f"{'Final SR':<12}"
    for label in labels_ordered:
        rounds_l, arr_l = data[label]
        x_l = np.array(rounds_l)
        pos = np.where(x_l == final_round)[0]
        if len(pos) == 0:
            final_row += "—".rjust(col_w)
        else:
            sv = arr_l[:, pos[0]]
            final_row += f"{np.nanmean(sv)*100:.1f}±{sample_std(sv)*100:.1f}".rjust(col_w)
    rows.append(final_row)

    annotation = "\n".join(rows)

    # ax.text(
    #     0.02, 0.03, annotation,
    #     transform=ax.transAxes,
    #     fontsize=7.5,
    #     verticalalignment="bottom",
    #     fontfamily="monospace",
    #     bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="lightgray", alpha=0.9),
    # )

    plt.tight_layout()
    save_figure(fig, args.output or cfg["out"])

    print(annotation)
    print()
    print_latex_table(data)
    print()
    print_latex_threshold_table(threshold_rows, LATEX_COLUMNS, data, final_round_idx=final_round)
    print()
    print_matched_success_savings(data, LATEX_COLUMNS)


if __name__ == "__main__":
    main()
