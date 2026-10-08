#!/usr/bin/env python
"""Failure-detection tables of the paper (simulation environments) from the stage-3 outputs.

    python tables/make_tables.py --results results            # writes tables/out/*.tex
    python tables/make_tables.py --results results --print    # also prints them

Expects <results>/{pusht,libero_plus}/<ensemble>/ as written by run_stage3.sh, for the ensembles s01, s23, s45.

Setting (as in the paper): constant threshold = the 0.90 quantile of the calibration rollouts' maximum scores
(`ct_quantile`), window 1. Each ensemble contributes the mean over tasks of the per-task metrics; tables show the
mean and the sample standard deviation over the three ensembles. LIBERO-Plus reports the perturbed (OOD) test
rollouts. A task whose test set has fewer than two rollouts of either outcome is left out of that ensemble's mean.

Marks: bold = best, underline = second best (the main-text table: bold only). TPR and TNR are never marked;
detection time (DT, lower is better) is ranked only among detectors with TPR >= 0.4 and TNR >= 0.4.

Outputs:
  failure_prediction_summary.tex     main text, Push-T and LIBERO-Plus columns (Acc., TWA)
  failure_prediction_pusht.tex       appendix, Push-T
  failure_prediction_liberoplus.tex  appendix, LIBERO-Plus by perturbation family, plus the pooled average
"""
from __future__ import annotations

import argparse
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

FD = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(FD))

SEEDS = ["s01", "s23", "s45"]
THRESHOLD, WINDOW, QUANTILE = "ct_quantile", 1, 0.90
MIN_PER_CLASS = 2
DT_RATE_FLOOR = 0.4

#: table column -> pipeline method name
METHODS = {"tc": "tc", "logpzo": "logpzo", "rnd_oe": "rnd_oe", "entropy": "entropy",
           "VFD": "bayesian_ensemble_inter_vel_diff"}
LABELS = {"tc": "STAC", "logpzo": "logpZO", "rnd_oe": "RND-OE", "entropy": "ACE", "VFD": r"\textbf{VFD (ours)}"}
SUMMARY_LABELS = {**LABELS, "VFD": r"\textbf{VFD}"}
#: (row label, results-CSV column, higher is better)
ROWS = [("Acc.", "Accuracy", True), ("AUROC", "AUROC", True), ("TWA", "TWA", True),
        ("DT", "Det. Time", False), ("TPR", "TPR", True), ("TNR", "TNR", True)]
HIGHER = {r: h for r, _, h in ROWS}
UNRANKED = ("TPR", "TNR")
ENVS = {"pusht": ("all", r"\pusht"), "libero_plus": ("ood", r"\liberoplus")}

FAMILIES = ("camera", "robot_init", "noise", "background", "light", "layout")
FAMILY_LABEL = {"camera": "Camera", "robot_init": "Robot init.", "noise": "Sensor noise",
                "background": "Background", "light": "Lighting", "layout": "Layout"}
N_PER_FAMILY = 30          # LIBERO-Plus test episode = family index * 30 + j; vanilla ID test episodes from 1000


# ---------------------------------------------------------------- environment blocks (from the results CSVs)

def _class_counts(seed_dir: Path, subtype: str) -> dict:
    """{task: (successes, failures)} among the test rollouts of `subtype` ("all", "id", "ood")."""
    with open(sorted(seed_dir.glob("method_results/*.pkl"))[0], "rb") as fh:
        res = pickle.load(fh)[0]
    out = {}
    for task, v in res.items():
        if not isinstance(v, dict) or "successful_test_rollouts" not in v:
            continue
        ok = np.asarray(v["successful_test_rollouts"], dtype=bool)
        mask = np.ones_like(ok) if subtype == "all" else np.asarray(v[f"{subtype}_test_rollouts"], dtype=bool)
        out[task] = (int((ok & mask).sum()), int((~ok & mask).sum()))
    return out


def seed_values(seed_dir: Path, method: str, subtype: str) -> dict:
    """{row label: mean over tasks} for one ensemble and detector."""
    prefix = {"all": "", "id": "ID ", "ood": "OOD "}[subtype]
    d = pd.read_csv(seed_dir / "complete_results.csv")
    d = d[(d.Method == method) & (d.Window == WINDOW) & (d.Threshold == THRESHOLD) & (d.Quantile.round(4) == QUANTILE)]
    if d.empty:
        raise SystemExit(f"no {method} rows in {seed_dir / 'complete_results.csv'}")
    drop = {t for t, (ns, nf) in _class_counts(seed_dir, subtype).items() if min(ns, nf) < MIN_PER_CLASS}
    d = d[~d.Task.isin(drop)]
    return {r: float(d[prefix + col].mean()) for r, col, _ in ROWS}


def env_block(results: Path, env: str):
    """({column: {row: mean}}, {column: {row: std}}) over the ensembles."""
    subtype = ENVS[env][0]
    per = {c: [seed_values(results / env / s, m, subtype) for s in SEEDS] for c, m in METHODS.items()}
    mean = {c: {r: float(np.mean([v[r] for v in per[c]])) for r, _, _ in ROWS} for c in METHODS}
    std = {c: {r: float(np.std([v[r] for v in per[c]], ddof=1)) for r, _, _ in ROWS} for c in METHODS}
    return mean, std


# ---------------------------------------------------------------- LIBERO-Plus per perturbation family

def _family(filename: str) -> str:
    ep = int(filename.split("_")[2])
    return "vanilla_id" if ep >= 1000 else FAMILIES[ep // N_PER_FAMILY]


def _cell_metrics(entry, idx):
    """The table metrics over the test rollouts `idx` of one task; NaN where an outcome class is missing."""
    from evaluation.utils import calculate_metrics
    success = np.asarray(entry["successful_test_rollouts"], dtype=bool)[idx]
    n_fail, n_succ = int((~success).sum()), int(success.sum())
    sbt = entry["test_scores_by_threshold"][THRESHOLD]
    sbt = sbt[next(k for k in sbt if round(float(k), 4) == QUANTILE)][WINDOW]
    stats = {"max_episode_length": entry["max_episode_length"], "successful_rollouts": success,
             "id_rollouts": np.asarray(entry["id_test_rollouts"], dtype=bool)[idx],
             "ood_rollouts": np.asarray(entry["ood_test_rollouts"], dtype=bool)[idx]}
    m = calculate_metrics([np.asarray(sbt[i]) for i in idx], stats)
    raw = entry["test_uncertainty_scores"][WINDOW]
    peaks = [np.nanmax(np.asarray(raw[i]["uncertainty_scores"], dtype=float)) for i in idx]
    both, nan = n_fail > 0 and n_succ > 0, float("nan")
    return {"Acc.": m["balanced_accuracy"] if both else nan,
            "AUROC": roc_auc_score(~success, peaks) if both else nan,
            "TWA": m["timestep_wise_accuracy"] if both else nan,
            "DT": m["avg_detection_time"] if n_fail > 0 else nan,
            "TPR": m["TPR"] if n_fail > 0 else nan,
            "TNR": m["TNR"] if n_succ > 0 else nan}, not both


def family_blocks(results: Path):
    """{family: (mean, std)}: per (task, family) cell of 30 rollouts, averaged over tasks, then over ensembles.

    Metrics needing both outcomes are left out for a cell where all 30 rollouts share one outcome; the number of
    such (task, ensemble) cells per family is returned too."""
    per_seed = {f: {c: defaultdict(list) for c in METHODS} for f in FAMILIES}
    degenerate = defaultdict(int)
    for seed in SEEDS:
        sd = results / "libero_plus" / seed
        files = defaultdict(list)
        for line in open(sd / "test_rollouts.txt"):
            task, name = line.split()
            files[task].append(name)
        for col, meth in METHODS.items():
            data = pickle.load(open(sd / "method_results" / f"{meth}_results.pkl", "rb"))[0]
            vals = {f: defaultdict(list) for f in FAMILIES}
            for t in range(10):
                entry = data[f"libero_10_task_{t:02d}"]
                fams = [_family(n) for n in files[f"task{t:02d}"]]
                if len(fams) != len(entry["successful_test_rollouts"]):
                    raise SystemExit(f"{seed} task{t:02d}: {len(fams)} listed test rollouts vs "
                                     f"{len(entry['successful_test_rollouts'])} in the results")
                for f in FAMILIES:
                    v, degen = _cell_metrics(entry, [i for i, x in enumerate(fams) if x == f])
                    degenerate[f] += degen and col == "VFD"
                    for r, _, _ in ROWS:
                        vals[f][r].append(v[r])
            for f in FAMILIES:
                for r, _, _ in ROWS:
                    per_seed[f][col][r].append(float(np.nanmean(vals[f][r])))
    out = {f: ({c: {r: float(np.mean(per_seed[f][c][r])) for r, _, _ in ROWS} for c in METHODS},
               {c: {r: float(np.std(per_seed[f][c][r], ddof=1)) for r, _, _ in ROWS} for c in METHODS})
           for f in FAMILIES}
    return out, degenerate


# ---------------------------------------------------------------- formatting

def cells(mean, std, keep, dec=2, bold_only=False):
    """{row: {column: LaTeX cell}} for the rows in `keep` (the DT rule still reads TPR/TNR when they are hidden)."""
    out = {}
    for r in keep:
        vals = {c: mean[c][r] for c in METHODS}
        if r in UNRANKED:
            eligible = {}
        elif r == "DT":
            eligible = {c: v for c, v in vals.items()
                        if mean[c]["TPR"] >= DT_RATE_FLOOR and mean[c]["TNR"] >= DT_RATE_FLOOR}
        else:
            eligible = dict(vals)
        bests, seconds = [], []
        if eligible:
            order = sorted(eligible, key=lambda c: eligible[c], reverse=HIGHER[r])
            bests = [c for c in eligible if round(eligible[c], dec) == round(eligible[order[0]], dec)]
            rest = [c for c in order if c not in bests]
            if rest:
                seconds = [c for c in eligible if round(eligible[c], dec) == round(eligible[rest[0]], dec)
                           and c not in bests]
        out[r] = {}
        for c in METHODS:
            t = f"{vals[c]:.{dec}f}"
            if c in bests:
                t = r"\mathbf{" + t + "}"
            elif c in seconds and not bold_only:
                t = r"\underline{" + t + "}"
            out[r][c] = f"${t}^{{\\pm {std[c][r]:.{dec}f}}}$"
    return out


def _arrow(r):
    return r"\uparrow" if HIGHER[r] else r"\downarrow"


def row_block(mean, std, keep):
    return [f"            & {r} ${_arrow(r)}$ & " + " & ".join(c[m] for m in METHODS) + r" \\"
            for r, c in cells(mean, std, keep).items()]


def summary_table(blocks, keep=("Acc.", "TWA")):
    """Main-text layout: environments as column groups, detectors as rows, bold = best only."""
    per_env = [(label, cells(m, s, keep, bold_only=True)) for label, m, s in blocks]
    k, n = len(keep), len(per_env)
    lines = [r"\begin{table}[tb!]", r"    \centering", r"    \setlength{\tabcolsep}{3.2pt}", r"    \small",
             r"    \caption{\textbf{Failure detection}. Balanced accuracy (Acc.) and timestep-wise accuracy (TWA).}",
             r"    \label{tab:failure_prediction_summary}",
             r"    \begin{tabular}{l " + " ".join("c" * k for _ in per_env) + "}", r"        \toprule"]
    lines += [f"        & \\multicolumn{{{k}}}{{c}}{{{label}}}" + (r" \\" if i == n - 1 else "")
              for i, (label, _) in enumerate(per_env)]
    lines += [f"        \\cmidrule(lr){{{2 + i * k}-{1 + (i + 1) * k}}}" for i in range(n)]
    lines.append(r"        Method")
    lines += ["        & " + " & ".join(f"{r} ${_arrow(r)}$" for r in keep) + (r" \\" if i == n - 1 else "")
              for i in range(n)]
    lines.append(r"        \midrule")
    for m in METHODS:
        lines.append(f"        {SUMMARY_LABELS[m]}")
        lines += ["        & " + " & ".join(c[r][m] for r in keep) + (r" \\" if i == n - 1 else "")
                  for i, (_, c) in enumerate(per_env)]
        lines.append("")
    lines += [r"        \bottomrule", r"    \end{tabular}", r"\end{table}"]
    return "\n".join(lines)


KEEP = ["Acc.", "TWA", "DT", "TPR", "TNR"]


def pusht_table(mean, std):
    heads = " & ".join(LABELS.values())
    lines = [r"\begin{table}[!htb]", r"    \centering", r"    \small",
             r"    \caption{\textbf{Failure detection results in \pusht.} Mean and standard deviation over three "
             r"ensembles.}",
             r"    \begin{tabular}{l l c c c c c}", r"        \toprule", f"        Metric & {heads} \\\\",
             r"        \midrule"]
    lines += [l.replace("            & ", "        ", 1) for l in row_block(mean, std, KEEP)]
    lines += [r"        \bottomrule", r"    \end{tabular}", r"    \label{tab:failure_prediction_pusht}", r"\end{table}"]
    return "\n".join(lines)


def libero_table(fam, degenerate, pooled):
    n_cells = 10 * len(SEEDS)
    degen = ", ".join(f"{FAMILY_LABEL[f].lower()}: {n} of \\num{{{n_cells}}}" for f, n in degenerate.items() if n)
    caption = (r"\textbf{Failure detection on \liberoplus.} We group the results by perturbation family and provide "
               r"the mean and standard deviation over three ensembles. Each family block computes the metrics per "
               r"task over the \num{30} rollouts of that family and averages them over tasks; the \emph{Average} "
               r"block computes them per task over all \num{180} perturbed rollouts. Tasks in which all \num{30} "
               f"rollouts of a family share the same outcome ({degen or 'none'}) are excluded from the metrics that "
               r"require both outcomes.")
    blocks = [(FAMILY_LABEL[f], fam[f]) for f in FAMILIES] + [("Average", pooled)]
    heads = " & ".join(LABELS.values())
    out = [r"\begin{table}[!htb]", r"    \centering", r"    \small", f"    \\caption{{{caption}}}",
           r"    \begin{tabular}{l l c c c c c}", r"        \toprule",
           f"        Perturbation & Metric & {heads} \\\\", r"        \midrule"]
    for i, (name, (mean, std)) in enumerate(blocks):
        out.append(f"        \\multirow{{{len(KEEP)}}}{{*}}{{{name}}}")
        out += row_block(mean, std, KEEP)
        out.append(r"        \midrule" if i < len(blocks) - 1 else r"        \bottomrule")
    out += [r"    \end{tabular}", r"    \label{tab:failure_prediction_liberoplus}", r"\end{table}"]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=str(FD / "results"), help="directory with pusht/ and libero_plus/")
    ap.add_argument("--out", default=str(FD / "tables" / "out"))
    ap.add_argument("--print", action="store_true")
    a = ap.parse_args()
    results, out = Path(a.results), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    pusht = env_block(results, "pusht")
    libero = env_block(results, "libero_plus")
    fam, degenerate = family_blocks(results)
    tables = {
        "failure_prediction_summary.tex": summary_table([(ENVS["pusht"][1], *pusht), (ENVS["libero_plus"][1], *libero)]),
        "failure_prediction_pusht.tex": pusht_table(*pusht),
        "failure_prediction_liberoplus.tex": libero_table(fam, degenerate, libero),
    }
    for name, tex in tables.items():
        (out / name).write_text(tex + "\n")
        if a.print:
            print(f"% ---- {name}\n{tex}\n")
    print(f"-> {out}: " + ", ".join(tables))


if __name__ == "__main__":
    main()
