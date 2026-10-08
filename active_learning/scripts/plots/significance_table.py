#!/usr/bin/env python3
"""Paper table tab:significance (overleaf tables/statistical_significance.tex): paired
comparison of SAVE w/ VFD against SAVE w/ GU on Push-T.

For each seed group both methods start from the same pretrained ensemble with the same
training seed, so the comparison is paired per seed. Per metric the row reports:

  * the mean and SAMPLE std over seeds of each method (ddof=1, as every paper table);
  * the mean paired difference VFD - GU with a 95 % percentile-bootstrap CI over seeds;
  * how many seeds VFD wins;
  * ONE-SIDED p-values (H1: VFD > GU) of the Wilcoxon signed-rank test and the paired t-test,
    which is the hypothesis the appendix text states.

AULC and Last-3 SR are the per-seed metrics of success_rate_paper.py (_per_seed_aulc,
_per_seed_last3), so the two VFD/GU columns are the same numbers as the Push-T columns of
tab:active_learning_comparison(_summary).

Below the table, `%` comment lines give the two-sided p-values and the Holm-adjusted ones over
the two metrics, which the appendix text quotes ("also hold for two-sided tests at a 2 % level
and after a Holm correction").

The default is all ten seed groups; `--seeds` restricts the comparison to a subset.

    PYTHONPATH=src python scripts/plots/significance_table.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import success_rate_paper as srp  # noqa: E402

BENCHMARK = "pusht"
VFD_EXP = "vfd"
GU_EXP = "gu"
METRICS = [("AULC", srp._per_seed_aulc), ("Last-3 SR", srp._per_seed_last3)]
N_BOOT = 10_000
BOOT_SEED = 0
SEED_WORDS = {9: "nine", 10: "ten"}


def per_seed(exp: str, seeds: list[str]) -> dict[str, np.ndarray]:
    cfg = srp.BENCHMARKS[BENCHMARK]
    srp.BASE, srp.SEEDS, srp.MAX_ROUND = cfg["base"], seeds, cfg.get("max_round")
    rounds, arr = srp.load_success_rates(exp)
    return {name: fn(rounds, arr) for name, fn in METRICS}


def bootstrap_ci(diff: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(BOOT_SEED)
    means = rng.choice(diff, size=(N_BOOT, len(diff)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def holm(pvals: list[float]) -> list[float]:
    order = np.argsort(pvals)
    adj, running = [0.0] * len(pvals), 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(pvals) - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", nargs="+", default=srp.BENCHMARKS[BENCHMARK]["seeds"],
                    help="Seed groups to pair over (default: all ten).")
    args = ap.parse_args()

    vfd, gu = per_seed(VFD_EXP, args.seeds), per_seed(GU_EXP, args.seeds)
    # Pair only seeds both methods finished; an unfinished run must not silently shrink n.
    ok = np.ones(len(args.seeds), bool)
    for name, _ in METRICS:
        ok &= np.isfinite(vfd[name]) & np.isfinite(gu[name])
    if not ok.all():
        missing = [s for s, k in zip(args.seeds, ok) if not k]
        sys.exit(f"missing VFD or GU results for seeds {missing}; sync them or pass --seeds")
    n = len(args.seeds)

    rows, notes, p_one = [], [], {"Wilcoxon": [], "t-test": []}
    for name, _ in METRICS:
        v, g = vfd[name], gu[name]
        d = v - g
        lo, hi = bootstrap_ci(d)
        pw = stats.wilcoxon(v, g, alternative="greater").pvalue
        pt = stats.ttest_rel(v, g, alternative="greater").pvalue
        pw2 = stats.wilcoxon(v, g).pvalue
        pt2 = stats.ttest_rel(v, g).pvalue
        p_one["Wilcoxon"].append(pw)
        p_one["t-test"].append(pt)
        rows.append(
            f"        {name:<9s} & ${v.mean():.1f}^{{\\pm {v.std(ddof=1):.1f}}}$"
            f" & ${g.mean():.1f}^{{\\pm {g.std(ddof=1):.1f}}}$"
            f" & ${d.mean():+.1f}$~$[{lo:+.1f}, {hi:+.1f}]$"
            f" & ${int((d > 0).sum())}/{n}$ & ${pw:.3f}$ & ${pt:.3f}$ \\\\")
        notes.append(f"% {name}: two-sided p_Wilcoxon {pw2:.4f}, p_t {pt2:.4f}; "
                     f"per-seed VFD-GU {np.array2string(d, precision=2, separator=' ')}")
    for test, ps in p_one.items():
        adj = holm(ps)
        notes.append(f"% Holm over the two metrics, one-sided {test}: "
                     + ", ".join(f"{m} {a:.4f}" for (m, _), a in zip(METRICS, adj)))

    words = SEED_WORDS.get(n, str(n))
    print(rf"""\begin{{table}}[tb!]
    \centering
    \small
    \setlength{{\tabcolsep}}{{4pt}}
    \caption{{\textbf{{Statistical significance on \pusht.}} Mean and sample standard deviation over {words} paired training seeds.}}
    \label{{tab:significance}}
    \begin{{tabular}}{{l c c c c c c}}
        \toprule
        Metric & \method w/ VFD & \method w/ GU & Difference & VFD better & $p_\text{{Wilcoxon}}$ & $p_{{t\text{{-test}}}}$ \\
        \midrule""")
    print("\n".join(rows))
    print(r"""        \bottomrule
    \end{tabular}
\end{table}""")
    print(f"% seeds: {' '.join(args.seeds)}; difference CI: 95 % percentile bootstrap "
          f"({N_BOOT} resamples, rng seed {BOOT_SEED}); p-values one-sided (VFD > GU)")
    print("\n".join(notes))


if __name__ == "__main__":
    main()
