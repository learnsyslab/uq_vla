#!/usr/bin/env python
"""Print the calibration table (tab:calibration_full; tab:calibration_summary is its Spearman rows) as a
LaTeX tabular.

Task-level protocol: per round, correlate the mean frame-0 uncertainty of a task's demonstrations with
that task's success rate over the 10 LIBERO-10 tasks; mean over rounds < 15 per seed pair, then mean +-
population std over the 3 seed pairs. -r is reported, so positive means "higher uncertainty goes with
lower success". Cells are NA where a method does not apply (Entropy and Perplexity need a VLM head,
which X-VLA and FastWAM do not have).

DECU saturates at log 2 on X-VLA and FastWAM and its scores are cached at bfloat16 precision, so in
some rounds every task mean collapses onto one value and the correlation is undefined.
jitter_degenerate_decu.py breaks those exact ties with +-1e-5 noise (two orders below the bf16 quantum,
so genuinely distinct values keep their order) before this table is computed; inside a fully tied round
the ranking is noise, so its correlation is an unbiased draw around zero.

    PYTHONPATH=src python scripts/calibration/calibration_table.py
"""
import argparse, sys, warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import calibration_utils as cs  # noqa: E402

IFT = Path("outputs")
COLUMNS = [("action_l2", "Action-L2"), ("ace", "ACE"), ("decu", "DECU"),
           ("ensemble_terminal_variance", "GU"), ("vlm_token_entropy", "Entropy"),
           ("vlm_perplexity", "Perplexity"), ("inter_vel_diff_2way", "VFD (ours)")]
SEEDS = ("s01", "s23", "s45")


def task_level(run_dirs: list[Path], max_round: int) -> dict[str, tuple[float, float, float, float]]:
    """{method: (spearman_mean, std, pearson_mean, std)} using the published task-level protocol."""
    out: dict[str, list[tuple[float, float]]] = {k: [] for k, _ in COLUMNS}
    for run in run_dirs:
        corrs = cs.correlations_for_seed(run, max_round=max_round)
        for key, _ in COLUMNS:
            c = corrs.get(key, {})
            if c.get("spearman"):
                out[key].append((-float(np.mean(c["spearman"])), -float(np.mean(c["pearson"]))))
    res = {}
    for key, vals in out.items():
        if not vals:
            continue
        a = np.array(vals)
        res[key] = (a[:, 0].mean(), a[:, 0].std(), a[:, 1].mean(), a[:, 1].std())
    return res


def emit(label: str, res: dict, note: str = "") -> list[str]:
    def cells(mean_i, std_i):
        vals = {k: res[k][mean_i] for k, _ in COLUMNS if k in res and np.isfinite(res[k][mean_i])}
        best = max(vals, key=vals.get) if vals else None
        out = []
        for key, _ in COLUMNS:
            if key not in res or not np.isfinite(res[key][mean_i]):
                out.append("NA"); continue
            m, s = res[key][mean_i], res[key][std_i]
            out.append(f"$\\mathbf{{{m:.2f}}}^{{\\pm \\num{{{s:.2f}}}}}$" if key == best
                       else f"\\num{{{m:.2f}}}$^{{\\pm \\num{{{s:.2f}}}}}$")
        return out
    lines = [f"        \\multirow{{2}}{{*}}{{{label}}}{('  % ' + note) if note else ''}",
             "            & $-$Spearman & " + " & ".join(cells(0, 1)) + " \\\\",
             "            & $-$Pearson  & " + " & ".join(cells(2, 3)) + " \\\\"]
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-round", type=int, default=15, help="use rounds < max-round (default 15)")
    a = ap.parse_args()

    rows = [
        (policy, task_level([IFT / "calibration" / key / f"random_{s}" for s in SEEDS], a.max_round),
         "task-level, 3 seed pairs")
        for policy, key in (("SmolVLA", "smolvla"), ("X-VLA", "xvla"), ("FastWAM", "fastwam"))
    ]
    print("\\begin{tabular}{l l c c c c c c c}")
    print("        \\toprule")
    print("        Policy & Metric & " + " & ".join(n for _, n in COLUMNS[:-1])
          + " & \\textbf{VFD (ours)} \\\\")
    print("        \\midrule")
    for i, (label, res, note) in enumerate(rows):
        print("\n".join(emit(label, res, note)))
        print("        \\midrule" if i < len(rows) - 1 else "        \\bottomrule")
    print("\\end{tabular}")


if __name__ == "__main__":
    main()
