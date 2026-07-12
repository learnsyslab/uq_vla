source .venv/bin/activate

python - <<'PY'
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

base = Path("plots/calibration_comparison/libero/uniform_leak3_weighted_pools_lr_schedule_history05_steps2000_ensemble_vs_laplace_rounds0_4/aggregated")
data = json.loads((base / "ensemble_vs_laplace_correlations.json").read_text())

for metric, ylabel in [("spearman", r"-Spearman $\rho$"), ("pearson", "- Pearson")]:
    fig, ax = plt.subplots(figsize=(4, 3))
    for label, color, marker in [("Ensemble", "tab:blue", "o"), ("Laplace", "tab:orange", "s")]:
        rounds = sorted(int(r) for r in data[metric][label])
        means = [-float(data[metric][label][str(r)]["mean"]) for r in rounds]
        stds = [float(data[metric][label][str(r)]["std"]) for r in rounds]
        x = [r + 1 for r in rounds]
        ax.plot(x, means, marker=marker, linewidth=2, label=label, color=color)
        ax.fill_between(
            x,
            [m - s for m, s in zip(means, stds)],
            [m + s for m, s in zip(means, stds)],
            color=color,
            alpha=0.18,
        )

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Round $r$")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(loc="lower right")
    fig.tight_layout()

    for name in [f"{metric}_across_rounds", f"{metric}_ensemble_vs_laplace"]:
        fig.savefig(base / f"{name}.png", dpi=150, bbox_inches="tight")
        fig.savefig(base / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
PY
