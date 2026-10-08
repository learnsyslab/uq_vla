"""Shared figure typography: the paper's own LaTeX type, at each figure's size on the page.

Imported by scripts/plots/*.py and scripts/calibration/*.py alike, so every paper figure is
typeset the same way. See paper_plots/README.md for what rel_width/font_scale mean and why
callers must not call tight_layout().
"""
import matplotlib.pyplot as plt


# On-page geometry of each figure: `rel` is the fraction of \textwidth its float occupies in
# overleaf/uq_vlas (read off the wrapfigure / includegraphics widths), and `ar` is the width /
# height of the version it replaces, kept so the migration to LaTeX type does not also restyle
# the layout. `font` corrects the type size for the rescale the old figure was getting: a figure
# drawn 2.42 in wide and placed in a 1.93 in float was shrunk by LaTeX to 0.8, so matching its
# on-page appearance at the true size means 0.8 of the font scale it used to carry.
PAGE = {
    # al_success_rate_comparison: \begin{figure}, width=\linewidth
    # Ticks and legend at the 9 pt axis-label size.
    # ar 1.19 (was 1.327): the two-row legend now sits inside the figure height, so the figure
    # grows by the legend and the panels keep their old size.
    "panels": dict(rel=1.0, ar=1.19, font=1.0, ticks_as_labels=True, legend_as_ticks=True),
    # libero_threshold_bars_2030: \begin{wrapfigure}[17]{r}{0.45\textwidth}
    # legend_font: the old figure carried a 7.5 pt legend in a 3.29 in canvas, i.e. ~5.6 pt at
    # the true 2.48 in; 6.16 is that +10%. Six rows still clear the bars.
    "bars_subset": dict(rel=0.45, ar=1.143, font=1.0, legend_font=6.16),
    # the full six-group bars are not in the paper; keep them full width
    "bars": dict(rel=1.0, ar=2.424, font=1.0),
    # ensemble_size_ablation: \begin{wrapfigure}[15]{r}{0.4\textwidth}
    "ensemble_ablation": dict(rel=0.40, ar=1.350, font=1.0),
    # Side-by-side Spearman + Pearson: 2.44 in tall (1.5x the single panel)
    # and 5.28 in wide -- twice the 2.64 in it was, so essentially the full text width. NOTE
    # figures/ensemble_size_ablation.tex still wraps it in a 0.4\textwidth wrapfigure; at this
    # width it has to become a normal full-width \begin{figure}, or LaTeX scales it back down.
    "ensemble_ablation_2panel": dict(rel=0.96, ar=5.28 / 2.44, font=1.0,
                                legend_font=9.0, ticks_as_labels=True),
    # ensemble_vs_laplace: Spearman and Pearson united into one full-width figure
    #, same geometry as ensemble_ablation_2panel so the two read alike.
    "ensemble_vs_laplace_2panel": dict(rel=0.96, ar=5.28 / 2.44, font=1.0,
                                  legend_font=9.0, ticks_as_labels=True),
    # success_share_entropy: \begin{figure}, width=\linewidth. The figure it replaces had
    # ar 3.71, but that was a 13.6 in canvas LaTeX shrank to 0.4; at the true 5.5 in it leaves
    # 1.4 in panels, too short for any in-panel legend (they covered the curves). 2.55 is the
    # flattest that fits all three legends -- a deliberate departure from the old proportions.
    "success_share": dict(rel=1.0, ar=2.55, font=1.0, legend_font=6.0),
    # exploration_vs_success_small: exploration vs success (once success_share_entropy's right panel), for a
    # wrapfigure; 0.45\textwidth like the other wrapped figures in the paper, no title.
    # tick_font 8: a step up from the bundle's 7 pt; the legend follows it.
    # It shares a figure row with a 0.45\textwidth panel of the same height (5.5 in / 1.85 in aspect
    # at full width), so both are built at their final size and LaTeX scales neither.
    "exploration_standalone": dict(rel=0.50, ar=0.50 * 5.5 / 1.85, font=1.0, tick_font=8.0,
                                   legend_as_ticks=True),
    # exploration_exploitation_entropy: \begin{figure}, width=0.95\linewidth; ar of the old one
    "pareto": dict(rel=0.95, ar=1.978, font=1.0, legend_font=7.0),
    # selection_uncertainty_heatmap: \begin{figure}, width=\linewidth. The old one was 0.863
    # for nine rows; tau = 3 is dropped, so eight rows at the same row height -> 0.97.
    "selection_heatmap": dict(rel=1.0, ar=0.776, font=1.0),  # 10 rows; was 0.97 at 8 rows
    # language_variation: \begin{wrapfigure}{r}{0.5\textwidth}
    "language_variation": dict(rel=0.50, ar=1.584, font=1.0, legend_font=5.5),
}
TEXTWIDTH_IN = 5.5   # ICLR 2024/2027 \textwidth


def page_style(key: str, **over) -> None:
    """use_paper_style() for one of the PAGE entries, preserving that figure's aspect ratio."""
    g = dict(PAGE[key], **over)
    width = TEXTWIDTH_IN * g["rel"]
    use_paper_style(rel_width=g["rel"], height=width / g["ar"],
                    font_scale=g["font"], line_scale=g.get("line", 1.0))
    if "legend_font" in g:
        plt.rcParams["legend.fontsize"] = g["legend_font"]
    # The bundle sets ticks two points below the axis labels; `ticks_as_labels` equalises them
    # for figures whose tick values are as much of the message as the axis name.
    if g.get("ticks_as_labels"):
        plt.rcParams["xtick.labelsize"] = plt.rcParams["ytick.labelsize"] = \
            plt.rcParams["axes.labelsize"]
    # `tick_font` pins the tick size to a number, for a figure that wants ticks between the
    # bundle's 7 pt and the 9 pt axis labels. Applied before legend_as_ticks so that follows.
    if "tick_font" in g:
        plt.rcParams["xtick.labelsize"] = plt.rcParams["ytick.labelsize"] = g["tick_font"]
    # `legend_as_ticks` sets the legend at the tick size, whatever that is after the two
    # options above -- so it tracks font_scale instead of being pinned to a number.
    if g.get("legend_as_ticks"):
        plt.rcParams["legend.fontsize"] = plt.rcParams["xtick.labelsize"]


def use_paper_style(rel_width: float = 1.0, height: float | None = None,
                    nrows: int = 1, ncols: int = 1,
                    font_scale: float = 1.0, line_scale: float = 1.0,
                    bold: bool = False) -> None:
    """Typeset a figure the way the paper is typeset: real LaTeX, Times, ICLR type sizes.

    `tueplots.bundles.iclr2024()` sets the body font sizes an ICLR submission uses (9 pt
    labels, 7 pt ticks and legend) and turns on constrained layout -- so callers must NOT also
    call tight_layout(). `text.usetex` renders through the LaTeX installation, so the figure's
    glyphs are the document's glyphs rather than matplotlib's lookalikes.

    The point of `rel_width` is that the figure is created at its FINAL size on the page: pass
    the same fraction the \includegraphics uses (0.45 for a wrapfigure at 0.45\textwidth) and
    LaTeX scales it by 1, so 7 pt in the figure is 7 pt in the paper. Scaling a wide figure
    down in LaTeX instead is what makes axis labels come out smaller than the caption.
    `height` overrides the bundle's golden-ratio height, which is too flat at small widths.
    """
    from tueplots import bundles, figsizes

    plt.rcParams.update(bundles.iclr2024())
    plt.rcParams.update(figsizes.iclr2024(rel_width=rel_width, nrows=nrows, ncols=ncols))
    plt.rcParams.update({
        "text.latex.preamble": r"\usepackage{times}\usepackage{amsmath}",
        "text.usetex": True,
    })
    if height is not None:
        w, _ = plt.rcParams["figure.figsize"]
        plt.rcParams["figure.figsize"] = (w, height)

    # `font_scale` lifts every type size off the bundle's together, for a figure that sits small
    # on the page and would otherwise carry body-text-sized labels in a third of the width.
    if font_scale != 1.0:
        for k in ("font.size", "axes.labelsize", "axes.titlesize",
                  "legend.fontsize", "xtick.labelsize", "ytick.labelsize"):
            plt.rcParams[k] = float(plt.rcParams[k]) * font_scale
    # `line_scale` thickens the frame with the type, so spines and ticks do not turn into
    # hairlines next to heavier labels and curves.
    if line_scale != 1.0:
        plt.rcParams.update({
            "axes.linewidth": 0.8 * line_scale,
            "xtick.major.width": 0.8 * line_scale,
            "ytick.major.width": 0.8 * line_scale,
            "xtick.minor.width": 0.6 * line_scale,
            "ytick.minor.width": 0.6 * line_scale,
            "grid.linewidth": 0.8 * line_scale,
            "patch.linewidth": 0.8 * line_scale,
        })
    if bold:
        plt.rcParams.update({"font.weight": "bold", "axes.labelweight": "bold"})


def tex(s: str) -> str:
    """Escape the characters LaTeX treats as special, when usetex is on."""
    if not plt.rcParams.get("text.usetex"):
        return s
    for ch in "#%&_":
        s = s.replace(ch, "\\" + ch)
    return s




def save_exact(fig, out) -> None:
    """Save at exactly the figure size, NOT cropped to the drawn content (bbox_inches='tight' would
    change the width by a few points, and with it the scale LaTeX applies). Constrained layout, on
    via the ICLR bundle, already keeps every label inside the canvas."""
    from pathlib import Path
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=150)
    print(f"Saved: {out}")
