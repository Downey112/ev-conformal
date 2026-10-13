"""Figures for the simulation study. Reads results/sim.csv, writes results/fig_*.png."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..", "results")
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
# fixed categorical order: colour follows the method everywhere
COLORS = {"CP": "#2a78d6", "CP-MNR (code quantile)": "#eb6834", "CP-MNR (n+1 quantile)": "#1baf7a",
          "Nested* (released code)": "#eda100", "Nested* (exact set)": "#e87ba4"}
MARKERS = {"CP": "o", "CP-MNR (code quantile)": "s", "CP-MNR (n+1 quantile)": "^",
           "Nested* (released code)": "D", "Nested* (exact set)": "v"}
LEVELS = ["national", "regional", "station"]

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10, "text.color": INK,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
})


def panels(df, xcol, xlabel, title, fname, ycol="coverage", ylabel="Coverage (active test days)",
           ref=0.9, logx=False):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.9), sharey=True)
    for ax, lv in zip(axes, LEVELS):
        d = df[df.cell == lv]
        for m, c in COLORS.items():
            s = d[d.method == m].sort_values(xcol)
            ax.plot(s[xcol], s[ycol], color=c, marker=MARKERS[m], lw=2, ms=6,
                    mec=SURFACE, mew=1.2, label=m)
        if ref is not None:
            ax.axhline(ref, color=INK2, lw=1, ls=(0, (4, 3)))
            ax.annotate("nominal 90%", (1, ref), xycoords=("axes fraction", "data"),
                        xytext=(-2, 3), textcoords="offset points", ha="right", color=INK2, fontsize=8)
        if logx:
            ax.set_xscale("log")
            ax.set_xticks(sorted(d[xcol].unique()), [str(v) for v in sorted(d[xcol].unique())])
            ax.minorticks_off()
        else:
            ax.set_xticks(sorted(d[xcol].unique()))
        ax.set_title(lv.capitalize(), loc="left", fontsize=10, color=INK2)
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel(ylabel)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    fig.savefig(os.path.join(ROOT, fname), dpi=160)
    plt.close(fig)


def regimes(df, title, fname):
    cells = list(dict.fromkeys(df.cell))
    fig, ax = plt.subplots(figsize=(9, 4.6))
    off = {m: (i - 2) * 0.14 for i, m in enumerate(COLORS)}
    for m, c in COLORS.items():
        s = df[df.method == m].set_index("cell").loc[cells]
        y = [i + off[m] for i in range(len(cells))]
        ax.errorbar(s.coverage, y, xerr=1.96 * s.coverage_se, fmt=MARKERS[m], color=c, ms=7,
                    mec=SURFACE, mew=1.2, elinewidth=1.2, label=m)
    ax.axvline(0.9, color=INK2, lw=1, ls=(0, (4, 3)))
    ax.set_yticks(range(len(cells)), cells)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Coverage conditional on the activity regime (bars: 95% interval)")
    ax.legend(frameon=False, fontsize=9, loc="center left", bbox_to_anchor=(1.0, 0.5))
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, fname), dpi=160)
    plt.close(fig)


def main():
    df = pd.read_csv(os.path.join(ROOT, "sim.csv"))
    lev = df[df.group == "level"]
    rho = lev[lev.sweep == "rho"]
    panels(rho, "rho", "Dependence of activity on demand (rho)",
           "Coverage as activity becomes more dependent on demand", "fig_coverage_vs_rho.png")
    panels(rho, "rho", "Dependence of activity on demand (rho)",
           "Share of nodes reported 'valid' (test coverage >= 90%)", "fig_validity_vs_rho.png",
           ycol="validity_rate", ylabel="Validity rate", ref=None)
    ncal = pd.concat([lev[lev.sweep == "n_cal"], rho[rho.rho == 0.6]])
    panels(ncal, "n_cal", "Calibration days", "Coverage by calibration size (rho = 0.6)",
           "fig_coverage_vs_ncal.png", logx=True)
    panels(ncal, "n_cal", "Calibration days", "Mean interval length by calibration size (rho = 0.6)",
           "fig_length_vs_ncal.png", ycol="mean_length", ylabel="Mean interval length", ref=None, logx=True)
    reg = df[(df.group == "regime") & (df.sweep == "rho")]
    for r in (0.0, 0.6):
        regimes(reg[reg.rho == r], f"Coverage by activity regime (rho = {r}, 200 calibration days)",
                f"fig_regimes_rho{r}.png")


if __name__ == "__main__":
    main()
