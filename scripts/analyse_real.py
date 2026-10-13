"""Tables and figures for the real-data runs (results/real_<name>_seed*.npz)."""
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binom

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from evconf.data import build_panel  # noqa: E402
from evconf.realdata import METHODS  # noqa: E402
import plot_sim as style  # noqa: E402  (shared rcParams and chrome colours)

ROOT = os.path.join(os.path.dirname(__file__), "..", "results")
ALPHA, MIN_ACTIVE = 0.1, 20
LEVELS = ["national", "regional", "station"]
COLORS = dict(style.COLORS, **{"Adaptive CP-MNR": "#008300"})
MARKERS = dict(style.MARKERS, **{"Adaptive CP-MNR": "P"})
PAPER_FOUR = ["CP", "CP-MNR (code quantile)", "Nested* (released code)", "Adaptive CP-MNR"]
PERIODS = {"paper period (2024-02-05 to 2024-08-18)": ("2024-02-05", "2024-08-18"),
           "later period (2024-08-19 to 2025-10-26)": ("2024-08-19", "2025-10-26"),
           "tariff period (2023-03-13 to 2023-08-06)": ("2023-03-13", "2023-08-06")}


def load(name):
    return [dict(np.load(f)) for f in sorted(glob.glob(os.path.join(ROOT, f"real_{name}_seed*.npz")))]


def covered(r, method):
    """(cov, defined): defined = node active and the method returned an interval."""
    lo, up = r[f"lo::{method}"], r[f"up::{method}"]
    d = r["A"] & ~np.isnan(lo) & ~np.isnan(up)
    return (r["Y"] >= lo) & (r["Y"] <= up) & d, d


def level_table(runs, level, period):
    lo_d, hi_d = (np.datetime64(d) for d in PERIODS[period])
    rows = []
    for method in METHODS:
        per_seed = {k: [] for k in ["cov", "valid", "relaxed", "bench", "len", "undef", "inf"]}
        for r in runs:
            sel = (r["date"] >= lo_d) & (r["date"] <= hi_d)
            if not sel.any():
                continue
            cov, d = covered(r, method)
            cov, d, A = cov[sel], d[sel], r["A"][sel]
            width = (r[f"up::{method}"] - r[f"lo::{method}"])[sel]
            for lv, label in enumerate(LEVELS):
                c, dd = cov[:, level == lv], d[:, level == lv]
                n_eff = dd.sum(0)
                ok = n_eff >= MIN_ACTIVE
                node_cov = c.sum(0)[ok] / n_eff[ok]
                # share of nodes an exactly calibrated method would call valid
                k_need = np.ceil((1 - ALPHA) * n_eff[ok] - 1e-9)
                bench = binom.sf(k_need - 1, n_eff[ok], 1 - ALPHA).mean() if ok.any() else np.nan
                w = np.where(dd & np.isfinite(width[:, level == lv]), width[:, level == lv], np.nan)
                per_seed["cov"].append((label, c.sum() / max(dd.sum(), 1)))
                per_seed["valid"].append((label, np.mean(node_cov >= 1 - ALPHA - 1e-12) if ok.any() else np.nan))
                per_seed["relaxed"].append((label, np.mean(node_cov >= 1 - 2 * ALPHA - 1e-12) if ok.any() else np.nan))
                per_seed["bench"].append((label, bench))
                per_seed["len"].append((label, np.nanmean(w)))
                a = A[:, level == lv]
                per_seed["undef"].append((label, (a & ~dd).sum() / max(a.sum(), 1)))
                per_seed["inf"].append((label, (dd & np.isinf(width[:, level == lv])).sum() / max(dd.sum(), 1)))
        for label in LEVELS:
            g = lambda k: np.array([v for l, v in per_seed[k] if l == label])
            if g("cov").size == 0:
                continue
            se = lambda x: 1.96 * np.nanstd(x, ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan
            rows.append(dict(period=period, level=label, method=method, seeds=len(g("cov")),
                             coverage=g("cov").mean(), validity=np.nanmean(g("valid")), validity_ci=se(g("valid")),
                             validity_if_exact=np.nanmean(g("bench")), relaxed_validity=np.nanmean(g("relaxed")),
                             mean_length=g("len").mean(), infinite=g("inf").mean(), undefined=g("undef").mean()))
    return rows


def monthly(runs, level, freq="MS"):
    """Pooled coverage per month x level x method, averaged over seeds."""
    out = []
    for s, r in enumerate(runs):
        month = pd.to_datetime(r["date"]).to_period("M" if freq == "MS" else "W").start_time
        for method in METHODS:
            cov, d = covered(r, method)
            for lv, label in enumerate(LEVELS):
                df = pd.DataFrame({"t": month, "c": cov[:, level == lv].sum(1), "n": d[:, level == lv].sum(1)})
                g = df.groupby("t").sum()
                out.append(pd.DataFrame({"t": g.index.to_numpy(), "seed": s, "method": method, "level": label,
                                         "coverage": (g.c / g.n).to_numpy(), "n": g.n.to_numpy()}))
    df = pd.concat(out)
    return df.groupby(["t", "method", "level"], as_index=False).agg(coverage=("coverage", "mean"), n=("n", "mean"))


def plot_time(df, title, fname, methods=PAPER_FOUR, vline=None):
    fig, axes = plt.subplots(3, 1, figsize=(10, 7.6), sharex=True)
    for ax, lv in zip(axes, LEVELS):
        d = df[df.level == lv]
        for m in methods:
            s = d[d.method == m].sort_values("t")
            ax.plot(s.t, s.coverage, color=COLORS[m], marker=MARKERS[m], lw=2, ms=5, mec=style.SURFACE, mew=1, label=m)
        ax.axhline(1 - ALPHA, color=style.INK2, lw=1, ls=(0, (4, 3)))
        if vline:
            ax.axvline(pd.Timestamp(vline[0]), color=style.MUTED, lw=1)
            if lv == LEVELS[0]:
                ax.annotate(vline[1], (pd.Timestamp(vline[0]), 1), xycoords=("data", "axes fraction"),
                            xytext=(4, -10), textcoords="offset points", fontsize=8, color=style.INK2)
        ax.set_title(lv.capitalize(), loc="left", fontsize=10, color=style.INK2)
        ax.set_ylabel("Coverage")
        ax.grid(axis="x", visible=False)
    axes[-1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(axes[-1].xaxis.get_major_locator()))
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(methods), frameon=False, fontsize=9)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(os.path.join(ROOT, fname), dpi=160)
    plt.close(fig)


def plot_node(r, j, name, fname, methods=("CP-MNR (code quantile)", "Adaptive CP-MNR"), vline=None):
    t = pd.to_datetime(r["date"])
    fig, axes = plt.subplots(len(methods), 1, figsize=(10, 2.9 * len(methods)), sharex=True, sharey=True)
    for ax, m in zip(np.atleast_1d(axes), methods):
        lo, up = r[f"lo::{m}"][:, j], r[f"up::{m}"][:, j]
        y = np.where(r["A"][:, j], r["Y"][:, j], np.nan)
        ax.fill_between(t, lo, up, color=COLORS[m], alpha=0.25, lw=0, label="90% interval")
        ax.plot(t, y, color=style.INK, lw=1.3, label="observed")
        miss = (y < lo) | (y > up)
        ax.plot(t[miss], y[miss], "o", color="#d03b3b", ms=4, label="outside interval")
        cov = 1 - np.nanmean(np.where(np.isnan(y), np.nan, miss))
        ax.set_title(f"{m}: coverage {cov:.0%}", loc="left", fontsize=10, color=style.INK2)
        ax.set_ylabel("kWh per day")
        ax.grid(axis="x", visible=False)
        if vline:
            ax.axvline(pd.Timestamp(vline), color=style.MUTED, lw=1)
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=8, ncol=3, loc="upper right")
    fig.suptitle(name, x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(ROOT, fname), dpi=160)
    plt.close(fig)


def main():
    panel = build_panel()
    R, n = len(panel["regions"]), len(panel["stations"])
    level = np.array([0] + [1] * R + [2] * n)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    tables = []
    ext, tar = load("extended"), load("tariff")
    if ext:
        for period in list(PERIODS)[:2]:
            tables += level_table(ext, level, period)
        plot_time(monthly(ext, level), "Monthly coverage on ChargePlace Scotland, February 2024 to October 2025",
                  "fig_real_monthly.png")
        plot_node(ext[0], 0, "Scotland total: day-ahead 90% intervals", "fig_real_scotland.png")
    if tar:
        tables += level_table(tar, level, list(PERIODS)[2])
        g = 1 + list(panel["regions"]).index("Glasgow City")
        plot_node(tar[0], g, "Glasgow City total around the tariff change of 4 April 2023",
                  "fig_tariff_glasgow.png", vline="2023-04-04")
        glasgow = np.where(np.arange(1 + R + n) == g, 1, np.where((level == 2) & (np.r_[[-1] * (1 + R), panel["region_of"]] == g - 1), 2, -1))
        glasgow[0] = 0
        w = monthly(tar, glasgow, freq="W")
        plot_time(w, "Weekly coverage in spring 2023: Scotland, Glasgow City, Glasgow stations",
                  "fig_tariff_weekly.png", vline=("2023-04-04", "Glasgow tariff"))
        lo_d = np.datetime64("2023-04-04")
        rows = []
        for m in METHODS:
            for lab, sel in (("Glasgow City node", glasgow == 1), ("Glasgow stations", glasgow == 2)):
                c = [(covered(r, m)[0][r["date"] >= lo_d][:, sel].sum() /
                      max(covered(r, m)[1][r["date"] >= lo_d][:, sel].sum(), 1)) for r in tar]
                rows.append(dict(method=m, node=lab, coverage_after_tariff=np.mean(c)))
        gl = pd.DataFrame(rows).pivot(index="method", columns="node", values="coverage_after_tariff").loc[METHODS]
        print("\nGlasgow coverage, 2023-04-04 to 2023-08-06\n", gl.round(3).to_string())
        gl.to_csv(os.path.join(ROOT, "real_glasgow.csv"))
    df = pd.DataFrame(tables)
    df.to_csv(os.path.join(ROOT, "real_summary.csv"), index=False)
    for period, d in df.groupby("period", sort=False):
        print("\n" + period)
        print(d.drop(columns="period").round(3).to_string(index=False))


if __name__ == "__main__":
    main()
