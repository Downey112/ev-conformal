"""Rolling-window conformal experiment on the Scottish hierarchy.

Protocol follows the authors' generate_rolling_windows(): 14-day test windows,
models refit for each window on a fixed-length history, calibration on a random
20% of the historical weeks plus the week before the test window.
Base models are deliberately light (gradient boosting, ridge at the top) so the
whole thing runs on a laptop; they replace the paper's RF/GAM/XGBoost/TabICL mix.
"""
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

from .conformal import adaptive_cp_mnr, cp_mnr_nested_star, split_quantile
from .hierarchy import make_fast_projector

METHODS = ["CP", "CP-MNR (code quantile)", "CP-MNR (n+1 quantile)",
           "Nested* (released code)", "Nested* (exact set)", "Nested* (corrected)",
           "Adaptive CP-MNR"]
DAY = np.timedelta64(1, "D")


@dataclass(frozen=True)
class Protocol:
    test_start: str = "2024-02-05"
    end_all: str = "2024-08-28"
    train_start: str = "2022-10-08"
    calibration_fraction: float = 0.2
    alpha: float = 0.1
    max_diff: int = 600
    gamma: float = 0.02


def rolling_windows(p, seed):
    """List of (train_dates, calib_dates, test_dates)."""
    rng = np.random.default_rng(seed)
    t0, tr0, end = (np.datetime64(d, "D") for d in (p.test_start, p.train_start, p.end_all))
    duration = t0 - tr0
    pool = np.arange(tr0, t0 - 20 * DAY, DAY)
    mondays = pool[(pool.astype(int) + 3) % 7 == 0]          # 1970-01-01 was a Thursday
    n_blocks = max(1, round(p.calibration_fraction * len(mondays)))
    offsets = rng.choice(mondays, n_blocks, replace=False) - tr0
    out = []
    for ts in np.arange(t0, end - 12 * DAY, 14 * DAY):
        start = ts - duration
        blocks = np.concatenate([start + offsets, [ts - 7 * DAY]])
        calib = np.unique((blocks[:, None] + np.arange(7) * DAY).ravel())
        hist = np.arange(start, ts, DAY)
        out.append((np.setdiff1d(hist, calib), calib, np.arange(ts, ts + 14 * DAY, DAY)))
    return out


def _fit_predict_global(X, Y, tr, pr, cat_cols):
    """One boosted model on log1p(y) across all series of a level."""
    Xtr, ytr = X[tr], Y[tr]
    ok = ~np.isnan(ytr)
    model = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.08,
                                          categorical_features=cat_cols, random_state=0)
    model.fit(Xtr[ok], np.log1p(ytr[ok]))
    shape = X[pr].shape[:2]
    return np.expm1(model.predict(X[pr].reshape(-1, X.shape[-1]))).reshape(shape)


def _fit_predict_national(X, Y, tr, pr):
    Xtr, ytr = X[tr, 0], Y[tr, 0]
    ok = ~np.isnan(ytr)
    num = list(range(1, X.shape[-1]))
    pre = ColumnTransformer([
        ("wd", make_pipeline(SimpleImputer(strategy="most_frequent"),
                             OneHotEncoder(handle_unknown="ignore")), [0]),
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num)])
    model = make_pipeline(pre, RidgeCV(alphas=np.logspace(-2, 2, 9)))
    model.fit(Xtr[ok], np.log1p(ytr[ok]))
    return np.expm1(model.predict(X[pr, 0]))[:, None]


def run_seed(panel, p, seed, log=print):
    """Run all methods over all windows. Returns a dict of arrays over test days."""
    dates, Y = panel["dates"], panel["Y"]
    region_of = panel["region_of"]
    R, n = len(panel["regions"]), len(region_of)
    m = 1 + R + n
    project = make_fast_projector(region_of)
    day_of = {d: i for i, d in enumerate(dates)}
    usable = ~np.isnan(Y[:, 0])                           # network-wide outage days are dropped
    names_s, names_r = list(panel["names_sta"]), list(panel["names_reg"])
    cat_s = [names_s.index("weekday_holiday")]
    cat_r = [names_r.index("weekday_holiday"), names_r.index("region")]

    idx = lambda ds: np.array([day_of[d] for d in ds if d in day_of and usable[day_of[d]]], int)
    res = {k: [] for k in ["date", "Y", "A", "pred"]}
    res.update({f"lo::{k}": [] for k in METHODS})
    res.update({f"up::{k}": [] for k in METHODS})
    state = None
    for w, (train, calib, test) in enumerate(rolling_windows(p, seed)):
        tr, ca, te = idx(train), idx(calib), idx(test)
        if len(te) == 0 or len(ca) == 0:
            continue
        pr = np.concatenate([ca, te])
        Yhat = np.concatenate([
            _fit_predict_national(panel["X_nat"], Y[:, :1], tr, pr),
            _fit_predict_global(panel["X_reg"], Y[:, 1:1 + R], tr, pr, cat_r),
            _fit_predict_global(panel["X_sta"], Y[:, 1 + R:], tr, pr, cat_s)], 1)
        A = ~np.isnan(Y[pr])
        Y0 = np.where(A, Y[pr], 0.0)
        Yhat = np.where(A, Yhat, 0.0)
        mu = project(A[:, -n:], Yhat)
        c, t = slice(0, len(ca)), slice(len(ca), None)

        s_all = np.abs(Y0[c] - mu[c])
        s_act = np.where(A[c], s_all, np.nan)
        lo, up = {}, {}
        for name, sc, rule in [("CP", s_all, "empirical"),
                               ("CP-MNR (code quantile)", s_act, "empirical"),
                               ("CP-MNR (n+1 quantile)", s_act, "conformal")]:
            q = split_quantile(sc, p.alpha, rule)
            lo[name], up[name] = mu[t] - q, mu[t] + q
        lo["Adaptive CP-MNR"], up["Adaptive CP-MNR"], state = adaptive_cp_mnr(
            s_act, mu[t], Y0[t], A[t], p.alpha, state, p.gamma)
        for v in ("released code", "exact set", "corrected"):
            lo[f"Nested* ({v})"] = np.full(mu[t].shape, np.nan)
            up[f"Nested* ({v})"] = np.full(mu[t].shape, np.nan)
        for j in range(len(te)):
            r = cp_mnr_nested_star(Y0[c], Yhat[c], A[c], Yhat[t][j], A[t][j], project, n,
                                   p.alpha, p.max_diff)
            lo["Nested* (released code)"][j], up["Nested* (released code)"][j] = r["released"]
            lo["Nested* (exact set)"][j], up["Nested* (exact set)"][j] = r["exact"]
            lo["Nested* (corrected)"][j], up["Nested* (corrected)"][j] = r["corrected"]

        res["date"].append(dates[te]); res["Y"].append(Y0[t]); res["A"].append(A[t]); res["pred"].append(mu[t])
        for k in METHODS:
            res[f"lo::{k}"].append(lo[k].astype(np.float32))
            res[f"up::{k}"].append(up[k].astype(np.float32))
        log(f"seed {seed} window {w} ({dates[te][0]}): {len(tr)} train / {len(ca)} calib / {len(te)} test days")
    return {k: np.concatenate(v) for k, v in res.items()}
