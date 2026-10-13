"""Synthetic time-varying hierarchy for stress-testing conformal coverage.

Days are i.i.d., so exchangeability (Assumption 1 of the paper) holds by
construction. The single knob rho controls how strongly a station's activity
depends on its own unpredictable demand, i.e. how badly Assumption 3
(activity independent of (x, y)) is violated. rho = 0 is the paper's ideal case.
"""
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from .conformal import cp_mnr_nested_star, split_quantile
from .hierarchy import all_projections, node_activity, pattern_code, summing_matrix

METHODS = ["CP", "CP-MNR (code quantile)", "CP-MNR (n+1 quantile)",
           "Nested* (released code)", "Nested* (exact set)"]
LEVELS = ["national", "regional", "station"]
REGIMES = ["nat: <=1 off", "nat: 2 off", "nat: >=3 off",
           "reg: 0 off", "reg: 1 off", "reg: >=2 off",
           "sta: 0 siblings off", "sta: >=1 sibling off"]


@dataclass(frozen=True)
class Config:
    n_regions: int = 3
    per_region: int = 4
    rho: float = 0.0          # corr(activity latent, demand noise)
    p_off: float = 0.15       # marginal probability a station is inactive
    common: float = 0.3       # share of demand-noise variance common to a region
    cv: float = 0.3           # noise sd relative to station mean
    n_train: int = 500
    n_cal: int = 200
    n_test: int = 200
    alpha: float = 0.1
    min_active: int = 20      # paper: nodes with fewer active test points are dropped
    hier_seed: int = 0


class Hierarchy:
    def __init__(self, cfg):
        self.cfg = cfg
        self.region_of = np.repeat(np.arange(cfg.n_regions), cfg.per_region)
        self.n = self.region_of.size
        self.S = summing_matrix(self.region_of)
        self.m = self.S.shape[0]
        self.P_all = all_projections(self.S)
        rng = np.random.default_rng(cfg.hier_seed)
        self.mu = np.exp(rng.normal(np.log(100.0), 0.5, self.n))
        R = cfg.n_regions
        self.level = np.array([0] + [1] * R + [2] * self.n)
        # parent region index of each node (-1 for national)
        self.node_region = np.concatenate([[-1], np.arange(R), self.region_of])

    def draw(self, T, rng):
        cfg = self.cfg
        x = rng.normal(size=T)
        g = rng.normal(size=(T, cfg.n_regions))[:, self.region_of]
        eps = np.sqrt(cfg.common) * g + np.sqrt(1 - cfg.common) * rng.normal(size=(T, self.n))
        u = cfg.rho * eps + np.sqrt(1 - cfg.rho ** 2) * rng.normal(size=(T, self.n))
        act = u > norm.ppf(cfg.p_off)
        y_sta = (self.mu * (1 + 0.3 * x[:, None]) + cfg.cv * self.mu * eps) * act
        return x, act, y_sta @ self.S.T


def _ols(X, y):
    return np.linalg.lstsq(X, y, rcond=None)[0]


def _design(h, x, act):
    """Per-node design matrices. Aggregates see how many of their stations are
    active (as the paper's charger-count features do), not which ones."""
    T = x.size
    one = np.ones(T)
    k_reg = np.stack([act[:, h.region_of == r].sum(1) for r in range(h.cfg.n_regions)], 1)
    k = np.concatenate([act.sum(1, keepdims=True), k_reg], 1).astype(float)
    agg = [np.stack([one, x, k[:, i], x * k[:, i]], 1) for i in range(k.shape[1])]
    sta = [np.stack([one, x], 1)] * h.n
    return agg + sta


def base_forecasts(h, train, x, act):
    """Fit one OLS model per node on its active training days; predict (T, m)."""
    x_tr, act_tr, Y_tr = train
    A_tr = node_activity(act_tr, h.region_of)
    D_tr, D = _design(h, x_tr, act_tr), _design(h, x, act)
    A = node_activity(act, h.region_of)
    Yhat = np.zeros((x.size, h.m))
    for i in range(h.m):
        rows = A_tr[:, i]
        Yhat[:, i] = D[i] @ _ols(D_tr[i][rows], Y_tr[rows, i])
    return Yhat * A


def one_replication(h, seed):
    """Run every method once. Returns per-node test coverage/length arrays and
    pooled (covered, count) sums per regime."""
    cfg = h.cfg
    rng = np.random.default_rng(seed)
    train = h.draw(cfg.n_train, rng)
    x_c, act_c, Y_c = h.draw(cfg.n_cal, rng)
    x_t, act_t, Y_t = h.draw(cfg.n_test, rng)
    A_c, A_t = node_activity(act_c, h.region_of), node_activity(act_t, h.region_of)
    code_c, code_t = pattern_code(act_c), pattern_code(act_t)
    Yh_c, Yh_t = base_forecasts(h, train, x_c, act_c), base_forecasts(h, train, x_t, act_t)

    mu_c = np.einsum("kij,kj->ki", h.P_all[code_c], Yh_c)
    mu_t = np.einsum("kij,kj->ki", h.P_all[code_t], Yh_t)
    s_all = np.abs(Y_c - mu_c)                       # inactive nodes give score 0
    s_act = np.where(A_c, s_all, np.nan)

    lo, up = {}, {}
    for name, sc, rule in [("CP", s_all, "empirical"),
                           ("CP-MNR (code quantile)", s_act, "empirical"),
                           ("CP-MNR (n+1 quantile)", s_act, "conformal")]:
        q = split_quantile(sc, cfg.alpha, rule)
        lo[name], up[name] = mu_t - q, mu_t + q
    for name in METHODS[3:]:
        lo[name], up[name] = np.full_like(mu_t, np.nan), np.full_like(mu_t, np.nan)
    project = lambda sta, V: np.einsum("kij,kj->ki", h.P_all[pattern_code(sta)], V)
    for t in range(cfg.n_test):
        r = cp_mnr_nested_star(Y_c, Yh_c, A_c, Yh_t[t], A_t[t], project, h.n, cfg.alpha)
        lo["Nested* (released code)"][t], up["Nested* (released code)"][t] = r["released"]
        lo["Nested* (exact set)"][t], up["Nested* (exact set)"][t] = r["exact"]

    # regime of each (test day, node)
    R = cfg.n_regions
    off_reg = np.stack([(~act_t[:, h.region_of == r]).sum(1) for r in range(R)], 1)
    off_nat = (~act_t).sum(1)
    regime = np.full((cfg.n_test, h.m), -1)
    regime[:, 0] = np.digitize(off_nat, [2, 3])                       # 0,1,2
    regime[:, 1:1 + R] = 3 + np.minimum(off_reg, 2)                   # 3,4,5
    sib_off = off_reg[:, h.region_of]                                 # station itself is active
    regime[:, 1 + R:] = 6 + (sib_off >= 1)

    out = {}
    n_act = A_t.sum(0)
    for name in METHODS:
        defined = A_t & ~np.isnan(lo[name]) & ~np.isnan(up[name])
        cov = (Y_t >= lo[name]) & (Y_t <= up[name]) & defined
        length = np.where(defined & np.isfinite(up[name] - lo[name]), up[name] - lo[name], np.nan)
        n_def = defined.sum(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            node_cov = np.where(n_def > 0, cov.sum(0) / n_def, np.nan)
        reg_cov = np.array([cov[regime == g].sum() for g in range(len(REGIMES))])
        reg_cnt = np.array([defined[regime == g].sum() for g in range(len(REGIMES))])
        out[name] = dict(
            node_cov=node_cov,
            node_len=np.nanmean(length, 0) if np.isfinite(length).any() else np.full(h.m, np.nan),
            node_ok=n_act >= cfg.min_active,
            infinite=float((np.isinf(up[name]) & A_t).sum() / max(A_t.sum(), 1)),
            undefined=float((A_t & ~defined).sum() / max(A_t.sum(), 1)),
            reg_cov=reg_cov, reg_cnt=reg_cnt,
        )
    return out


def summarise(h, reps):
    """Aggregate replications into tidy rows (one per method x level / regime)."""
    cfg = h.cfg
    rows = []
    for name in METHODS:
        cov = np.stack([r[name]["node_cov"] for r in reps])       # (reps, m)
        ln = np.stack([r[name]["node_len"] for r in reps])
        ok = np.stack([r[name]["node_ok"] for r in reps])
        for lv, label in enumerate(LEVELS):
            sel = ok & (h.level == lv)
            c = cov[sel]
            per_rep = np.array([np.nanmean(cov[i][sel[i]]) for i in range(len(reps)) if sel[i].any()])
            rows.append(dict(
                method=name, group="level", cell=label,
                coverage=np.nanmean(c),
                coverage_se=np.nanstd(per_rep, ddof=1) / np.sqrt(per_rep.size),
                validity_rate=np.nanmean(c >= 1 - cfg.alpha - 1e-12),
                validity_rate_relaxed=np.nanmean(c >= 1 - 2 * cfg.alpha - 1e-12),
                mean_length=np.nanmean(ln[sel]),
                infinite=np.mean([r[name]["infinite"] for r in reps]),
                undefined=np.mean([r[name]["undefined"] for r in reps]),
            ))
        rc = np.stack([r[name]["reg_cov"] for r in reps]).astype(float)
        rn = np.stack([r[name]["reg_cnt"] for r in reps]).astype(float)
        for g, label in enumerate(REGIMES):
            with np.errstate(invalid="ignore", divide="ignore"):
                per_rep = rc[:, g] / rn[:, g]
            rows.append(dict(
                method=name, group="regime", cell=label,
                coverage=rc[:, g].sum() / max(rn[:, g].sum(), 1),
                coverage_se=np.nanstd(per_rep, ddof=1) / np.sqrt(np.isfinite(per_rep).sum()),
            ))
    return rows
