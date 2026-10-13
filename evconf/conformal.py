"""Conformal procedures from Principato et al. (arXiv:2610.13588), following the
released R code (R/conformal_methods.R) so that deviations are deliberate.

Conventions: Y0 is the observed node vector with inactive nodes set to 0, A is
the boolean node-activity matrix, scores for nodes that must be ignored are NaN.
"""
import math

import numpy as np

_EPS = 1e-9


def split_quantile(scores, alpha, rule="conformal"):
    """Column-wise calibration quantile, ignoring NaN.

    rule="empirical": order statistic ceil((1-alpha) n), i.e. R's
        quantile(probs = 1 - alpha, type = 1) as in the released code.
    rule="conformal": order statistic ceil((1-alpha)(n+1)), the finite-sample
        valid choice; +inf when it exceeds n.
    """
    scores = np.asarray(scores, float)
    n = (~np.isnan(scores)).sum(0)
    srt = np.sort(scores, axis=0)  # NaN sorted last
    target = (1 - alpha) * (n if rule == "empirical" else n + 1)
    k = np.ceil(target - _EPS).astype(int)
    q = np.full(scores.shape[1], np.inf)
    ok = (k >= 1) & (k <= n)
    q[ok] = srt[k[ok] - 1, np.nonzero(ok)[0]]
    q[n == 0] = np.nan
    return q


def nested_interval(lo, up, thr):
    """Convex hull of {y : #{k : lo_k <= y <= up_k} >= thr}.

    Returns {"released": (lower, upper), "exact": (lower, upper)}.

    After sorting the 2n endpoints, c[i] is the number of intervals covering
    (x[i], x[i+1]), so the closed set is the union of [x[i], x[i+1]] over
    c[i] >= thr. "exact" does that. "released" transcribes nested_component()
    in the authors' R/conformal_methods.R, whose closure step
    `inside | c(inside[-1], FALSE)` extends one endpoint to the left instead of
    to the right, shifting both bounds down by one order statistic.
    """
    nan = (np.nan, np.nan)
    ok = ~(np.isnan(lo) | np.isnan(up))
    lo, up = lo[ok], up[ok]
    n = lo.size
    if n == 0:
        return {"released": nan, "exact": nan}
    x = np.concatenate([lo, up])
    d = np.concatenate([np.ones(n), -np.ones(n)])
    order = np.lexsort((-d, x))  # by x, openings before closings on ties
    x, inside = x[order], np.cumsum(d[order]) >= thr
    if not inside.any():
        return {"released": nan, "exact": nan}
    idx = np.nonzero(inside)[0]
    first, last = idx[0], idx[-1]
    return {"released": (x[max(first - 1, 0)], x[last]),
            "exact": (x[first], x[min(last + 1, 2 * n - 1)])}


def nested_threshold(alpha, n):
    """floor(alpha (n+1)): y is kept when it lies in strictly more than
    alpha (n+1) - 1 calibration intervals, the jackknife+-style rule behind the
    1 - 2 alpha guarantee."""
    return math.floor(alpha * (n + 1) + _EPS)


def cp_mnr_nested_star(Y0_cal, Yhat_cal, A_cal, yhat_t, a_t, project, n_sta, alpha,
                       max_diff=None):
    """CP-MNR-Nested* interval for one test point.

    A_* are node-activity masks whose last n_sta columns are the stations.
    project(station_active (k, n_sta), V (k, m)) applies the refined projection
    of each pattern to the matching row of V.
    Returns {"released" | "exact": (lower, upper)} with arrays of length m.
    """
    keep = np.ones(len(A_cal), bool)
    if max_diff is not None:
        keep = (A_cal != a_t).sum(1) <= max_diff
    m = Y0_cal.shape[1]
    res = {v: (np.full(m, np.nan), np.full(m, np.nan)) for v in ("released", "exact")}
    if not keep.any():
        return res
    A_comb = A_cal[keep] & a_t                     # node-level pmax of the missing masks
    sta = A_comb[:, -n_sta:]
    score = Y0_cal[keep] - project(sta, Yhat_cal[keep])
    score[(score == 0) | ~A_comb] = np.nan
    mu = project(sta, np.broadcast_to(yhat_t, score.shape))
    lo, up = mu - np.abs(score), mu + np.abs(score)
    n_valid = (~np.isnan(score)).sum(0)
    for j in np.nonzero(a_t & (n_valid > 0))[0]:
        iv = nested_interval(lo[:, j], up[:, j], nested_threshold(alpha, int(n_valid[j])))
        for v in res:
            res[v][0][j], res[v][1][j] = iv[v]
    return res


def adaptive_cp_mnr(scores_cal, mu_test, Y_test, A_test, alpha, state=None, gamma=0.02):
    """Adaptive CP-MNR (transcribes Adaptive_CP_MNR in the released code).

    scores_cal: calibration scores, NaN where the node was inactive.
    state: (a_t, alpha_t) carried over from the previous window, or None.
    Returns lower, upper, new state. Only active nodes update their level.
    """
    n_cal, m = scores_cal.shape
    a_t, alpha_t = (np.full(m, alpha), np.full(m, alpha)) if state is None else state
    srt = np.sort(scores_cal, axis=0)
    n = (~np.isnan(scores_cal)).sum(0)
    top = np.where(n > 0, srt[np.maximum(n - 1, 0), np.arange(m)], np.nan)
    lo, up = np.full(mu_test.shape, np.nan), np.full(mu_test.shape, np.nan)
    for j in range(len(mu_test)):
        k = np.clip(np.ceil((1 - a_t) * n - _EPS).astype(int), 1, np.maximum(n, 1))
        q = np.where(n > 0, srt[k - 1, np.arange(m)], np.nan)
        q = np.where(a_t >= 1, 0.0, np.where(a_t <= 0, 2 * top, q))
        lo[j], up[j] = mu_test[j] - q, mu_test[j] + q
        miss = np.abs(Y_test[j] - mu_test[j]) > q
        upd = A_test[j] & ~np.isnan(q)
        alpha_t = alpha_t - gamma * np.where(upd, miss - alpha, 0.0)
        rounded = 1 - np.ceil((n_cal + 1) * (1 - alpha_t) - _EPS) / (n_cal + 1)
        a_t = np.where((alpha_t >= 0) & (alpha_t <= 1), rounded, np.clip(alpha_t, 0, 1))
    return lo, up, (a_t, alpha_t)
