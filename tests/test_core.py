import numpy as np

from evconf.conformal import cp_mnr_nested_star, nested_interval, split_quantile
from evconf.hierarchy import (all_projections, make_fast_projector, node_activity,
                              pattern_code, refined_projection, summing_matrix)

REGION_OF = np.array([0, 0, 1, 1, 1])


def test_projection_is_coherent_and_idempotent():
    S = summing_matrix(REGION_OF)
    active = np.array([1, 0, 1, 1, 0], bool)
    P = refined_projection(S, active)
    assert np.allclose(P @ P, P)
    y = P @ np.random.default_rng(0).normal(size=S.shape[0])
    assert np.allclose(y[3:][~active], 0)                      # inactive stations -> 0
    assert np.isclose(y[0], y[3:].sum())                       # national = sum of stations
    assert np.isclose(y[1], y[3:5].sum()) and np.isclose(y[2], y[5:].sum())


def test_projection_matches_authors_formula():
    # t(H_t) ginv(H_t t(H_t)) H_t with rows of inactive stations zeroed
    S = summing_matrix(REGION_OF)
    active = np.array([1, 1, 0, 1, 0], bool)
    H_t = S.T * active[:, None]
    H_t[:, 3:][:, ~active] = 0
    ref = H_t.T @ np.linalg.pinv(H_t @ H_t.T) @ H_t
    assert np.allclose(refined_projection(S, active), ref)


def test_all_projections_indexing():
    S = summing_matrix(REGION_OF)
    P_all = all_projections(S)
    active = np.array([[0, 1, 1, 0, 1]], bool)
    assert np.allclose(P_all[pattern_code(active)[0]], refined_projection(S, active[0]))


def test_node_activity():
    A = node_activity(np.array([[0, 0, 1, 0, 0]]), REGION_OF)
    assert A.tolist() == [[True, False, True, False, False, True, False, False]]


def test_split_quantile_rules():
    s = np.arange(1.0, 11.0)[:, None]                          # n = 10
    assert split_quantile(s, 0.1, "empirical")[0] == 9         # ceil(0.9 * 10)
    assert split_quantile(s, 0.1, "conformal")[0] == 10        # ceil(0.9 * 11)
    assert np.isinf(split_quantile(s[:5], 0.1, "conformal")[0])
    s_nan = np.concatenate([s, [[np.nan]]])
    assert split_quantile(s_nan, 0.1, "empirical")[0] == 9


def test_nested_interval_exact_matches_brute_force():
    rng = np.random.default_rng(1)
    for _ in range(50):
        n = rng.integers(3, 30)
        c, w = rng.normal(size=n), rng.uniform(0.1, 2, size=n)
        lo, up = c - w, c + w
        thr = int(rng.integers(1, n))
        got = nested_interval(lo, up, thr)
        pts = np.sort(np.concatenate([lo, up]))
        counts = ((lo[None] <= pts[:, None]) & (pts[:, None] <= up[None])).sum(1)
        keep = pts[counts >= thr]
        if keep.size == 0:
            assert np.isnan(got["exact"]).all()
            continue
        assert np.allclose(got["exact"], (keep.min(), keep.max()))
        # the released closure sits one endpoint lower at both ends
        i, j = np.searchsorted(pts, keep.min()), np.searchsorted(pts, keep.max())
        assert np.allclose(got["released"], (pts[max(i - 1, 0)], pts[j - 1]))


def test_fast_projector_matches_dense():
    rng = np.random.default_rng(3)
    region_of = np.repeat(np.arange(4), [1, 3, 5, 2])
    S = summing_matrix(region_of)
    act = rng.random((20, region_of.size)) > 0.4
    act[0] = False
    act[1, region_of == 2] = False
    V = rng.normal(size=(20, S.shape[0]))
    got = make_fast_projector(region_of)(act, V)
    want = np.stack([refined_projection(S, a) @ v for a, v in zip(act, V)])
    assert np.allclose(got, want)



def test_nested_star_corrected_whole_line_below_threshold():
    region_of = np.array([0, 0, 1, 1])
    S = summing_matrix(region_of)
    project = make_fast_projector(region_of)
    rng = np.random.default_rng(4)
    sta = np.ones((30, 4), bool)
    A = node_activity(sta, region_of)
    Y = (S @ rng.uniform(1, 2, (30, 4)).T).T
    Yhat = Y + rng.normal(0, 0.1, Y.shape)
    a_t, yhat_t = A[0], S @ np.full(4, 1.5)
    # 30 scores: threshold floor(0.1 * 31) = 3, corrected = exact, both finite
    r = cp_mnr_nested_star(Y, Yhat, A, yhat_t, a_t, project, 4, 0.1)
    assert np.allclose(r["corrected"], r["exact"]) and np.isfinite(r["exact"]).all()
    # 8 scores: threshold 0, every y qualifies; released and exact stay finite
    r8 = cp_mnr_nested_star(Y[:8], Yhat[:8], A[:8], yhat_t, a_t, project, 4, 0.1)
    assert (r8["corrected"][0] == -np.inf).all() and (r8["corrected"][1] == np.inf).all()
    assert np.isfinite(r8["exact"]).all() and np.isfinite(r8["released"]).all()
