"""Three-level hierarchy (national / regions / stations) with time-varying activity.

Node order is [national, region_1..region_R, station_1..station_n]. An activity
pattern is a boolean vector over stations; aggregate nodes are active when at
least one of their stations is.
"""
import numpy as np


def summing_matrix(region_of):
    """S (m x n): maps station values to the full node vector."""
    region_of = np.asarray(region_of)
    n = region_of.size
    R = region_of.max() + 1
    S = np.zeros((1 + R + n, n))
    S[0] = 1.0
    S[1 + region_of, np.arange(n)] = 1.0
    S[1 + R + np.arange(n), np.arange(n)] = 1.0
    return S


def node_activity(station_active, region_of):
    """(T, n) station activity -> (T, m) node activity."""
    station_active = np.atleast_2d(station_active).astype(bool)
    R = np.max(region_of) + 1
    reg = np.stack([station_active[:, region_of == r].any(1) for r in range(R)], 1)
    return np.concatenate([station_active.any(1, keepdims=True), reg, station_active], 1)


def refined_projection(S, active):
    """OLS projection onto the coherent subspace spanned by the active stations.

    Same operator as the paper's refined OLS, t(H_t) ginv(H_t t(H_t)) H_t, where
    H_t keeps only the rows of active stations. Inactive nodes are mapped to 0.
    """
    Sa = S[:, np.asarray(active, bool)]
    if Sa.shape[1] == 0:
        return np.zeros((S.shape[0], S.shape[0]))
    return Sa @ np.linalg.pinv(Sa)


def pattern_code(station_active):
    """Encode (T, n) boolean patterns as integers (n <= 20)."""
    station_active = np.atleast_2d(station_active).astype(np.int64)
    return station_active @ (1 << np.arange(station_active.shape[1], dtype=np.int64))


def all_projections(S):
    """Projection for every one of the 2^n activity patterns, indexed by pattern_code."""
    m, n = S.shape
    if n > 14:
        raise ValueError("enumerating all patterns is only meant for small hierarchies")
    out = np.zeros((1 << n, m, m))
    bits = np.arange(n)
    for code in range(1, 1 << n):
        out[code] = refined_projection(S, (code >> bits) & 1)
    return out


def make_fast_projector(region_of):
    """Batched refined projection without forming any m x m matrix.

    For active stations, S_a' S_a = I + U U' with U = [region indicators, 1],
    so the normal equations reduce by Woodbury to an (R+1) x (R+1) solve.
    Returns project(station_active (B, n), V (B, m)) -> (B, m).
    """
    region_of = np.asarray(region_of)
    n, R = region_of.size, int(region_of.max()) + 1
    G = np.zeros((n, R))
    G[np.arange(n), region_of] = 1.0

    def project(station_active, V):
        a = np.asarray(station_active, float)
        V = np.asarray(V, float)
        w = (V[:, 1 + R:] + V[:, 1:1 + R][:, region_of] + V[:, :1]) * a
        k = a @ G
        M = np.zeros((a.shape[0], R + 1, R + 1))
        idx = np.arange(R)
        M[:, idx, idx] = 1 + k
        M[:, :R, R] = k
        M[:, R, :R] = k
        M[:, R, R] = 1 + k.sum(1)
        rhs = np.concatenate([w @ G, w.sum(1, keepdims=True)], 1)
        z = np.linalg.solve(M, rhs[..., None])[..., 0]
        b = w - (z[:, :R][:, region_of] + z[:, R:]) * a
        return np.concatenate([b.sum(1, keepdims=True), b @ G, b], 1)

    return project
