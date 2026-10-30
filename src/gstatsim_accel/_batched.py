"""Batched kriging that reproduces GStatSim 1.2.0.

Written once against the NumPy API: call with xp=numpy for a vectorized CPU
version or xp=cupy for the GPU. Each step mirrors a GStatSim function (named
in the comments) so the two can be checked line by line.

Known, documented difference: GStatSim sorts neighbors with an unstable sort,
so among points at exactly the same distance its choice is arbitrary. Here
ties are broken by data row order.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import gamma, kv

import gstatsim as gs

# GStatSim's octant bin edges (NearestNeighbor.nearest_neighbor_search),
# built from the same float expressions so the edges are bit-identical.
OCTANT_EDGES = [-math.pi, -3 * math.pi / 4, -math.pi / 2, -math.pi / 4, 0,
                math.pi / 4, math.pi / 2, 3 * math.pi / 4, math.pi + 1e-3]

# Elements per (batch x candidates) array in the neighbor search; bounds GPU memory.
_SEARCH_ELEMS = 2 ** 23


def to_numpy(xp, a):
    return a.get() if xp is not np else a


def covariance(xp, lag, vario):
    """Covariance.covar from GStatSim 1.2.0."""
    nug, sill, vtype = float(vario[1]), float(vario[4]), vario[5].lower()
    c_struct = sill - nug
    if vtype == "exponential":
        return c_struct * xp.exp(-3 * lag)
    if vtype == "gaussian":
        return c_struct * xp.exp(-3 * xp.square(lag))
    if vtype == "spherical":
        return c_struct * xp.where(lag <= 1.0, 1.0 - 1.5 * lag + 0.5 * xp.power(lag, 3), 0.0)
    if vtype == "matern":
        if xp is not np:
            raise NotImplementedError("Matern covariance needs scipy.special.kv, which CuPy lacks")
        s = float(vario[6])
        scale = 0.45246434 * np.exp(-0.70449189 * s) + 1.7863836
        eff = np.where(lag == 0.0, 1e-8, lag)
        c = c_struct * 2 / gamma(s) * np.power(scale * eff * np.sqrt(s), s) * kv(s, 2 * scale * eff * np.sqrt(s))
        return np.where(np.isnan(c), c_struct, c)
    raise ValueError(f"unknown vtype {vario[5]!r}")


def find_neighbors(xp, px, py, X, Y, radius, num_points):
    """Octant search (NearestNeighbor.nearest_neighbor_search) for a batch of points.

    Returns
    -------
    nbr : (B, 8 * (num_points // 8)) int array
        Row indices into X/Y in GStatSim's layout: octant-major, nearest first
        within an octant; -1 marks an empty slot.
    coincident : (B,) int array
        First data row at exactly the point's location, or -1.
    """
    B, oct_count = len(px), num_points // 8
    nbr = xp.full((B, 8 * oct_count), -1, dtype=xp.int64)
    if len(X) == 0:
        return nbr, xp.full(B, -1, dtype=xp.int64)
    dx = X[None, :] - px[:, None]
    dy = Y[None, :] - py[:, None]
    hit = (dx == 0) & (dy == 0)
    coincident = xp.where(hit.any(axis=1), xp.argmax(hit, axis=1), -1)

    # Squares added as separate arrays (not one expression) so the GPU can't
    # fuse them into an FMA and round differently from NumPy.
    d2 = dx * dx
    d2 += dy * dy
    dist = xp.sqrt(d2)
    angle = xp.arctan2(dx, dy)
    # pd.cut with right-closed bins: octant k holds edges[k] < angle <= edges[k+1].
    octant = xp.searchsorted(xp.asarray(OCTANT_EDGES), angle, side="left") - 1
    valid = (dist < radius) & (octant >= 0)
    del dx, dy, d2, angle

    width = int(valid.sum(axis=1).max()) if B else 0
    if width == 0:
        return nbr, coincident
    order = xp.argsort(xp.where(valid, dist, xp.inf), axis=1, kind="stable")[:, :width]
    oct_sorted = xp.take_along_axis(octant, order, axis=1)
    valid_sorted = xp.take_along_axis(valid, order, axis=1)
    for k in range(8):
        in_k = valid_sorted & (oct_sorted == k)
        rank = xp.cumsum(in_k, axis=1)
        rows, cols = xp.nonzero(in_k & (rank <= oct_count))
        nbr[rows, k * oct_count + rank[rows, cols] - 1] = order[rows, cols]
    return nbr, coincident


def _rotate(x, y, R):
    # coord @ rotation_matrix, written out per component.
    return x * R[0, 0] + y * R[1, 0], x * R[0, 1] + y * R[1, 1]


def kriging_systems(xp, valid, nx, ny, px, py, vario, R, ordinary):
    """Build the batched kriging matrices (Covariance.make_covariance_matrix/_array).

    nx, ny are the neighbors' coordinates, shape (B, K); `valid` marks real
    neighbors. Empty slots get an identity row/column and a zero right-hand
    side, which gives them zero weight and leaves the real solution unchanged.
    """
    vf = valid.astype(xp.float64)
    m0, m1 = _rotate(nx, ny, R)
    p0, p1 = _rotate(px, py, R)

    d0 = m0[:, :, None] - m0[:, None, :]
    d1 = m1[:, :, None] - m1[:, None, :]
    C = covariance(xp, xp.sqrt(d0 * d0 + d1 * d1), vario)
    del d0, d1
    C *= vf[:, :, None] * vf[:, None, :]
    K = valid.shape[1]
    diag = xp.arange(K)
    # GStatSim 1.2.0 sets the diagonal to the full sill (nugget included).
    C[:, diag, diag] = xp.where(valid, float(vario[4]), 1.0)

    e0, e1 = m0 - p0[:, None], m1 - p1[:, None]
    c0 = covariance(xp, xp.sqrt(e0 * e0 + e1 * e1), vario) * vf

    if not ordinary:
        return C, c0
    B = len(valid)
    A = xp.zeros((B, K + 1, K + 1))
    A[:, :K, :K] = C
    A[:, K, :K] = vf
    A[:, :K, K] = vf
    b = xp.concatenate([c0, xp.ones((B, 1))], axis=1)
    return A, b


def solve(xp, A, b, valid, ordinary):
    """Batched solve; systems that come out singular are redone with lstsq
    on the CPU, as GStatSim does for every system."""
    try:
        w = xp.linalg.solve(A, b[..., None])[..., 0]
        bad = ~xp.isfinite(w).all(axis=1)
    except np.linalg.LinAlgError:  # NumPy raises; CuPy returns NaN instead
        w = xp.zeros_like(b)
        bad = xp.ones(len(b), dtype=bool)
    if bad.any():
        for i in to_numpy(xp, xp.nonzero(bad)[0]):
            keep = np.append(to_numpy(xp, valid[i]), True) if ordinary else to_numpy(xp, valid[i])
            Ai, bi = to_numpy(xp, A[i])[np.ix_(keep, keep)], to_numpy(xp, b[i])[keep]
            wi = np.zeros(len(keep))
            wi[keep] = np.linalg.lstsq(Ai, bi, rcond=None)[0]
            w[i] = xp.asarray(wi)
    return w


def estimate(xp, w, c0, z, valid, vario, ordinary, global_mean):
    """Estimate and (raw, unclipped) variance, as in skrige/okrige."""
    vf = valid.astype(xp.float64)
    K = valid.shape[1]
    if ordinary:
        w, c0 = w[:, :K], c0[:, :K]
        # Local mean of the neighbors; max(..., 1) only guards rows with none,
        # which are points on a datum and get overwritten by the caller.
        mean = (z * vf).sum(axis=1) / xp.maximum(vf.sum(axis=1), 1)
        centered = z - mean[:, None]
    else:
        mean = global_mean
        centered = z - mean
    est = mean + xp.sum(w * centered * vf, axis=1)
    var = float(vario[4]) - xp.sum(w * c0, axis=1)
    return est, var


def krige(xp, points, X, Y, Z, vario, num_points, radius, ordinary, global_mean, batch_size=None):
    """Simple or ordinary kriging of every point (Interpolation.skrige/okrige).

    points, X, Y, Z are NumPy arrays; returns NumPy (estimate, variance).
    """
    R = gs.make_rotation_matrix(vario[0], vario[2], vario[3])
    Xd, Yd, Zd = xp.asarray(X), xp.asarray(Y), xp.asarray(Z)
    est = np.empty(len(points))
    var = np.empty(len(points))
    if batch_size is None:
        batch_size = int(min(4096, max(64, _SEARCH_ELEMS // max(len(X), 1))))

    for start in range(0, len(points), batch_size):
        p = points[start:start + batch_size]
        # Only data within `radius` of the batch's bounding box can be neighbors.
        near = np.nonzero((X >= p[:, 0].min() - radius) & (X <= p[:, 0].max() + radius) &
                          (Y >= p[:, 1].min() - radius) & (Y <= p[:, 1].max() + radius))[0]
        sub = xp.asarray(near)
        Xs, Ys, Zs = Xd[sub], Yd[sub], Zd[sub]
        px, py = xp.asarray(p[:, 0]), xp.asarray(p[:, 1])

        nbr, coincident = find_neighbors(xp, px, py, Xs, Ys, radius, num_points)
        valid = nbr >= 0
        empty = ~valid.any(axis=1) & (coincident < 0)
        if bool(empty.any()):
            raise ValueError("Unable to find nearest neighbors for some prediction points. "
                             "Try increasing the search radius.")

        j = xp.where(valid, nbr, 0)
        A, b = kriging_systems(xp, valid, Xs[j], Ys[j], px, py, vario, R, ordinary)
        w = solve(xp, A, b, valid, ordinary)
        z = Zs[j]
        e, v = estimate(xp, w, b, z, valid, vario, ordinary, global_mean)

        # A point on top of a datum takes its value with zero variance.
        on_datum = coincident >= 0
        e = xp.where(on_datum, Zs[xp.where(on_datum, coincident, 0)], e)
        v = xp.where(on_datum, 0.0, v)
        est[start:start + len(p)] = to_numpy(xp, e)
        var[start:start + len(p)] = to_numpy(xp, v)
    return est, var
