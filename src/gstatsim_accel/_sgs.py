"""Sequential Gaussian simulation that reproduces GStatSim 1.2.0
(Interpolation.skrige_sgs / okrige_sgs), for NumPy or CuPy.

SGS is sequential: each node is conditioned on the data *and* every node
simulated before it, so nodes within one realization can't run in parallel
without changing the algorithm. Realizations, though, are independent. This
steps many realizations forward together, one node each per step, and keeps
all state on the device so the loop never waits on the CPU. Each realization
is exactly what GStatSim produces with that seed (up to neighbor ties).

Same seed -> same path and same random draws as GStatSim: GStatSim shuffles
the path, then calls rng.normal(est, sd) once per simulated node, which
consumes the generator exactly like drawing standard normals up front.
"""

from __future__ import annotations

import gstatsim as gs
import numpy as np

from ._batched import OCTANT_EDGES, estimate, kriging_systems, to_numpy

_CHUNK = 2048  # columns added to the conditioning-set slice at a time


def _row_neighbors(xp, px, py, X, Y, radius, oct_count, edges):
    """Octant search for B points, each against its own candidate row
    (X, Y have shape (B, n)). Same layout as find_neighbors, but written so the
    GPU never has to report a size back to the CPU. NaN candidates are ignored."""
    B, n = X.shape
    dx = X - px[:, None]
    dy = Y - py[:, None]
    d2 = dx * dx
    d2 += dy * dy
    dist = xp.sqrt(d2)
    octant = xp.searchsorted(edges, xp.arctan2(dx, dy), side="left") - 1
    valid = (dist < radius) & (octant >= 0)
    octant = xp.where(valid, octant, 8)  # 8 = not a candidate

    # Stable sort by distance, then stable sort by octant: octant-major,
    # nearest first, ties by row order.
    by_dist = xp.argsort(xp.where(valid, dist, xp.inf), axis=1, kind="stable")
    by_oct = xp.argsort(xp.take_along_axis(octant, by_dist, axis=1), axis=1, kind="stable")
    order = xp.take_along_axis(by_dist, by_oct, axis=1)

    counts = (octant[:, :, None] == xp.arange(8)).sum(axis=1)       # (B, 8)
    starts = xp.cumsum(counts, axis=1) - counts
    slot = xp.arange(oct_count)
    pos = xp.minimum(starts[:, :, None] + slot, n - 1).reshape(B, -1)
    picked = xp.take_along_axis(order, pos, axis=1).reshape(B, 8, oct_count)
    return xp.where(slot < counts[:, :, None], picked, -1).reshape(B, -1)


def sgs_ensemble(xp, points, X, Y, Z, vario, num_points, radius, ordinary, seeds, global_mean):
    """One SGS realization per seed, simulated together.

    points, X, Y, Z are NumPy arrays; returns a NumPy array (len(seeds), M).
    """
    S, M, N = len(seeds), len(points), len(X)
    oct_count = num_points // 8

    # GStatSim copies a datum's value when a node sits exactly on it, but only
    # when the nugget is 0 (1.2.0 kriges those nodes otherwise).
    first_datum = {}
    if float(vario[1]) == 0:
        for i, xy in enumerate(zip(X, Y, strict=True)):
            first_datum.setdefault(xy, i)
    datum_of = np.array([first_datum.get((x, y), -1) for x, y in points])

    # Per realization: its path, which steps copy a datum, and the random draw
    # each step uses (the k-th simulated node takes the k-th normal).
    node = np.empty((S, M), dtype=np.int64)
    eps = np.zeros((S, M))
    for s, seed in enumerate(seeds):
        rng = gs.get_random_generator(seed)
        path = np.arange(M)
        rng.shuffle(path)
        draws = rng.standard_normal(M)
        simulated = datum_of[path] < 0
        node[s] = path
        eps[s, simulated] = draws[:simulated.sum()]
    datum = datum_of[node]

    R = gs.make_rotation_matrix(vario[0], vario[2], vario[3])
    edges = xp.asarray(OCTANT_EDGES)
    node_d, datum_d, eps_d = xp.asarray(node), xp.asarray(datum), xp.asarray(eps)
    P, Zdata = xp.asarray(points), xp.asarray(Z)

    # Conditioning set per realization: the data, then one column per step.
    # A step that copies a datum adds a NaN placeholder, which never counts as
    # a neighbor, so every realization keeps the same column count.
    # Unfilled columns are NaN too, so the search can run over a slice wider
    # than needed. The slice grows in chunks: growing it by one column per
    # step would give every step new array sizes, which CuPy's memory pool
    # can't reuse, and GPU memory fills up within a few thousand steps.
    Xa, Ya = xp.full((S, N + M), xp.nan), xp.full((S, N + M), xp.nan)
    Za = xp.zeros((S, N + M))
    Xa[:, :N], Ya[:, :N], Za[:, :N] = xp.asarray(X), xp.asarray(Y), Zdata
    out = xp.empty((S, M))
    rows = xp.arange(S)
    no_neighbors = xp.zeros((), dtype=bool)

    for i in range(M):
        n = N + i
        width = min(N + M, N + (i // _CHUNK + 1) * _CHUNK)
        Xw, Yw, Zw = Xa[:, :width], Ya[:, :width], Za[:, :width]
        px, py = P[node_d[:, i], 0], P[node_d[:, i], 1]
        nbr = _row_neighbors(xp, px, py, Xw, Yw, radius, oct_count, edges)
        valid = nbr >= 0
        j = xp.where(valid, nbr, 0)
        A, b = kriging_systems(xp, valid, xp.take_along_axis(Xw, j, axis=1),
                               xp.take_along_axis(Yw, j, axis=1), px, py, vario, R, ordinary)
        w = xp.linalg.solve(A, b[..., None])[..., 0]
        zn = xp.take_along_axis(Zw, j, axis=1)
        est, var = estimate(xp, w, b, zn, valid, vario, ordinary, global_mean)
        value = est + xp.sqrt(xp.abs(var)) * eps_d[:, i]

        on_datum = datum_d[:, i] >= 0
        no_neighbors |= (~valid.any(axis=1) & ~on_datum).any()
        value = xp.where(on_datum, Zdata[xp.maximum(datum_d[:, i], 0)], value)
        out[rows, node_d[:, i]] = value
        Xa[:, n] = xp.where(on_datum, xp.nan, px)
        Ya[:, n] = xp.where(on_datum, xp.nan, py)
        Za[:, n] = value

    # Checked once at the end; checking per step would stall the GPU every step.
    if bool(no_neighbors):
        raise ValueError("Unable to find nearest neighbors for some nodes. Try increasing the search radius.")
    result = to_numpy(xp, out)
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("SGS produced non-finite values (singular kriging system)")
    return result


def sgs(xp, points, X, Y, Z, vario, num_points, radius, ordinary, seed, global_mean):
    """A single realization (NumPy arrays in, NumPy array out)."""
    return sgs_ensemble(xp, points, X, Y, Z, vario, num_points, radius, ordinary, [seed], global_mean)[0]
