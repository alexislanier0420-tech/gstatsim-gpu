"""Batched kriging (NumPy and GPU) against GStatSim 1.2.0.

Checks the acceptance criteria in TRACKER.md: exact agreement on scattered
data, and on gridded data every difference must be an equidistant-neighbor swap.
"""

import math

import gstatsim as gs
import numpy as np
import pandas as pd
import pytest

from gstatsim_accel import _batched, gpu_status, kriging

XPS = [pytest.param(np, id="numpy")]
if gpu_status().available:
    import cupy as cp
    XPS.append(pytest.param(cp, id="gpu"))
needs_gpu = pytest.mark.skipif(not gpu_status().available, reason="no usable CUDA GPU")

VARIOS = [
    [0, 0.0, 6000.0, 6000.0, 1.0, "Exponential"],
    [30, 0.1, 6000.0, 3000.0, 1.0, "Spherical"],
    [120, 0.05, 5000.0, 2500.0, 0.8, "Gaussian"],
]


def gstatsim_krige(points, df, vario, k, radius, ordinary):
    f = gs.Interpolation.okrige if ordinary else gs.Interpolation.skrige
    return f(points, df.copy(), "X", "Y", "Z", k, vario, radius, quiet=True)


def batched_krige(xp, points, df, vario, k, radius, ordinary):
    est, var = _batched.krige(xp, points, df.X.to_numpy(), df.Y.to_numpy(), df.Z.to_numpy(),
                              vario, k, radius, ordinary, float(df.Z.mean()))
    return est, np.where(var < 0, 0, var)


@pytest.fixture(scope="module")
def scattered():
    rng = np.random.default_rng(1)
    x, y = rng.uniform(0, 2e4, 300), rng.uniform(0, 2e4, 300)
    z = np.sin(x / 5e3) + np.cos(y / 7e3) + 0.2 * rng.standard_normal(300)
    pts = np.column_stack([rng.uniform(0, 2e4, 150), rng.uniform(0, 2e4, 150)])
    pts[:3] = np.column_stack([x[:3], y[:3]])  # three points sit on data
    return pd.DataFrame({"X": x, "Y": y, "Z": z}), pts


@pytest.mark.parametrize("xp", XPS)
@pytest.mark.parametrize("vario", VARIOS, ids=lambda v: f"{v[5]}-nug{v[1]}")
@pytest.mark.parametrize("ordinary", [False, True], ids=["SK", "OK"])
def test_scattered_matches_gstatsim(xp, vario, ordinary, scattered):
    """Criterion 1: no ties, so every cell must agree to 1e-6."""
    df, pts = scattered
    ref_est, ref_var = gstatsim_krige(pts, df, vario, 32, 8000, ordinary)
    est, var = batched_krige(xp, pts, df, vario, 32, 8000, ordinary)
    np.testing.assert_allclose(est, ref_est, rtol=0, atol=1e-6)
    np.testing.assert_allclose(var, ref_var, rtol=0, atol=1e-6)


@pytest.mark.parametrize("xp", XPS)
def test_ties_go_to_earlier_row(xp):
    # Four data points at the same distance in octant 4 (0 < angle <= pi/4);
    # only one slot per octant, so row order must decide.
    r = 1000.0
    angles = [0.1, 0.3, 0.5, 0.7]
    X = np.array([r * math.sin(a) for a in angles])
    Y = np.array([r * math.cos(a) for a in angles])
    nbr, _ = _batched.find_neighbors(xp, xp.zeros(1), xp.zeros(1), xp.asarray(X), xp.asarray(Y), 5000, 8)
    picked = [int(j) for j in _batched.to_numpy(xp, nbr[0]) if j >= 0]
    dist = np.hypot(X, Y)
    assert picked == [int(np.flatnonzero(dist == dist.min())[0])]


@pytest.mark.parametrize("xp", XPS)
def test_gridded_differences_are_only_equidistant_swaps(xp):
    """Criterion 2 (proposed form): on gridded data a cell may differ from
    GStatSim only when equidistant points swap places in the neighbor set."""
    rng = np.random.default_rng(5)
    gx, gy = np.meshgrid(np.arange(0, 30_000, 1000.0), np.arange(0, 30_000, 1000.0))
    keep = rng.random(gx.size) < 0.4  # sparse, survey-like coverage
    x, y = gx.ravel()[keep], gy.ravel()[keep]
    df = pd.DataFrame({"X": x, "Y": y, "Z": np.sin(x / 7e3) + np.cos(y / 9e3)})
    pts = np.column_stack([gx.ravel(), gy.ravel()])[::7]
    vario = VARIOS[0]
    ref_est, _ = gstatsim_krige(pts, df, vario, 16, 6000, False)
    est, _ = batched_krige(xp, pts, df, vario, 16, 6000, False)

    differs = np.flatnonzero(np.abs(est - ref_est) > 1e-6)
    nbr, _ = _batched.find_neighbors(np, pts[differs, 0], pts[differs, 1], x, y, 6000, 16)
    for i, row in zip(differs, nbr, strict=True):
        ours = {(x[j], y[j]) for j in row if j >= 0}
        theirs = {(a, b) for a, b, _ in gs.NearestNeighbor.nearest_neighbor_search(6000, 16, pts[i], df)}
        assert ours != theirs, f"cell {i}: same neighbors but values differ"
        def dists(s, p=pts[i]):
            return sorted(round(math.hypot(a - p[0], b - p[1]), 6) for a, b in s)
        assert dists(ours - theirs) == dists(theirs - ours), f"cell {i}: neighbor sets differ beyond ties"


@pytest.mark.parametrize("xp", XPS)
def test_no_neighbors_raises(xp):
    df = pd.DataFrame({"X": [0.0], "Y": [0.0], "Z": [1.0]})
    with pytest.raises(ValueError, match="search radius"):
        batched_krige(xp, np.array([[1e6, 1e6]]), df, VARIOS[0], 8, 1000, False)


@needs_gpu
@pytest.mark.parametrize("method", ["simple", "ordinary"])
def test_predict_gpu_matches_cpu(method, scattered):
    df, pts = scattered
    cpu = kriging.predict(df, pts, VARIOS[1], num_points=32, radius=8000, method=method)
    gpu = kriging.predict(df, pts, VARIOS[1], num_points=32, radius=8000, method=method, backend="gpu")
    np.testing.assert_allclose(gpu[0], cpu[0], rtol=0, atol=1e-6)
    np.testing.assert_allclose(gpu[1], cpu[1], rtol=0, atol=1e-6)


@pytest.fixture(scope="module")
def with_secondary():
    rng = np.random.default_rng(3)
    x, y = rng.uniform(0, 3e4, 300), rng.uniform(0, 3e4, 300)
    z = np.sin(x / 5e3) + np.cos(y / 7e3) + 0.2 * rng.standard_normal(300)
    g = np.arange(500, 30000, 1000.0)
    gx, gy = np.meshgrid(g, g)
    x2, y2 = gx.ravel() + 3.7, gy.ravel() - 11.1   # offset so no point coincides with the grid
    z2 = np.sin(x2 / 5e3) + 0.5 * np.cos(y2 / 9e3) + 0.3 * rng.standard_normal(len(x2))
    pts = np.column_stack([rng.uniform(0, 3e4, 150), rng.uniform(0, 3e4, 150)])
    pts[:2] = np.column_stack([x[:2], y[:2]])
    return pd.DataFrame({"X": x, "Y": y, "Z": z}), pd.DataFrame({"X": x2, "Y": y2, "Z": z2}), pts


@pytest.mark.parametrize("xp", XPS)
@pytest.mark.parametrize("vario", VARIOS[:2], ids=lambda v: v[5])
def test_cokrige_mm1_matches_gstatsim(xp, vario, with_secondary):
    df1, df2, pts = with_secondary
    ref_est, ref_var = gs.Interpolation.cokrige_mm1(pts, df1.copy(), "X", "Y", "Z", df2.copy(), "X", "Y", "Z",
                                                    32, list(vario), 9000, 0.6, quiet=True)
    est, var = _batched.cokrige_mm1(xp, pts, *(df1[c].to_numpy() for c in "XYZ"),
                                    *(df2[c].to_numpy() for c in "XYZ"), vario, 32, 9000, 0.6)
    np.testing.assert_allclose(est, ref_est, rtol=0, atol=1e-6)
    np.testing.assert_allclose(var, ref_var, rtol=0, atol=1e-6)


@needs_gpu
def test_cosim_mm1_matches_gstatsim_and_leaves_vario_alone(with_secondary):
    from gstatsim_accel import simulation

    df1, df2, pts = with_secondary
    vario = [30, 0.1, 8000.0, 4000.0, 1.0, "Spherical"]
    kw = dict(num_points=32, radius=9000, corrcoef=0.6)
    gpu = simulation.cosim_mm1(df1, df2, pts, vario, seeds=[7, 8], backend="gpu", **kw)
    for row, seed in zip(gpu, (7, 8), strict=True):
        np.testing.assert_allclose(row, simulation.cosim_mm1(df1, df2, pts, vario, seed=seed, **kw),
                                   rtol=0, atol=1e-6)
    assert vario[4] == 1.0  # GStatSim itself would have overwritten the sill


def test_corrcoef_validated(with_secondary):
    df1, df2, pts = with_secondary
    with pytest.raises(ValueError, match="corrcoef"):
        kriging.cokrige_mm1(df1, df2, pts, VARIOS[0], num_points=16, radius=5000, corrcoef=1.0)
