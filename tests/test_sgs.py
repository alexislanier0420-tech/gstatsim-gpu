"""SGS (NumPy and GPU) against GStatSim 1.2.0 with a fixed seed (criterion 3)."""

import numpy as np
import pandas as pd
import pytest

from gstatsim_accel import _sgs, _sgs_cuda, gpu_status, simulation

# Three implementations: CuPy-style code on NumPy, on CuPy, and the CUDA kernel.
XPS = [pytest.param(np, id="numpy")]
if gpu_status().available:
    import cupy as cp
    XPS += [pytest.param(cp, id="cupy"), pytest.param("cuda", id="cuda-kernel")]


@pytest.fixture(scope="module")
def field():
    rng = np.random.default_rng(2)
    x, y = rng.uniform(0, 2e4, 200), rng.uniform(0, 2e4, 200)
    z = np.sin(x / 5e3) + np.cos(y / 7e3) + 0.2 * rng.standard_normal(200)
    pts = np.column_stack([rng.uniform(0, 2e4, 120), rng.uniform(0, 2e4, 120)])
    return pd.DataFrame({"X": x, "Y": y, "Z": z}), pts


def ensemble(xp, df, pts, vario, method, seeds):
    args = (pts, df.X.to_numpy(), df.Y.to_numpy(), df.Z.to_numpy(), vario, 16, 8000,
            method == "ordinary", seeds, float(df.Z.mean()))
    return _sgs_cuda.sgs_ensemble(*args) if xp == "cuda" else _sgs.sgs_ensemble(xp, *args)


def run(xp, df, pts, vario, method, seed=7):
    return ensemble(xp, df, pts, vario, method, [seed])[0]


@pytest.mark.parametrize("xp", XPS)
@pytest.mark.parametrize("method", ["simple", "ordinary"])
@pytest.mark.parametrize("vario", [
    [20, 0.0, 6000.0, 3000.0, 1.0, "Exponential"],
    [0, 0.1, 6000.0, 6000.0, 1.0, "Spherical"],
], ids=["exp-nug0", "sph-nug0.1"])
def test_fixed_seed_matches_gstatsim(xp, method, vario, field):
    df, pts = field
    pts = pts.copy()
    if vario[1] == 0:
        # Nodes on data take the datum's value (nugget 0 only).
        pts[:2] = df[["X", "Y"]].to_numpy()[:2]
    # With nugget > 0, GStatSim 1.2.0 simulates a node sitting on a datum and
    # adds it to the conditioning set, creating two points at one location;
    # those always tie, so that case can't be compared node by node.
    ref = simulation.sgs(df, pts, vario, num_points=16, radius=8000, method=method, seed=7)
    np.testing.assert_allclose(run(xp, df, pts, vario, method), ref, rtol=0, atol=1e-6)


@pytest.mark.parametrize("xp", XPS)
def test_ensemble_rows_match_gstatsim_per_seed(xp, field):
    """Stepping realizations together must not change any of them."""
    df, pts = field
    pts = pts.copy()
    pts[:3] = df[["X", "Y"]].to_numpy()[:3]  # some steps copy a datum, others don't
    vario = [20, 0.0, 6000.0, 3000.0, 1.0, "Exponential"]
    seeds = [3, 11, 42]
    ens = ensemble(xp, df, pts, vario, "simple", seeds)
    for row, seed in zip(ens, seeds):
        ref = simulation.sgs(df, pts, vario, num_points=16, radius=8000, seed=seed)
        np.testing.assert_allclose(row, ref, rtol=0, atol=1e-6)


@pytest.mark.parametrize("xp", XPS)
def test_seed_controls_realization(xp, field):
    df, pts = field
    vario = [0, 0.0, 6000.0, 6000.0, 1.0, "Exponential"]
    a, b, c = (run(xp, df, pts, vario, "simple", seed=s) for s in (1, 1, 2))
    np.testing.assert_array_equal(a, b)
    assert not np.allclose(a, c)


def test_bad_seed_rejected(field):
    df, pts = field
    with pytest.raises(ValueError, match="seed"):
        simulation.sgs(df, pts, [0, 0, 1e3, 1e3, 1.0, "Exponential"], num_points=8, radius=1e3, seed=1.5)
