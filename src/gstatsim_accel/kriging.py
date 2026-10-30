"""Notebook-facing kriging API with explicit CPU/GPU backend selection.

The CPU backend calls GStatSim unchanged, so it is the reference that the GPU
backend is validated against.
"""

from __future__ import annotations

import numpy as np

from . import validation
from .backend import resolve_backend


def predict(
    conditioning_data,
    prediction_points,
    variogram,
    *,
    num_points: int,
    radius: float,
    method: str = "simple",
    backend: str = "cpu",
    xx: str = "X",
    yy: str = "Y",
    zz: str = "Z",
    quiet: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Krige values at prediction_points.

    Parameters
    ----------
    conditioning_data : pandas.DataFrame
        Data with coordinate columns xx, yy and value column zz. Units must
        match prediction_points and the variogram ranges (e.g. metres).
    prediction_points : array-like, shape (M, 2)
        x, y coordinates to estimate, as made by gstatsim.Gridding.prediction_grid.
    variogram : list
        GStatSim format: [azimuth, nugget, major_range, minor_range, sill, vtype],
        plus smoothness s for Matern.
    num_points : int
        Neighbors to use, split evenly across 8 octants (GStatSim's search),
        so the effective maximum is 8 * (num_points // 8).
    radius : float
        Neighbor search radius.
    method : {"simple", "ordinary"}
    backend : {"cpu", "gpu"}
        "cpu" runs GStatSim itself. "gpu" runs a batched CuPy version that
        matches it to ~1e-12, except where equidistant points tie for a
        neighbor slot (GStatSim's pick is arbitrary there; ties here go to the
        earlier data row). "gpu" raises BackendUnavailableError if CuPy or a
        CUDA device is missing; it never falls back to the CPU on its own.

    Returns
    -------
    estimate, variance : numpy.ndarray, shape (M,)
        Same order as prediction_points. Points that coincide with a
        conditioning point take its value and zero variance.
    """
    backend = resolve_backend(backend)
    method = validation.check_method(method)
    points = validation.check_prediction_points(prediction_points)
    df = validation.check_conditioning_data(conditioning_data, xx, yy, zz)
    vario = validation.check_variogram(variogram)
    validation.check_search(num_points, radius)

    if len(points) == 0:
        return np.empty(0), np.empty(0)

    if backend == "cpu":
        return _predict_cpu(df, points, vario, num_points, radius, method, quiet)
    return _predict_gpu(df, points, vario, num_points, radius, method)


def _predict_cpu(df, points, vario, num_points, radius, method, quiet):
    import gstatsim as gs

    krige = gs.Interpolation.skrige if method == "simple" else gs.Interpolation.okrige
    est, var = krige(points, df, "X", "Y", "Z", num_points, vario, radius, quiet=quiet)
    return np.asarray(est, dtype=np.float64), np.asarray(var, dtype=np.float64)


def _predict_gpu(df, points, vario, num_points, radius, method):
    if vario[5].lower() == "matern":
        raise NotImplementedError("Matern covariance isn't supported on the GPU yet; use backend='cpu'")
    import cupy as cp

    from . import _batched
    est, var = _batched.krige(
        cp, points, df["X"].to_numpy(), df["Y"].to_numpy(), df["Z"].to_numpy(), vario,
        num_points, radius, ordinary=(method == "ordinary"),
        # Same pandas mean GStatSim uses, so simple kriging's mean is bit-identical.
        global_mean=float(df["Z"].mean()),
    )
    var[var < 0] = 0
    return est, var


def cokrige_mm1(
    conditioning_data,
    secondary_data,
    prediction_points,
    variogram,
    *,
    num_points: int,
    radius: float,
    corrcoef: float,
    backend: str = "cpu",
    xx: str = "X",
    yy: str = "Y",
    zz: str = "Z",
    xx2: str = "X",
    yy2: str = "Y",
    zz2: str = "Z",
    quiet: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Simple cokriging with a secondary variable, Markov model 1
    (GStatSim's cokrige_mm1).

    Each point uses its octant neighbors from conditioning_data plus the single
    nearest point of secondary_data (no radius limit), whose covariances are the
    primary ones times corrcoef. As in GStatSim, the variogram's sill is
    replaced by the variance of the primary data; unlike GStatSim, the caller's
    variogram list is left unchanged. The variance is returned unclipped, as
    GStatSim returns it.
    """
    backend = resolve_backend(backend)
    points = validation.check_prediction_points(prediction_points)
    df1 = validation.check_conditioning_data(conditioning_data, xx, yy, zz)
    df2 = validation.check_conditioning_data(secondary_data, xx2, yy2, zz2)
    vario = validation.check_variogram(variogram)
    validation.check_search(num_points, radius)
    corrcoef = validation.check_corrcoef(corrcoef)
    if len(points) == 0:
        return np.empty(0), np.empty(0)

    if backend == "cpu":
        import gstatsim as gs

        est, var = gs.Interpolation.cokrige_mm1(points, df1, "X", "Y", "Z", df2, "X", "Y", "Z",
                                                num_points, list(vario), radius, corrcoef, quiet=quiet)
        return np.asarray(est, dtype=np.float64), np.asarray(var, dtype=np.float64)

    if vario[5].lower() == "matern":
        raise NotImplementedError("Matern covariance isn't supported on the GPU yet; use backend='cpu'")
    import cupy as cp

    from . import _batched

    return _batched.cokrige_mm1(cp, points, *(df1[c].to_numpy() for c in "XYZ"),
                                *(df2[c].to_numpy() for c in "XYZ"), vario, num_points, radius, corrcoef)
