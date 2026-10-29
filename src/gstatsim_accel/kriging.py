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
        "gpu" raises BackendUnavailableError if CuPy or a CUDA device is
        missing; it never falls back to the CPU on its own.

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
    raise NotImplementedError(
        "GPU kriging is not implemented yet (Phase 2, Nov 9-15). "
        "Use backend='cpu' for now."
    )
