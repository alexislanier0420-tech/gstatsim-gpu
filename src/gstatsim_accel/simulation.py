"""Notebook-facing sequential Gaussian simulation with explicit CPU/GPU backend."""

from __future__ import annotations

import numbers

import numpy as np

from . import validation
from .backend import resolve_backend


def sgs(
    conditioning_data,
    prediction_points,
    variogram,
    *,
    num_points: int,
    radius: float,
    method: str = "simple",
    seed: int | None = None,
    backend: str = "cpu",
    xx: str = "X",
    yy: str = "Y",
    zz: str = "Z",
    quiet: bool = True,
) -> np.ndarray:
    """One SGS realization at prediction_points (same inputs as kriging.predict).

    seed : int or None
        Same seed and inputs -> same random path and random draws on either
        backend. Realizations then agree to ~1e-12 except where equidistant
        neighbors tie (see kriging.predict); a tie changes that node and,
        through it, later nodes, so compare gridded runs as ensembles.

    Returns
    -------
    numpy.ndarray, shape (M,)
        Simulated values in the units of zz (usually normal scores).
    """
    backend = resolve_backend(backend)
    method = validation.check_method(method)
    points = validation.check_prediction_points(prediction_points)
    df = validation.check_conditioning_data(conditioning_data, xx, yy, zz)
    vario = validation.check_variogram(variogram)
    validation.check_search(num_points, radius)
    if seed is not None and (not isinstance(seed, numbers.Integral) or isinstance(seed, bool)):
        raise ValueError(f"seed must be an integer or None, got {seed!r}")

    if len(points) == 0:
        return np.empty(0)

    if backend == "cpu":
        import gstatsim as gs

        f = gs.Interpolation.skrige_sgs if method == "simple" else gs.Interpolation.okrige_sgs
        return np.asarray(f(points, df, "X", "Y", "Z", num_points, vario, radius,
                            seed=seed, quiet=quiet), dtype=np.float64)

    if vario[5].lower() == "matern":
        raise NotImplementedError("Matern covariance isn't supported on the GPU yet; use backend='cpu'")
    return _gpu_ensemble(df, points, vario, num_points, radius, method, [seed])[0]


def _gpu_ensemble(df, points, vario, num_points, radius, method, seeds):
    """CUDA kernel when it supports the settings, otherwise the CuPy version.
    Both give GStatSim's result for each seed (up to neighbor ties)."""
    from . import _sgs_cuda

    args = (points, df["X"].to_numpy(), df["Y"].to_numpy(), df["Z"].to_numpy(), vario,
            num_points, radius)
    kw = dict(ordinary=(method == "ordinary"), seeds=list(seeds), global_mean=float(df["Z"].mean()))
    if _sgs_cuda.supports(vario, num_points):
        return _sgs_cuda.sgs_ensemble(*args, **kw)
    import cupy as cp

    from . import _sgs

    return _sgs.sgs_ensemble(cp, *args, **kw)


def realizations(
    conditioning_data,
    prediction_points,
    variogram,
    *,
    seeds,
    num_points: int,
    radius: float,
    method: str = "simple",
    backend: str = "cpu",
    xx: str = "X",
    yy: str = "Y",
    zz: str = "Z",
    quiet: bool = True,
) -> np.ndarray:
    """One SGS realization per seed; returns shape (len(seeds), M).

    Row i equals sgs(..., seed=seeds[i]). On the GPU all realizations are
    simulated together, one node each per step, which is where the speedup
    comes from; on the CPU they run one after another through GStatSim.
    The GPU path is a CUDA kernel that runs each realization as one thread
    block (exponential/Gaussian/spherical, up to 96 neighbors); other settings
    use a slower CuPy version.
    """
    seeds = list(seeds)
    if not seeds or not all(isinstance(s, numbers.Integral) and not isinstance(s, bool) for s in seeds):
        raise ValueError("seeds must be a non-empty list of integers")
    backend = resolve_backend(backend)
    if backend == "cpu":
        return np.stack([
            sgs(conditioning_data, prediction_points, variogram, num_points=num_points, radius=radius,
                method=method, seed=s, xx=xx, yy=yy, zz=zz, quiet=quiet)
            for s in seeds
        ])

    method = validation.check_method(method)
    points = validation.check_prediction_points(prediction_points)
    df = validation.check_conditioning_data(conditioning_data, xx, yy, zz)
    vario = validation.check_variogram(variogram)
    validation.check_search(num_points, radius)
    if vario[5].lower() == "matern":
        raise NotImplementedError("Matern covariance isn't supported on the GPU yet; use backend='cpu'")
    if len(points) == 0:
        return np.empty((len(seeds), 0))
    return _gpu_ensemble(df, points, vario, num_points, radius, method, seeds)


def cluster_sgs(
    conditioning_data,
    prediction_points,
    variograms,
    *,
    num_points: int,
    radius: float,
    cluster_col: str = "K",
    seed: int | None = None,
    seeds=None,
    backend: str = "cpu",
    xx: str = "X",
    yy: str = "Y",
    zz: str = "Z",
    quiet: bool = True,
) -> np.ndarray:
    """Non-stationary SGS with a variogram per cluster (GStatSim's cluster_sgs).

    variograms : list of variogram lists, or GStatSim's df_gamma DataFrame
        (column "Variogram"). Cluster k uses variograms[k].
    cluster_col : column of conditioning_data with each point's cluster
        (integers 0 .. len(variograms) - 1). Each simulated node takes the
        cluster of its nearest neighbor. Simple kriging only, as in GStatSim.
    seed / seeds : one realization for `seed` (returns shape (M,)), or one per
        entry of `seeds` (returns shape (len(seeds), M); simulated side by side
        on the GPU).
    """
    if seeds is not None and seed is not None:
        raise ValueError("pass either seed or seeds, not both")
    many = seeds is not None
    seeds = list(seeds) if many else [seed]
    if many and (not seeds or not all(isinstance(s, numbers.Integral) and not isinstance(s, bool) for s in seeds)):
        raise ValueError("seeds must be a non-empty list of integers")
    if not many and seed is not None and (not isinstance(seed, numbers.Integral) or isinstance(seed, bool)):
        raise ValueError(f"seed must be an integer or None, got {seed!r}")

    backend = resolve_backend(backend)
    points = validation.check_prediction_points(prediction_points)
    df = validation.check_conditioning_data(conditioning_data, xx, yy, zz)
    validation.check_search(num_points, radius)
    if hasattr(variograms, "columns"):
        variograms = list(variograms["Variogram"])
    varios = [validation.check_variogram(v) for v in variograms]
    if cluster_col not in conditioning_data.columns:
        raise ValueError(f"conditioning_data is missing the cluster column {cluster_col!r}")
    labels = np.asarray(conditioning_data[cluster_col], dtype=np.float64)
    if (not np.all(np.isfinite(labels)) or np.any(labels != np.round(labels))
            or labels.min() < 0 or labels.max() >= len(varios)):
        raise ValueError(f"{cluster_col!r} must hold integer cluster ids from 0 to {len(varios) - 1}, "
                         "one per variogram")
    labels = labels.astype(np.int64)

    if len(points) == 0:
        out = np.empty((len(seeds), 0))
    elif backend == "cpu":
        import gstatsim as gs
        import pandas as pd

        dfk = df.assign(K=labels)
        gamma = pd.DataFrame({"Variogram": varios})
        out = np.stack([np.asarray(gs.Interpolation.cluster_sgs(
            points, dfk, "X", "Y", "Z", "K", num_points, gamma, radius, seed=s, quiet=quiet), dtype=np.float64)
            for s in seeds])
    else:
        from . import _sgs_cuda

        out = _sgs_cuda.cluster_sgs_ensemble(
            points, df["X"].to_numpy(), df["Y"].to_numpy(), df["Z"].to_numpy(), labels, varios,
            num_points, radius, seeds,
            # cluster_sgs uses np.average for the simple-kriging mean
            global_mean=float(np.average(df["Z"].to_numpy())))
    return out if many else out[0]
