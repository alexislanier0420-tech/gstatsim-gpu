"""SGS on the Rubble Glacier inputs: one full GStatSim realization vs the GPU.

Prepares the inputs exactly as notebooks/rubble_glacier_workflow.ipynb
(sections 1-3), then times GStatSim (one realization) and the GPU with 1 and
with N realizations stepped together. Run from the repo root:

    python benchmarks/bench_rubble_sgs.py --gpu-ensemble 50
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import gstatsim as gs
import numpy as np
import pandas as pd
import skgstat as skg
from sklearn.preprocessing import QuantileTransformer

from gstatsim_accel import bench, simulation

ROOT = Path(__file__).resolve().parents[1]


def prepare_rubble(res=1000):
    """Same steps as the notebook: grid radar + rock, RBF trend, normal scores, variogram.
    Returns (gridded data, simulation nodes, variogram)."""
    w = rubble_workflow(res)
    return w["df_grid"], w["pred_xy"][w["land"]], w["vario"]


def rubble_workflow(res=1000):
    """Everything sections 1-3 of the notebook produce, as a dict: gridded data,
    prediction grid and its shape, trend, normal-score transform, land mask,
    variogram, and the study-area arrays."""
    return region_workflow("rubble", res)


# Above this many gridded data points, GStatSim's rbf_trend (dense N x N RBF system)
# and the all-pairs variogram no longer fit in memory, so both are done coarser.
LARGE = 15_000


def coarse_rbf_trend(data, res, shape, trend_res=10_000, smooth=100_000):
    """GStatSim's rbf_trend fitted on a trend_res grid, then interpolated to the res grid.

    The trend is smooth over ~smooth metres, so fitting it on a 10 km grid changes it
    very little, and the RBF system shrinks from N^2 to (N / 100)^2 entries.
    Returns the trend on the same top-down grid as grid_data's grid_matrix.
    """
    from scipy.interpolate import RegularGridInterpolator

    _, coarse, rows_c, cols_c = gs.Gridding.grid_data(data, "X", "Y", "Bed", trend_res)
    # GStatSim 1.2.0's rbf_trend evaluates on meshgrid(arange(ny), arange(nx)), which is
    # only right for square grids: pad with NaN (ignored by the fit) to square, then crop.
    n = max(rows_c, cols_c)
    square = np.full((n, n), np.nan)
    square[:rows_c, :cols_c] = coarse
    tc = np.flipud(gs.rbf_trend(square, smooth, trend_res)[:rows_c, :cols_c])   # bottom-up
    x0, y0 = data["X"].min(), data["Y"].min()
    f = RegularGridInterpolator((y0 + np.arange(rows_c) * trend_res, x0 + np.arange(cols_c) * trend_res),
                                tc, bounds_error=False, fill_value=None)
    rows, cols = shape
    yy, xx = np.meshgrid(y0 + np.arange(rows) * res, x0 + np.arange(cols) * res, indexing="ij")
    return np.flipud(f(np.column_stack([yy.ravel(), xx.ravel()])).reshape(rows, cols))


def region_workflow(region="rubble", res=1000):
    """region_workflow("rubble") is the notebook's workflow. Other regions come from
    figures/prep_map_data.py <data dir> <region>; large ones fit the trend on a 10 km
    grid and the variogram on a 10,000-point sample (see LARGE)."""
    name = "study_area.npz" if region == "rubble" else f"study_area_{region}.npz"
    d = np.load(ROOT / "figures" / "data" / name)
    data = pd.concat([
        pd.DataFrame({"X": d["radar_x"], "Y": d["radar_y"], "Bed": d["radar_bed"]}),
        pd.DataFrame({"X": d["rock_x"], "Y": d["rock_y"], "Bed": d["rock_bed"]}),
    ], ignore_index=True).astype(float)
    df_grid, grid_matrix, rows, cols = gs.Gridding.grid_data(data, "X", "Y", "Bed", res)
    xmin, xmax, ymin, ymax = df_grid.X.min(), df_grid.X.max(), df_grid.Y.min(), df_grid.Y.max()
    df_grid = df_grid[df_grid["Z"].notna()].rename(columns={"Z": "Bed"}).reset_index(drop=True)
    pred_xy = gs.Gridding.prediction_grid(xmin, xmax, ymin, ymax, res)

    large = len(df_grid) > LARGE
    trend = (coarse_rbf_trend(data, res, grid_matrix.shape) if large
             else gs.rbf_trend(grid_matrix, 100_000, res))
    resid = np.flipud(grid_matrix - trend)
    df_grid["Residual"] = resid[~np.isnan(resid)]
    nst = QuantileTransformer(n_quantiles=500, output_distribution="normal", random_state=0)
    df_grid["NRes"] = nst.fit_transform(df_grid[["Residual"]].to_numpy()).ravel()

    ix = np.rint((pred_xy[:, 0] - d["x"][0]) / 500).astype(int)
    iy = np.rint((d["y"][0] - pred_xy[:, 1]) / 500).astype(int)
    land = np.isin(d["mask"][iy, ix], (1, 2))

    vdf = df_grid.sample(10_000, random_state=0) if large else df_grid
    V = skg.Variogram(vdf[["X", "Y"]].to_numpy(), vdf["NRes"].to_numpy(),
                      bin_func="even", n_lags=70, maxlag=50_000, normalize=False)
    V.model = "exponential"
    vrange, sill, nugget = (float(p) for p in V.parameters)
    return {"df_grid": df_grid, "pred_xy": pred_xy, "rows": int(rows), "cols": int(cols),
            "trend": trend, "nst": nst, "land": land, "node_ix": ix, "node_iy": iy,
            "vario": [0.0, nugget, vrange, vrange, sill, "Exponential"], "variogram_model": V,
            "study": d}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gpu-ensemble", type=int, default=50)
    p.add_argument("--skip-gstatsim", action="store_true")
    p.add_argument("--out", type=Path, default=ROOT / "results" / "sgs")
    a = p.parse_args(argv)

    df_grid, nodes, vario = prepare_rubble()
    kw = dict(num_points=100, radius=50_000, zz="NRes")
    config = {"workload": "rubble_sgs", "n_conditioning": len(df_grid), "n_nodes": len(nodes),
              "variogram": vario, "num_points": 100, "radius": 50_000}
    print(f"{len(df_grid):,} conditioning cells, {len(nodes):,} nodes", flush=True)

    runs = [("gpu", [0]), ("gpu", list(range(a.gpu_ensemble)))]
    if not a.skip_gstatsim:
        runs.insert(0, ("cpu", [0]))
    simulation.realizations(df_grid, nodes[:200], vario, seeds=[0], backend="gpu", **kw)  # GPU warm-up
    for backend, seeds in runs:
        t = time.perf_counter()
        simulation.realizations(df_grid, nodes, vario, seeds=seeds, backend=backend, **kw)
        dt = time.perf_counter() - t
        name = f"rubble_sgs_{'gstatsim' if backend == 'cpu' else 'gpu'}_x{len(seeds)}"
        timing = {"runs_s": [dt], "median_s": dt, "per_realization_s": dt / len(seeds)}
        bench.record(name, {**config, "backend": backend, "realizations": len(seeds)}, timing, a.out)
        print(f"{name}: {dt:.1f} s total, {dt / len(seeds):.1f} s per realization", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
