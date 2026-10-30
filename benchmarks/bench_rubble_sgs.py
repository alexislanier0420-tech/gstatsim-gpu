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
    """Same steps as the notebook: grid radar + rock, RBF trend, normal scores, variogram."""
    d = np.load(ROOT / "figures" / "data" / "study_area.npz")
    data = pd.concat([
        pd.DataFrame({"X": d["radar_x"], "Y": d["radar_y"], "Bed": d["radar_bed"]}),
        pd.DataFrame({"X": d["rock_x"], "Y": d["rock_y"], "Bed": d["rock_bed"]}),
    ], ignore_index=True).astype(float)
    df_grid, grid_matrix, rows, cols = gs.Gridding.grid_data(data, "X", "Y", "Bed", res)
    xmin, xmax, ymin, ymax = df_grid.X.min(), df_grid.X.max(), df_grid.Y.min(), df_grid.Y.max()
    df_grid = df_grid[df_grid["Z"].notna()].rename(columns={"Z": "Bed"}).reset_index(drop=True)
    pred_xy = gs.Gridding.prediction_grid(xmin, xmax, ymin, ymax, res)

    trend = gs.rbf_trend(grid_matrix, 100_000, res)
    resid = np.flipud(grid_matrix - trend)
    df_grid["Residual"] = resid[~np.isnan(resid)]
    nst = QuantileTransformer(n_quantiles=500, output_distribution="normal", random_state=0)
    df_grid["NRes"] = nst.fit_transform(df_grid[["Residual"]].to_numpy()).ravel()

    ix = np.rint((pred_xy[:, 0] - d["x"][0]) / 500).astype(int)
    iy = np.rint((d["y"][0] - pred_xy[:, 1]) / 500).astype(int)
    land = np.isin(d["mask"][iy, ix], (1, 2))

    V = skg.Variogram(df_grid[["X", "Y"]].to_numpy(), df_grid["NRes"].to_numpy(),
                      bin_func="even", n_lags=70, maxlag=50_000, normalize=False)
    V.model = "exponential"
    vrange, sill, nugget = (float(p) for p in V.parameters)
    return df_grid, pred_xy[land], [0.0, nugget, vrange, vrange, sill, "Exponential"]


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
