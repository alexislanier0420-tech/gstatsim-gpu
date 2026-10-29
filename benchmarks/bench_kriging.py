"""Tutorial 3 kriging benchmark (GStatSim demo 3: simple/ordinary kriging).

Reproduces the notebook's workload: Greenland bed picks, gridded to `res`,
normal-score transformed, exponential variogram, k neighbors within `radius`.
Saves a JSON record (environment + config + timings) and the estimates as
.npz, so a later GPU run can be checked against this CPU reference.

Example:
    python benchmarks/bench_kriging.py --csv path/to/greenland_test_data.csv \
        --backend cpu --method simple --max-points 2000 --repeats 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import gstatsim as gs
from gstatsim_accel import bench, kriging


def prepare(csv: Path, res: float, value_col: str, max_value: float | None, seed: int):
    df = pd.read_csv(csv)
    if max_value is not None:
        # Tutorial 3 drops bed picks above 700 m as bad picks.
        df = df[df[value_col] <= max_value]
    df_grid, _, rows, cols = gs.Gridding.grid_data(df, "X", "Y", value_col, res)
    df_grid = df_grid[df_grid["Z"].notna()].reset_index(drop=True)

    from sklearn.preprocessing import QuantileTransformer
    # random_state pins the quantile subsample so reruns transform identically.
    nst = QuantileTransformer(n_quantiles=500, output_distribution="normal", random_state=seed)
    df_grid["NZ"] = nst.fit_transform(df_grid[["Z"]].to_numpy()).ravel()

    pred = gs.Gridding.prediction_grid(df_grid["X"].min(), df_grid["X"].max(),
                                       df_grid["Y"].min(), df_grid["Y"].max(), res)
    return df_grid, pred, (rows, cols)


def fit_variogram(df_grid, maxlag: float, n_lags: int) -> list:
    import skgstat as skg

    v = skg.Variogram(df_grid[["X", "Y"]].to_numpy(), df_grid["NZ"].to_numpy(),
                      bin_func="even", n_lags=n_lags, maxlag=maxlag, normalize=False)
    v.model = "exponential"
    rng, sill, nugget = (float(p) for p in v.parameters)
    return [0.0, nugget, rng, rng, sill, "Exponential"]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", type=Path, required=True, help="Tutorial 3 data (greenland_test_data.csv)")
    p.add_argument("--value-col", default="Bed")
    p.add_argument("--max-value", type=float, default=700.0, help="drop values above this (Tutorial 3: 700)")
    p.add_argument("--res", type=float, default=1000.0)
    p.add_argument("--method", choices=["simple", "ordinary"], default="simple")
    p.add_argument("--backend", choices=["cpu", "gpu"], default="cpu")
    p.add_argument("-k", "--num-points", type=int, default=100)
    p.add_argument("--radius", type=float, default=50_000.0)
    p.add_argument("--vario", nargs=6, metavar=("AZ", "NUG", "MAJ", "MIN", "SILL", "VTYPE"),
                   help="fixed variogram instead of fitting one with scikit-gstat")
    p.add_argument("--max-points", type=int, default=None,
                   help="krige only the first N prediction points (quick runs)")
    p.add_argument("--warmup", type=int, default=0)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("results/kriging"))
    p.add_argument("--notes", default="")
    a = p.parse_args(argv)

    df_grid, pred, shape = prepare(a.csv, a.res, a.value_col, a.max_value, a.seed)
    if a.vario:
        vario = [float(x) for x in a.vario[:5]] + [a.vario[5]]
    else:
        vario = fit_variogram(df_grid, maxlag=50_000, n_lags=70)
    if a.max_points is not None:
        pred = pred[: a.max_points]

    result = {}

    def run():
        result["est"], result["var"] = kriging.predict(
            df_grid, pred, vario, num_points=a.num_points, radius=a.radius,
            method=a.method, backend=a.backend, zz="NZ")

    timing = bench.time_call(run, backend=a.backend, warmup=a.warmup, repeats=a.repeats)

    config = {
        "workload": "tutorial3_kriging",
        "data_file": a.csv.name,  # name only: no local paths in shared records
        "value_col": a.value_col, "max_value": a.max_value, "res": a.res,
        "n_conditioning": int(len(df_grid)), "grid_shape": list(shape),
        "n_prediction_points": int(len(pred)), "method": a.method, "backend": a.backend,
        "num_points": a.num_points, "radius": a.radius, "variogram": vario,
        "nst_seed": a.seed, "warmup": a.warmup, "repeats": a.repeats,
    }
    name = f"tut3_{a.method}_{a.backend}_n{len(pred)}"
    path = bench.record(name, config, timing, a.out, notes=a.notes)
    np.savez_compressed(path.with_suffix(".npz"), pred=pred, est=result["est"], var=result["var"])

    per_pt_ms = 1000 * timing["median_s"] / max(len(pred), 1)
    print(f"{name}: median {timing['median_s']:.3f} s ({per_pt_ms:.3f} ms/point) -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
