"""Compute and save the Rubble Glacier results the poster figures draw.

Same workflow as notebooks/rubble_glacier_workflow.ipynb: GPU simple kriging of
the whole grid, then an ensemble of SGS realizations on rock and grounded ice,
back-transformed and with the trend added. Writes results/poster/rubble.npz.

    python figures/poster/compute_results.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks"))
from bench_rubble_sgs import rubble_workflow  # noqa: E402

from gstatsim_accel import kriging, simulation  # noqa: E402

N_REAL = 20
KW = dict(num_points=100, radius=50_000, zz="NRes")


def main() -> int:
    w = rubble_workflow()
    df, pred, land, nst, trend = w["df_grid"], w["pred_xy"], w["land"], w["nst"], w["trend"]
    rows, cols = w["rows"], w["cols"]

    t = time.perf_counter()
    est, var = kriging.predict(df, pred, w["vario"], backend="gpu", **KW)
    print(f"kriging: {time.perf_counter() - t:.1f} s", flush=True)

    nodes = np.flatnonzero(land)
    t = time.perf_counter()
    sims = simulation.realizations(df, pred[nodes], w["vario"], seeds=list(range(N_REAL)),
                                   backend="gpu", **KW)
    print(f"{N_REAL} realizations: {time.perf_counter() - t:.1f} s", flush=True)

    def to_bed(normal_scores, idx):
        shape = normal_scores.shape
        return nst.inverse_transform(normal_scores.reshape(-1, 1)).reshape(shape) + trend.ravel()[idx]

    d = w["study"]
    out = ROOT / "results" / "poster"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "rubble.npz",
        pred_xy=pred, rows=rows, cols=cols, land=land, nodes=nodes,
        sk_bed=to_bed(est, np.arange(len(pred))), sk_var=var,
        sims=sims, beds=to_bed(sims, nodes),
        bedmachine=d["bedmachine_bed"][w["node_iy"], w["node_ix"]].astype(float),
        data_x=df["X"].to_numpy(), data_y=df["Y"].to_numpy(), data_nres=df["NRes"].to_numpy(),
        vario=np.array(w["vario"][:5], dtype=float),
        vario_bins=w["variogram_model"].bins, vario_exp=w["variogram_model"].experimental,
    )
    print("wrote", out / "rubble.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
