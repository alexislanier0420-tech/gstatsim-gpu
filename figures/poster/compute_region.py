"""Compute and save the large-region (Amundsen Sea Embayment) kriged bed for the poster.

Same inputs as benchmarks/bench_region.py: region_workflow("ase", 1000), then GPU
simple kriging of every grounded node (as in the benchmark), back-transformed and with the trend added.
Writes results/poster/ase.npz.

    python figures/poster/compute_region.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks"))
from bench_rubble_sgs import region_workflow  # noqa: E402

from gstatsim_accel import kriging  # noqa: E402

KW = dict(num_points=100, radius=50_000, zz="NRes")


def main(region="ase", res=1000) -> int:
    w = region_workflow(region, res)
    df, pred, nst, trend, land = w["df_grid"], w["pred_xy"], w["nst"], w["trend"], w["land"]
    nodes = np.flatnonzero(land)

    t = time.perf_counter()
    est, var = kriging.predict(df, pred[nodes], w["vario"], backend="gpu", **KW)
    print(f"kriging {len(nodes):,} nodes: {time.perf_counter() - t:.1f} s", flush=True)

    d = w["study"]
    out = ROOT / "results" / "poster"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / f"{region}.npz",
        pred_xy=pred, rows=w["rows"], cols=w["cols"], land=land, nodes=nodes, res=res,
        sk_bed=nst.inverse_transform(est.reshape(-1, 1)).ravel() + trend.ravel()[nodes],
        sk_var=var, data_x=df["X"].to_numpy(), data_y=df["Y"].to_numpy(),
        x=d["x"], y=d["y"], mask=d["mask"],
    )
    print("wrote", out / f"{region}.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
