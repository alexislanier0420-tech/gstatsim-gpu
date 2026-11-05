"""GPU vs CPU on a large region (default: Amundsen Sea Embayment, Thwaites + Pine Island).

Prepares the inputs with region_workflow (bench_rubble_sgs.py), then times one part:

    --part gpu   GPU kriging of every node, then SGS with each --batches count of
                 realizations stepped together (for an L4 GPU job)
    --part cpu   our NumPy version (kriging, one SGS realization), then GStatSim
                 itself (kriging, one SGS realization), slowest last (for a CPU-only job)

The record is rewritten after every step, so a job that hits its time limit keeps what
finished. GStatSim runs with its progress bar on, so the log shows its rate even then.

    python benchmarks/bench_region.py --part gpu --res 1000 --out results/hpg/region_gpu_1km.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_rubble_sgs import region_workflow  # noqa: E402

from gstatsim_accel import _batched, _sgs, envinfo, kriging, simulation  # noqa: E402

KW = dict(num_points=100, radius=50_000)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--region", default="ase")
    p.add_argument("--res", type=float, default=1000.0)
    p.add_argument("--part", choices=["gpu", "cpu"], required=True)
    p.add_argument("--batches", type=int, nargs="+", default=[1, 32, 64],
                   help="GPU part: realizations stepped together, one SGS run per value")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-nodes", type=int, help="time only the first N nodes (smoke test)")
    a = p.parse_args(argv)

    t = time.perf_counter()
    w = region_workflow(a.region, a.res)
    df, nodes, vario = w["df_grid"], w["pred_xy"][w["land"]], w["vario"]
    nodes = nodes[: a.max_nodes]
    X, Y, Z = (df[c].to_numpy() for c in ("X", "Y", "NRes"))
    mean = float(df["NRes"].mean())
    out = {"environment": envinfo.collect(), "region": a.region, "res_m": a.res, "part": a.part,
           "n_conditioning": len(df), "n_nodes": len(nodes), "variogram": vario, **KW,
           "prepare_s": time.perf_counter() - t, "steps": {}}
    print(f"{a.region} at {a.res:.0f} m: {len(df):,} gridded data, {len(nodes):,} nodes, "
          f"prepared in {out['prepare_s']:.0f} s", flush=True)

    def save():
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(out, indent=2, default=str))

    def step(name, fn, sync=False, realizations=None):
        print(f"{name} ...", flush=True)
        if sync:
            import cupy as cp
            cp.cuda.Device().synchronize()
        t = time.perf_counter()
        fn()
        if sync:
            cp.cuda.Device().synchronize()
        dt = time.perf_counter() - t
        rec = {"seconds": dt, "ms_per_node": dt / len(nodes) * 1e3}
        if realizations:
            rec |= {"realizations": realizations, "seconds_per_realization": dt / realizations}
        out["steps"][name] = rec
        save()
        print(f"    {dt:.1f} s" + (f" ({dt / realizations:.1f} s per realization)" if realizations else ""),
              flush=True)

    kw = dict(zz="NRes", **KW)
    if a.part == "gpu":
        # Warm-up (CUDA context, kernel compilation) kept out of the timings
        kriging.predict(df, nodes[:200], vario, backend="gpu", **kw)
        simulation.realizations(df, nodes[:200], vario, seeds=[0], backend="gpu", **kw)
        step("gpu_kriging", lambda: kriging.predict(df, nodes, vario, backend="gpu", **kw), sync=True)
        for S in a.batches:
            step(f"gpu_sgs_x{S}", lambda S=S: simulation.realizations(
                df, nodes, vario, seeds=list(range(S)), backend="gpu", **kw), sync=True, realizations=S)
    else:
        step("numpy_kriging", lambda: _batched.krige(np, nodes, X, Y, Z, vario, KW["num_points"],
                                                     KW["radius"], False, mean))
        step("gstatsim_kriging", lambda: kriging.predict(df, nodes, vario, backend="cpu", quiet=False, **kw))
        step("numpy_sgs_x1", lambda: _sgs.sgs(np, nodes, X, Y, Z, vario, KW["num_points"], KW["radius"],
                                              False, 0, mean), realizations=1)
        step("gstatsim_sgs_x1", lambda: simulation.sgs(df, nodes, vario, seed=0, backend="cpu", quiet=False,
                                                       **kw), realizations=1)
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
