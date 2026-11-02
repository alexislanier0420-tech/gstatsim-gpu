"""Speed measurements for the poster, all on one machine in one run.

GStatSim (CPU) vs this project (GPU) on four workloads, plus how the time per
SGS realization falls as more realizations run side by side on the GPU.
Writes results/poster/speed.json. Keep the computer awake while it runs
(~20 min, mostly GStatSim).

    python figures/poster/compute_speed.py
    python figures/poster/compute_speed.py --out /blue/.../speed_hpg_l4.json   # HiPerGator
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from functools import partial
from pathlib import Path

import cupy as cp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks"))
from bench_kriging import fit_variogram, prepare  # noqa: E402
from bench_rubble_sgs import rubble_workflow  # noqa: E402

from gstatsim_accel import envinfo, kriging, simulation  # noqa: E402

KW = dict(num_points=100, radius=50_000)


def timed(fn, sync=True):
    if sync:
        cp.cuda.Device().synchronize()
    t = time.perf_counter()
    fn()
    if sync:
        cp.cuda.Device().synchronize()
    return time.perf_counter() - t


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=ROOT / "results" / "poster" / "speed.json")
    args = p.parse_args(argv)
    out = {"environment": envinfo.collect(), "workloads": [], "scaling": {}}

    tut, pred3, _ = prepare(ROOT / "data" / "greenland_test_data.csv", 1000.0, "Bed", 700.0, 0)
    v3 = fit_variogram(tut, 50_000, 70)
    w = rubble_workflow()
    rub, predr, land, vr = w["df_grid"], w["pred_xy"], w["land"], w["vario"]
    nodes = predr[land]
    cases = [
        ("Kriging · Tutorial 3", "kriging", tut, pred3, v3, "NZ"),
        ("SGS · Tutorial 3", "sgs", tut, pred3, v3, "NZ"),
        ("Kriging · Rubble Glacier", "kriging", rub, predr, vr, "NRes"),
        ("SGS · Rubble Glacier", "sgs", rub, nodes, vr, "NRes"),
    ]
    # GPU warm-up (CUDA context, kernel compilation) kept out of the timings
    kriging.predict(tut, pred3[:200], v3, backend="gpu", zz="NZ", **KW)
    simulation.sgs(tut, pred3[:200], v3, seed=0, backend="gpu", zz="NZ", **KW)

    for label, kind, df, pts, vario, zz in cases:
        print(label, f"({len(pts):,} cells)", flush=True)
        if kind == "kriging":
            krige = partial(kriging.predict, df, pts, vario, zz=zz, **KW)
            cpu = timed(partial(krige, backend="cpu"), sync=False)
            gpu = min(timed(partial(krige, backend="gpu")) for _ in range(3))
            row = {"label": label, "cells": len(pts), "gstatsim_s": cpu, "gpu_s": gpu}
        else:
            sgs = partial(simulation.sgs, df, pts, vario, seed=0, zz=zz, **KW)
            cpu = timed(partial(sgs, backend="cpu"), sync=False)
            one = timed(partial(sgs, backend="gpu"))
            many = timed(partial(simulation.realizations, df, pts, vario, seeds=list(range(32)),
                                 backend="gpu", zz=zz, **KW)) / 32
            row = {"label": label, "cells": len(pts), "gstatsim_s": cpu, "gpu_s": one,
                   "gpu_per_realization_32_s": many}
        out["workloads"].append(row)
        print("   ", {k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()}, flush=True)

    # Time per realization vs. how many run side by side (Rubble Glacier SGS)
    for S in (1, 2, 4, 8, 16, 32, 64):
        t = timed(lambda S=S: simulation.realizations(rub, nodes, vr, seeds=list(range(S)),
                                                      backend="gpu", zz="NRes", **KW))
        out["scaling"][S] = t / S
        print(f"    {S:3d} side by side: {t / S:.2f} s per realization", flush=True)

    path = args.out
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, default=str))
    print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
