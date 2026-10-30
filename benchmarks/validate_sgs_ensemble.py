"""Criterion 4: SGS on a grid, GPU ensemble vs GStatSim ensembles.

On a grid, equidistant-neighbor ties make single realizations diverge from
GStatSim's, so whole ensembles are compared instead. Three ensembles with
non-overlapping seeds:
    A, B = GStatSim (two independent sets: their difference is the chance spread)
    C    = GPU
For each statistic, the GPU passes if |C - GStatSim| is no larger than the
GStatSim-vs-GStatSim spread, judged with a two-sample test on the per-realization
values (Welch t-test, alpha 0.01 after Bonferroni correction).

    python benchmarks/validate_sgs_ensemble.py --csv data/greenland_test_data.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from bench_kriging import prepare
from scipy import stats

from gstatsim_accel import simulation

LAGS = [2000, 5000, 10000, 20000]  # m, for the realization variogram


def semivariance(points, values, lag, tol=500.0):
    """Mean of 0.5 (z_i - z_j)^2 over pairs whose separation is within tol of lag."""
    d = np.hypot(points[:, None, 0] - points[None, :, 0], points[:, None, 1] - points[None, :, 1])
    i, j = np.nonzero(np.triu(np.abs(d - lag) <= tol, k=1))
    return 0.5 * np.mean((values[i] - values[j]) ** 2)


def summarize(points, real):
    """Per-realization statistics: mean, variance, semivariance at LAGS."""
    out = {"mean": real.mean(axis=1), "variance": real.var(axis=1)}
    for lag in LAGS:
        out[f"gamma_{lag // 1000}km"] = np.array([semivariance(points, r, lag) for r in real])
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--n", type=int, default=20, help="realizations per ensemble")
    p.add_argument("--box", type=int, default=30, help="grid side length (cells) of the test area")
    p.add_argument("--out", type=Path, default=Path("results/validation"))
    a = p.parse_args(argv)

    df, pred, (rows, cols) = prepare(a.csv, 1000.0, "Bed", 700.0, 0)
    vario = [0.0, 0.0, 31852.81332372462, 31852.81332372462, 0.7027482239926033, "Exponential"]
    r0, c0 = int(rows) // 3, int(cols) // 3
    pts = pred.reshape(int(rows), int(cols), 2)[r0:r0 + a.box, c0:c0 + a.box].reshape(-1, 2)
    kw = dict(num_points=100, radius=50_000, zz="NZ")

    ens = {}
    ensembles = (("A_gstatsim", 0, "cpu"), ("B_gstatsim", 1000, "cpu"), ("C_gpu", 2000, "gpu"))
    for name, start, backend in ensembles:
        seeds = list(range(start, start + a.n))
        print(f"{name}: {a.n} realizations x {len(pts)} nodes ...", flush=True)
        ens[name] = simulation.realizations(df, pts, vario, seeds=seeds, backend=backend, **kw)
    summaries = {k: summarize(pts, v) for k, v in ens.items()}

    n_tests = len(summaries["A_gstatsim"])
    alpha = 0.01 / n_tests
    rows_out, passed = [], True
    gstat = {k: np.concatenate([summaries["A_gstatsim"][k], summaries["B_gstatsim"][k]])
             for k in summaries["A_gstatsim"]}
    for stat in summaries["A_gstatsim"]:
        a_, b_, c_ = (summaries[e][stat] for e in ("A_gstatsim", "B_gstatsim", "C_gpu"))
        p_ab = stats.ttest_ind(a_, b_, equal_var=False).pvalue
        p_gpu = stats.ttest_ind(c_, gstat[stat], equal_var=False).pvalue
        ok = bool(p_gpu > alpha)
        passed &= ok
        rows_out.append({"statistic": stat, "gstatsim_A": [a_.mean(), a_.std()],
                         "gstatsim_B": [b_.mean(), b_.std()], "gpu": [c_.mean(), c_.std()],
                         "p_A_vs_B": p_ab, "p_gpu_vs_gstatsim": p_gpu, "pass": ok})
        print(f"{stat:12s} A {a_.mean():8.4f}±{a_.std():.4f}  B {b_.mean():8.4f}±{b_.std():.4f}  "
              f"GPU {c_.mean():8.4f}±{c_.std():.4f}  p(A,B)={p_ab:.3f}  p(GPU,GStatSim)={p_gpu:.3f}  "
              f"{'ok' if ok else 'DIFFERENT'}")
    print(f"Criterion 4 (alpha={alpha:.4f} per test): {'PASS' if passed else 'FAIL'}")

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "sgs_ensemble_criterion4.json").write_text(json.dumps(
        {"nodes": len(pts), "realizations_per_ensemble": a.n, "variogram": vario,
         "alpha_per_test": alpha, "pass": passed, "statistics": rows_out}, indent=2, default=float))
    np.savez_compressed(a.out / "sgs_ensembles.npz", points=pts, **ens)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
