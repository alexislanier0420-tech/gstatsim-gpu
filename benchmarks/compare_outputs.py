"""Compare two kriging outputs saved by bench_kriging.py (e.g. CPU vs GPU).

    python benchmarks/compare_outputs.py results/kriging/<cpu>.npz results/kriging/<gpu>.npz --atol 1e-6
"""

from __future__ import annotations

import argparse
import sys

import numpy as np


def compare(ref: dict, other: dict) -> dict:
    if not np.array_equal(ref["pred"], other["pred"]):
        raise ValueError("prediction points differ; the runs are not comparable")
    out = {}
    for key in ("est", "var"):
        diff = np.abs(ref[key] - other[key])
        out[key] = {
            "max_abs": float(diff.max()) if diff.size else 0.0,
            "mean_abs": float(diff.mean()) if diff.size else 0.0,
            "p99_abs": float(np.percentile(diff, 99)) if diff.size else 0.0,
        }
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("reference")
    p.add_argument("candidate")
    p.add_argument("--atol", type=float, required=True,
                   help="max allowed absolute difference (normal-score units); set it before looking")
    a = p.parse_args(argv)

    stats = compare(dict(np.load(a.reference)), dict(np.load(a.candidate)))
    ok = all(s["max_abs"] <= a.atol for s in stats.values())
    for key, s in stats.items():
        print(f"{key}: max {s['max_abs']:.3e}  p99 {s['p99_abs']:.3e}  mean {s['mean_abs']:.3e}")
    print("PASS" if ok else f"FAIL (atol={a.atol})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
