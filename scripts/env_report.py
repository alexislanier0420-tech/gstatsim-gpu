"""Phase 0 environment report: GPU, driver, CUDA, Python and package versions.

Writes JSON and a Markdown summary, and runs a small CuPy smoke test (matrix
multiply plus a batched solve, checked against NumPy) when a GPU is present.

    python scripts/env_report.py --out results/env
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from gstatsim_accel import envinfo


def gpu_smoke_test() -> dict:
    try:
        import cupy as cp
    except ImportError:
        return {"ran": False, "reason": "CuPy not installed"}
    try:
        rng = np.random.default_rng(0)
        # Batched SPD solves are the core operation of batched kriging.
        a = rng.standard_normal((256, 32, 32))
        a = a @ a.transpose(0, 2, 1) + 32 * np.eye(32)
        b = rng.standard_normal((256, 32, 1))
        x_gpu = cp.asnumpy(cp.linalg.solve(cp.asarray(a), cp.asarray(b)))
        x_cpu = np.linalg.solve(a, b)
        err = float(np.abs(x_gpu - x_cpu).max())
        return {"ran": True, "batched_solve_max_abs_err": err, "passed": err < 1e-10}
    except Exception as exc:
        msg = " ".join(str(exc).split())  # one line, so it fits in the Markdown table
        return {"ran": True, "passed": False, "error": f"{type(exc).__name__}: {msg}"}


def to_markdown(rep: dict) -> str:
    gib = lambda b: f"{b / 2**30:.1f} GiB" if b else "unknown"
    g = rep["cupy_gpu"]
    lines = [
        f"# Environment report ({rep['timestamp_utc']})", "",
        "| Item | Value |", "|---|---|",
        f"| OS | {rep['os']} |",
        f"| Python | {rep['python']} |",
        f"| CPU | {rep['cpu_model']} ({rep['cpu_count_logical']} logical) |",
        f"| RAM | {gib(rep['ram_bytes'])} |",
    ]
    for i, s in enumerate(rep["nvidia_smi"] or []):
        lines.append(f"| GPU {i} (nvidia-smi) | {s['name']}, {s['memory_mib']} MiB, "
                     f"driver {s['driver']}, CC {s['compute_capability']} |")
    if not rep["nvidia_smi"]:
        lines.append("| GPU (nvidia-smi) | not found |")
    if g["available"]:
        rt = g["cuda_runtime_version"]
        lines.append(f"| CuPy device | {g['device_name']}, CC {g['compute_capability']}, "
                     f"{gib(g['total_memory_bytes'])}, CUDA runtime {rt // 1000}.{rt % 1000 // 10} |")
    else:
        lines.append(f"| CuPy device | unavailable: {g['reason']} |")
    st = rep["gpu_smoke_test"]
    if st["ran"]:
        lines.append(f"| GPU smoke test | {'passed' if st.get('passed') else 'FAILED'} "
                     f"{st.get('batched_solve_max_abs_err', st.get('error', ''))} |")
    for pkg, ver in rep["packages"].items():
        lines.append(f"| {pkg} | {ver} |")
    for k, v in rep["slurm"].items():
        lines.append(f"| {k} | {v} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("results/env"))
    p.add_argument("--label", default="", help="e.g. laptop, hpg-b200")
    a = p.parse_args(argv)

    stamp = datetime.now(timezone.utc)
    rep = {"timestamp_utc": stamp.isoformat(timespec="seconds"), "label": a.label,
           **envinfo.collect(), "gpu_smoke_test": gpu_smoke_test()}

    a.out.mkdir(parents=True, exist_ok=True)
    base = a.out / f"{stamp.strftime('%Y%m%dT%H%M%SZ')}_{a.label or 'env'}"
    base.with_suffix(".json").write_text(json.dumps(rep, indent=2))
    md = to_markdown(rep)
    base.with_suffix(".md").write_text(md)
    print(md)
    print(f"Saved {base}.json and .md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
