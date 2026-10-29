"""Benchmark harness: warm-up, repeated timing, and a JSON record per run.

Kept separate from the notebook API so production calls carry no timing
overhead. Each record holds the environment, the workload configuration and
the raw timings, so a run can be compared or repeated later.
"""

from __future__ import annotations

import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from . import envinfo


def _gpu_sync() -> None:
    # GPU work is asynchronous; without a sync the timer stops before the kernels finish.
    import cupy as cp
    cp.cuda.Device().synchronize()


def _gpu_pool_bytes() -> int | None:
    try:
        import cupy as cp
    except ImportError:
        return None
    return int(cp.get_default_memory_pool().total_bytes())


def time_call(fn: Callable[[], Any], *, backend: str, warmup: int = 1, repeats: int = 5) -> dict:
    """Run fn warmup + repeats times; return wall times in seconds.

    Warm-up runs absorb one-off costs (CUDA context, kernel compilation) and
    are reported separately rather than dropped, since a notebook user pays them once.
    """
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    sync = _gpu_sync if backend == "gpu" else (lambda: None)

    def timed() -> float:
        t0 = time.perf_counter()
        fn()
        sync()
        return time.perf_counter() - t0

    warmup_s = [timed() for _ in range(warmup)]
    runs_s = [timed() for _ in range(repeats)]
    return {
        "warmup_s": warmup_s,
        "runs_s": runs_s,
        "median_s": statistics.median(runs_s),
        "min_s": min(runs_s),
        "stdev_s": statistics.stdev(runs_s) if len(runs_s) > 1 else 0.0,
        "gpu_pool_bytes": _gpu_pool_bytes() if backend == "gpu" else None,
    }


def record(name: str, config: dict, timing: dict, out_dir: str | Path,
           correctness: dict | None = None, notes: str = "") -> Path:
    """Write one benchmark record to out_dir/<timestamp>_<name>.json."""
    stamp = datetime.now(timezone.utc)
    rec = {
        "name": name,
        "timestamp_utc": stamp.isoformat(timespec="seconds"),
        "config": config,
        "timing": timing,
        # None means "not checked"; a benchmark without a correctness check
        # must not be quoted as a speedup.
        "correctness": correctness,
        "environment": envinfo.collect(),
        "notes": notes,
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{stamp.strftime('%Y%m%dT%H%M%SZ')}_{name}.json"
    path.write_text(json.dumps(rec, indent=2, default=str))
    return path
