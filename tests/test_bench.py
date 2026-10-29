import json
import sys
from pathlib import Path

import numpy as np

from gstatsim_accel import bench, envinfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import compare_outputs  # noqa: E402


def test_time_call_counts_runs():
    calls = []
    t = bench.time_call(lambda: calls.append(1), backend="cpu", warmup=2, repeats=3)
    assert len(calls) == 5
    assert len(t["warmup_s"]) == 2 and len(t["runs_s"]) == 3
    assert t["min_s"] <= t["median_s"]


def test_record_has_environment_and_no_correctness_claim(tmp_path):
    path = bench.record("unit", {"n": 1}, {"median_s": 0.1}, tmp_path)
    rec = json.loads(path.read_text())
    assert rec["correctness"] is None
    for key in ("python", "os", "packages", "cupy_gpu"):
        assert key in rec["environment"]
    assert "numpy" in rec["environment"]["packages"]


def test_envinfo_has_no_hostname():
    import platform
    text = json.dumps(envinfo.collect())
    assert platform.node() not in text


def test_compare_outputs():
    pred = np.zeros((3, 2))
    a = {"pred": pred, "est": np.array([1.0, 2.0, 3.0]), "var": np.zeros(3)}
    b = {"pred": pred, "est": np.array([1.0, 2.0, 3.5]), "var": np.zeros(3)}
    stats = compare_outputs.compare(a, b)
    assert stats["est"]["max_abs"] == 0.5
    assert stats["var"]["max_abs"] == 0.0
