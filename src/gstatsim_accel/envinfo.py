"""Collect the hardware and software details every benchmark must record.

Deliberately leaves out hostnames, usernames and file paths so reports can be
shared (RFC section 9).
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from dataclasses import asdict
from importlib import metadata

from .backend import gpu_status

PACKAGES = (
    "gstatsim", "gstatsim-accel", "numpy", "scipy", "pandas", "scikit-learn",
    "cupy", "cupy-cuda12x", "cupy-cuda13x", "psutil",
)

# Scheduler context is useful on HiPerGator; job IDs and partitions are not sensitive.
SLURM_VARS = ("SLURM_JOB_ID", "SLURM_JOB_PARTITION", "SLURM_CPUS_PER_TASK",
              "SLURM_GPUS", "SLURM_JOB_GPUS", "SLURM_MEM_PER_NODE")


def _cpu_model() -> str:
    if platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or platform.machine()


def _ram_bytes() -> int | None:
    try:
        import psutil
        return int(psutil.virtual_memory().total)
    except ImportError:
        pass
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, ValueError, OSError):
        return None


def _nvidia_smi() -> list[dict] | None:
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    fields = "name,memory.total,driver_version,compute_cap"
    try:
        out = subprocess.run(
            [exe, f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20, check=True,
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return None
    gpus = []
    for line in out.strip().splitlines():
        name, mem, driver, cc = (p.strip() for p in line.split(","))
        gpus.append({"name": name, "memory_mib": int(mem), "driver": driver,
                     "compute_capability": cc})
    return gpus


def _versions() -> dict[str, str]:
    found = {}
    for pkg in PACKAGES:
        try:
            found[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            pass
    return found


def collect() -> dict:
    return {
        "os": platform.platform(),
        "python": platform.python_version(),
        "cpu_model": _cpu_model(),
        "cpu_count_logical": os.cpu_count(),
        "ram_bytes": _ram_bytes(),
        "nvidia_smi": _nvidia_smi(),
        "cupy_gpu": asdict(gpu_status()),
        "packages": _versions(),
        "slurm": {k: os.environ[k] for k in SLURM_VARS if k in os.environ},
    }
