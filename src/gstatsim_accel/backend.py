"""Backend selection and GPU availability checks.

The CPU backend is always available. The GPU backend needs CuPy and a working
CUDA device; when either is missing, an explicit GPU request fails with
BackendUnavailableError instead of silently running on the CPU.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass

BACKENDS = ("cpu", "gpu")


class BackendUnavailableError(RuntimeError):
    """Raised when a requested backend cannot run in this environment."""


@dataclass(frozen=True)
class GpuStatus:
    available: bool
    reason: str
    device_name: str | None = None
    compute_capability: str | None = None
    total_memory_bytes: int | None = None
    cuda_runtime_version: int | None = None
    cupy_version: str | None = None


def _import_cupy():
    # Indirection so tests can simulate a missing or broken CuPy install.
    return importlib.import_module("cupy")


def gpu_status() -> GpuStatus:
    """Probe for CuPy and a usable CUDA device without raising."""
    try:
        cp = _import_cupy()
    except ImportError as exc:
        return GpuStatus(
            False,
            f"CuPy is not installed ({exc}). Install the 'gpu' extra, "
            "e.g. pip install -e .[gpu]",
        )

    try:
        count = cp.cuda.runtime.getDeviceCount()
    except Exception as exc:  # CUDARuntimeError, missing driver, etc.
        return GpuStatus(
            False,
            f"CuPy is installed but no CUDA device is usable ({exc}). "
            "Check the NVIDIA driver, or that the job was given a GPU "
            "(e.g. SLURM --gpus=1).",
            cupy_version=cp.__version__,
        )
    if count == 0:
        return GpuStatus(False, "No CUDA devices found.", cupy_version=cp.__version__)

    device = cp.cuda.Device(0)
    props = cp.cuda.runtime.getDeviceProperties(device.id)
    name = props["name"]
    if isinstance(name, bytes):
        name = name.decode()
    major, minor = device.compute_capability[:-1], device.compute_capability[-1]
    return GpuStatus(
        True,
        "ok",
        device_name=name,
        compute_capability=f"{major}.{minor}",
        total_memory_bytes=int(props["totalGlobalMem"]),
        cuda_runtime_version=int(cp.cuda.runtime.runtimeGetVersion()),
        cupy_version=cp.__version__,
    )


def available_backends() -> tuple[str, ...]:
    """Backends that can run here, CPU first."""
    return ("cpu", "gpu") if gpu_status().available else ("cpu",)


def resolve_backend(backend: str) -> str:
    """Validate a backend name and confirm it can run; returns the name."""
    if not isinstance(backend, str) or backend.lower() not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
    backend = backend.lower()
    if backend == "gpu":
        status = gpu_status()
        if not status.available:
            raise BackendUnavailableError(f"GPU backend requested but unavailable: {status.reason}")
    return backend
