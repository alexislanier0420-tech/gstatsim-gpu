"""Opt-in CPU/GPU backends for GStatSim workflows."""

from . import kriging
from .backend import BackendUnavailableError, available_backends, gpu_status

__all__ = ["kriging", "BackendUnavailableError", "available_backends", "gpu_status"]
__version__ = "0.1.0"
