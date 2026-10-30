"""Opt-in CPU/GPU backends for GStatSim workflows."""

from . import kriging, simulation
from .backend import BackendUnavailableError, available_backends, gpu_status

__all__ = ["kriging", "simulation", "BackendUnavailableError", "available_backends", "gpu_status"]
__version__ = "0.1.0"
