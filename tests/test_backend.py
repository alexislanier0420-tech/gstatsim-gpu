import pytest

from gstatsim_accel import backend
from gstatsim_accel.backend import BackendUnavailableError, resolve_backend


def test_cpu_always_available():
    assert resolve_backend("cpu") == "cpu"
    assert resolve_backend("CPU") == "cpu"
    assert "cpu" in backend.available_backends()


@pytest.mark.parametrize("bad", ["cuda", "", None, 1])
def test_unknown_backend_rejected(bad):
    with pytest.raises(ValueError, match="backend must be one of"):
        resolve_backend(bad)


def test_missing_cupy_gives_actionable_error(monkeypatch):
    def no_cupy():
        raise ImportError("No module named 'cupy'")

    monkeypatch.setattr(backend, "_import_cupy", no_cupy)
    assert backend.available_backends() == ("cpu",)
    with pytest.raises(BackendUnavailableError, match="CuPy is not installed"):
        resolve_backend("gpu")


def test_cupy_without_device_gives_actionable_error(monkeypatch):
    class FakeRuntime:
        @staticmethod
        def getDeviceCount():
            raise RuntimeError("cudaErrorNoDevice")

    class FakeCupy:
        __version__ = "0.0-fake"

        class cuda:
            runtime = FakeRuntime

    monkeypatch.setattr(backend, "_import_cupy", lambda: FakeCupy)
    status = backend.gpu_status()
    assert not status.available
    assert "no CUDA device is usable" in status.reason
    with pytest.raises(BackendUnavailableError):
        resolve_backend("gpu")
