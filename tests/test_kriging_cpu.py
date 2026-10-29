import numpy as np
import pandas as pd
import pytest

import gstatsim as gs
from gstatsim_accel import BackendUnavailableError, backend, kriging


@pytest.mark.parametrize("method", ["simple", "ordinary"])
def test_cpu_backend_matches_gstatsim_exactly(small_field, method):
    df, pred, vario = small_field
    est, var = kriging.predict(df, pred, vario, num_points=16, radius=5000, method=method)

    ref = gs.Interpolation.skrige if method == "simple" else gs.Interpolation.okrige
    ref_est, ref_var = ref(pred, df.copy(), "X", "Y", "Z", 16, vario, 5000, quiet=True)
    np.testing.assert_array_equal(est, ref_est)
    np.testing.assert_array_equal(var, ref_var)
    assert est.shape == var.shape == (len(pred),)
    assert np.all(var >= 0)


def test_custom_column_names_and_input_left_unchanged(small_field):
    df, pred, vario = small_field
    renamed = df.rename(columns={"X": "east", "Y": "north", "Z": "bed"})
    before = renamed.copy()
    est, _ = kriging.predict(renamed, pred, vario, num_points=16, radius=5000,
                             xx="east", yy="north", zz="bed")
    ref, _ = kriging.predict(df, pred, vario, num_points=16, radius=5000)
    np.testing.assert_array_equal(est, ref)
    pd.testing.assert_frame_equal(renamed, before)


def test_point_on_datum_returns_datum(small_field):
    df, _, vario = small_field
    pred = df[["X", "Y"]].to_numpy()[:3]
    est, var = kriging.predict(df, pred, vario, num_points=16, radius=5000)
    np.testing.assert_array_equal(est, df["Z"].to_numpy()[:3])
    np.testing.assert_array_equal(var, 0)


def test_empty_prediction_points(small_field):
    df, _, vario = small_field
    est, var = kriging.predict(df, np.empty((0, 2)), vario, num_points=16, radius=5000)
    assert est.shape == var.shape == (0,)


def test_gpu_request_never_falls_back_to_cpu(small_field, monkeypatch):
    df, pred, vario = small_field

    def no_cupy():
        raise ImportError("No module named 'cupy'")

    monkeypatch.setattr(backend, "_import_cupy", no_cupy)
    with pytest.raises(BackendUnavailableError):
        kriging.predict(df, pred, vario, num_points=16, radius=5000, backend="gpu")
