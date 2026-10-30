import numpy as np
import pandas as pd
import pytest

from gstatsim_accel import kriging

GOOD_VARIO = [0, 0.05, 4000.0, 4000.0, 1.0, "Exponential"]


def predict(df=None, pred=None, vario=None, **kw):
    df = pd.DataFrame({"X": [0.0, 100.0], "Y": [0.0, 100.0], "Z": [1.0, 2.0]}) if df is None else df
    pred = np.array([[50.0, 50.0]]) if pred is None else pred
    kw.setdefault("num_points", 8)
    kw.setdefault("radius", 1000.0)
    return kriging.predict(df, pred, GOOD_VARIO if vario is None else vario, **kw)


@pytest.mark.parametrize("pred, msg", [
    (np.zeros((3, 3)), "shape"),
    (np.zeros(4), "shape"),
    (np.array([[np.nan, 0.0]]), "NaN"),
])
def test_bad_prediction_points(pred, msg):
    with pytest.raises(ValueError, match=msg):
        predict(pred=pred)


@pytest.mark.parametrize("df, err, msg", [
    (np.zeros((2, 3)), TypeError, "DataFrame"),
    (pd.DataFrame({"X": [0.0], "Y": [0.0]}), ValueError, "missing columns"),
    (pd.DataFrame({"X": [], "Y": [], "Z": []}), ValueError, "empty"),
    (pd.DataFrame({"X": [0.0], "Y": [0.0], "Z": [np.nan]}), ValueError, "NaN"),
])
def test_bad_conditioning_data(df, err, msg):
    with pytest.raises(err, match=msg):
        predict(df=df)


@pytest.mark.parametrize("vario, msg", [
    ([0, 0, 1, 1, 1], "variogram must be"),
    ([0, 0, -1, 1, 1, "Exponential"], "ranges must be positive"),
    ([0, 0, 1, 1, 0, "Exponential"], "sill must be positive"),
    ([0, 2, 1, 1, 1, "Exponential"], "nugget"),
    ([0, 0, 1, 1, 1, "Cubic"], "vtype"),
    ([0, 0, 1, 1, 1, "Matern"], "smoothness"),
    ([0, 0, float("nan"), 1, 1, "Exponential"], "finite"),
])
def test_bad_variogram(vario, msg):
    with pytest.raises(ValueError, match=msg):
        predict(vario=vario)


@pytest.mark.parametrize("kw, msg", [
    ({"num_points": 7}, "num_points"),
    ({"num_points": 8.0}, "num_points"),
    ({"radius": 0}, "radius"),
    ({"radius": float("inf")}, "radius"),
    ({"method": "universal"}, "method"),
])
def test_bad_search_and_method(kw, msg):
    with pytest.raises(ValueError, match=msg):
        predict(**kw)


def test_matern_rejected_on_gpu(monkeypatch):
    # Pretend a GPU is present so the call reaches the GPU path.
    monkeypatch.setattr("gstatsim_accel.kriging.resolve_backend", lambda b: b)
    with pytest.raises(NotImplementedError, match="Matern"):
        predict(vario=[0, 0, 1000.0, 1000.0, 1.0, "Matern", 1.5], backend="gpu")
