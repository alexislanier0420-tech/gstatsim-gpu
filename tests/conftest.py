import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def small_field():
    """200 scattered points with a smooth trend plus noise, on a 10 km square."""
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 10_000, 200)
    y = rng.uniform(0, 10_000, 200)
    z = np.sin(x / 3000) + np.cos(y / 4000) + 0.1 * rng.standard_normal(200)
    df = pd.DataFrame({"X": x, "Y": y, "Z": z})
    gx, gy = np.meshgrid(np.arange(250, 10_000, 1000.0), np.arange(250, 10_000, 1000.0))
    pred = np.column_stack([gx.ravel(), gy.ravel()])
    vario = [0, 0.05, 4000.0, 4000.0, 1.0, "Exponential"]
    return df, pred, vario
