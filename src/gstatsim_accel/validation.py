"""Input checks shared by every backend.

Both backends see inputs that already passed these checks, so a bad input
fails the same way whichever backend was requested.
"""

from __future__ import annotations

import numbers

import numpy as np
import pandas as pd

VTYPES = ("exponential", "gaussian", "spherical", "matern")
METHODS = ("simple", "ordinary")


def check_prediction_points(points) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"prediction_points must have shape (M, 2), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("prediction_points contains NaN or infinite coordinates")
    return arr


def check_conditioning_data(df, xx: str, yy: str, zz: str) -> pd.DataFrame:
    """Return a copy with columns renamed to X, Y, Z (GStatSim's convention)."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"conditioning_data must be a pandas DataFrame, got {type(df).__name__}")
    missing = [c for c in (xx, yy, zz) if c not in df.columns]
    if missing:
        raise ValueError(f"conditioning_data is missing columns {missing}")
    if len(df) == 0:
        raise ValueError("conditioning_data is empty")
    out = df[[xx, yy, zz]].astype(np.float64)
    out.columns = ["X", "Y", "Z"]
    if not np.all(np.isfinite(out.to_numpy())):
        raise ValueError("conditioning_data contains NaN or infinite values; drop or fill them first")
    return out.reset_index(drop=True)


def check_variogram(vario) -> list:
    """Validate GStatSim's variogram list:
    [azimuth, nugget, major_range, minor_range, sill, vtype(, smoothness for Matern)]
    """
    if not isinstance(vario, (list, tuple)) or len(vario) not in (6, 7):
        raise ValueError(
            "variogram must be [azimuth, nugget, major_range, minor_range, sill, vtype]"
            " plus smoothness s for Matern"
        )
    azimuth, nugget, major, minor, sill, vtype = vario[:6]
    for name, val in (("azimuth", azimuth), ("nugget", nugget), ("major_range", major),
                      ("minor_range", minor), ("sill", sill)):
        if not isinstance(val, numbers.Real) or not np.isfinite(val):
            raise ValueError(f"variogram {name} must be a finite number, got {val!r}")
    if major <= 0 or minor <= 0:
        raise ValueError("variogram ranges must be positive")
    if sill <= 0:
        raise ValueError("variogram sill must be positive")
    if not 0 <= nugget <= sill:
        raise ValueError("variogram nugget must be between 0 and the sill")
    if not isinstance(vtype, str) or vtype.lower() not in VTYPES:
        raise ValueError(f"variogram vtype must be one of {VTYPES}, got {vtype!r}")
    if vtype.lower() == "matern":
        if len(vario) != 7 or not isinstance(vario[6], numbers.Real) or vario[6] <= 0:
            raise ValueError("Matern variogram needs a positive smoothness s as its 7th entry")
    return list(vario)


def check_search(num_points, radius) -> None:
    # GStatSim's octant search takes num_points // 8 neighbors per octant,
    # so anything below 8 would find no neighbors at all.
    if not isinstance(num_points, numbers.Integral) or num_points < 8:
        raise ValueError(f"num_points must be an integer >= 8, got {num_points!r}")
    if not isinstance(radius, numbers.Real) or not np.isfinite(radius) or radius <= 0:
        raise ValueError(f"radius must be a positive finite number, got {radius!r}")


def check_method(method: str) -> str:
    if not isinstance(method, str) or method.lower() not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    return method.lower()
