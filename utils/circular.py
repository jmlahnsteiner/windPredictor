"""utils/circular.py — Circular statistics for wind direction data."""
import math

import numpy as np
import pandas as pd


def circular_std(angles) -> float:
    """
    Circular standard deviation of wind directions (degrees).

    Accepts a pd.Series or list of degree values. NaN values are dropped.
    Returns NaN if no valid values remain.
    """
    if not isinstance(angles, pd.Series):
        angles = pd.Series(angles)
    clean = angles.dropna()
    if clean.empty:
        return float("nan")
    if len(clean) == 1:
        return 0.0
    rad = np.radians(clean)
    R = np.hypot(np.sin(rad).mean(), np.cos(rad).mean())
    return float(np.degrees(np.sqrt(-2 * np.log(np.clip(R, 1e-9, 1)))))
