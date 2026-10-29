"""Event detection: free-fall release/impact window.

Spec: docs/plan/05-physics-and-uncertainty.md §2.1 (+ M1 finding V12: SAM reports a static
object only from the frame after it starts moving, so the release is usually before the first
mask and the window simply starts at the first valid point).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import savgol_filter


@dataclass
class Window:
    start: int  # first index (inclusive) of the fit window, into the valid-point arrays
    end: int  # last index (inclusive)
    release_index: int | None  # last "in hand" index, None if before the first point
    impact_index: int | None  # contact frame, None if the object left the frame
    v_max_px_s: float
    warnings: list[str] = field(default_factory=list)

    @property
    def n(self) -> int:
        return self.end - self.start + 1


def smooth_velocity(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Savitzky–Golay derivative (window ≤ 7, quadratic)."""
    n = len(y)
    if n < 5:
        return np.gradient(y, t)
    win = min(7, n if n % 2 == 1 else n - 1)
    win = max(win, 5)
    dt = float(np.median(np.diff(t)))
    return savgol_filter(y, win, 2, deriv=1, delta=dt)


def freefall_window(t: np.ndarray, y: np.ndarray) -> Window:
    """Fit window between release and the first impact (image y points down).

    1. v = smoothed velocity; i_vmax = argmax(v).
    2. Impact = first local maximum of the RAW position after i_vmax (the frame closest to
       contact); if none, the object left the frame (warning ``object_left_frame``).
    3. Release = last index before i_vmax with v < 0.05·v_max (still in the hand); if none, the
       release happened before the first point (normal with SAM, M1 V12).
    4. Window = [release + 1, impact − 1] (contact frames excluded).
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    warnings: list[str] = []
    n = len(y)
    if n < 4:
        return Window(0, n - 1, None, None, 0.0, ["too_few_points"])
    v = smooth_velocity(t, y)
    i_vmax = int(np.argmax(v))
    v_max = float(v[i_vmax])
    if v_max <= 0:
        return Window(0, n - 1, None, None, v_max, ["no_downward_motion"])

    impact = None
    for j in range(max(i_vmax, 1), n - 1):
        if y[j] >= y[j - 1] and y[j] > y[j + 1]:
            impact = j
            break
    if impact is None:
        warnings.append("object_left_frame")
        end = n - 1
    else:
        end = impact - 1

    release = None
    for j in range(i_vmax - 1, -1, -1):
        if v[j] < 0.05 * v_max:
            release = j
            break
    start = 0 if release is None else release + 1
    if end - start + 1 < 8:
        warnings.append("too_few_points")
    return Window(start, end, release, impact, v_max, warnings)
