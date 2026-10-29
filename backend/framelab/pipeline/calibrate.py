"""Scale calibration (ruler, object diameter, spoken length) and pendulum pivot.

Spec: docs/plan/04-pipeline-and-measurement.md §6 + M1 findings (docs/plan/11 §1):
- V11: SAM masks are ~1 px larger per side → subtract δ (``mask_dilation_px``) from object
  diameters; a ruler's LONG axis needs no correction (PCA extent matched truth to 0.01 px).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from framelab.pipeline.measure import extent_along_major, mask_pixels
from framelab.schemas import Calibration, Observation, ScaleEstimate

PRIORITY = ("ruler", "object_diameter", "spoken_length", "manual")
EDGE_SIGMA_PX = math.sqrt(2.0)  # ±1 px at each end of a measured length


def ruler_scale(
    record: Any,
    length_m: float,
    *,
    length_sigma_m: float = 0.001,
    frame_long_edge_px: int | None = None,
) -> ScaleEstimate:
    """m/px from the long-axis extent of the reference mask (04 §6.1)."""
    xs, ys, sx, sy = mask_pixels(record)
    if xs.size < 20:
        raise ValueError("reference mask too small for calibration")
    length_px = extent_along_major(xs, ys, pixel=max(sx, sy))
    m_per_px = length_m / length_px
    rel = math.sqrt((EDGE_SIGMA_PX / length_px) ** 2 + (length_sigma_m / length_m) ** 2)
    details: dict[str, Any] = {
        "length_px": length_px,
        "length_m": length_m,
        "length_sigma_m": length_sigma_m,
        "rel_sigma": rel,
    }
    if frame_long_edge_px and length_px < 0.15 * frame_long_edge_px:
        details["warning"] = "reference_short"
    return ScaleEstimate(
        method="ruler", m_per_px=m_per_px, sigma_m_per_px=m_per_px * rel, details=details
    )


def diameter_px(observations: list[Observation], dilation_px: float) -> tuple[float, float, int]:
    """Blur-robust object diameter in px: (median, σ, n) after subtracting the mask dilation.

    Motion blur stretches a ball along its motion, so when minor/major < 0.85 the MINOR axis
    (perpendicular to the motion) is used, else the equivalent-area diameter (04 §6.2).
    """
    widths = []
    for o in observations:
        if o.major_axis_px <= 0:
            continue
        w = o.minor_axis_px if o.minor_axis_px / o.major_axis_px < 0.85 else o.eq_diameter_px
        widths.append(w - dilation_px)
    if not widths:
        raise ValueError("no observations for the diameter scale")
    arr = np.asarray(widths)
    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median)))
    sigma = 1.4826 * mad / math.sqrt(arr.size) + 1.0  # + 1 px edge term (04 §6.2)
    return median, sigma, int(arr.size)


def diameter_scale(
    observations: list[Observation],
    diameter_m: float,
    *,
    dilation_px: float = 1.5,
    diameter_sigma_m: float = 0.001,
) -> ScaleEstimate:
    d_px, sigma_px, n = diameter_px(observations, dilation_px)
    m_per_px = diameter_m / d_px
    rel = math.sqrt((sigma_px / d_px) ** 2 + (diameter_sigma_m / diameter_m) ** 2)
    return ScaleEstimate(
        method="object_diameter",
        m_per_px=m_per_px,
        sigma_m_per_px=m_per_px * rel,
        details={
            "diameter_px": d_px,
            "diameter_sigma_px": sigma_px,
            "diameter_m": diameter_m,
            "dilation_px": dilation_px,
            "n": n,
            "rel_sigma": rel,
        },
    )


def fit_circle(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Kåsa algebraic circle fit → (cx, cy, r) (04 §6.3)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    a = np.column_stack([x, y, np.ones_like(x)])
    b = -(x**2 + y**2)
    (d, e, f), *_ = np.linalg.lstsq(a, b, rcond=None)
    cx, cy = -d / 2, -e / 2
    r = math.sqrt(max(cx**2 + cy**2 - f, 0.0))
    return float(cx), float(cy), float(r)


def combine(estimates: list[ScaleEstimate]) -> Calibration:
    """Primary/secondary by priority ruler > object_diameter > spoken_length (04 §6.4)."""
    if not estimates:
        raise ValueError("no scale estimates")
    ordered = sorted(estimates, key=lambda e: PRIORITY.index(e.method))
    primary = ordered[0]
    secondary = ordered[1] if len(ordered) > 1 else None
    agreement = (
        abs(primary.m_per_px - secondary.m_per_px) / primary.m_per_px * 100.0 if secondary else None
    )
    return Calibration(primary=primary, secondary=secondary, agreement_pct=agreement)
