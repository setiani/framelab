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


MIN_PIVOT_SPAN_DEG = 8.0


def fit_circle_geometric(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Kåsa start + geometric refinement (minimise Σ(|p − c| − r)²).

    Kåsa alone is biased towards a small radius on short noisy arcs, which a pendulum swing
    always is (±10–20°); the geometric fit removes that bias.
    """
    from scipy.optimize import least_squares

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    cx0, cy0, _r0 = fit_circle(x, y)

    def resid(c: np.ndarray) -> np.ndarray:
        d = np.hypot(x - c[0], y - c[1])
        return d - d.mean()

    sol = least_squares(resid, [cx0, cy0], method="lm")
    cx, cy = sol.x
    r = float(np.mean(np.hypot(x - cx, y - cy)))
    return float(cx), float(cy), r


def pivot_from_track(
    x: np.ndarray,
    y: np.ndarray,
    *,
    string_top: tuple[float, float] | None = None,
    n_boot: int = 200,
    seed: int = 1234,
) -> dict[str, Any]:
    """Pendulum pivot and radius (pivot → bob centre) from the bob track (04 §6.3).

    Returns cx, cy, r_px, sigma_r_px (bootstrap over points), span_deg, reliable, and the
    cross-check against the topmost point of the "string" mask when available.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    out: dict[str, Any] = {"warnings": []}
    if x.size < 10:
        out.update(reliable=False, r_px=None)
        out["warnings"].append("pivot_too_few_points")
        return out
    cx, cy, r = fit_circle_geometric(x, y)
    angles = np.degrees(np.arctan2(x - cx, y - cy))  # 0° = straight down
    span = float(np.percentile(angles, 99) - np.percentile(angles, 1))
    rng = np.random.default_rng(seed)
    radii = []
    for _ in range(n_boot):
        idx = rng.integers(0, x.size, x.size)
        radii.append(fit_circle_geometric(x[idx], y[idx])[2])
    sigma_r = float(np.std(radii))
    reliable = span >= MIN_PIVOT_SPAN_DEG and cy < float(np.min(y))
    out.update(
        cx=cx,
        cy=cy,
        r_px=r,
        sigma_r_px=sigma_r,
        span_deg=span,
        reliable=bool(reliable),
        rms_px=float(np.sqrt(np.mean((np.hypot(x - cx, y - cy) - r) ** 2))),
    )
    if not reliable:
        out["warnings"].append("pivot_unreliable_small_swing" if span < MIN_PIVOT_SPAN_DEG
                               else "pivot_below_bob")  # fmt: skip
    if string_top is not None:
        dist = float(math.hypot(string_top[0] - cx, string_top[1] - cy))
        out["string_top"] = list(string_top)
        out["string_pivot_distance_px"] = dist
        if dist > 0.1 * r:
            out["warnings"].append("pivot_string_mismatch")
    return out


def string_top_point(record: Any) -> tuple[float, float] | None:
    """Topmost pixel (centre of the top row) of the string mask, if any."""
    xs, ys, _sx, _sy = mask_pixels(record)
    if xs.size == 0:
        return None
    top = float(np.min(ys))
    row = xs[ys <= top + 0.5]
    return float(np.mean(row)), top


def spoken_length_scale(
    length_m: float, r_px: float, sigma_r_px: float, *, length_sigma_m: float = 0.005
) -> ScaleEstimate:
    """m/px from a spoken string length (pivot → bob centre) and the fitted radius (04 §6.3)."""
    m_per_px = length_m / r_px
    rel = math.sqrt((sigma_r_px / r_px) ** 2 + (length_sigma_m / length_m) ** 2)
    return ScaleEstimate(
        method="spoken_length",
        m_per_px=m_per_px,
        sigma_m_per_px=m_per_px * rel,
        details={"length_m": length_m, "length_sigma_m": length_sigma_m, "r_px": r_px,
                 "sigma_r_px": sigma_r_px, "rel_sigma": rel},
    )  # fmt: skip


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
