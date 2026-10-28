"""Derived quantities: g from fits, amplitude correction, hypothesis test, series.

Spec: docs/plan/05-physics-and-uncertainty.md §2–§4
Status: skeleton (M0). Implement in milestone M3–M4.
"""

from __future__ import annotations

import math
from typing import Any


def g_from_quadratic(c2_px: float, m_per_px: float) -> float:
    """g = 2·c2·s (image y points down, so g > 0)."""
    return 2.0 * c2_px * m_per_px


def amplitude_correction(period_s: float, theta_rad: float) -> float:
    """T0 = T / (1 + θ²/16 + 11θ⁴/3072)."""
    return period_s / (1.0 + theta_rad**2 / 16.0 + 11.0 * theta_rad**4 / 3072.0)


def g_from_pendulum(length_m: float, period0_s: float) -> float:
    """g = 4π² L / T0²."""
    return 4.0 * math.pi**2 * length_m / period0_s**2


def g_reference(lat_deg: float, height_m: float = 0.0) -> float:
    """International Gravity Formula (comparison only, never used to alter data)."""
    phi = math.radians(lat_deg)
    return (
        9.780327 * (1 + 0.0053024 * math.sin(phi) ** 2 - 0.0000058 * math.sin(2 * phi) ** 2)
        - 3.086e-6 * height_m
    )


def hypothesis_test(prediction: float, measured: float, lo: float, hi: float) -> dict[str, Any]:
    """supported | close | not_supported (05 §3.4)."""
    if lo <= prediction <= hi:
        outcome = "supported"
    elif abs(prediction - measured) / abs(measured) <= 0.05:
        outcome = "close"
    else:
        outcome = "not_supported"
    rel = (prediction - measured) / measured * 100.0
    return {
        "p": prediction,
        "m": measured,
        "lo": lo,
        "hi": hi,
        "rel_diff_pct": rel,
        "outcome": outcome,
    }


def pendulum_series(points: list[dict[str, float]]) -> dict[str, Any]:
    """Weighted T0² = kL + b, g = 4π²/k, power-law exponent, pairs bootstrap (05 §4)."""
    raise NotImplementedError("Milestone M3–M4: see docs/plan/09-milestones-and-tasks.md")
