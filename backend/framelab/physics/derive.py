"""Derived quantities: g from fits, amplitude correction, hypothesis test, series.

Spec: docs/plan/05-physics-and-uncertainty.md §2–§4
Status: implemented (free fall M3, pendulum + series M4).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def g_from_quadratic(c2_px: float, m_per_px: float) -> float:
    """g = 2·c2·s (image y points down, so g > 0)."""
    return 2.0 * c2_px * m_per_px


def compare_reference(value: float, lo: float, hi: float, ref: float) -> dict[str, Any]:
    """Comparison with a reference value (diagnosis only; never used to alter data)."""
    return {
        "ref": ref,
        "value": value,
        "lo": lo,
        "hi": hi,
        "rel_diff_pct": (value - ref) / ref * 100.0,
        "within_ci": lo <= ref <= hi,
    }


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


def _weighted_line(x: np.ndarray, y: np.ndarray, sigma: np.ndarray) -> dict[str, float]:
    """Weighted y = k·x + b with known σ_y. cov is scaled by max(1, χ²_red) (honest when the
    stated errors are too small); returns k, b, σ_k, σ_b, χ²_red."""
    w = 1.0 / sigma**2
    sw, sx, sy = w.sum(), (w * x).sum(), (w * y).sum()
    sxx, sxy = (w * x * x).sum(), (w * x * y).sum()
    det = sw * sxx - sx**2
    k = (sw * sxy - sx * sy) / det
    b = (sxx * sy - sx * sxy) / det
    var_k, var_b = sw / det, sxx / det
    dof = x.size - 2
    chi2 = float(np.sum(w * (y - (k * x + b)) ** 2))
    chi2_red = chi2 / dof if dof > 0 else float("nan")
    scale = max(1.0, chi2_red) if dof > 0 else 1.0
    return {
        "k": float(k),
        "b": float(b),
        "sigma_k": float(math.sqrt(var_k * scale)),
        "sigma_b": float(math.sqrt(var_b * scale)),
        "chi2_red": chi2_red,
        "cov_kb": float(-sx / det * scale),
    }


def pendulum_series(
    points: list[dict[str, float]], *, n_boot: int = 2000, seed: int = 1234
) -> dict[str, Any]:
    """Weighted T0² = kL + b, g = 4π²/k, power law T ∝ L^n, pairs bootstrap (05 §4).

    ``points``: dicts with L, T0, sigma_T, sigma_L (SI). Needs ≥ 3 runs with ≥ 2 lengths.
    """
    if len(points) < 3:
        raise ValueError(f"a series needs at least 3 runs, got {len(points)}")
    L = np.array([p["L"] for p in points], dtype=float)
    T = np.array([p["T0"] for p in points], dtype=float)
    sT = np.array([max(p.get("sigma_T", 0.0), 1e-6) for p in points], dtype=float)
    sL = np.array([max(p.get("sigma_L", 0.0), 0.0) for p in points], dtype=float)
    if np.unique(np.round(L, 4)).size < 2:
        raise ValueError("a series needs at least two different lengths")

    def regress(Li, Ti, sTi, sLi):
        T2 = Ti**2
        s_t2 = 2 * Ti * sTi
        k0 = np.polyfit(Li, T2, 1)[0]
        lin = _weighted_line(Li, T2, np.sqrt(s_t2**2 + (k0 * sLi) ** 2))
        lin = _weighted_line(Li, T2, np.sqrt(s_t2**2 + (lin["k"] * sLi) ** 2))
        lnL, lnT = np.log(Li), np.log(Ti)
        s_ln = np.sqrt((sTi / Ti) ** 2 + (0.5 * sLi / Li) ** 2)
        pw = _weighted_line(lnL, lnT, s_ln)
        return lin, pw

    lin, pw = regress(L, T, sT, sL)
    g = 4 * math.pi**2 / lin["k"]
    sigma_g = g * lin["sigma_k"] / lin["k"]
    n_exp, sigma_n = pw["k"], pw["sigma_k"]

    rng = np.random.default_rng(seed)
    g_boot, n_boot_vals = [], []
    idx_all = np.arange(L.size)
    for _ in range(n_boot):
        idx = rng.choice(idx_all, size=L.size)
        if np.unique(np.round(L[idx], 4)).size < 2:
            continue
        lb, pb = regress(L[idx], T[idx], sT[idx], sL[idx])
        g_boot.append(4 * math.pi**2 / lb["k"])
        n_boot_vals.append(pb["k"])
    boot_ci_g = tuple(np.percentile(g_boot, [2.5, 97.5]).tolist()) if g_boot else None
    boot_ci_n = tuple(np.percentile(n_boot_vals, [2.5, 97.5]).tolist()) if n_boot_vals else None

    flags = []
    if abs(lin["b"]) > 2 * lin["sigma_b"]:
        flags.append("length_offset")
    supported = abs(n_exp - 0.5) <= 2 * sigma_n
    return {
        "n_runs": int(L.size),
        "points": [
            {"L": float(a), "T0": float(b), "sigma_T": float(c), "sigma_L": float(d)}
            for a, b, c, d in zip(L, T, sT, sL, strict=True)
        ],  # fmt: skip
        "linear": lin,
        "g": g,
        "sigma_g": sigma_g,
        "ci_g": [g - 1.96 * sigma_g, g + 1.96 * sigma_g],
        "boot_ci_g": list(boot_ci_g) if boot_ci_g else None,
        "power_law": pw,
        "n": n_exp,
        "sigma_n": sigma_n,
        "ci_n": [n_exp - 1.96 * sigma_n, n_exp + 1.96 * sigma_n],
        "boot_ci_n": list(boot_ci_n) if boot_ci_n else None,
        "sqrt_law_supported": bool(supported),
        "n_boot_used": len(g_boot),
        "flags": flags,
    }
