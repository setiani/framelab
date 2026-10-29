"""Free-fall analysis: window → quadratic fit → bootstrap → g with combined uncertainty.

Spec: docs/plan/05-physics-and-uncertainty.md §2 and §6. Pure function of the track arrays
and the scale; the pipeline/agent tools turn the result into ledger evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats

from framelab.config import G_REF_DEFAULT
from framelab.physics import derive
from framelab.physics.events import Window, freefall_window
from framelab.physics.models import FitResult, fit_cubic, fit_quadratic, polyfit_params, polyval
from framelab.physics.uncertainty import (
    block_bootstrap,
    combine_sigma,
    percentile_ci,
    sigma_from_ci,
)

MAX_TRIM_PER_END = 3
VFR_REL_SIGMA = 0.005


@dataclass
class FreeFallAnalysis:
    window: Window
    trimmed_start: int
    trimmed_end: int
    fit: FitResult
    cubic: FitResult | None
    systematic_curvature: bool
    g: float
    v0: float
    ci_stat: tuple[float, float]
    sigma_boot: float
    sigma_analytic: float
    sigma_window: float
    sigma_stat: float
    sigma_scale: float
    sigma_vfr: float
    sigma_total: float
    ci95: tuple[float, float]
    comparison: dict[str, Any]
    bootstrap_n: int
    warnings: list[str] = field(default_factory=list)
    t_fit: np.ndarray = field(default_factory=lambda: np.zeros(0), repr=False)
    y_fit: np.ndarray = field(default_factory=lambda: np.zeros(0), repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "window": {
                "start": self.window.start,
                "end": self.window.end,
                "release_index": self.window.release_index,
                "impact_index": self.window.impact_index,
                "trimmed_start": self.trimmed_start,
                "trimmed_end": self.trimmed_end,
                "n": int(self.t_fit.size),
                "t_start": float(self.t_fit[0]) if self.t_fit.size else None,
                "t_end": float(self.t_fit[-1]) if self.t_fit.size else None,
            },
            "fit": self.fit.summary(),
            "cubic": self.cubic.summary() if self.cubic else None,
            "systematic_curvature": self.systematic_curvature,
            "g": self.g,
            "v0": self.v0,
            "ci_stat": list(self.ci_stat),
            "sigma_boot": self.sigma_boot,
            "sigma_analytic": self.sigma_analytic,
            "sigma_window": self.sigma_window,
            "sigma_stat": self.sigma_stat,
            "sigma_scale": self.sigma_scale,
            "sigma_vfr": self.sigma_vfr,
            "sigma_total": self.sigma_total,
            "ci95": list(self.ci95),
            "comparison": self.comparison,
            "bootstrap_n": self.bootstrap_n,
            "warnings": self.warnings,
        }


def trim_contact_points(t: np.ndarray, y: np.ndarray, start: int, end: int) -> tuple[int, int]:
    """Drop up to 3 points at each end whose residual is an outlier (hand contact, impact)."""
    for _ in range(2 * MAX_TRIM_PER_END):
        if end - start + 1 < 8:
            break
        fit = fit_quadratic(t[start : end + 1], y[start : end + 1])
        resid = fit.residuals
        inner = resid[1:-1]
        limit = max(3.0 * float(np.sqrt(np.mean(inner**2))), 1.0)
        if abs(resid[0]) > limit and abs(resid[0]) >= abs(resid[-1]):
            start += 1
        elif abs(resid[-1]) > limit:
            end -= 1
        else:
            break
    return start, end


def _window_variants(
    t: np.ndarray, y: np.ndarray, start: int, end: int, m_per_px: float
) -> list[float]:
    """g for the window with its first and/or last point removed (window-choice spread)."""
    out = []
    for a, b in ((start + 1, end), (start, end - 1), (start + 1, end - 1)):
        if b - a + 1 >= 6:
            out.append(2.0 * fit_quadratic(t[a : b + 1], y[a : b + 1]).params["c2"] * m_per_px)
    return out


def analyze_free_fall(
    t: np.ndarray,
    y_px: np.ndarray,
    m_per_px: float,
    sigma_m_per_px: float,
    *,
    bootstrap_n: int = 1000,
    is_vfr: bool = False,
    g_ref: float = G_REF_DEFAULT,
    seed: int = 1234,
) -> FreeFallAnalysis:
    """Analyse valid track points (t in s, y in px, image y down) of one free fall."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y_px, dtype=float)
    win = freefall_window(t, y)
    warnings = list(win.warnings)
    start, end = trim_contact_points(t, y, win.start, win.end)
    if (start, end) != (win.start, win.end):
        warnings.append(f"trimmed_contact_points:{start - win.start}+{win.end - end}")
    tf, yf = t[start : end + 1], y[start : end + 1]
    fit = fit_quadratic(tf, yf)

    cubic = None
    curvature = False
    if tf.size >= 8:
        cubic = fit_cubic(tf, yf)
        c3_sigma = np.sqrt(cubic.cov[3][3]) if cubic.cov else float("inf")
        improves = cubic.rms_px < 0.7 * fit.rms_px
        curvature = bool(abs(cubic.params["c3"]) > 3 * c3_sigma and improves)
        if curvature:
            warnings.append("systematic_curvature")

    t0 = fit.t_offset

    def fit_fn(tt: np.ndarray, yy: np.ndarray) -> np.ndarray:
        return polyfit_params(tt - t0, yy, 2)

    def predict_fn(tt: np.ndarray, params: np.ndarray) -> np.ndarray:
        return polyval(tt - t0, params)

    samples = block_bootstrap(tf, yf, fit_fn, predict_fn, n=bootstrap_n, seed=seed)
    g_samples = 2.0 * samples[:, 2] * m_per_px
    g = derive.g_from_quadratic(fit.params["c2"], m_per_px)
    ci_stat = percentile_ci(g_samples)
    # σ_stat (05 §6.1, calibrated on synthetic data in M3 to reach ≥90 % CI coverage):
    #   max(bootstrap σ, analytic σ from the fit covariance) × Student-t small-sample factor,
    #   combined with the spread from moving the window edges by one frame.
    sigma_boot = sigma_from_ci(*ci_stat)
    sigma_analytic = 2.0 * float(np.sqrt(fit.cov[2][2])) * m_per_px if fit.cov else sigma_boot
    t_factor = float(stats.t.ppf(0.975, max(fit.n - 3, 1)) / 1.96)
    variants = [g_val for g_val in _window_variants(t, y, start, end, m_per_px)]
    sigma_window = (
        float(np.std([*variants, 2.0 * fit.params["c2"] * m_per_px])) if variants else 0.0
    )
    sigma_stat = combine_sigma(max(sigma_boot, sigma_analytic) * t_factor, sigma_window)
    sigma_scale = abs(g) * sigma_m_per_px / m_per_px if m_per_px > 0 else 0.0
    sigma_vfr = abs(g) * VFR_REL_SIGMA if is_vfr else 0.0
    sigma_total = combine_sigma(sigma_stat, sigma_scale, sigma_vfr)
    ci95 = (g - 1.96 * sigma_total, g + 1.96 * sigma_total)
    return FreeFallAnalysis(
        window=win,
        trimmed_start=start,
        trimmed_end=end,
        fit=fit,
        cubic=cubic,
        systematic_curvature=curvature,
        g=g,
        v0=fit.params["c1"] * m_per_px,
        ci_stat=ci_stat,
        sigma_boot=sigma_boot,
        sigma_analytic=sigma_analytic,
        sigma_window=sigma_window,
        sigma_stat=sigma_stat,
        sigma_scale=sigma_scale,
        sigma_vfr=sigma_vfr,
        sigma_total=sigma_total,
        ci95=ci95,
        comparison=derive.compare_reference(g, ci95[0], ci95[1], g_ref),
        bootstrap_n=bootstrap_n,
        warnings=warnings,
        t_fit=tf,
        y_fit=yf,
    )
