"""Pendulum analysis: damped-sine fit → period → amplitude correction → g, with σ budget.

Spec: docs/plan/05-physics-and-uncertainty.md §3 and §6.3. Pure function of the cleaned
track (pixels, seconds) plus optional length and scale; numbers reach the ledger via
``pipeline/analysis_auto.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats

from framelab.config import G_REF_DEFAULT
from framelab.physics.derive import compare_reference, g_from_pendulum
from framelab.physics.models import (
    DAMPED_SINE_PARAMS,
    FitResult,
    damped_sine,
    damped_sine_params,
    fit_damped_sine,
)
from framelab.physics.uncertainty import block_bootstrap, combine_sigma, sigma_from_ci

MIN_CYCLES_WARN = 5.0
MIN_CYCLES_FAIL = 3.0
MIN_POINTS = 60
VFR_REL_SIGMA = 0.005
MAX_TRIM_FRACTION = 0.1


def period_from_omega(omega: float, gamma: float) -> float:
    """Undamped period T = 2π/√(ω_d² + γ²) (05 §3.3)."""
    return 2.0 * math.pi / math.sqrt(omega**2 + gamma**2)


def amplitude_factor(theta_rad: np.ndarray) -> float:
    """Mean of 1 + θ²/16 + 11θ⁴/3072 over the window's envelope (T = T0 · factor).

    Averaging the factor (not θ) is exact to first order while the amplitude decays.
    """
    th = np.asarray(theta_rad, dtype=float)
    return float(np.mean(1.0 + th**2 / 16.0 + 11.0 * th**4 / 3072.0))


def trim_ends(t: np.ndarray, x: np.ndarray) -> tuple[int, int, FitResult]:
    """Drop leading/trailing points with large residuals (hand at release or catching the bob).

    At most 10 % of the points per end; threshold max(4·rms, 2 px). Returns (start, end, fit).
    """
    start, end = 0, len(t) - 1
    fit = fit_damped_sine(t, x)
    limit_n = int(MAX_TRIM_FRACTION * len(t))
    thr = max(4.0 * fit.rms_px, 2.0)
    resid = np.abs(fit.residuals)
    lead = 0
    while lead < limit_n and resid[lead] > thr:
        lead += 1
    trail = 0
    while trail < limit_n and resid[-1 - trail] > thr:
        trail += 1
    if lead or trail:
        start, end = lead, len(t) - 1 - trail
        fit = fit_damped_sine(t[start : end + 1], x[start : end + 1])
    return start, end, fit


@dataclass
class PendulumAnalysis:
    fit: FitResult
    start: int
    end: int
    period_damped: float  # T_d = 2π/ω_d
    period: float  # T = 2π/√(ω_d²+γ²), measured (finite amplitude)
    period0: float  # T0, amplitude-corrected (small-angle) period
    amplitude_px: float  # mean envelope amplitude over the window
    theta_mean_deg: float | None
    theta_source: str | None  # "circle_fit" | "length_scale" | None
    amp_factor: float
    n_cycles: float
    duration_s: float
    sigma_period_boot: float
    sigma_period_analytic: float
    sigma_period_stat: float
    sigma_period_vfr: float
    sigma_period: float  # σ of T (stat ⊕ vfr)
    sigma_corr_rel: float  # half of the amplitude correction (relative, on T0)
    sigma_period0: float
    ci_period0: tuple[float, float]
    length_m: float | None
    sigma_length_m: float | None
    length_source: str | None
    g: float | None
    sigma_g: float | None
    ci_g: tuple[float, float] | None
    g_uncorrected: float | None
    comparison: dict[str, Any] | None
    drift_px_s: float
    bootstrap_n: int
    warnings: list[str] = field(default_factory=list)
    t_fit: np.ndarray = field(default_factory=lambda: np.zeros(0), repr=False)
    x_fit: np.ndarray = field(default_factory=lambda: np.zeros(0), repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fit": self.fit.summary(),
            "window": {
                "start_index": self.start,
                "end_index": self.end,
                "t_start": float(self.t_fit[0]) if self.t_fit.size else None,
                "t_end": float(self.t_fit[-1]) if self.t_fit.size else None,
                "n_points": self.fit.n,
                "duration_s": self.duration_s,
                "n_cycles": self.n_cycles,
            },
            "period_damped": self.period_damped,
            "period": self.period,
            "period0": self.period0,
            "amplitude_px": self.amplitude_px,
            "theta_mean_deg": self.theta_mean_deg,
            "theta_source": self.theta_source,
            "amp_factor": self.amp_factor,
            "sigma_period_boot": self.sigma_period_boot,
            "sigma_period_analytic": self.sigma_period_analytic,
            "sigma_period_stat": self.sigma_period_stat,
            "sigma_period_vfr": self.sigma_period_vfr,
            "sigma_period": self.sigma_period,
            "sigma_corr_rel": self.sigma_corr_rel,
            "sigma_period0": self.sigma_period0,
            "ci_period0": list(self.ci_period0),
            "length_m": self.length_m,
            "sigma_length_m": self.sigma_length_m,
            "length_source": self.length_source,
            "g": self.g,
            "sigma_g": self.sigma_g,
            "ci_g": list(self.ci_g) if self.ci_g else None,
            "g_uncorrected": self.g_uncorrected,
            "comparison": self.comparison,
            "drift_px_s": self.drift_px_s,
            "bootstrap_n": self.bootstrap_n,
            "warnings": self.warnings,
        }


def analyze_pendulum(
    t: np.ndarray,
    x_px: np.ndarray,
    *,
    length_m: float | None = None,
    sigma_length_m: float | None = None,
    length_source: str | None = None,
    radius_px: float | None = None,
    m_per_px: float | None = None,
    bootstrap_n: int = 300,
    is_vfr: bool = False,
    g_ref: float = G_REF_DEFAULT,
    seed: int = 1234,
) -> PendulumAnalysis:
    """Period (and g when the length is known) from the horizontal bob position."""
    t = np.asarray(t, dtype=float)
    x = np.asarray(x_px, dtype=float)
    warnings: list[str] = []
    if t.size < 12:
        raise ValueError(f"only {t.size} valid points: too few for a pendulum fit")
    if t.size < MIN_POINTS:
        warnings.append("too_few_points")

    start, end, fit = trim_ends(t, x)
    tw, xw = t[start : end + 1], x[start : end + 1]
    p = fit.params
    omega, gamma = p["omega"], p["gamma"]
    period_d = 2.0 * math.pi / omega
    period = period_from_omega(omega, gamma)
    duration = float(tw[-1] - tw[0])
    n_cycles = duration / period
    if n_cycles < MIN_CYCLES_FAIL:
        warnings.append("too_few_cycles")
    elif n_cycles < MIN_CYCLES_WARN:
        warnings.append("few_cycles")

    # --- statistical σ of T: block bootstrap ⊕ analytic (Jacobian), t-scaled -----------------
    tt = tw - tw[0]
    p0 = np.array([p[k] for k in DAMPED_SINE_PARAMS])

    def fit_fn(ts: np.ndarray, xs: np.ndarray) -> np.ndarray:
        return damped_sine_params(ts, xs, p0)

    samples = block_bootstrap(tt, xw, fit_fn, damped_sine, n=bootstrap_n, seed=seed)
    t_samples = 2.0 * math.pi / np.sqrt(samples[:, 2] ** 2 + samples[:, 1] ** 2)
    lo, hi = np.percentile(t_samples, [2.5, 97.5])
    sigma_boot = sigma_from_ci(float(lo), float(hi))
    sigma_analytic = sigma_boot
    if fit.cov is not None:
        cov = np.asarray(fit.cov)
        s2 = omega**2 + gamma**2
        grad = np.zeros(len(DAMPED_SINE_PARAMS))
        grad[2] = -2 * math.pi * omega / s2**1.5
        grad[1] = -2 * math.pi * gamma / s2**1.5
        sigma_analytic = float(math.sqrt(max(grad @ cov @ grad, 0.0)))
    t_factor = float(stats.t.ppf(0.975, max(fit.n - 6, 1)) / 1.96)
    sigma_stat = max(sigma_boot, sigma_analytic) * t_factor
    sigma_vfr = period * VFR_REL_SIGMA if is_vfr else 0.0
    sigma_period = combine_sigma(sigma_stat, sigma_vfr)

    # --- amplitude correction -------------------------------------------------------------
    env = p["A"] * np.exp(-gamma * tt)
    amplitude_px = float(np.mean(env))
    theta = None
    theta_source = None
    if radius_px and radius_px > 0:
        theta = np.arcsin(np.clip(env / radius_px, 0.0, 1.0))
        theta_source = "circle_fit"
    elif length_m and m_per_px:
        theta = np.arcsin(np.clip(env * m_per_px / length_m, 0.0, 1.0))
        theta_source = "length_scale"
    if theta is None:
        factor = 1.0
        warnings.append("no_amplitude_correction")
    else:
        factor = amplitude_factor(theta)
        if float(np.max(theta)) > math.radians(40):
            warnings.append("large_amplitude")
    period0 = period / factor
    sigma_corr_rel = (factor - 1.0) / 2.0 if theta is not None else 0.0
    sigma_period0 = period0 * math.sqrt((sigma_period / period) ** 2 + sigma_corr_rel**2)
    ci_t0 = (period0 - 1.96 * sigma_period0, period0 + 1.96 * sigma_period0)

    drift = p["d"]
    if abs(drift) * duration > 0.2 * amplitude_px:
        warnings.append("camera_drift")

    g = sigma_g = g_unc = None
    ci_g = comparison = None
    if length_m:
        g = g_from_pendulum(length_m, period0)
        g_unc = g_from_pendulum(length_m, period)
        rel_l = (sigma_length_m or 0.0) / length_m
        rel = math.sqrt(rel_l**2 + (2 * sigma_period / period) ** 2 + (2 * sigma_corr_rel) ** 2)
        sigma_g = g * rel
        ci_g = (g - 1.96 * sigma_g, g + 1.96 * sigma_g)
        comparison = compare_reference(g, ci_g[0], ci_g[1], g_ref)

    return PendulumAnalysis(
        fit=fit,
        start=start,
        end=end,
        period_damped=period_d,
        period=period,
        period0=period0,
        amplitude_px=amplitude_px,
        theta_mean_deg=float(np.degrees(np.mean(theta))) if theta is not None else None,
        theta_source=theta_source,
        amp_factor=factor,
        n_cycles=n_cycles,
        duration_s=duration,
        sigma_period_boot=sigma_boot,
        sigma_period_analytic=sigma_analytic,
        sigma_period_stat=sigma_stat,
        sigma_period_vfr=sigma_vfr,
        sigma_period=sigma_period,
        sigma_corr_rel=sigma_corr_rel,
        sigma_period0=sigma_period0,
        ci_period0=ci_t0,
        length_m=length_m,
        sigma_length_m=sigma_length_m,
        length_source=length_source if length_m else None,
        g=g,
        sigma_g=sigma_g,
        ci_g=ci_g,
        g_uncorrected=g_unc,
        comparison=comparison,
        drift_px_s=drift,
        bootstrap_n=bootstrap_n,
        warnings=warnings,
        t_fit=tw,
        x_fit=xw,
    )
