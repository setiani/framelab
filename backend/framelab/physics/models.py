"""Model functions and least-squares fitters.

Spec: docs/plan/05-physics-and-uncertainty.md §2 (free fall, M3) and §3 (pendulum, M4).
All fits work in PIXELS and SECONDS so fit quality is independent of the scale calibration.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class FitResult:
    model: str
    params: dict[str, float]
    cov: list[list[float]] | None
    r2: float
    rms_px: float
    durbin_watson: float
    n: int
    t_offset: float = 0.0  # fits use t − t_offset for numerical stability
    residuals: np.ndarray = field(default_factory=lambda: np.zeros(0), repr=False)

    def summary(self) -> dict:
        return {
            "model": self.model,
            "params": {k: float(v) for k, v in self.params.items()},
            "cov": self.cov,
            "r2": float(self.r2),
            "rms_px": float(self.rms_px),
            "durbin_watson": float(self.durbin_watson),
            "n": int(self.n),
            "t_offset": float(self.t_offset),
            "max_abs_resid_px": float(np.max(np.abs(self.residuals)))
            if self.residuals.size
            else 0.0,
        }


def durbin_watson(residuals: np.ndarray) -> float:
    """Σ(e_i − e_{i−1})² / Σ e_i². ≈2 for white noise, ≪1 for systematic residuals."""
    e = np.asarray(residuals, dtype=float)
    denom = float(np.sum(e**2))
    return float(np.sum(np.diff(e) ** 2) / denom) if denom > 0 else float("nan")


def r_squared(y: np.ndarray, yhat: np.ndarray) -> float:
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def polyfit_params(t: np.ndarray, y: np.ndarray, degree: int) -> np.ndarray:
    """Coefficients c0..c_degree (ascending) of a least-squares polynomial. Fast (bootstrap)."""
    return np.polynomial.polynomial.polyfit(t, y, degree)


def polyval(t: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    return np.polynomial.polynomial.polyval(t, coeffs)


def fit_polynomial(t: np.ndarray, y: np.ndarray, degree: int, name: str) -> FitResult:
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if t.size < degree + 2:
        raise ValueError(f"{name} fit needs at least {degree + 2} points, got {t.size}")
    t0 = float(t[0])
    tt = t - t0
    coeffs = polyfit_params(tt, y, degree)
    yhat = polyval(tt, coeffs)
    resid = y - yhat
    cov = None
    dof = t.size - (degree + 1)
    if dof > 0:
        vander = np.vander(tt, degree + 1, increasing=True)
        sigma2 = float(np.sum(resid**2)) / dof
        try:
            cov = (np.linalg.inv(vander.T @ vander) * sigma2).tolist()
        except np.linalg.LinAlgError:
            cov = None
    return FitResult(
        model=name,
        params={f"c{i}": float(c) for i, c in enumerate(coeffs)},
        cov=cov,
        r2=r_squared(y, yhat),
        rms_px=float(np.sqrt(np.mean(resid**2))),
        durbin_watson=durbin_watson(resid),
        n=int(t.size),
        t_offset=t0,
        residuals=resid,
    )


def fit_quadratic(t: np.ndarray, y: np.ndarray) -> FitResult:
    """y = c0 + c1·(t−t0) + c2·(t−t0)² in pixels; free fall a_px = 2·c2 (05 §2.2)."""
    return fit_polynomial(t, y, 2, "quadratic")


def fit_cubic(t: np.ndarray, y: np.ndarray) -> FitResult:
    """Cubic control fit used to detect systematic curvature (05 §2.3)."""
    return fit_polynomial(t, y, 3, "cubic")


def predict(fit: FitResult, t: np.ndarray) -> np.ndarray:
    coeffs = np.array([fit.params[f"c{i}"] for i in range(len(fit.params))])
    return polyval(np.asarray(t, dtype=float) - fit.t_offset, coeffs)


def guess_damped_sine(t: np.ndarray, x: np.ndarray) -> dict[str, float]:
    """FFT (zero-pad ×8, parabolic peak), Hilbert envelope for gamma (05 §3.2). M4."""
    raise NotImplementedError("Milestone M4: see docs/plan/09-milestones-and-tasks.md")


def fit_damped_sine(t: np.ndarray, x: np.ndarray) -> FitResult:
    """x = A e^(−γt) cos(ω_d t + φ) + c + d t via least_squares (05 §3.3). M4."""
    raise NotImplementedError("Milestone M4: see docs/plan/09-milestones-and-tasks.md")
