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
    if fit.model == "damped_sine":
        return predict_damped_sine(fit, t)
    coeffs = np.array([fit.params[f"c{i}"] for i in range(len(fit.params))])
    return polyval(np.asarray(t, dtype=float) - fit.t_offset, coeffs)


DAMPED_SINE_PARAMS = ("A", "gamma", "omega", "phi", "c", "d")
F_MIN_HZ, F_MAX_HZ = 0.2, 5.0


def damped_sine(t: np.ndarray, p: np.ndarray) -> np.ndarray:
    """x = A·e^(−γt)·cos(ω_d·t + φ) + c + d·t (t already shifted to start at 0)."""
    a, gamma, omega, phi, c, d = p
    return a * np.exp(-gamma * t) * np.cos(omega * t + phi) + c + d * t


def _damped_sine_jac(t: np.ndarray, p: np.ndarray) -> np.ndarray:
    a, gamma, omega, phi, _c, _d = p
    env = np.exp(-gamma * t)
    cos = np.cos(omega * t + phi)
    sin = np.sin(omega * t + phi)
    return np.column_stack(
        [env * cos, -a * t * env * cos, -a * t * env * sin, -a * env * sin, np.ones_like(t), t]
    )


def guess_damped_sine(t: np.ndarray, x: np.ndarray) -> dict[str, float]:
    """Initial guesses (05 §3.2): FFT peak (Hann, zero-pad ×8, parabolic), Hilbert envelope."""
    from scipy.signal import hilbert

    t = np.asarray(t, dtype=float) - float(t[0])
    x = np.asarray(x, dtype=float)
    c0 = float(np.median(x))
    xd = x - c0
    dt = float(np.median(np.diff(t)))
    n = xd.size
    nfft = 1 << int(np.ceil(np.log2(n * 8)))
    spec = np.abs(np.fft.rfft(xd * np.hanning(n), nfft))
    freqs = np.fft.rfftfreq(nfft, dt)
    band = (freqs >= F_MIN_HZ) & (freqs <= F_MAX_HZ)
    if not band.any():
        raise ValueError("no frequency in the 0.2–5 Hz band (clip too short or fps too low)")
    idx = int(np.flatnonzero(band)[np.argmax(spec[band])])
    f0 = float(freqs[idx])
    if 0 < idx < spec.size - 1:  # parabolic interpolation of the peak
        a_, b_, c_ = np.log(spec[idx - 1 : idx + 2] + 1e-300)
        denom = a_ - 2 * b_ + c_
        if denom != 0:
            f0 += 0.5 * (a_ - c_) / denom * (freqs[1] - freqs[0])
    amp = float((np.percentile(xd, 95) - np.percentile(xd, 5)) / 2)
    analytic = hilbert(xd)
    env = np.abs(analytic)
    inner = slice(n // 10, n - n // 10) if n >= 20 else slice(None)
    try:
        slope = np.polyfit(t[inner], np.log(np.maximum(env[inner], 1e-9)), 1)[0]
    except (np.linalg.LinAlgError, ValueError):
        slope = 0.0
    gamma0 = float(np.clip(-slope, 0.0, 1.0))
    phi0 = float(np.angle(analytic[0]))
    return {"A": max(amp, 1e-6), "gamma": gamma0, "omega": 2 * np.pi * f0, "phi": phi0,
            "c": c0, "d": 0.0}  # fmt: skip


def _ls_damped_sine(tt: np.ndarray, x: np.ndarray, p0: np.ndarray):
    from scipy.optimize import least_squares

    lower = [0.0, 0.0, 2 * np.pi * F_MIN_HZ, -np.inf, -np.inf, -np.inf]
    upper = [np.inf, 5.0, 2 * np.pi * F_MAX_HZ, np.inf, np.inf, np.inf]
    p0 = np.clip(p0, np.array(lower) + 1e-12, np.array(upper) - 1e-12)
    return least_squares(
        lambda p: damped_sine(tt, p) - x,
        p0,
        jac=lambda p: _damped_sine_jac(tt, p),
        bounds=(lower, upper),
        method="trf",
        loss="linear",
        x_scale="jac",
    )


def damped_sine_params(t: np.ndarray, x: np.ndarray, p0: np.ndarray) -> np.ndarray:
    """Fast refit (bootstrap): t already shifted, warm start from ``p0``."""
    return _ls_damped_sine(np.asarray(t, float), np.asarray(x, float), np.asarray(p0, float)).x


def fit_damped_sine(
    t: np.ndarray, x: np.ndarray, guess: dict[str, float] | None = None
) -> FitResult:
    """x = A e^(−γt) cos(ω_d t + φ) + c + d t via least_squares (05 §3.3), pixels and seconds.

    The fit is started from the FFT guess and, as a guard against a wrong phase, from the
    guess with φ + π and A unchanged; the lower-cost solution wins.
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    if t.size < 12:
        raise ValueError(f"damped-sine fit needs at least 12 points, got {t.size}")
    t0 = float(t[0])
    tt = t - t0
    g0 = guess or guess_damped_sine(t, x)
    p0 = np.array([g0[k] for k in DAMPED_SINE_PARAMS])
    best = None
    for dphi in (0.0, np.pi, np.pi / 2, -np.pi / 2):
        start = p0.copy()
        start[3] += dphi
        res = _ls_damped_sine(tt, x, start)
        if best is None or res.cost < best.cost:
            best = res
    assert best is not None
    p = best.x.copy()
    p[3] = (p[3] + np.pi) % (2 * np.pi) - np.pi
    resid = x - damped_sine(tt, p)
    dof = t.size - p.size
    cov = None
    if dof > 0:
        jac = _damped_sine_jac(tt, p)
        try:
            cov = (np.linalg.inv(jac.T @ jac) * float(np.sum(resid**2)) / dof).tolist()
        except np.linalg.LinAlgError:
            cov = None
    return FitResult(
        model="damped_sine",
        params={k: float(v) for k, v in zip(DAMPED_SINE_PARAMS, p, strict=True)},
        cov=cov,
        r2=r_squared(x, x - resid),
        rms_px=float(np.sqrt(np.mean(resid**2))),
        durbin_watson=durbin_watson(resid),
        n=int(t.size),
        t_offset=t0,
        residuals=resid,
    )


def predict_damped_sine(fit: FitResult, t: np.ndarray) -> np.ndarray:
    p = np.array([fit.params[k] for k in DAMPED_SINE_PARAMS])
    return damped_sine(np.asarray(t, dtype=float) - fit.t_offset, p)
