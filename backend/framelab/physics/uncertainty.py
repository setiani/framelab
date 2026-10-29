"""Block bootstrap, error propagation and number formatting.

Spec: docs/plan/05-physics-and-uncertainty.md §6–§7. ``fmt`` is the ONLY way numbers reach
the UI, overlay and reports.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

UNIT_DISPLAY = {
    "m/s^2": "m/s²",
    "m/s2": "m/s²",
    "s^-1": "s⁻¹",
    "1/s": "s⁻¹",
    "m^2": "m²",
    "deg": "°",
}


def block_bootstrap(
    t: np.ndarray,
    y: np.ndarray,
    fit_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    predict_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    n: int = 1000,
    block: int | None = None,
    seed: int = 1234,
) -> np.ndarray:
    """Residual moving-block bootstrap (05 §6.1); returns an (n, n_params) array.

    Residuals of a time series are autocorrelated, so they are resampled in blocks of length
    ``block`` (default max(3, round(N^(1/3)))). The seed is fixed for reproducibility.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    rng = np.random.default_rng(seed)
    params = fit_fn(t, y)
    yhat = predict_fn(t, params)
    resid = y - yhat
    length = block or max(3, int(round(len(resid) ** (1 / 3))))
    length = min(length, len(resid))
    starts = np.arange(len(resid) - length + 1)
    n_blocks = int(math.ceil(len(resid) / length))
    out = np.empty((n, len(params)))
    for i in range(n):
        chosen = rng.choice(starts, size=n_blocks)
        idx = (chosen[:, None] + np.arange(length)[None, :]).ravel()[: len(resid)]
        out[i] = fit_fn(t, yhat + resid[idx])
    return out


def percentile_ci(samples: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    alpha = (1 - level) / 2 * 100
    lo, hi = np.percentile(np.asarray(samples, dtype=float), [alpha, 100 - alpha])
    return float(lo), float(hi)


def sigma_from_ci(lo: float, hi: float) -> float:
    """σ of a normal distribution whose 95 % interval is [lo, hi]."""
    return (hi - lo) / 3.92


def combine_sigma(sigma_stat: float, *sigma_sys: float) -> float:
    """√(σ_stat² + Σσ_sys²)."""
    return float(np.sqrt(sigma_stat**2 + sum(s**2 for s in sigma_sys)))


# --------------------------------------------------------------------------------------------
# Formatting (05 §7)
# --------------------------------------------------------------------------------------------


def _decimals_for_sigma(sigma: float) -> int:
    """Decimal places so σ keeps 2 significant figures if it starts with 1 or 2, else 1."""
    if sigma <= 0 or not math.isfinite(sigma):
        return 3
    exponent = math.floor(math.log10(sigma))
    leading = int(sigma / 10**exponent)
    sig_figs = 2 if leading in (1, 2) else 1
    decimals = sig_figs - 1 - exponent
    # rounding may bump σ to the next decade (e.g. 0.0996 → 0.10): recompute once
    rounded = round(sigma, decimals)
    if rounded > 0 and math.floor(math.log10(rounded)) != exponent:
        return _decimals_for_sigma(rounded)
    return max(decimals, 0)


def _num(value: float, decimals: int, lang: str) -> str:
    text = f"{value:.{decimals}f}"
    if text.startswith("-") and float(text) == 0:
        text = text[1:]
    if lang == "id":
        text = text.replace(".", ",")
    return text


def unit_display(unit: str) -> str:
    return UNIT_DISPLAY.get(unit, unit)


def fmt(value: float, sigma: float | None, unit: str, lang: str = "en") -> str:
    """'9.74 ± 0.12 m/s²' (EN) / '9,74 ± 0,12 m/s²' (ID).

    σ keeps 2 significant figures when its first digit is 1 or 2, otherwise 1; the value is
    rounded to the same decimal place. Without σ the value is shown with 3 significant figures.
    """
    u = unit_display(unit)
    sep = "" if u in ("°", "%") else " "
    if sigma is None or not math.isfinite(sigma) or sigma <= 0:
        decimals = 2 if value == 0 else max(0, 2 - math.floor(math.log10(abs(value))))
        return f"{_num(value, decimals, lang)}{sep}{u}".rstrip()
    d = _decimals_for_sigma(sigma)
    return f"{_num(value, d, lang)} ± {_num(sigma, d, lang)}{sep}{u}".rstrip()


def fmt_interval(lo: float, hi: float, unit: str, sigma: float, lang: str = "en") -> str:
    """'[9.62, 9.86] m/s²' / '[9,62; 9,86] m/s²' with the decimals implied by σ."""
    d = _decimals_for_sigma(sigma)
    sep = "; " if lang == "id" else ", "
    return f"[{_num(lo, d, lang)}{sep}{_num(hi, d, lang)}] {unit_display(unit)}"


def fmt_pct(value: float, lang: str = "en", decimals: int = 1) -> str:
    sign = "+" if value > 0 else ""
    return (
        f"{sign}{_num(value, decimals, lang)} %" if lang == "id" else f"{sign}{value:.{decimals}f}%"
    )
