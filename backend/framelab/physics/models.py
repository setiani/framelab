"""Model functions and fitters (quadratic, cubic, damped sine, linear area).

Spec: docs/plan/05-physics-and-uncertainty.md §2–§3
Status: skeleton (M0). Implement in milestone M3–M4.
"""

from __future__ import annotations

from dataclasses import dataclass

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


def fit_quadratic(t: np.ndarray, y: np.ndarray) -> FitResult:
    """y = c0 + c1 t + c2 t² in pixels (t shifted to start at 0)."""
    raise NotImplementedError("Milestone M3–M4: see docs/plan/09-milestones-and-tasks.md")


def guess_damped_sine(t: np.ndarray, x: np.ndarray) -> dict[str, float]:
    """FFT (zero-pad ×8, parabolic peak), Hilbert envelope for gamma (05 §3.2)."""
    raise NotImplementedError("Milestone M3–M4: see docs/plan/09-milestones-and-tasks.md")


def fit_damped_sine(t: np.ndarray, x: np.ndarray) -> FitResult:
    """x = A e^(−γt) cos(ω_d t + φ) + c + d t via least_squares (05 §3.3)."""
    raise NotImplementedError("Milestone M3–M4: see docs/plan/09-milestones-and-tasks.md")


def durbin_watson(residuals: np.ndarray) -> float:
    """Σ(e_i − e_{i−1})² / Σ e_i²."""
    e = np.asarray(residuals, dtype=float)
    denom = float(np.sum(e**2))
    return float(np.sum(np.diff(e) ** 2) / denom) if denom > 0 else float("nan")
