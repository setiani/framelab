"""Block bootstrap, error propagation and number formatting.

Spec: docs/plan/05-physics-and-uncertainty.md §6–§7
Status: skeleton (M0). Implement in milestone M3.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def block_bootstrap(
    t: np.ndarray,
    y: np.ndarray,
    fit_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    predict_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    n: int = 1000,
    block: int | None = None,
    seed: int = 1234,
) -> np.ndarray:
    """Residual moving-block bootstrap; returns an (n, n_params) array."""
    raise NotImplementedError("Milestone M3: see docs/plan/09-milestones-and-tasks.md")


def combine_sigma(sigma_stat: float, *sigma_sys: float) -> float:
    """√(σ_stat² + Σσ_sys²)."""
    return float(np.sqrt(sigma_stat**2 + sum(s**2 for s in sigma_sys)))


def fmt(value: float, sigma: float | None, unit: str, lang: str = "en") -> str:
    """'9.74 ± 0.12 m/s²' (EN) / '9,74 ± 0,12 m/s²' (ID). The ONLY way numbers are displayed."""
    raise NotImplementedError("Milestone M3: see docs/plan/09-milestones-and-tasks.md")
