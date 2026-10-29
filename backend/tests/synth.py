"""Synthetic data generators for physics tests (docs/plan/05 §8)."""

from __future__ import annotations

import numpy as np


def make_freefall(
    g: float = 9.78,
    m_per_px: float = 0.0011,
    fps: float = 240,
    drop_m: float = 1.0,
    noise_px: float = 0.4,
    release_hold: int = 20,
    bounce: bool = True,
    restitution: float = 0.5,
    after_s: float = 0.3,
    seed: int = 0,
    y0_px: float = 80.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Ball held for ``release_hold`` frames, dropped from rest, optional bounce. Image y down."""
    rng = np.random.default_rng(seed)
    g_px = g / m_per_px
    drop_px = drop_m / m_per_px
    t_fall = np.sqrt(2 * drop_px / g_px)
    total = release_hold / fps + t_fall + after_s
    t = np.arange(int(total * fps)) / fps
    y = np.empty_like(t)
    t_rel = release_hold / fps
    v_up = g_px * t_fall * restitution
    for i, ti in enumerate(t):
        if ti <= t_rel:
            y[i] = y0_px
        elif ti <= t_rel + t_fall:
            y[i] = y0_px + 0.5 * g_px * (ti - t_rel) ** 2
        elif bounce:
            tb = ti - t_rel - t_fall
            y[i] = y0_px + drop_px - max(0.0, v_up * tb - 0.5 * g_px * tb**2)
        else:
            y[i] = y0_px + drop_px
    return t, y + rng.normal(0, noise_px, size=y.size)
