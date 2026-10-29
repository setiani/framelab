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


def pendulum_theta(
    L: float, theta0_deg: float, gamma: float, g: float, t: np.ndarray
) -> np.ndarray:
    """θ(t) of the exact nonlinear damped pendulum θ'' = −(g/L)·sin θ − 2γ·θ', released at rest."""
    from scipy.integrate import solve_ivp

    def rhs(_t, s):
        th, om = s
        return [om, -(g / L) * np.sin(th) - 2 * gamma * om]

    sol = solve_ivp(
        rhs, (0.0, float(t[-1]) + 1e-9), [np.radians(theta0_deg), 0.0], t_eval=t,
        rtol=1e-11, atol=1e-12, method="DOP853",
    )  # fmt: skip
    return sol.y[0]


def make_pendulum(
    L: float = 0.5,
    theta0_deg: float = 10.0,
    gamma: float = 0.03,
    fps: float = 60,
    seconds: float = 15.0,
    noise_px: float = 0.5,
    drift_px_s: float = 0.0,
    m_per_px: float = 1 / 460,
    g: float = 9.78,
    pivot_px: tuple[float, float] = (480.0, 40.0),
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """t, x_px, y_px of the bob centre on its true arc (image y down). Released at t = 0."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(round(seconds * fps))) / fps
    th = pendulum_theta(L, theta0_deg, gamma, g, t)
    r_px = L / m_per_px
    x = pivot_px[0] + r_px * np.sin(th) + drift_px_s * t
    y = pivot_px[1] + r_px * np.cos(th)
    return t, x + rng.normal(0, noise_px, t.size), y + rng.normal(0, noise_px, t.size)


def small_angle_period(L: float, g: float = 9.78) -> float:
    return 2 * np.pi * np.sqrt(L / g)


def exact_period(L: float, theta0_deg: float, g: float = 9.78) -> float:
    """Undamped finite-amplitude period via the complete elliptic integral K(k), k = sin(θ0/2)."""
    from scipy.special import ellipk

    k = np.sin(np.radians(theta0_deg) / 2)
    return 4 * np.sqrt(L / g) * ellipk(k**2)
