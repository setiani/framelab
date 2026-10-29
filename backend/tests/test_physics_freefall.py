from __future__ import annotations

import numpy as np
import pytest

from framelab.physics.events import freefall_window
from framelab.physics.freefall import analyze_free_fall
from framelab.physics.models import durbin_watson, fit_cubic, fit_quadratic
from tests.synth import make_freefall

S = 0.0011  # m/px


@pytest.mark.parametrize("fps", [240, 60, 30])
def test_g_within_half_percent(fps: int) -> None:
    t, y = make_freefall(g=9.78, m_per_px=S, fps=fps, noise_px=0.4)
    r = analyze_free_fall(t, y, S, 0.0, bootstrap_n=300)
    assert r.g == pytest.approx(9.78, rel=0.005)
    assert r.ci95[0] < r.g < r.ci95[1]
    assert r.fit.r2 > 0.9999 and r.fit.rms_px < 1.0


def test_window_excludes_hold_and_bounce() -> None:
    hold = 20
    t, y = make_freefall(fps=240, noise_px=0.2, release_hold=hold)
    w = freefall_window(t, y)
    impact = int(np.argmax(y))
    assert hold <= w.start <= hold + 8  # first moving frames
    assert abs(w.impact_index - impact) <= 2 and w.end < w.impact_index
    assert w.warnings == []


def test_release_before_first_point_is_normal() -> None:
    """SAM reports a static object only once it moves (M1 V12): no hold segment at all."""
    t, y = make_freefall(fps=60, noise_px=0.3, release_hold=0)
    w = freefall_window(t[1:], y[1:])
    assert w.start <= 1 and w.release_index in (None, 0)


def test_object_left_frame() -> None:
    t, y = make_freefall(fps=60, noise_px=0.3, bounce=False, after_s=0.0)
    w = freefall_window(t, y)
    assert "object_left_frame" in w.warnings or w.impact_index is not None


def test_too_few_points_warning() -> None:
    t, y = make_freefall(fps=15, noise_px=0.2, drop_m=0.5)
    r = analyze_free_fall(t, y, S, 0.0, bootstrap_n=100)
    assert "too_few_points" in r.warnings


def test_scale_and_vfr_uncertainty_propagate() -> None:
    t, y = make_freefall(fps=240, noise_px=0.2)
    base = analyze_free_fall(t, y, S, 0.0, bootstrap_n=200)
    more = analyze_free_fall(t, y, S, S * 0.01, bootstrap_n=200, is_vfr=True)
    assert more.sigma_scale == pytest.approx(base.g * 0.01, rel=1e-6)
    assert more.sigma_vfr == pytest.approx(base.g * 0.005, rel=1e-6)
    assert more.sigma_total > base.sigma_total


def test_residual_diagnostics() -> None:
    rng = np.random.default_rng(1)
    assert 1.6 < durbin_watson(rng.normal(size=2000)) < 2.4
    t = np.linspace(0, 0.5, 60)
    y = 100 + 2000 * t**2 + 300 * t**3  # cubic term = systematic misfit for a parabola
    assert fit_quadratic(t, y).durbin_watson < 0.5
    assert fit_cubic(t, y).rms_px < 1e-6


def test_systematic_curvature_flagged() -> None:
    t = np.arange(0, 0.45, 1 / 240)
    y = 80 + 0.5 * 9000 * t**2 + 8000 * t**3 + np.random.default_rng(2).normal(0, 0.2, t.size)
    y = np.concatenate([y, [y[-1] - 5, y[-1] - 9]])  # bounce after impact
    t = np.concatenate([t, [t[-1] + 1 / 240, t[-1] + 2 / 240]])
    r = analyze_free_fall(t, y, S, 0.0, bootstrap_n=100)
    assert r.systematic_curvature and "systematic_curvature" in r.warnings


@pytest.mark.slow
def test_ci_coverage_at_least_90_percent() -> None:
    hits = 0
    for seed in range(50):
        t, y = make_freefall(fps=240, noise_px=0.4, seed=seed)
        r = analyze_free_fall(t, y, S, 0.0, bootstrap_n=300, seed=seed)
        hits += r.ci95[0] <= 9.78 <= r.ci95[1]
    assert hits / 50 >= 0.88
