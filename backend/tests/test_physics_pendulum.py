"""Pendulum physics on the exact nonlinear ODE (05 §3, §8; 08 test_physics_pendulum)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from framelab.physics.derive import hypothesis_test, pendulum_series
from framelab.physics.models import fit_damped_sine, guess_damped_sine
from framelab.physics.pendulum import amplitude_factor, analyze_pendulum, period_from_omega
from framelab.pipeline.calibrate import pivot_from_track, spoken_length_scale
from tests.synth import exact_period, make_pendulum, small_angle_period

S = 1 / 460
L = 0.5
R_PX = L / S


def test_guess_finds_the_frequency() -> None:
    t, x, _y = make_pendulum(theta0_deg=10, gamma=0.03)
    g0 = guess_damped_sine(t, x)
    f_true = 1 / exact_period(L, 10)
    assert g0["omega"] / (2 * math.pi) == pytest.approx(f_true, rel=0.01)
    assert 0.0 <= g0["gamma"] < 0.1


@pytest.mark.parametrize("theta0", [5, 20])
def test_measured_period_matches_exact_nonlinear_period(theta0: float) -> None:
    t, x, _y = make_pendulum(theta0_deg=theta0, gamma=0.0)
    fit = fit_damped_sine(t, x)
    period = period_from_omega(fit.params["omega"], fit.params["gamma"])
    assert period == pytest.approx(exact_period(L, theta0), rel=0.001)


@pytest.mark.parametrize("theta0", [5, 20])
def test_t0_and_g_after_amplitude_correction(theta0: float) -> None:
    t, x, _y = make_pendulum(theta0_deg=theta0, gamma=0.03)
    r = analyze_pendulum(t, x, length_m=L, sigma_length_m=0.0, radius_px=R_PX, bootstrap_n=150)
    assert r.period0 == pytest.approx(small_angle_period(L), rel=0.001)
    assert r.g == pytest.approx(9.78, rel=0.003)
    assert r.ci_g[0] <= 9.78 <= r.ci_g[1]
    assert r.n_cycles > 9 and r.warnings == []


def test_amplitude_correction_matters_at_20_degrees() -> None:
    t, x, _y = make_pendulum(theta0_deg=20, gamma=0.0)
    r = analyze_pendulum(t, x, length_m=L, sigma_length_m=0.0, radius_px=R_PX, bootstrap_n=100)
    assert abs(r.g / 9.78 - 1) < 0.003  # with correction
    assert abs(r.g_uncorrected / 9.78 - 1) > 0.005  # without: ~1.5 % low
    assert 18 < r.theta_mean_deg < 21 and r.theta_source == "circle_fit"


def test_amplitude_from_length_and_scale_when_no_pivot() -> None:
    t, x, _y = make_pendulum(theta0_deg=15, gamma=0.02)
    r = analyze_pendulum(t, x, length_m=L, sigma_length_m=0.005, m_per_px=S, bootstrap_n=100)
    assert r.theta_source == "length_scale" and r.g == pytest.approx(9.78, rel=0.003)


def test_period_only_without_length() -> None:
    t, x, _y = make_pendulum(theta0_deg=8)
    r = analyze_pendulum(t, x, bootstrap_n=100)
    assert r.g is None and r.length_source is None
    assert "no_amplitude_correction" in r.warnings
    assert r.period == pytest.approx(exact_period(L, 8), rel=0.002)


def test_drift_term_absorbs_camera_drift() -> None:
    t, x, _y = make_pendulum(theta0_deg=10, drift_px_s=3.0)
    r = analyze_pendulum(t, x, length_m=L, sigma_length_m=0.0, radius_px=R_PX, bootstrap_n=100)
    assert r.drift_px_s == pytest.approx(3.0, abs=0.2)
    assert r.g == pytest.approx(9.78, rel=0.003)


def test_few_cycles_warnings() -> None:
    t, x, _y = make_pendulum(theta0_deg=10, seconds=5.0)
    assert "few_cycles" in analyze_pendulum(t, x, bootstrap_n=50).warnings
    t, x, _y = make_pendulum(theta0_deg=10, seconds=3.5)
    assert "too_few_cycles" in analyze_pendulum(t, x, bootstrap_n=50).warnings


def test_trims_hand_catching_the_bob() -> None:
    t, x, _y = make_pendulum(theta0_deg=10)
    x = x.copy()
    x[-20:] = x[-21] + np.linspace(0, 40, 20)  # caught and pulled aside
    r = analyze_pendulum(t, x, length_m=L, sigma_length_m=0.0, radius_px=R_PX, bootstrap_n=50)
    assert r.end <= len(t) - 10  # the first pulled points deviate < threshold
    assert r.g == pytest.approx(9.78, rel=0.003)


def test_amplitude_factor_series() -> None:
    assert amplitude_factor(np.array([0.0])) == 1.0
    th = math.radians(20)
    exact = exact_period(1.0, 20) / small_angle_period(1.0)
    assert amplitude_factor(np.array([th])) == pytest.approx(exact, rel=2e-5)


@pytest.mark.parametrize("theta0", [10, 20])
def test_pivot_circle_fit(theta0: float) -> None:
    t, x, y = make_pendulum(theta0_deg=theta0, noise_px=0.5, pivot_px=(480.0, 40.0))
    pv = pivot_from_track(x, y, string_top=(480.5, 41.0), n_boot=100)
    assert pv["reliable"]
    assert pv["r_px"] == pytest.approx(R_PX, abs=3 * pv["sigma_r_px"] + 0.5)
    assert pv["cx"] == pytest.approx(480, abs=1.0)
    assert "pivot_string_mismatch" not in pv["warnings"]
    est = spoken_length_scale(L, pv["r_px"], pv["sigma_r_px"])
    assert est.m_per_px == pytest.approx(S, rel=0.02)


def test_pivot_unreliable_on_tiny_swing_and_string_mismatch() -> None:
    _t, x, y = make_pendulum(theta0_deg=2, noise_px=0.5)
    pv = pivot_from_track(x, y, string_top=(300.0, 40.0), n_boot=20)
    assert not pv["reliable"] and "pivot_unreliable_small_swing" in pv["warnings"]
    assert "pivot_string_mismatch" in pv["warnings"]


def test_hypothesis_outcomes() -> None:
    assert hypothesis_test(1.42, 1.418, 1.41, 1.43)["outcome"] == "supported"
    assert hypothesis_test(1.47, 1.418, 1.41, 1.43)["outcome"] == "close"
    h = hypothesis_test(2.0, 1.418, 1.41, 1.43)
    assert h["outcome"] == "not_supported" and h["rel_diff_pct"] > 40


def _series_points(lengths, rel_sigma_t=0.0005, sigma_l=0.002, offset=0.0, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for length in lengths:
        t0 = small_angle_period(length + offset)
        noisy_t0 = t0 * (1 + rng.normal(0, rel_sigma_t))
        noisy_l = length + rng.normal(0, sigma_l)
        out.append({"L": noisy_l, "T0": noisy_t0, "sigma_T": t0 * rel_sigma_t, "sigma_L": sigma_l})
    return out


def test_series_three_lengths() -> None:
    r = pendulum_series(_series_points([0.3, 0.5, 0.8]), n_boot=500)
    assert r["g"] == pytest.approx(9.78, rel=0.005)
    assert r["n"] == pytest.approx(0.5, abs=0.02) and r["sqrt_law_supported"]
    assert r["ci_g"][0] <= 9.78 <= r["ci_g"][1]
    assert r["flags"] == []


def test_series_flags_length_offset() -> None:
    # L measured to the top of the bob instead of its centre: every L is 3 cm short.
    pts = _series_points([0.3, 0.45, 0.6, 0.8], offset=0.03, sigma_l=0.001)
    r = pendulum_series(pts, n_boot=200)
    assert "length_offset" in r["flags"]
    assert r["g"] == pytest.approx(9.78, rel=0.01)  # slope is unaffected by the offset


def test_series_input_validation() -> None:
    with pytest.raises(ValueError):
        pendulum_series(_series_points([0.3, 0.5]))
    with pytest.raises(ValueError):
        pendulum_series(_series_points([0.5, 0.5, 0.5], sigma_l=0.0))


@pytest.mark.slow
def test_period_ci_coverage() -> None:
    hits = 0
    for seed in range(40):
        t, x, _y = make_pendulum(theta0_deg=2, gamma=0.0, seed=seed)
        r = analyze_pendulum(t, x, bootstrap_n=150, seed=seed)
        hits += abs(r.period - exact_period(L, 2)) <= 1.96 * r.sigma_period
    assert hits / 40 >= 0.88
