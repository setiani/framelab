"""Tests for the small helpers that are already implemented in the M0 skeleton."""

from __future__ import annotations

import math

import numpy as np
import pytest

from framelab.agent.auditor import finalize_verdict
from framelab.agent.gates import slowmo_factor
from framelab.config import mask_secret
from framelab.costs import sam_cost, token_cost, voice_cost
from framelab.physics.derive import (
    amplitude_correction,
    g_from_pendulum,
    g_from_quadratic,
    g_reference,
    hypothesis_test,
)
from framelab.physics.models import durbin_watson
from framelab.pipeline.segment import slug
from framelab.schemas import GateResult, Verdict


def test_costs() -> None:
    assert token_cost("muse-spark-1.3", 1_000_000, 0) == pytest.approx(1.25)
    assert token_cost("muse-spark-1.3", 1_000_000, 0, cached_tokens=1_000_000) == pytest.approx(
        0.15
    )
    assert token_cost("muse-wonder-1.0", 0, 1_000_000) == pytest.approx(20.0)
    assert sam_cost(frames=10) == pytest.approx(50 * 0.2 / 1000)  # 50-frame minimum
    assert sam_cost(images=2) == pytest.approx(0.005)
    assert voice_cost(3600) == pytest.approx(0.18)


def test_g_reference_jakarta() -> None:
    assert g_reference(-6.2, 8) == pytest.approx(9.781, abs=0.001)


def test_pendulum_helpers() -> None:
    L, g = 0.5, 9.781
    T0 = 2 * math.pi * math.sqrt(L / g)
    assert g_from_pendulum(L, T0) == pytest.approx(g)
    assert amplitude_correction(T0, 0.0) == pytest.approx(T0)
    assert amplitude_correction(T0, math.radians(20)) < T0
    assert g_from_quadratic(4350.0, 0.001124) == pytest.approx(9.7788, rel=1e-4)


def test_hypothesis_test_outcomes() -> None:
    assert hypothesis_test(1.43, 1.42, 1.40, 1.44)["outcome"] == "supported"
    assert hypothesis_test(1.47, 1.42, 1.41, 1.43)["outcome"] == "close"
    assert hypothesis_test(1.5, 1.42, 1.41, 1.43)["outcome"] == "not_supported"


def test_slowmo_factor() -> None:
    assert slowmo_factor(9.781 / 64, 9.781) == 8
    assert slowmo_factor(9.70, 9.781) is None


def test_durbin_watson() -> None:
    rng = np.random.default_rng(0)
    assert 1.6 < durbin_watson(rng.normal(size=2000)) < 2.4
    assert durbin_watson(np.linspace(-1, 1, 200)) < 0.1


def test_finalize_verdict() -> None:
    ok = GateResult(gate="G1", name="track", status="pass", hard=True)
    bad = GateResult(gate="G2", name="scale", status="fail", hard=True, remedy="rerecord")
    assert finalize_verdict(Verdict(status="PASS"), [ok]) == ("PASS", [])
    assert finalize_verdict(Verdict(status="PASS"), [ok, bad])[0] == "NEEDS_RERECORD"
    status, warnings = finalize_verdict(Verdict(status="REJECT"), [ok])
    assert status == "PASS_WITH_WARNINGS" and "auditor_unresolved" in warnings


def test_slug_and_mask_secret() -> None:
    assert slug("Orange Ball!") == "orange_ball"
    assert mask_secret(None) == "<missing>"
    assert "secret" not in mask_secret("LLM|123456|supersecretvalue")
