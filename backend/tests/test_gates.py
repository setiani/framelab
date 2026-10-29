"""M6 gates: each threshold boundary, the diagnosis codes, EN/ID messages, and G5 (06 §7)."""

from __future__ import annotations

import numpy as np
import pytest

from framelab.agent import gates
from framelab.agent.gates import TEMPLATES, GateThresholds, slowmo_factor
from framelab.agent.ledger import Ledger
from framelab.schemas import Calibration, ReportDraft, ScaleEstimate, Track, TrackPoint

THR = GateThresholds()


def track(
    n_valid: int,
    n_total: int | None = None,
    *,
    steps: np.ndarray | None = None,
    id_switches: int = 0,
) -> Track:
    """A straight fall of ``n_valid`` valid points with 1 px steps (or the given steps)."""
    n_total = n_total or n_valid
    steps = np.ones(n_valid - 1) if steps is None else steps
    y = np.concatenate([[0.0], np.cumsum(steps)])
    points = []
    for i in range(n_total):
        ok = i < n_valid
        points.append(
            TrackPoint(
                t=i / 60,
                x_px=0.0,
                y_px=float(y[i]) if ok else 0.0,
                area_px=100.0,
                valid=ok,
                flag="ok" if ok else "missing",
            )
        )
    return Track(
        track_id="orange_ball_1",
        concept="orange ball",
        object_id="1",
        fps=60.0,
        points=points,
        coverage=n_valid / n_total,
        id_switches_suspected=id_switches,
    )


def calibration(agreement: float | None, *, two: bool = True) -> Calibration:
    primary = ScaleEstimate(method="ruler", m_per_px=0.001, sigma_m_per_px=0.00001)
    secondary = (
        ScaleEstimate(method="object_diameter", m_per_px=0.001, sigma_m_per_px=0.00001)
        if two
        else None
    )
    return Calibration(primary=primary, secondary=secondary, agreement_pct=agreement)


# --- G1 ---------------------------------------------------------------------------------------


def test_g1_coverage_boundary() -> None:
    assert gates.gate_track(track(89, 100), THR, min_points=8).diagnosis == "low_coverage_blur"
    assert gates.gate_track(track(90, 100), THR, min_points=8).status == "pass"


def test_g1_too_few_points_is_a_rerecord() -> None:
    result = gates.gate_track(track(7), THR, min_points=8)
    assert (result.status, result.diagnosis, result.remedy) == (
        "fail",
        "too_few_points",
        "rerecord",
    )
    assert result.hard


def test_g1_jump_and_id_switch_ask_for_reanalysis() -> None:
    steps = np.ones(40)
    steps[20] = 9.0  # one step nine times the median
    result = gates.gate_track(track(41, steps=steps), THR, min_points=8)
    assert (result.diagnosis, result.remedy) == ("id_switch", "reanalyze")
    assert gates.gate_track(track(40, id_switches=1), THR, min_points=8).diagnosis == "id_switch"


def test_g1_pendulum_needs_three_cycles() -> None:
    assert gates.gate_track(track(60), THR, min_points=8, cycles=2.5).diagnosis == "too_few_cycles"
    assert gates.gate_track(track(60), THR, min_points=8, cycles=3.0).status == "pass"


# --- G2 ---------------------------------------------------------------------------------------


def test_g2_disagreement_boundary_and_diagnosis() -> None:
    assert gates.gate_scale(calibration(5.0), THR).status == "pass"
    failed = gates.gate_scale(calibration(5.1), THR)
    assert (failed.status, failed.diagnosis, failed.hard) == ("fail", "ruler_offplane", True)
    assert failed.remedy == "rerecord"
    assert "mm/px" in failed.message_en


def test_g2_pendulum_uses_its_own_limit() -> None:
    # the caller passes the experiment's thresholds (REGISTRY: pendulum 8 %)
    pendulum_thr = GateThresholds(g2_max_scale_disagreement_pct=8.0)
    assert gates.gate_scale(calibration(6.0), THR).status == "fail"
    assert gates.gate_scale(calibration(6.0), pendulum_thr, pendulum=True).status == "pass"


def test_g2_one_method_warns_and_no_scale_fails() -> None:
    warn = gates.gate_scale(calibration(None, two=False), THR)
    assert (warn.status, warn.hard) == ("warn", False)
    missing = gates.gate_scale(None, THR)
    assert (missing.status, missing.diagnosis, missing.hard) == ("fail", "no_scale", True)


# --- G3 ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("r2", "rms", "dw", "status", "diagnosis"),
    [
        (0.994, 0.5, 1.5, "fail", "wrong_window"),  # R² below 0.995
        (0.999, 2.5, 1.5, "fail", "wrong_window"),  # RMS above 2 px
        (0.999, 0.5, 0.4, "fail", "systematic_residuals"),  # DW below 0.5
        (0.999, 0.5, 0.8, "warn", "systematic_residuals"),  # DW below 1.0
        (0.999, 0.5, 1.5, "pass", None),
    ],
)
def test_g3_boundaries(r2, rms, dw, status, diagnosis) -> None:
    result = gates.gate_fit({"r2": r2, "rms_px": rms, "durbin_watson": dw}, THR)
    assert result.status == status
    assert result.diagnosis == diagnosis


# --- G4 ---------------------------------------------------------------------------------------


def test_g4_plausibility_is_soft_and_names_slowmo() -> None:
    assert gates.gate_plausibility(9.81, THR).status == "pass"
    out = gates.gate_plausibility(8.0, THR)
    assert (out.status, out.diagnosis, out.hard) == ("warn", "out_of_range", False)
    slow = gates.gate_plausibility(9.81 / 64, THR)  # g·8² = g_ref
    assert slow.diagnosis == "slowmo_retimed"
    assert "fps = 240" in slow.message_en


def test_slowmo_factor_finds_k() -> None:
    assert slowmo_factor(9.81 / 16, 9.81) == 4  # g·4² = g_ref
    assert slowmo_factor(9.81, 9.81) is None


# --- G5 ---------------------------------------------------------------------------------------


def draft(**changes) -> dict:
    base = {
        "lang": "en",
        "title": "Free fall",
        "aim": "Measure g.",
        "setup": "A ball was dropped.",
        "method": "A quadratic fit was used.",
        "results": [{"heading": "Result", "body": "The value is {{c01}}."}],
        "hypothesis_evaluation": None,
        "error_analysis": "The spread is small.",
        "conclusion": "See {{c01}}.",
        "claims_used": ["c01"],
    }
    base.update(changes)
    return base


@pytest.fixture
def ledger_with_claim(tmp_path) -> Ledger:
    ledger = Ledger(tmp_path / "ledger.jsonl")
    ev = ledger.add_evidence("der", "derive", {"quantity": "g", "value": 9.8, "unit": "m/s^2",
                                              "ci95": [9.7, 9.9]}, "g")  # fmt: skip
    ledger.add_claim("derived", ev.id, "g = {value}", "g = {value}", quantity="g")
    return ledger


def test_g5_passes_a_clean_draft(ledger_with_claim) -> None:
    assert gates.gate_g5(draft(), ledger_with_claim).status == "pass"


def test_g5_flags_a_bare_number(ledger_with_claim) -> None:
    result = gates.gate_g5(draft(conclusion="g is about 9.8."), ledger_with_claim)
    assert result.status == "fail" and result.diagnosis == "bare_number"
    assert "9" in result.message_en


def test_g5_allow_list_lets_standard_tokens_through(ledger_with_claim) -> None:
    text = "Muse Spark 1.3 and SAM 3.1 with G1 to G5, at 95% confidence. T₀² is fine.\n1. Method"
    assert gates.gate_g5(draft(method=text), ledger_with_claim).status == "pass"


def test_g5_unknown_claim_and_claims_used_mismatch(ledger_with_claim) -> None:
    unknown = gates.gate_g5(
        draft(conclusion="See {{c09}}.", claims_used=["c09"]), ledger_with_claim
    )
    assert unknown.diagnosis == "unknown_claim"
    mismatch = gates.gate_g5(draft(claims_used=[]), ledger_with_claim)
    assert mismatch.diagnosis == "claims_mismatch"


def test_g5_missing_hypothesis_claim(ledger_with_claim) -> None:
    result = gates.gate_g5(draft(), ledger_with_claim, hypothesis=True)
    assert result.diagnosis == "missing_hypothesis_claim"


def test_every_gate_diagnosis_has_english_and_indonesian_text() -> None:
    for key, (en, idn) in TEMPLATES.items():
        assert en and idn, key


def test_report_draft_validates_the_same_shape() -> None:
    assert ReportDraft.model_validate(draft()).claims_used == ["c01"]
