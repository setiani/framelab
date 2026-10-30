"""Fixes from the M5–M9 audit (2026-10-30): pendulum agent path, render order, early
re-recording, fall-time hypothesis, G5 hypothesis evidence, cost guard, fallback cap, diameter
traceability and the new gate diagnoses. Agents are scripted fakes; SAM is replayed."""

from __future__ import annotations

import json
import shutil

import pytest

from framelab.agent import auditor, gates, planner
from framelab.agent.ledger import Ledger
from framelab.clients import meta as meta_client
from framelab.config import Settings
from framelab.pipeline import report
from framelab.pipeline.context import CostGuardExceeded, RunContext
from framelab.pipeline.run import execute_run
from framelab.schemas import (
    ExperimentSpec,
    Quantity,
    ReferenceObject,
    Track,
    TrackPoint,
    Verdict,
)
from framelab.store import RunStore
from tests.fakes import MEDIA, FakeChat, freefall_fake, pendulum_fake
from tests.test_agent_loop import CLAIM_CMP, CLAIM_G, HAPPY_PATH, REPORT, planner_spec
from tests.test_revise_loop import verdict

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")


def _capture_pdfs(monkeypatch) -> dict[str, str]:
    """Keep the HTML that would become each PDF (Chromium is not needed for these checks)."""
    captured: dict[str, str] = {}

    def fake_pdf(html, out_pdf):
        captured[out_pdf.name] = html
        out_pdf.parent.mkdir(parents=True, exist_ok=True)
        out_pdf.write_bytes(b"%PDF-1.4 test")
        return out_pdf

    monkeypatch.setattr(report, "html_to_pdf", fake_pdf)
    return captured


async def _run(tmp_path, monkeypatch, spec, routes, *, video="freefall_60fps_5s.mp4", sam=None):
    (sam or freefall_fake()).install(monkeypatch)

    async def fake_plan(file_id, transcript, probe, lang, note, *, emit=None, ledger=None):
        return spec

    monkeypatch.setattr(planner, "run", fake_plan)

    async def routed(**kw):
        return await routes[kw["purpose"]](**kw)

    monkeypatch.setattr(meta_client, "chat", routed)
    settings = Settings(runs_dir=tmp_path / "runs")
    store = RunStore(settings.runs_dir)
    state = store.create_run(video, lang="en")
    shutil.copyfile(MEDIA / video, store.input_video_path(state.run_id))
    final = await execute_run(state.run_id, settings)
    return final, store.run_dir(state.run_id)


DESIGNER_FF = [
    [("predict", {"changed_variable": "height", "values": [0.5, 1.0]})],
    json.dumps({"title": "Fall time against height", "changed_variable": "height",
                "controlled_variables": ["ball"], "prediction_en": "From {{p1}} it takes {{p2}}.",
                "prediction_id": "Dari {{p1}} butuh {{p2}}.", "protocol_steps_en": ["Drop it."],
                "protocol_steps_id": ["Jatuhkan."], "safety_notes_en": ["Adult supervision."],
                "safety_notes_id": ["Pengawasan orang dewasa."]}),
]  # fmt: skip


# --- 1. render order: the verdict and the Designer before RENDERING ------------------------------
@needs_ffmpeg
async def test_live_report_has_verdict_badge_and_next_experiment(tmp_path, monkeypatch) -> None:
    html = _capture_pdfs(monkeypatch)
    routes = {"analyst": FakeChat([*HAPPY_PATH, [("submit_report", REPORT)]]),
              "auditor": FakeChat([verdict("PASS")]), "translator": FakeChat(["{}"]),
              "designer": FakeChat(DESIGNER_FF)}  # fmt: skip
    final, run_dir = await _run(tmp_path, monkeypatch, planner_spec(), routes)
    assert final.verdict == "PASS_WITH_WARNINGS" and final.stage == "done"
    page = html["report_en.pdf"]
    assert "PASS WITH WARNINGS" in page  # the badge, rendered during the live run
    assert "Fall time against height" in page  # the Designer's proposal is in the report
    assert "Built with Meta SAM 3.1 · Muse Spark" in page
    assert "Muse Spark · Muse Wonder" not in page  # Wonder did not run: not credited


@needs_ffmpeg
async def test_live_rerecord_gets_feedback_page_only(tmp_path, monkeypatch) -> None:
    html = _capture_pdfs(monkeypatch)
    rerecord = [[("request_rerecord", {"reasons": ["G1"], "tips_en": ["More light."],
                                       "tips_id": ["Tambah cahaya."]})]]  # fmt: skip
    routes = {"analyst": FakeChat(rerecord), "auditor": FakeChat([verdict("NEEDS_RERECORD")]),
              "translator": FakeChat([]), "designer": FakeChat([])}  # fmt: skip
    final, run_dir = await _run(tmp_path, monkeypatch, planner_spec(), routes)
    assert final.verdict == "NEEDS_RERECORD" and final.headline is None
    assert "recording_feedback.pdf" in html and "report_en.pdf" not in html
    assert "More light." in html["recording_feedback.pdf"]


# --- 2. NEEDS_RERECORD before analysis, not FAILED ---------------------------------------------
@needs_ffmpeg
async def test_no_scale_ends_as_rerecord_with_feedback(tmp_path, monkeypatch) -> None:
    html = _capture_pdfs(monkeypatch)
    spec = planner_spec().model_copy(update={"reference": None})
    routes = {k: FakeChat([]) for k in ("analyst", "auditor", "translator", "designer")}
    final, run_dir = await _run(tmp_path, monkeypatch, spec, routes)
    assert final.stage == "done" and final.verdict == "NEEDS_RERECORD"
    gate_ev = [e for e in Ledger(run_dir / "analysis" / "ledger.jsonl").evidence()
               if e.id.startswith("gate_")]  # fmt: skip
    assert gate_ev[-1].data["diagnosis"] == "no_scale"
    assert "recording_feedback.pdf" in html


@needs_ffmpeg
async def test_planner_stop_ends_with_feedback_page(tmp_path, monkeypatch) -> None:
    html = _capture_pdfs(monkeypatch)
    spec = planner_spec().model_copy(update={"experiment": "unknown", "confidence": 0.2,
                                             "warnings": ["The camera moves."]})  # fmt: skip
    routes = {k: FakeChat([]) for k in ("analyst", "auditor", "translator", "designer")}
    final, run_dir = await _run(tmp_path, monkeypatch, spec, routes)
    assert final.stage == "done" and final.verdict == "NEEDS_RERECORD"
    assert "The camera moves." in html["recording_feedback.pdf"]


# --- 3. pendulum through the agents --------------------------------------------------------------
def pendulum_spec(**update) -> ExperimentSpec:
    spec = ExperimentSpec(
        experiment="pendulum", target_concept="red ball", target_alt_concepts=["ball"],
        reference=ReferenceObject(concept="ruler", length_m=1.0, length_source="spoken"),
        spoken_params={"string_length": Quantity(value=0.60, unit="m", source_turn=0)},
        analysis_window_s=(0.0, 12.0), motion_axis="x", hypothesis="I think T is 1.55 s",
        hypothesis_prediction=Quantity(value=1.55, unit="s", source_turn=0), language="en",
        setup_description="A red ball on a string swings in front of a ruler.", confidence=0.9,
    )  # fmt: skip
    return spec.model_copy(update=update)


PENDULUM_REPORT = {
    "lang": "en", "title": "Pendulum measurement", "aim": "Measure the period and g.",
    "setup": "A red ball swings on a string.", "method": "A damped sine fit of the swing.",
    "results": [{"heading": "Result",
                 "body": "The period is {{c03}} and g is {{c01}}. Reference: {{c02}}."}],
    "hypothesis_evaluation": "The prediction is {{c04}}.",
    "error_analysis": "The length uncertainty dominates.", "conclusion": "See {{c01}}.",
    "claims_used": ["c01", "c02", "c03", "c04"],
}  # fmt: skip
# trk_01; cal_01 pivot; cal_02 ruler; cal_03 spoken length (secondary scale)
PENDULUM_PATH = [
    [("get_run_summary", {})],
    [("set_fit_window", {"mode": "auto", "reason": "whole swing"})],
    [("fit_model", {"window_id": "win_01", "model": "damped_sine"})],
    [("bootstrap_ci", {"fit_id": "fit_01", "quantity": "period"})],
    [("derive", {"quantity": "period", "fit_id": "fit_01"}),
     ("derive", {"quantity": "period0", "fit_id": "fit_01"}),
     ("derive", {"quantity": "g", "fit_id": "fit_01", "length_source": "spoken"})],
    [("compare_reference", {"derived_id": "der_04"}),
     ("compare_hypothesis", {"derived_id": "der_01"}), ("run_gates", {})],
    [("make_chart", {"kind": "trajectory", "fit_id": "fit_01"})],
    [("add_claim", {**CLAIM_G, "evidence_id": "der_04"}), ("add_claim", CLAIM_CMP),
     ("add_claim", {"claim_type": "derived", "evidence_id": "der_01", "quantity": "period",
                    "statement_en": "T = {value}", "statement_id": "T = {value}"}),
     ("add_claim", {"claim_type": "hypothesis_test", "evidence_id": "cmp_02",
                    "statement_en": "The prediction is {outcome}.",
                    "statement_id": "Prediksi {outcome}."})],
    [("submit_report", PENDULUM_REPORT)],
]  # fmt: skip
DESIGNER_PEND = [
    [("predict", {"changed_variable": "string_length", "values": [0.25, 1.0]})],
    json.dumps({"title": "Period against length", "changed_variable": "string_length",
                "controlled_variables": ["bob"], "prediction_en": "At {{p1}}: {{p2}}.",
                "prediction_id": "Pada {{p1}}: {{p2}}.", "protocol_steps_en": ["Swing it."],
                "protocol_steps_id": ["Ayunkan."], "safety_notes_en": ["Adult supervision."],
                "safety_notes_id": ["Pengawasan orang dewasa."]}),
]  # fmt: skip


@needs_ffmpeg
async def test_pendulum_runs_through_analyst_auditor_designer_and_report(tmp_path, monkeypatch):
    html = _capture_pdfs(monkeypatch)
    analyst_chat = FakeChat(PENDULUM_PATH)
    routes = {"analyst": analyst_chat, "auditor": FakeChat([verdict("PASS")]),
              "translator": FakeChat(["{}"]), "designer": FakeChat(DESIGNER_PEND)}  # fmt: skip
    final, run_dir = await _run(tmp_path, monkeypatch, pendulum_spec(), routes,
                                video="pendulum_60fps_12s.mp4", sam=pendulum_fake())  # fmt: skip
    tool_results = [m["content"] for c in analyst_chat.calls for m in c["messages"]
                    if m["role"] == "tool"]  # fmt: skip
    assert all('"ok": true' in r for r in tool_results), tool_results
    assert final.stage == "done" and final.verdict in ("PASS", "PASS_WITH_WARNINGS")
    h = final.headline
    assert h["experiment"] == "pendulum" and h["quantity"] == "g"
    assert h["value"] == pytest.approx(9.81, rel=0.01) and h["length_source"] == "spoken"
    ledger = Ledger(run_dir / "analysis" / "ledger.jsonl")
    hyp = ledger.get("cmp_02").data
    assert hyp["outcome"] == "close" and hyp["quantity"] == "period"
    g1 = [e for e in ledger.evidence() if e.data.get("gate") == "G1"][-1].data
    assert g1["metrics"]["cycles"] > 5 and g1["metrics"]["coverage_scope"] == "fit_window"
    g2 = [e for e in ledger.evidence() if e.data.get("gate") == "G2"][-1].data
    assert g2["status"] == "pass"  # ruler vs spoken length agree
    page = html["report_en.pdf"]
    assert "Period against length" in page and "g = 9.8" in page
    for name in ("chart_trajectory.png", "overlay.mp4", "data.csv"):
        assert (run_dir / "outputs" / name).exists(), name
    chart = json.loads((run_dir / "outputs" / "chart_trajectory.json").read_text("utf-8"))
    assert "Pendulum" in chart["layout"]["title"]["text"]


@needs_ffmpeg
async def test_pendulum_without_length_falls_back_to_the_period(tmp_path, monkeypatch) -> None:
    """Budget exhausted → deterministic fallback; no length → the headline is the period."""
    _capture_pdfs(monkeypatch)
    spec = pendulum_spec(reference=None, spoken_params={}, hypothesis=None,
                         hypothesis_prediction=None)  # fmt: skip
    routes = {"analyst": FakeChat([[("get_run_summary", {})]] * 12),
              "auditor": FakeChat([verdict("PASS")]), "translator": FakeChat(["{}"]),
              "designer": FakeChat([])}  # fmt: skip
    final, run_dir = await _run(tmp_path, monkeypatch, spec, routes,
                                video="pendulum_60fps_12s.mp4", sam=pendulum_fake())  # fmt: skip
    assert final.verdict == "PASS_WITH_WARNINGS"  # fallback: capped (06 §3.4)
    assert final.headline["quantity"] == "period"
    assert final.headline["period0"] == pytest.approx(1.5539, rel=0.002)
    assert (run_dir / "outputs" / "data.csv").read_text("utf-8").count("x_px") == 1


# --- 4. fall-time hypothesis and the G5 hypothesis evidence ---------------------------------
@needs_ffmpeg
async def test_fall_time_prediction_is_tested(tmp_path, monkeypatch) -> None:
    _capture_pdfs(monkeypatch)
    spec = planner_spec().model_copy(update={
        "hypothesis": "It falls in about 0.45 s",
        "hypothesis_prediction": Quantity(value=0.45, unit="s", source_turn=0)})  # fmt: skip
    steps = [
        *HAPPY_PATH[:5],
        [("derive", {"quantity": "fall_time", "fit_id": "fit_01"})],
        [("compare_reference", {"derived_id": "der_01"}),
         ("compare_hypothesis", {"derived_id": "der_02"})],
        [("add_claim", CLAIM_G), ("add_claim", CLAIM_CMP),
         ("add_claim", {"claim_type": "hypothesis_test", "evidence_id": "cmp_02",
                        "statement_en": "The prediction is {outcome}.",
                        "statement_id": "Prediksi {outcome}."})],
        [("submit_report", {**REPORT, "hypothesis_evaluation": "It is {{c03}}.",
                            "claims_used": ["c01", "c02", "c03"]})],
    ]  # fmt: skip
    routes = {"analyst": FakeChat(steps), "auditor": FakeChat([verdict("PASS")]),
              "translator": FakeChat(["{}"]), "designer": FakeChat(DESIGNER_FF)}  # fmt: skip
    final, run_dir = await _run(tmp_path, monkeypatch, spec, routes)
    assert "analyst_fallback" not in final.warnings
    ledger = Ledger(run_dir / "analysis" / "ledger.jsonl")
    fall = ledger.get("der_02").data
    assert fall["quantity"] == "fall_time"
    assert fall["value"] == pytest.approx(0.418, abs=0.01)  # 0.857 m drop at g = 9.81
    assert ledger.get("cmp_02").data["outcome"] in ("close", "not_supported")


def test_hypothesis_claim_must_cite_hypothesis_evidence(tmp_path) -> None:
    ledger = Ledger(tmp_path / "ledger.jsonl")
    ref = ledger.add_evidence("cmp", "compare_reference", {"rel_diff_pct": 0.4}, "ref")
    ledger.add_claim("hypothesis_test", ref.id, "Prediction {rel_diff}.", "Prediksi {rel_diff}.")
    draft = {**REPORT, "results": [{"heading": "R", "body": "{{c01}}"}], "claims_used": ["c01"],
             "conclusion": "Done."}  # fmt: skip
    g5 = gates.gate_g5(draft, ledger, hypothesis=True)
    assert g5.status == "fail" and g5.diagnosis == "missing_hypothesis_claim"
    hyp = ledger.add_evidence("cmp", "hypothesis_test", {"outcome": "close", "m": 1.0}, "hyp")
    ledger.add_claim("hypothesis_test", hyp.id, "It is {outcome}.", "Ini {outcome}.")
    draft["claims_used"] = ["c01", "c02"]
    draft["conclusion"] = "Done {{c02}}."
    assert gates.gate_g5(draft, ledger, hypothesis=True).status == "pass"


# --- 5–8. cost guard, fallback cap, diameter traceability ---------------------------------------
def test_cost_guard_refuses_model_calls(tmp_path) -> None:
    settings = Settings(runs_dir=tmp_path / "runs")
    state = RunStore(settings.runs_dir).create_run("x.mp4")
    ctx = RunContext.load(state.run_id, settings)
    ctx.emit("model_result", cost_usd=1.49)
    ctx.emit("model_call", purpose="analyst")  # still under the limit
    ctx.emit("model_result", cost_usd=0.02)
    with pytest.raises(CostGuardExceeded, match="cost_guard"):
        ctx.emit("model_call", purpose="auditor")


def test_fallback_report_is_capped_at_pass_with_warnings() -> None:
    status, warnings = auditor.finalize_verdict(Verdict(status="PASS"), [], fallback=True)
    assert status == "PASS_WITH_WARNINGS" and warnings
    assert auditor.finalize_verdict(Verdict(status="PASS"), [])[0] == "PASS"


def test_object_diameter_only_from_the_spoken_value() -> None:
    spec = planner_spec().model_copy(update={"object_diameter_m": 0.065})
    synced, warnings = planner.sync_object_diameter(spec)
    assert synced.object_diameter_m is None and warnings  # nothing spoken: dropped
    spoken = spec.model_copy(
        update={"spoken_params": {"ball_diameter": Quantity(value=10, unit="cm", source_turn=0)}}
    )
    synced, warnings = planner.sync_object_diameter(spoken)
    assert synced.object_diameter_m == pytest.approx(0.10) and warnings


# --- 10. gate diagnoses ---------------------------------------------------------------------------
def _track(n: int, fps: float = 60.0, gap: tuple[int, int] | None = None) -> Track:
    pts = []
    for i in range(n):
        valid = not (gap and gap[0] <= i < gap[1])
        pts.append(TrackPoint(t=i / fps, x_px=100.0 + i, y_px=100.0 + 2 * i, area_px=500.0,
                              valid=valid, flag="ok" if valid else "missing"))  # fmt: skip
    return Track(track_id="b_0", concept="ball", object_id="0", fps=fps, points=pts,
                 coverage=sum(p.valid for p in pts) / n, n_outliers=0, n_interpolated=0,
                 max_gap_frames=0)  # fmt: skip


def test_g1_uses_fit_window_coverage_and_object_left_frame() -> None:
    th = gates.GateThresholds()
    track = _track(100, gap=(60, 100))  # lost after the fall: whole-track coverage 0.6
    assert gates.gate_track(track, th, min_points=8).status == "fail"
    in_window = gates.gate_track(track, th, min_points=8, window=(0.1, 0.9))
    assert in_window.status == "pass" and in_window.metrics["coverage_scope"] == "fit_window"
    left = gates.gate_track(track, th, min_points=8, window=(0.1, 0.9),
                            window_warnings=["object_left_frame"])  # fmt: skip
    assert left.status == "warn" and left.diagnosis == "object_left_frame"


def test_g3_camera_motion_and_g4_air_drag_and_g2_pendulum_skip() -> None:
    th = gates.GateThresholds(g3_min_r2=0.98)
    fit = {"r2": 0.999, "rms_px": 0.5, "durbin_watson": 1.8, "params": {"A": 50.0}}
    assert gates.gate_fit(fit, th, drift_px_s=0.5, period_s=1.5).status == "pass"
    moved = gates.gate_fit(fit, th, drift_px_s=5.0, duration_s=10.0, period_s=1.5)
    assert moved.status == "warn" and moved.diagnosis == "camera_motion"
    drag = gates.gate_plausibility(9.30, gates.GateThresholds(), durbin_watson=0.8,
                                   free_fall=True)  # fmt: skip
    assert drag.diagnosis == "air_drag_likely" and drag.status == "warn"
    assert gates.gate_scale(None, th, pendulum=True).status == "skipped"
    assert gates.gate_scale(None, th).diagnosis == "no_scale"


def test_series_accepts_agent_path_pendulum_runs(tmp_path) -> None:
    """Agent-path runs keep L and T0 in the headline (no analysis/pendulum.json)."""
    from framelab.pipeline import series as series_mod

    store = RunStore(tmp_path / "runs")
    state = store.create_run("p.mp4")
    state.experiment = "pendulum"
    state.verdict = "PASS"
    state.headline = {"experiment": "pendulum", "length_m": 0.6, "length_sigma_m": 0.005,
                      "period0": 1.554, "period0_sigma": 0.003, "length_source": "spoken",
                      "theta_mean_deg": 12.0}  # fmt: skip
    store.save_state(state)
    point = series_mod.run_point(store, state.run_id)
    assert point["L"] == 0.6 and point["T0"] == 1.554
    state.verdict = "NEEDS_RERECORD"
    store.save_state(state)
    with pytest.raises(series_mod.SeriesError):
        series_mod.run_point(store, state.run_id)
