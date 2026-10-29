"""M6 Analyst loop with a scripted FakeChat (06 §3): happy path, G5 rejection and fix, budget
exhaustion and API failure fallbacks, re-recording, invalid tool arguments.

The pipeline runs for real up to ANALYZING (SAM and Spark replayed from fixtures); only the
Analyst's chat turns are scripted.
"""

from __future__ import annotations

import json
import shutil

import pytest

from framelab.agent import planner
from framelab.agent.ledger import Ledger
from framelab.agent.narrative import bare_numbers
from framelab.clients.meta import MetaClientError
from framelab.config import Settings
from framelab.pipeline.run import execute_run
from framelab.schemas import ExperimentSpec, ReferenceObject, Stage
from framelab.store import RunStore
from tests.fakes import MEDIA, FakeChat, freefall_fake

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")

CLAIM_G = {"claim_type": "derived", "evidence_id": "der_01", "quantity": "g",
           "statement_en": "g = {value}", "statement_id": "g = {value}"}  # fmt: skip
CLAIM_CMP = {"claim_type": "comparison", "evidence_id": "cmp_01", "quantity": "g",
             "statement_en": "Compared with the reference value, the difference is {rel_diff}.",
             "statement_id": "Dibandingkan nilai referensi, selisihnya {rel_diff}."}  # fmt: skip

REPORT = {
    "lang": "en",
    "title": "Free-fall measurement",
    "aim": "Measure the acceleration from the video.",
    "setup": "An orange ball was dropped in front of a ruler.",
    "method": "A quadratic fit between release and impact, with the scale from the ruler.",
    "results": [{"heading": "Result",
                 "body": "The acceleration is {{c01}}. Against the reference: {{c02}}."}],
    "hypothesis_evaluation": None,
    "error_analysis": "The uncertainty includes the spread of the fit and of the scale.",
    "conclusion": "The data support the measurement in {{c01}}.",
    "claims_used": ["c01", "c02"],
}  # fmt: skip

# The Analyst's turns up to the claims. Ids are allocated in order: trk_01 and cal_01 come from
# the pipeline; then win_01, fit_01, ci_01, der_01, cmp_01, gate_01..04, fig_01, c01, c02.
HAPPY_PATH = [
    [("get_run_summary", {})],
    [("set_fit_window", {"mode": "auto", "reason": "release to impact"})],
    [("fit_model", {"window_id": "win_01", "model": "quadratic"})],
    [("bootstrap_ci", {"fit_id": "fit_01", "quantity": "g"})],
    [("derive", {"quantity": "g", "fit_id": "fit_01", "scale_id": "cal_01"})],
    [("compare_reference", {"derived_id": "der_01"}), ("run_gates", {})],
    [("make_chart", {"kind": "trajectory", "fit_id": "fit_01"})],
    [("add_claim", CLAIM_G), ("add_claim", CLAIM_CMP)],
]


def planner_spec() -> ExperimentSpec:
    return ExperimentSpec(
        experiment="free_fall",
        target_concept="orange ball",
        target_alt_concepts=["ball"],
        reference=ReferenceObject(
            concept="ruler", length_m=1.0, length_source="spoken", frame_hint_s=0.5
        ),  # fmt: skip
        analysis_window_s=(0.2, 2.0),
        motion_axis="y",
        language="en",
        setup_description="Orange ball dropped in front of a ruler.",
        confidence=0.9,
    )


async def run_analyst(tmp_path, monkeypatch, steps) -> tuple[dict, object, FakeChat, object]:
    """Pipeline to ANALYZING with the Analyst's turns scripted. Returns state, run dir, chat."""
    fake_sam = freefall_fake().install(monkeypatch)

    async def fake_plan(file_id, transcript, probe, lang, note, *, emit=None, ledger=None):
        return planner_spec()

    monkeypatch.setattr(planner, "run", fake_plan)
    chat = FakeChat(steps).install(monkeypatch)
    settings = Settings(runs_dir=tmp_path / "runs")
    store = RunStore(settings.runs_dir)
    state = store.create_run("freefall.mp4", lang="en")
    shutil.copyfile(MEDIA / "freefall_60fps_5s.mp4", store.input_video_path(state.run_id))
    final = await execute_run(state.run_id, settings, until=Stage.ANALYZING)
    assert fake_sam.calls  # segmentation ran before the Analyst
    run_dir = store.run_dir(state.run_id)
    return json.loads((run_dir / "state.json").read_text(encoding="utf-8")), run_dir, chat, final


def events(run_dir) -> list[dict]:
    return [json.loads(line) for line in (run_dir / "events.jsonl").read_text(encoding="utf-8")
            .splitlines()]  # fmt: skip


def tool_messages(chat: FakeChat) -> list[str]:
    """Every tool result the model received, in order."""
    return [m["content"] for call in chat.calls for m in call["messages"] if m["role"] == "tool"]


async def test_happy_path_ends_with_report_claims_and_gates(tmp_path, monkeypatch) -> None:
    steps = HAPPY_PATH + [[("submit_report", REPORT)]]
    state, run_dir, chat, _ = await run_analyst(tmp_path, monkeypatch, steps)

    assert len(chat.calls) == len(steps)
    assert "analyst_fallback" not in state["warnings"]
    draft = json.loads((run_dir / "draft_report.json").read_text(encoding="utf-8"))
    assert draft["claims_used"] == ["c01", "c02"]
    claims = Ledger(run_dir / "analysis" / "ledger.jsonl").claims()
    assert [c.id for c in claims] == ["c01", "c02"]
    assert all(c.evidence for c in claims)
    assert claims[0].value is not None and abs(claims[0].value - 9.81) < 0.3  # from der_01
    assert state["headline"]["quantity"] == "g"
    names = [e["type"] for e in events(run_dir)]
    assert names.count("tool_call") == sum(len(step) for step in steps)
    assert names.count("claim_added") == 2
    gate_status = {e["gate"]: e["status"] for e in events(run_dir) if e["type"] == "gate_result"}
    assert gate_status == {"G1": "pass", "G2": "warn", "G3": "pass", "G4": "pass", "G5": "pass"}


async def test_every_tool_result_fits_the_model_budget(tmp_path, monkeypatch) -> None:
    steps = HAPPY_PATH + [[("submit_report", REPORT)]]
    _, _, chat, _ = await run_analyst(tmp_path, monkeypatch, steps)
    results = tool_messages(chat)
    assert results and all(len(r.encode("utf-8")) <= 2048 for r in results)


async def test_bare_number_is_rejected_then_fixed(tmp_path, monkeypatch) -> None:
    bad = {**REPORT, "conclusion": "The value is about 9.8 m/s²."}
    steps = HAPPY_PATH + [[("submit_report", bad)], [("submit_report", REPORT)]]
    state, run_dir, chat, _ = await run_analyst(tmp_path, monkeypatch, steps)

    results = [json.loads(r) for r in tool_messages(chat)]
    rejected = [r for r in results if r["summary"].startswith("report rejected by G5")]
    assert len(rejected) == 1 and rejected[0]["ok"] is False
    assert "bare_number" in rejected[0]["summary"] or "9.8" in rejected[0]["warnings"][0]
    assert (run_dir / "draft_report.json").exists()  # the corrected draft was accepted
    assert "analyst_fallback" not in state["warnings"]


async def test_budget_exhaustion_falls_back_to_the_deterministic_report(
    tmp_path, monkeypatch
) -> None:
    steps = [[("get_run_summary", {})]] * 12  # never submits
    state, run_dir, chat, _ = await run_analyst(tmp_path, monkeypatch, steps)

    assert len(chat.calls) == 12  # the step budget, then no thirteenth call
    budget_notes = [m["content"] for m in chat.calls[9]["messages"] if m["role"] == "user"]
    assert any("Three steps are left" in note for note in budget_notes)
    assert "analyst_fallback" in state["warnings"]
    draft = json.loads((run_dir / "draft_report.json").read_text(encoding="utf-8"))
    texts = [
        draft["title"],
        draft["aim"],
        draft["setup"],
        draft["method"],
        draft["error_analysis"],
        draft["conclusion"],
    ]
    texts += [section["body"] for section in draft["results"]]
    assert all(bare_numbers(text) == [] for text in texts)  # only placeholders carry numbers
    assert draft["claims_used"] and set(draft["claims_used"]) <= {c.id for c in
        Ledger(run_dir / "analysis" / "ledger.jsonl").claims()}  # fmt: skip
    assert state["headline"]["quantity"] == "g"


async def test_two_api_failures_fall_back(tmp_path, monkeypatch) -> None:
    steps = [MetaClientError(500, "upstream down"), MetaClientError(503, "busy")]
    state, run_dir, chat, _ = await run_analyst(tmp_path, monkeypatch, steps)

    assert len(chat.calls) == 2
    assert "analyst_fallback" in state["warnings"]
    assert (run_dir / "draft_report.json").exists()


async def test_request_rerecord_ends_the_run(tmp_path, monkeypatch) -> None:
    steps = [[("request_rerecord", {"reasons": ["low_coverage_blur"],
                                    "tips_en": ["Use slow motion."],
                                    "tips_id": ["Gunakan slow motion."]})]]  # fmt: skip
    state, run_dir, _, _ = await run_analyst(tmp_path, monkeypatch, steps)

    # the run stops after ANALYZING (M7): the verdict comes from the review stages, see
    # test_revise_loop.py for the NEEDS_RERECORD verdict with --until auditing
    assert state["verdict"] is None and state["stage"] == "analyzing"
    request = json.loads((run_dir / "analysis" / "rerecord.json").read_text(encoding="utf-8"))
    assert request["tips_en"] == ["Use slow motion."]


async def test_invalid_arguments_and_plain_text_are_answered(tmp_path, monkeypatch) -> None:
    steps = [
        [("fit_model", {"window_id": "win_01", "model": "linear"})],  # bad enum, no window yet
        "I think the answer is clear.",  # no tool call: the loop asks for a tool
        [("request_rerecord", {"reasons": [], "tips_en": [], "tips_id": []})],
    ]
    _, _, chat, _ = await run_analyst(tmp_path, monkeypatch, steps)

    first = json.loads(tool_messages(chat)[0])
    assert first["ok"] is False and "must be one of" in first["summary"]
    nudges = [m["content"] for m in chat.calls[2]["messages"] if m["role"] == "user"]
    assert any("must act through tools" in text for text in nudges)
