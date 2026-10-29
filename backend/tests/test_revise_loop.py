"""M7 revise loop (04 §1, 06 §8): REJECT → revise in round 2, a tampered draft overridden by the
post-checks, rounds exhausted, and a re-recording request. Both agents are scripted fakes; the
pipeline up to the Analyst is real (SAM replayed from fixtures)."""

from __future__ import annotations

import json
import shutil

import pytest

from framelab.agent import analyst, planner
from framelab.clients import meta as meta_client
from framelab.config import Settings
from framelab.pipeline.run import execute_run
from framelab.schemas import Stage
from framelab.store import RunStore
from tests.fakes import MEDIA, FakeChat, freefall_fake
from tests.test_agent_loop import HAPPY_PATH, REPORT, planner_spec

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")

# Round 1 creates fresh evidence: win_02, fit_02, ci_02, der_02, cmp_02 and claims c03, c04.
REPORT_R1 = {
    **REPORT,
    "results": [
        {"heading": "Result",
         "body": "The acceleration is {{c03}}. Against the reference: {{c04}}."}
    ],
    "conclusion": "The data support the measurement in {{c03}}.",
    "claims_used": ["c03", "c04"],
}  # fmt: skip
ROUND1 = [
    [("get_run_summary", {})],
    [("set_fit_window", {"mode": "auto", "reason": "fix the conclusion"})],
    [("fit_model", {"window_id": "win_02", "model": "quadratic"})],
    [("bootstrap_ci", {"fit_id": "fit_02", "quantity": "g"})],
    [("derive", {"quantity": "g", "fit_id": "fit_02", "scale_id": "cal_01"})],
    [("compare_reference", {"derived_id": "der_02"}), ("run_gates", {})],
    [("add_claim", {"claim_type": "derived", "evidence_id": "der_02", "quantity": "g",
                    "statement_en": "g = {value}", "statement_id": "g = {value}"}),
     ("add_claim", {"claim_type": "comparison", "evidence_id": "cmp_02", "quantity": "g",
                    "statement_en":
                    "Compared with the reference value, the difference is {rel_diff}.",
                    "statement_id": "Dibandingkan nilai referensi, selisihnya {rel_diff}."})],
    [("submit_report", REPORT_R1)],
]  # fmt: skip
ROUND0 = [*HAPPY_PATH, [("submit_report", REPORT)]]


def verdict(status: str, fix: list[str] | None = None) -> str:
    return json.dumps({
        "status": status, "unsupported_claims": [], "overreaching_statements": [],
        "gate_failures": [], "fix_instructions": fix or [], "rerecord_tips_en": [],
        "rerecord_tips_id": [], "notes": "",
    })  # fmt: skip


async def run_pipeline(
    tmp_path,
    monkeypatch,
    *,
    analyst_steps,
    auditor_steps,
    until,
    max_revise_rounds=2,
    wrap_analyst=None,
):
    freefall_fake().install(monkeypatch)

    async def fake_plan(file_id, transcript, probe, lang, note, *, emit=None, ledger=None):
        return planner_spec()

    monkeypatch.setattr(planner, "run", fake_plan)
    analyst_chat = FakeChat(analyst_steps)
    auditor_chat = FakeChat(auditor_steps)

    async def routed(**kw):
        chat = analyst_chat if kw["purpose"] == "analyst" else auditor_chat
        return await chat(**kw)

    monkeypatch.setattr(meta_client, "chat", routed)
    if wrap_analyst is not None:
        real_run = analyst.run

        async def tampered(session, **kw):
            return await wrap_analyst(real_run, session, **kw)

        monkeypatch.setattr(analyst, "run", tampered)
    settings = Settings(runs_dir=tmp_path / "runs", max_revise_rounds=max_revise_rounds)
    store = RunStore(settings.runs_dir)
    state = store.create_run("freefall.mp4", lang="en")
    shutil.copyfile(MEDIA / "freefall_60fps_5s.mp4", store.input_video_path(state.run_id))
    await execute_run(state.run_id, settings, until=until)
    run_dir = store.run_dir(state.run_id)
    final = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
    return final, run_dir, analyst_chat, auditor_chat


def read_verdict(run_dir, round_: int) -> dict:
    return json.loads(
        (run_dir / "analysis" / f"verdict_r{round_:02d}.json").read_text(encoding="utf-8")
    )


async def test_reject_is_revised_in_the_second_round(tmp_path, monkeypatch) -> None:
    fix = "Rewrite the conclusion so that each claim is cited."
    state, run_dir, analyst_chat, auditor_chat = await run_pipeline(
        tmp_path, monkeypatch,
        analyst_steps=ROUND0 + ROUND1,
        auditor_steps=[verdict("REJECT", [fix]), verdict("PASS")],
        until=Stage.AUDITING,
    )  # fmt: skip

    assert state["revise_rounds"] == 1
    assert read_verdict(run_dir, 0)["status"] == "REJECT"
    second = read_verdict(run_dir, 1)
    assert second["status"] == "PASS_WITH_WARNINGS"  # the single-scale warning downgrades PASS
    assert any("PASS → PASS_WITH_WARNINGS" in o for o in second["overrides"])
    assert state["verdict"] == "PASS_WITH_WARNINGS" and state["stage"] == "auditing"
    round1_context = [m["content"] for m in analyst_chat.calls[len(ROUND0)]["messages"]
                      if m["role"] == "user"]  # fmt: skip
    assert any(fix in text for text in round1_context)  # the Analyst saw the Auditor's fix
    assert any(c["purpose"] == "auditor" for c in auditor_chat.calls)


async def test_a_bare_number_is_overridden_then_fixed_in_round_two(tmp_path, monkeypatch) -> None:
    async def inject(real_run, session, **kw):
        outcome = await real_run(session, **kw)
        if session.round == 0 and outcome.draft is not None:  # a draft that bypassed G5
            bad = outcome.draft.conclusion + " The value is about 9.8 m/s²."
            outcome.draft = outcome.draft.model_copy(update={"conclusion": bad})
        return outcome

    state, run_dir, _, _ = await run_pipeline(
        tmp_path, monkeypatch,
        analyst_steps=ROUND0 + ROUND1,
        auditor_steps=[verdict("PASS"), verdict("PASS")],  # the model misses it both times
        until=Stage.AUDITING,
        wrap_analyst=inject,
    )  # fmt: skip

    first = read_verdict(run_dir, 0)
    assert first["status"] == "REJECT"  # the deterministic G5 check overrode the model's PASS
    assert any("G5" in o for o in first["overrides"])
    assert state["revise_rounds"] == 1
    assert state["verdict"] == "PASS_WITH_WARNINGS"
    final_draft = json.loads((run_dir / "draft_report.json").read_text(encoding="utf-8"))
    assert "9.8" not in final_draft["conclusion"]


async def test_exhausted_rounds_end_as_pass_with_warnings(tmp_path, monkeypatch) -> None:
    state, run_dir, _, _ = await run_pipeline(
        tmp_path, monkeypatch,
        analyst_steps=ROUND0 + ROUND1,
        auditor_steps=[verdict("REJECT"), verdict("REJECT")],
        until=Stage.AUDITING,
        max_revise_rounds=1,
    )  # fmt: skip

    assert read_verdict(run_dir, 1)["status"] == "REJECT"  # the model's last word, a soft REJECT
    assert state["verdict"] == "PASS_WITH_WARNINGS"
    assert "auditor_unresolved" in state["warnings"]


async def test_an_analyst_rerecord_request_skips_the_auditor(tmp_path, monkeypatch) -> None:
    rerecord = [
        [
            (
                "request_rerecord",
                {
                    "reasons": ["low_coverage_blur"],
                    "tips_en": ["Film in brighter light."],
                    "tips_id": ["Rekam dengan cahaya lebih terang."],
                },
            )
        ]
    ]
    state, run_dir, _, auditor_chat = await run_pipeline(
        tmp_path, monkeypatch,
        analyst_steps=rerecord,
        auditor_steps=[],
        until=Stage.AUDITING,
    )  # fmt: skip

    assert state["verdict"] == "NEEDS_RERECORD"
    assert auditor_chat.calls == []
    verdict_json = json.loads((run_dir / "analysis" / "verdict.json").read_text(encoding="utf-8"))
    assert "Film in brighter light." in verdict_json["rerecord_tips_en"]


async def test_rendering_stage_writes_both_languages_and_registers_artifacts(
    tmp_path, monkeypatch
) -> None:
    """Analyst → Auditor PASS → translation → outputs and PDFs → Designer, all in one run."""
    from framelab.schemas import ReportDraft

    translated = ReportDraft.model_validate({
        **REPORT, "lang": "id", "title": "Pengukuran jatuh bebas", "aim": "Mengukur g.",
        "setup": "Bola dijatuhkan.", "method": "Fit kuadratik.",
        "results": [{"heading": "Hasil", "body": "Nilainya {{c01}}. Dibandingkan: {{c02}}."}],
        "error_analysis": "Sebaran kecil.", "conclusion": "Lihat {{c01}}.",
    })  # fmt: skip
    designer_steps = [
        [("predict", {"changed_variable": "height", "values": [0.5, 1.0]})],
        json.dumps({
            "title": "Fall time against height", "changed_variable": "height",
            "controlled_variables": ["ball", "release from rest"],
            "prediction_en": "The fall from {{p1}} takes {{p2}}.",
            "prediction_id": "Jatuh dari {{p1}} memakan {{p2}}.",
            "protocol_steps_en": ["Drop the ball from each height."],
            "protocol_steps_id": ["Jatuhkan bola dari tiap tinggi."],
            "safety_notes_en": ["Adult supervision for children."],
            "safety_notes_id": ["Pengawasan orang dewasa untuk anak."],
        }),
    ]  # fmt: skip
    routes = {
        "analyst": FakeChat(ROUND0),
        "auditor": FakeChat([verdict("PASS")]),
        "translator": FakeChat([json.dumps(translated.model_dump(mode="json"))]),
        "designer": FakeChat(designer_steps),
    }

    async def routed(**kw):
        return await routes[kw["purpose"]](**kw)

    freefall_fake().install(monkeypatch)

    async def fake_plan(file_id, transcript, probe, lang, note, *, emit=None, ledger=None):
        return planner_spec()

    monkeypatch.setattr(planner, "run", fake_plan)
    monkeypatch.setattr(meta_client, "chat", routed)
    settings = Settings(runs_dir=tmp_path / "runs")
    store = RunStore(settings.runs_dir)
    state = store.create_run("freefall.mp4", lang="en")
    shutil.copyfile(MEDIA / "freefall_60fps_5s.mp4", store.input_video_path(state.run_id))
    await execute_run(state.run_id, settings, until=Stage.DESIGNING)

    run_dir = store.run_dir(state.run_id)
    final = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
    assert final["verdict"] == "PASS_WITH_WARNINGS"  # the single-scale warning downgrades PASS
    artifacts = final["artifacts"]
    for name in ("report_en", "report_id", "chart_trajectory_png", "data_csv", "overlay"):
        assert name in artifacts, name
        assert (run_dir / artifacts[name]).exists(), name
    assert (run_dir / "analysis" / "designer.json").exists()
    proposal = json.loads((run_dir / "analysis" / "designer.json").read_text(encoding="utf-8"))
    assert [p["id"] for p in proposal["predicted_values"]] == ["p1", "p2"]
