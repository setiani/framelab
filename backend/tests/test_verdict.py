"""M7 verdicts: deterministic post-checks override the model, finalisation (06 §8), the Auditor's
input (no Analyst conversation), output retries and the Wonder fallback (02 §7)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from framelab.agent import auditor
from framelab.agent.auditor import AuditorOutput, PreCheck, post_check
from framelab.agent.ledger import Ledger
from framelab.config import MODEL_SPARK, MODEL_WONDER, Settings
from framelab.schemas import ExperimentSpec, GateResult, ReportDraft, Verdict
from tests.fakes import FakeChat


def gate(
    name: str,
    status: str = "pass",
    *,
    hard: bool = True,
    remedy: str = "none",
    diagnosis: str | None = None,
    message: str = "",
) -> GateResult:
    return GateResult(gate=name, name=name, status=status, hard=hard, remedy=remedy,  # type: ignore[arg-type]
                      diagnosis=diagnosis, message_en=message or f"{name} {status}",
                      message_id=f"{name} {status} (id)")  # fmt: skip


def out(status: str, **changes) -> AuditorOutput:
    base = {"status": status, "unsupported_claims": [], "overreaching_statements": [],
            "gate_failures": [], "fix_instructions": [], "rerecord_tips_en": [],
            "rerecord_tips_id": [], "notes": ""}  # fmt: skip
    base.update(changes)
    return AuditorOutput(**base)


def pre(
    hard: list[GateResult] | None = None,
    soft: list[GateResult] | None = None,
    g5_status: str = "pass",
) -> PreCheck:
    g5 = gate("G5", g5_status, message="bare number")
    return PreCheck(g5=g5, hard_failures=list(hard or []), soft_warnings=list(soft or []))


# --- post-checks (06 §5) ----------------------------------------------------------------------


def test_clean_pass_stays_pass() -> None:
    verdict = post_check(out("PASS"), pre(), rounds_left=True, model="m")
    assert verdict.status == "PASS" and verdict.overrides == []


def test_soft_warning_downgrades_pass() -> None:
    verdict = post_check(
        out("PASS"), pre(soft=[gate("G2", "warn", hard=False)]), rounds_left=True, model="m"
    )
    assert verdict.status == "PASS_WITH_WARNINGS"
    assert any("PASS → PASS_WITH_WARNINGS" in o for o in verdict.overrides)


def test_hard_rerecord_failure_forces_needs_rerecord_over_a_pass() -> None:
    failed = gate(
        "G2",
        "fail",
        remedy="rerecord",
        diagnosis="ruler_offplane",
        message="Ruler scale and ball scale disagree.",
    )
    verdict = post_check(out("PASS"), pre(hard=[failed]), rounds_left=True, model="m")
    assert verdict.status == "NEEDS_RERECORD"
    assert verdict.rerecord_tips_en == ["Ruler scale and ball scale disagree."]
    assert verdict.gate_failures == ["G2"]
    assert "hard gates ['G2'] failed: PASS → NEEDS_RERECORD" in verdict.overrides[0]


def test_hard_reanalyze_failure_forces_reject_while_rounds_remain() -> None:
    failed = gate("G1", "fail", remedy="reanalyze", diagnosis="id_switch")
    verdict = post_check(out("PASS_WITH_WARNINGS"), pre(hard=[failed]), rounds_left=True, model="m")
    assert verdict.status == "REJECT"
    assert verdict.fix_instructions and "G1" in verdict.fix_instructions[0]


def test_hard_reanalyze_failure_on_the_last_round_is_needs_rerecord() -> None:
    failed = gate("G1", "fail", remedy="reanalyze", diagnosis="id_switch")
    verdict = post_check(
        out("REJECT", fix_instructions=["fix it"]), pre(hard=[failed]), rounds_left=False, model="m"
    )
    assert verdict.status == "NEEDS_RERECORD"


def test_bare_number_in_the_draft_forces_reject_even_when_the_model_passes() -> None:
    bad = pre(g5_status="fail")
    bad.hard_failures.append(bad.g5)
    verdict = post_check(out("PASS"), bad, rounds_left=True, model="m")
    assert verdict.status == "REJECT"
    assert verdict.overrides and "G5" in verdict.overrides[0]


def test_reject_without_fixes_gets_a_generic_instruction() -> None:
    verdict = post_check(out("REJECT"), pre(), rounds_left=True, model="m")
    assert verdict.fix_instructions == ["Revise the flagged statements so each one cites a claim."]
    assert any("generic instruction" in o for o in verdict.overrides)


def test_gate_messages_are_always_in_the_rerecord_tips() -> None:
    failed = gate("G2", "fail", remedy="rerecord", diagnosis="ruler_offplane",
                  message="Ruler scale and ball scale disagree.")  # fmt: skip
    verdict = post_check(
        out("NEEDS_RERECORD", rerecord_tips_en=["Use a longer swing."],
            rerecord_tips_id=["Gunakan ayunan lebih panjang."]),
        pre(hard=[failed]), rounds_left=True, model="m",
    )  # fmt: skip
    assert "Ruler scale and ball scale disagree." in verdict.rerecord_tips_en
    assert "Use a longer swing." in verdict.rerecord_tips_en


def test_model_needs_rerecord_without_a_hard_failure_is_kept() -> None:
    verdict = post_check(
        out(
            "NEEDS_RERECORD",
            rerecord_tips_en=["Use a longer swing."],
            rerecord_tips_id=["Gunakan ayunan lebih panjang."],
        ),
        pre(),
        rounds_left=True,
        model="m",
    )
    assert verdict.status == "NEEDS_RERECORD" and verdict.overrides == []
    assert verdict.rerecord_tips_en == ["Use a longer swing."]


# --- finalisation (06 §8) ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("auditor_status", "hard_fail", "expected", "warning"),
    [
        ("PASS", False, "PASS", []),
        ("PASS_WITH_WARNINGS", False, "PASS_WITH_WARNINGS", []),
        ("REJECT", False, "PASS_WITH_WARNINGS", ["auditor_unresolved"]),
        ("PASS", True, "NEEDS_RERECORD", []),
        ("NEEDS_RERECORD", False, "NEEDS_RERECORD", []),
    ],
)  # fmt: skip
def test_finalize_verdict(auditor_status, hard_fail, expected, warning) -> None:
    gates = [gate("G1", "fail" if hard_fail else "pass", remedy="reanalyze")]
    status, warnings = auditor.finalize_verdict(Verdict(status=auditor_status), gates)
    assert status == expected and warnings == warning


def test_finalize_never_returns_reject() -> None:
    for status in ("PASS", "PASS_WITH_WARNINGS", "REJECT", "NEEDS_RERECORD"):
        final, _ = auditor.finalize_verdict(Verdict(status=status), [])
        assert final != "REJECT"


# --- the Auditor's input (06 §5) --------------------------------------------------------------


@pytest.fixture
def ledger(tmp_path) -> Ledger:
    ledger = Ledger(tmp_path / "analysis" / "ledger.jsonl")
    ev = ledger.add_evidence("der", "derive", {"quantity": "g", "value": 9.74, "unit": "m/s^2",
                                              "ci95": [9.62, 9.86]}, "g")  # fmt: skip
    ledger.add_claim("derived", ev.id, "g = {value}", "g = {value}", quantity="g")
    return ledger


def draft(text: str = "The value is {{c01}}.") -> ReportDraft:
    return ReportDraft(lang="en", title="T", aim="A", setup="S", method="M",
                       results=[{"heading": "Result", "body": text}], error_analysis="E",
                       conclusion="C", claims_used=["c01"])  # fmt: skip


def spec() -> ExperimentSpec:
    return ExperimentSpec(
        experiment="free_fall",
        target_concept="orange ball",
        analysis_window_s=(0, 2),
        motion_axis="y",
        confidence=0.9,
    )


def test_the_auditor_sees_claims_inline_and_no_analyst_conversation(ledger) -> None:
    message = auditor.build_message(
        draft(), SimpleNamespace(ledger=ledger, spec=spec()),
        [gate("G1")], pre(),
    )  # fmt: skip
    assert "[c01: g = 9.74 ± 0.06 m/s²]" in message
    assert "CLAIMS:" in message and "EVIDENCE SUMMARIES:" in message and "GATES:" in message
    assert "analyst" not in message.lower()  # the Analyst's messages are never copied in


def test_bare_number_in_the_draft_fails_the_precheck(ledger) -> None:
    result = auditor.pre_check(draft("The value is about 9.8 m/s²."), ledger, [gate("G1")], spec())
    assert result.g5.status == "fail"
    assert [g.gate for g in result.hard_failures] == ["G5"]


async def test_invalid_verdict_is_retried_once_with_the_errors(monkeypatch) -> None:
    good = out("PASS").model_dump_json()
    chat = FakeChat(['{"status": "PASS", "surprise": 1}', good]).install(monkeypatch)
    session = _session()
    result, model = await auditor.ask("message", session)
    assert result.status == "PASS" and model == MODEL_SPARK
    retry = chat.calls[1]["messages"]
    assert retry[-2]["role"] == "assistant" and "surprise" in retry[-1]["content"]


async def test_wonder_failure_falls_back_to_spark(monkeypatch) -> None:
    chat = FakeChat(["not json", "still not json", out("PASS_WITH_WARNINGS").model_dump_json()])
    chat.install(monkeypatch)
    session = _session(use_wonder=True)
    result, model = await auditor.ask("message", session)
    assert model == MODEL_SPARK and result.status == "PASS_WITH_WARNINGS"
    assert [c["model"] for c in chat.calls] == [MODEL_WONDER, MODEL_WONDER, MODEL_SPARK]


def test_run_uses_the_unavailable_fallback_as_a_flagged_warning() -> None:
    fallback = auditor.unavailable_output(pre())
    assert fallback.status == "PASS_WITH_WARNINGS"
    assert fallback.notes.startswith("auditor_unavailable")


def _session(*, use_wonder: bool = False):
    """Only what the Auditor reads from the session: settings and the event emitter."""
    events: list[tuple] = []
    ctx = SimpleNamespace(
        settings=Settings(use_wonder=use_wonder), emit=lambda t, **p: events.append((t, p))
    )
    return SimpleNamespace(ctx=ctx)
