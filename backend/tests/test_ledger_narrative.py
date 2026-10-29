"""M6 ledger and narrative: claims copy numbers from evidence, placeholders render, tool results
stay small (06 §1, §4, §6)."""

from __future__ import annotations

import json

import pytest

from framelab.agent import narrative
from framelab.agent.ledger import Ledger
from framelab.agent.tools import MAX_RESULT_BYTES, ToolResult, dispatch, tool_schemas, validate_args


@pytest.fixture
def ledger(tmp_path) -> Ledger:
    ledger = Ledger(tmp_path / "analysis" / "ledger.jsonl")
    ledger.add_evidence(
        "der", "derive",
        {"quantity": "g", "value": 9.81, "unit": "m/s^2", "ci95": [9.75, 9.87]},
        "g = 9.81 ± 0.03 m/s²",
    )  # fmt: skip
    ledger.add_evidence(
        "cmp", "compare_reference",
        {"value": 9.81, "lo": 9.75, "hi": 9.87, "rel_diff_pct": 0.1, "unit": "m/s^2"},
        "compared",
    )  # fmt: skip
    ledger.add_evidence(
        "cmp", "hypothesis_test",
        {"p": 1.5, "m": 1.48, "lo": 1.4, "hi": 1.6, "outcome": "supported", "unit": "s"},
        "hypothesis",
    )  # fmt: skip
    return ledger


def test_add_claim_copies_value_ci_and_unit_from_evidence(ledger) -> None:
    claim = ledger.add_claim("derived", "der_01", "g = {value}", "g = {value}", quantity="g")
    assert claim.id == "c01"
    assert claim.value == 9.81
    assert claim.ci95 == (9.75, 9.87)
    assert claim.unit == "m/s^2"
    assert claim.evidence == ["der_01"]
    # the claim is persisted, so a reader gets the same numbers back
    assert ledger.get_claim("c01").value == 9.81


def test_add_claim_takes_the_measured_value_from_a_hypothesis_test(ledger) -> None:
    claim = ledger.add_claim("hypothesis_test", "cmp_02", "{outcome}", "{outcome}")
    assert claim.value == 1.48 and claim.unit == "s"


def test_add_claim_rejects_unknown_evidence_and_claim_types(ledger) -> None:
    with pytest.raises(KeyError):
        ledger.add_claim("derived", "der_99", "g = {value}", "g = {value}")
    with pytest.raises(ValueError, match="unknown claim_type"):
        ledger.add_claim("opinion", "der_01", "g = {value}", "g = {value}")


@pytest.mark.parametrize(
    ("statement", "message"),
    [
        ("g is 9.8", "bare number"),
        ("g = {value}, see {foo}", "unknown template field"),
        ("g = {{c01}}", "cannot contain"),
    ],
)  # fmt: skip
def test_add_claim_rejects_numbers_and_unknown_fields(ledger, statement, message) -> None:
    with pytest.raises(ValueError, match=message):
        ledger.add_claim("derived", "der_01", statement, "g = {value}")


def test_statement_templates_may_use_the_allowed_fields(ledger) -> None:
    claim = ledger.add_claim(
        "derived", "der_01", "g = {value}, at 95% confidence {ci}", "{value} {unit}"
    )
    assert claim.id == "c01"


def test_render_statement_en_and_id(ledger) -> None:
    claim = ledger.add_claim("derived", "der_01", "g = {value}", "g = {value}")
    evidence = ledger.get("der_01")
    assert narrative.render_statement(claim, evidence, "en") == "g = 9.81 ± 0.03 m/s²"
    assert narrative.render_statement(claim, evidence, "id") == "g = 9,81 ± 0,03 m/s²"


def test_render_statement_uses_outcome_and_rel_diff(ledger) -> None:
    hyp = ledger.add_claim("hypothesis_test", "cmp_02", "{outcome}", "{outcome}")
    text = narrative.render_statement(hyp, ledger.get("cmp_02"), "id")
    assert text == "didukung"
    cmp_claim = ledger.add_claim("comparison", "cmp_01", "off by {rel_diff}", "{rel_diff}")
    assert narrative.render_statement(cmp_claim, ledger.get("cmp_01"), "en") == "off by +0.10%"


def test_render_fills_claim_placeholders_and_rejects_unknown_ids() -> None:
    assert narrative.render("Result {{c01}}; see {{c02}}.", {"c01": "A", "c02": "B"}) == \
        "Result A; see B."  # fmt: skip
    with pytest.raises(KeyError):
        narrative.render("Result {{c09}}", {"c01": "A"})


def test_bare_numbers_respects_the_allow_list() -> None:
    allowed = "Muse Spark 1.3, SAM 3.1, G5, at 95% confidence, T₀²\n1. Aim\n2. Method"
    assert narrative.bare_numbers(allowed) == []
    assert narrative.bare_numbers("The mass is 2 kg") == ["2"]
    assert narrative.bare_numbers("{{c01}} then 0.5") == ["0", "5"]


def test_tool_result_is_at_most_two_kilobytes() -> None:
    rows = [{"t": i, "y": i * 1.5} for i in range(2000)]
    huge = ToolResult(True, "summary " * 400, evidence_ids=[f"fit_{i:02d}" for i in range(200)],
                      data_excerpt={"rows": rows})  # fmt: skip
    text = huge.to_json()
    assert len(text.encode("utf-8")) <= MAX_RESULT_BYTES
    body = json.loads(text)
    assert body["ok"] is True  # the status survives the cut


def test_small_tool_result_is_returned_whole() -> None:
    text = ToolResult(False, "bad window", data_excerpt={"n": 3}).to_json()
    assert json.loads(text) == {"ok": False, "summary": "bad window", "evidence_ids": [],
                                "data": {"n": 3}, "warnings": []}  # fmt: skip


def test_validate_args_checks_required_types_and_enums() -> None:
    schema = next(
        t["function"]["parameters"] for t in tool_schemas() if t["function"]["name"] == "fit_model"
    )
    assert validate_args(schema, {"window_id": "win_01", "model": "quadratic"}) == []
    assert validate_args(schema, {"window_id": "win_01"}) == ["missing model"]
    assert validate_args(schema, {"window_id": 3, "model": "quadratic"}) == [
        "window_id must be string"
    ]
    assert "must be one of" in validate_args(schema, {"window_id": "w", "model": "linear"})[0]


def test_every_tool_has_a_schema_in_openai_format() -> None:
    schemas = tool_schemas()
    assert len(schemas) == 15
    for item in schemas:
        assert item["type"] == "function"
        assert item["function"]["parameters"]["type"] == "object"


async def test_unknown_tool_is_an_error_result(tmp_path, monkeypatch) -> None:
    from framelab.agent.session import AnalysisSession

    class Ctx:
        emit = staticmethod(lambda *args, **kwargs: None)

    session = AnalysisSession.__new__(AnalysisSession)
    session.ctx = Ctx()
    result = await dispatch(session, "delete_everything", {})
    assert result.ok is False and "unknown tool" in result.summary


def test_scale_evidence_gives_a_claim_with_its_value(ledger) -> None:
    data = {"method": "ruler", "value": 2.17, "unit": "mm/px"}
    ledger.add_evidence("cal", "calibrate_scale", data, "ruler scale")
    claim = ledger.add_claim("quality", "cal_01", "The scale is {value}.", "Skala {value}.")
    assert claim.value == 2.17 and claim.unit == "mm/px"
    assert (
        narrative.render_statement(claim, ledger.get("cal_01"), "en") == "The scale is 2.17 mm/px."
    )


def test_missing_field_error_lists_what_the_evidence_provides(ledger) -> None:
    claim = ledger.add_claim("comparison", "cmp_02", "{rel_diff}", "{rel_diff}")
    with pytest.raises(ValueError, match="provides"):
        narrative.render_statement(claim, ledger.get("cmp_02"), "en")
