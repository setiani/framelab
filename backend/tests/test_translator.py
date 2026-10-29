"""M7 translator (04 §9, 06 §9.5): the second language is used only when the placeholders, the
claim ids and the numbers are unchanged and G5 passes."""

from __future__ import annotations

import json

import pytest

from framelab.agent import translator
from framelab.agent.ledger import Ledger
from framelab.schemas import ReportDraft
from tests.fakes import FakeChat


@pytest.fixture
def ledger(tmp_path) -> Ledger:
    ledger = Ledger(tmp_path / "ledger.jsonl")
    ev = ledger.add_evidence(
        "der", "derive",
        {"quantity": "g", "value": 9.74, "unit": "m/s^2", "ci95": [9.62, 9.86]},
        "g",
    )  # fmt: skip
    ledger.add_claim("derived", ev.id, "g = {value}", "g = {value}", quantity="g")
    return ledger


def draft() -> ReportDraft:
    return ReportDraft(
        lang="en",
        title="Free fall",
        aim="Measure g.",
        setup="A ball fell.",
        method="Quadratic fit.",
        results=[{"heading": "Result", "body": "The value is {{c01}}."}],
        error_analysis="Small spread.",
        conclusion="See {{c01}}.",
        claims_used=["c01"],
    )  # fmt: skip


def translated(**changes) -> str:
    base = draft().model_dump(mode="json")
    base.update(
        lang="id",
        title="Jatuh bebas",
        aim="Mengukur g.",
        setup="Bola jatuh.",
        method="Fit kuadratik.",
        results=[{"heading": "Hasil", "body": "Nilainya {{c01}}."}],
        error_analysis="Sebaran kecil.",
        conclusion="Lihat {{c01}}.",
    )  # fmt: skip
    base.update(changes)
    return json.dumps(base)


async def test_a_faithful_translation_is_used(ledger, monkeypatch) -> None:
    FakeChat([translated()]).install(monkeypatch)
    out, problem = await translator.translate(draft(), ledger, dst="id")
    assert problem == "" and out is not None
    assert out.lang == "id" and out.claims_used == ["c01"]


async def test_a_dropped_placeholder_keeps_one_language(ledger, monkeypatch) -> None:
    FakeChat([translated(conclusion="Lihat hasilnya.")]).install(monkeypatch)
    out, problem = await translator.translate(draft(), ledger, dst="id")
    assert out is None and "placeholders differ" in problem


async def test_a_new_number_keeps_one_language(ledger, monkeypatch) -> None:
    FakeChat([translated(conclusion="Lihat {{c01}}, sekitar 9.8.")]).install(monkeypatch)
    out, problem = await translator.translate(draft(), ledger, dst="id")
    assert out is None and "adds numbers" in problem


async def test_an_invalid_translation_keeps_one_language(ledger, monkeypatch) -> None:
    FakeChat(["not a draft"]).install(monkeypatch)
    out, problem = await translator.translate(draft(), ledger, dst="id")
    assert out is None and "not a valid draft" in problem


async def test_same_language_is_not_translated(ledger) -> None:
    out, problem = await translator.translate(draft(), ledger, dst="en")
    assert out is None and problem == "same language"
