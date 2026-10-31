"""M7 Designer (06 §6, §9.4): code computes the predictions, the safety limit holds, the texts
use placeholders only, and an invalid proposal is retried once."""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

import pytest

from framelab.agent import designer
from framelab.agent.ledger import Ledger
from framelab.clients import baseten as baseten_client
from framelab.config import MODEL_SPARK
from framelab.schemas import ExperimentSpec
from tests.fakes import FakeChat

G = 9.81


def proposal(
    prediction_en: str = "The fall from {{p1}} takes {{p2}}.",
    prediction_id: str = "Jatuh dari {{p1}} memakan {{p2}}.",
) -> str:
    return json.dumps({
        "title": "Fall time against height",
        "changed_variable": "height",
        "controlled_variables": ["ball", "release from rest", "same camera"],
        "prediction_en": prediction_en,
        "prediction_id": prediction_id,
        "protocol_steps_en": ["Drop the ball from each height and time the fall."],
        "protocol_steps_id": ["Jatuhkan bola dari tiap tinggi dan ukur waktunya."],
        "safety_notes_en": ["Adult supervision for children."],
        "safety_notes_id": ["Pengawasan orang dewasa untuk anak."],
    })  # fmt: skip


def session(tmp_path, experiment: str = "free_fall"):
    events: list[tuple] = []
    ctx = SimpleNamespace(
        settings=SimpleNamespace(),
        run_dir=tmp_path,
        emit=lambda t, **p: events.append((t, p)),
    )
    spec = ExperimentSpec(experiment=experiment, target_concept="ball",
                          analysis_window_s=(0, 2), motion_axis="y", confidence=0.9)  # fmt: skip
    return SimpleNamespace(ctx=ctx, spec=spec, ledger=Ledger(tmp_path / "ledger.jsonl")), events


def test_predictions_are_computed_by_code() -> None:
    fall = designer.predict("free_fall", "height", [0.5, 1.0], G)
    assert fall[0]["predicted"] == pytest.approx(math.sqrt(2 * 0.5 / G))
    pend = designer.predict("pendulum", "string_length", [1.0], G)
    assert pend[0]["predicted"] == pytest.approx(2 * math.pi * math.sqrt(1.0 / G))


@pytest.mark.parametrize(
    ("experiment", "variable", "values", "message"),
    [
        ("free_fall", "height", [2.5], "safety limit"),
        ("free_fall", "height", [0.0], "safety limit"),
        ("free_fall", "height", [0.5] * 6, "values"),
        ("free_fall", "string_length", [0.5], "varies"),
        ("pendulum", "height", [0.5], "varies"),
    ],
)  # fmt: skip
def test_predict_refuses_unsafe_or_wrong_input(experiment, variable, values, message) -> None:
    with pytest.raises(ValueError, match=message):
        designer.predict(experiment, variable, values, G)


async def test_a_proposal_gets_its_numbers_from_code(tmp_path, monkeypatch) -> None:
    sess, _ = session(tmp_path)
    chat = FakeChat([
        [("predict", {"changed_variable": "height", "values": [0.5, 1.0]})],
        proposal(),
    ]).install(monkeypatch)  # fmt: skip

    result = await designer.run(sess, headline={"value": G}, gate_warnings=["G2 single scale"])

    assert result is not None
    assert result.changed_variable == "height"
    assert [q.value for q in result.new_values] == [0.5, 1.0]
    assert result.predicted_values[0]["id"] == "p1"
    assert result.predicted_values[1]["predicted"] == pytest.approx(math.sqrt(2 / G))
    assert [e.tool for e in sess.ledger.evidence()] == ["predict", "predict"]
    text = designer.render_text(result.prediction_en, result.predicted_values, "en")
    assert text.startswith("The fall from 0.319 s")  # rendered from code, not from the model
    assert (tmp_path / "analysis" / "designer.json").exists()
    assert chat.calls[0]["purpose"] == "designer"


async def test_an_unsafe_predict_call_is_answered_and_retried(tmp_path, monkeypatch) -> None:
    sess, _ = session(tmp_path)
    FakeChat([
        [("predict", {"changed_variable": "height", "values": [3.0]})],  # above 2 m
        [("predict", {"changed_variable": "height", "values": [0.8]})],
        proposal(prediction_en="The fall takes {{p1}}.", prediction_id="Jatuh {{p1}}."),
    ]).install(monkeypatch)  # fmt: skip

    result = await designer.run(sess, headline={"value": G}, gate_warnings=[])

    assert result is not None and [q.value for q in result.new_values] == [0.8]


async def test_bare_numbers_in_prediction_text_are_retried(tmp_path, monkeypatch) -> None:
    sess, _ = session(tmp_path)
    chat = FakeChat([
        [("predict", {"changed_variable": "height", "values": [0.5]})],
        proposal(prediction_en="The fall takes 0.32 s.", prediction_id="Jatuh 0,32 s."),
        proposal(prediction_en="The fall takes {{p1}}.", prediction_id="Jatuh {{p1}}."),
    ]).install(monkeypatch)  # fmt: skip

    result = await designer.run(sess, headline={"value": G}, gate_warnings=[])

    assert result is not None and result.prediction_en == "The fall takes {{p1}}."
    retry = chat.calls[2]["messages"][-1]["content"]
    assert "bare numbers" in retry


async def test_no_valid_proposal_means_no_designer_output(tmp_path, monkeypatch) -> None:
    sess, events = session(tmp_path)
    FakeChat([
        [("predict", {"changed_variable": "height", "values": [0.5]})],
        proposal(prediction_en="Takes {{p9}}.", prediction_id="Ambil {{p9}}."),
        proposal(prediction_en="Takes {{p9}}.", prediction_id="Ambil {{p9}}."),
    ]).install(monkeypatch)  # fmt: skip

    result = await designer.run(sess, headline={"value": G}, gate_warnings=[])

    assert result is None
    assert any(t == "warning" and "unavailable" in p["message"] for t, p in events)


async def test_no_headline_means_no_proposal(tmp_path) -> None:
    sess, _ = session(tmp_path)
    assert await designer.run(sess, headline=None, gate_warnings=[]) is None


# --- S-3: Muse Spark Open via Baseten ------------------------------------------------------

OPEN = designer.SPARK_OPEN.model


def open_session(tmp_path):
    sess, events = session(tmp_path)
    sess.ctx.settings = SimpleNamespace(designer_backend="baseten", baseten_api_key="k")
    return sess, events


def install_open(monkeypatch, steps) -> FakeChat:
    fake = FakeChat(steps)
    monkeypatch.setattr(baseten_client, "chat", fake)
    return fake


def test_provider_choice() -> None:
    def chain(**kw):
        return [p.model for p in designer.providers(SimpleNamespace(**kw))]

    assert chain(designer_backend="auto", baseten_api_key=None) == [MODEL_SPARK]
    assert chain(designer_backend="auto", baseten_api_key="k") == [OPEN, MODEL_SPARK]
    assert chain(designer_backend="meta", baseten_api_key="k") == [MODEL_SPARK]
    assert chain(designer_backend="baseten", baseten_api_key=None) == [OPEN, MODEL_SPARK]
    assert chain() == [MODEL_SPARK]


async def test_spark_open_submits_the_proposal_through_a_tool(tmp_path, monkeypatch) -> None:
    sess, _ = open_session(tmp_path)
    fake = install_open(monkeypatch, [
        [("predict", {"changed_variable": "height", "values": [0.5, 1.0]})],
        [("submit_proposal", json.loads(proposal()))],
    ])  # fmt: skip
    FakeChat([]).install(monkeypatch)  # Muse Spark 1.3 must not be called

    result = await designer.run(sess, headline={"value": G}, gate_warnings=[])

    assert result is not None and result.models == [OPEN]
    assert fake.calls[0]["model"] == OPEN
    write = fake.calls[1]
    assert "response_format" not in write  # Baseten has no structured outputs for Spark Open
    assert [t["function"]["name"] for t in write["tools"]] == ["submit_proposal"]
    saved = json.loads((tmp_path / "analysis" / "designer.json").read_text(encoding="utf-8"))
    assert saved["models"] == [OPEN]


async def test_spark_open_json_reply_without_the_tool_is_accepted(tmp_path, monkeypatch) -> None:
    sess, _ = open_session(tmp_path)
    install_open(monkeypatch, [
        [("predict", {"changed_variable": "height", "values": [0.5, 1.0]})],
        "Here is the proposal:\n```json\n" + proposal() + "\n```",
    ])  # fmt: skip

    result = await designer.run(sess, headline={"value": G}, gate_warnings=[])

    assert result is not None and result.title == "Fall time against height"


async def test_spark_open_failure_falls_back_to_muse_spark(tmp_path, monkeypatch) -> None:
    sess, events = open_session(tmp_path)
    install_open(monkeypatch, [baseten_client.BasetenError(503, "overloaded")])
    meta = FakeChat([
        [("predict", {"changed_variable": "height", "values": [0.5, 1.0]})],
        proposal(),
    ]).install(monkeypatch)  # fmt: skip

    result = await designer.run(sess, headline={"value": G}, gate_warnings=[])

    assert result is not None and result.models == [MODEL_SPARK]
    assert meta.calls[0]["model"] == MODEL_SPARK
    warning = next(p["message"] for t, p in events if t == "warning")
    assert "Muse Spark Open" in warning and "trying Muse Spark 1.3" in warning


async def test_a_failed_write_step_is_finished_by_muse_spark(tmp_path, monkeypatch) -> None:
    """The predictions from Spark Open are kept (no second set of pred_ evidence)."""
    sess, _ = open_session(tmp_path)
    bad = json.loads(proposal(prediction_en="Takes 0.3 s.", prediction_id="0,3 s."))
    install_open(monkeypatch, [
        [("predict", {"changed_variable": "height", "values": [0.5, 1.0]})],
        [("submit_proposal", bad)],
        [("submit_proposal", bad)],
    ])  # fmt: skip
    FakeChat([proposal()]).install(monkeypatch)

    result = await designer.run(sess, headline={"value": G}, gate_warnings=[])

    assert result is not None and result.models == [OPEN, MODEL_SPARK]
    assert [e.tool for e in sess.ledger.evidence()] == ["predict", "predict"]


def test_credits_name_spark_open_only_when_it_designed() -> None:
    from framelab.pipeline import overlay, report

    assert "Muse Spark Open" in overlay.built_with("en", None, [OPEN])
    assert "Muse Spark Open" not in overlay.built_with("en", None, [MODEL_SPARK])
    roles = {r["role"]: r["use"] for r in report.models_used(None, None, [OPEN])}  # type: ignore[arg-type]
    assert roles["Muse Spark Open (open weights, via Baseten)"] == "Designer"
    assert "Designer" not in roles["Muse Spark 1.3"]
