"""M7 Designer (06 §6, §9.4): code computes the predictions, the safety limit holds, the texts
use placeholders only, and an invalid proposal is retried once."""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

import pytest

from framelab.agent import designer
from framelab.agent.ledger import Ledger
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
    assert any(t == "warning" and "designer unavailable" in p["message"] for t, p in events)


async def test_no_headline_means_no_proposal(tmp_path) -> None:
    sess, _ = session(tmp_path)
    assert await designer.run(sess, headline=None, gate_warnings=[]) is None
