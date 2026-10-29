"""M5 Planner offline tests: strict schema, validation, retry, drop, evidence, overrides."""

from __future__ import annotations

import json
from typing import Any

import pytest

from framelab.agent import planner
from framelab.agent.ledger import Ledger
from framelab.agent.planner import PlannerError, PlannerOutput, to_spec, validate_spec
from framelab.clients.meta import strict_schema
from framelab.config import MODEL_SPARK
from framelab.schemas import ExperimentSpec, ProbeInfo, Quantity, Transcript, TranscriptTurn
from tests.fakes import FakeResponses

TRANSCRIPT = Transcript(
    text="Tinggi jatuh 0.5 meter. Hipotesis saya 1,5 detik.",
    turns=[
        TranscriptTurn(index=0, text="Tinggi jatuh 0.5 meter.", start_ms=0, end_ms=1800),
        TranscriptTurn(index=1, text="Hipotesis saya 1,5 detik.", start_ms=1800, end_ms=3500),
    ],
)
PROBE = ProbeInfo(
    width=960, height=540, duration_s=5.0, avg_fps=60.0, r_fps=60.0, nb_frames=300,
    is_vfr=False, has_audio=False, codec="h264",
)  # fmt: skip


def output_dict(**changes: Any) -> dict[str, Any]:
    """A valid Spark answer; ``changes`` replace top-level keys."""
    base: dict[str, Any] = {
        "experiment": "free_fall",
        "target_concept": "Orange Ball",
        "target_alt_concepts": ["ball"],
        "reference": {
            "concept": "ruler",
            "length_m": 1.0,
            "length_source": "spoken",
            "frame_hint_s": 0.5,
        },
        "object_diameter_m": None,
        "spoken_params": [{"key": "drop_height", "value": 0.5, "unit": "m", "source_turn": 0}],
        "analysis_window_start_s": 0.2,
        "analysis_window_end_s": 2.0,
        "motion_axis": "y",
        "hypothesis": "Hipotesis saya 1,5 detik.",
        "hypothesis_prediction": {"value": 1.5, "unit": "s", "source_turn": 1},
        "setup_description": "Bola oranye dijatuhkan dari ketinggian.",
        "confidence": 0.9,
        "warnings": [],
    }
    base.update(changes)
    return base


def as_text(**changes: Any) -> str:
    return json.dumps(output_dict(**changes))


def spoken(value: float = 0.5, unit: str = "m", turn: int = 0, key: str = "drop_height") -> dict:
    return {"key": key, "value": value, "unit": unit, "source_turn": turn}


def good_spec() -> ExperimentSpec:
    return to_spec(PlannerOutput.model_validate(output_dict()), "id")


def _walk_objects(node: Any, path: str = "") -> list[str]:
    """Paths of JSON-schema objects that a strict schema does not close properly."""
    problems: list[str] = []
    if isinstance(node, dict):
        if node.get("type") == "object" or "properties" in node:
            props = set(node.get("properties", {}))
            if node.get("additionalProperties") is not False:
                problems.append(f"{path}: additionalProperties is not false")
            if set(node.get("required", [])) != props:
                problems.append(f"{path}: required does not list every property")
        for key, value in node.items():
            problems += _walk_objects(value, f"{path}/{key}")
    elif isinstance(node, list):
        for i, value in enumerate(node):
            problems += _walk_objects(value, f"{path}/{i}")
    return problems


def test_strict_schema_closes_every_object() -> None:
    schema = strict_schema(PlannerOutput)
    assert _walk_objects(schema) == []
    # spoken params are a list of closed objects, not a dict (a dict would allow no keys)
    assert schema["properties"]["spoken_params"]["type"] == "array"


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ("Hipotesis saya 1,5 detik.", 1.5),  # decimal comma
        ("Tinggi jatuh 0.5 meter", 0.5),
        ("satu meter", 1.0),  # Voice writes words too (M5 live check)
        ("diameter bola sepuluh sentimeter", 10.0),
        ("tinggi jatuh setengah meter", 0.5),
        ("nol koma empat lima detik", 0.45),
        ("zero point four five seconds", 0.45),
        ("dua puluh sentimeter", 20.0),
        ("dua belas", 12.0),
        ("twenty five centimeters", 25.0),
    ],
)
def test_value_in_text_reads_digits_and_words(text: str, value: float) -> None:
    assert planner.value_in_text(value, text)


def test_value_in_text_rejects_other_numbers() -> None:
    assert not planner.value_in_text(0.7, "Tinggi jatuh 0.5 meter")
    assert not planner.value_in_text(2.0, "satu meter")
    assert not planner.value_in_text(0.5, "tinggi jatuh banyak sekali")


def test_to_spec_lowercases_and_builds_quantities() -> None:
    spec = good_spec()
    assert spec.target_concept == "orange ball"
    assert spec.spoken_params["drop_height"] == Quantity(value=0.5, unit="m", source_turn=0)
    assert spec.analysis_window_s == (0.2, 2.0)
    assert spec.language == "id"


def test_valid_spec_has_no_errors() -> None:
    assert validate_spec(good_spec(), TRANSCRIPT, PROBE.duration_s) == []


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({"target_concept": "a big bright orange rubber ball"}, "1–4 words"),
        ({"target_concept": "orange ball!"}, "lowercase words without punctuation"),
        ({"analysis_window_end_s": 6.0}, "must lie inside [0, 5.00] s"),
        ({"analysis_window_start_s": 1.9, "analysis_window_end_s": 2.0}, "minimum is 0.5 s"),
        ({"spoken_params": [spoken(value=0.7)]}, "0.7 is not in turn 0"),
        ({"spoken_params": [spoken(turn=9)]}, "source_turn 9 is not a transcript turn"),
        ({"spoken_params": [spoken(unit="furlong")]}, "unknown unit 'furlong'"),
    ],
)  # fmt: skip
def test_validate_spec_reports_each_rule(changes: dict[str, Any], expected: str) -> None:
    spec = to_spec(PlannerOutput.model_validate(output_dict(**changes)), "id")
    errors = validate_spec(spec, TRANSCRIPT, PROBE.duration_s)
    assert any(expected in error for error in errors), errors


def test_run_happy_path_records_spoken_evidence(tmp_path, monkeypatch) -> None:
    responses = FakeResponses([as_text()]).install(monkeypatch)
    ledger = Ledger(tmp_path / "ledger.jsonl")

    import asyncio

    spec = asyncio.run(planner.run("file-1", TRANSCRIPT, PROBE, "id", "note", ledger=ledger))

    assert spec.target_concept == "orange ball"
    assert spec.spoken_params["drop_height"].value == 0.5
    call = responses.calls[0]
    assert call["model"] == MODEL_SPARK and call["purpose"] == "planner"
    assert call["text"]["format"]["strict"] is True
    user_content = call["input"][0]["content"]
    assert user_content[1] == {"type": "input_file", "file_id": "file-1"}
    assert "Tinggi jatuh" in user_content[0]["text"]  # transcript travels with the video
    evidence = {e.data["key"]: e for e in ledger.evidence() if e.tool == "planner"}
    assert evidence["drop_height"].data["turn_text"] == "Tinggi jatuh 0.5 meter."
    assert evidence["hypothesis_prediction"].data["value"] == 1.5
    assert evidence["drop_height"].id.startswith("spk_")


def test_retry_sends_the_validation_errors_back(monkeypatch) -> None:
    bad = as_text(spoken_params=[spoken(value=0.7)])
    responses = FakeResponses([bad, as_text()]).install(monkeypatch)

    import asyncio

    spec = asyncio.run(planner.run("file-1", TRANSCRIPT, PROBE, "id", None))

    assert len(responses.calls) == 2
    assert spec.spoken_params["drop_height"].value == 0.5  # the corrected answer won
    retry_input = responses.calls[1]["input"]
    assert retry_input[1] == {"role": "assistant", "content": bad}
    assert "0.7 is not in turn 0" in retry_input[2]["content"]


def test_untraceable_value_is_dropped_after_the_retry(monkeypatch) -> None:
    bad = as_text(spoken_params=[spoken(value=0.7)])
    FakeResponses([bad, bad]).install(monkeypatch)

    import asyncio

    spec = asyncio.run(planner.run("file-1", TRANSCRIPT, PROBE, "id", None))

    assert "drop_height" not in spec.spoken_params
    assert any("Dropped a value the transcript does not support" in w for w in spec.warnings)
    assert spec.target_concept == "orange ball"  # the rest of the spec is kept


def test_hypothesis_prediction_is_dropped_when_untraceable(monkeypatch) -> None:
    bad = as_text(hypothesis_prediction={"value": 9.9, "unit": "s", "source_turn": 1})
    FakeResponses([bad, bad]).install(monkeypatch)

    import asyncio

    spec = asyncio.run(planner.run("file-1", TRANSCRIPT, PROBE, "id", None))

    assert spec.hypothesis_prediction is None


def test_structural_problem_after_retry_raises(monkeypatch) -> None:
    bad = as_text(target_concept="a big bright orange rubber ball")
    FakeResponses([bad, bad]).install(monkeypatch)

    import asyncio

    with pytest.raises(PlannerError, match="1–4 words"):
        asyncio.run(planner.run("file-1", TRANSCRIPT, PROBE, "id", None))


def test_unparseable_output_raises_after_retry(monkeypatch) -> None:
    FakeResponses(["not json", "still not json"]).install(monkeypatch)

    import asyncio

    with pytest.raises(PlannerError, match="Planner output invalid"):
        asyncio.run(planner.run("file-1", TRANSCRIPT, PROBE, "id", None))


@pytest.mark.parametrize(
    ("changes", "stop"),
    [
        ({"experiment": "unknown"}, True),
        ({"confidence": 0.39}, True),
        ({"confidence": 0.4}, False),
    ],
)
def test_needs_rerecord_gate(changes: dict[str, Any], stop: bool) -> None:
    spec = to_spec(PlannerOutput.model_validate(output_dict(**changes)), "id")
    assert planner.needs_rerecord(spec) is stop


def test_apply_overrides_wins_and_merges_spoken_params() -> None:
    spec = good_spec()
    merged = planner.apply_overrides(
        spec,
        {
            "target_concept": "red ball",
            "spoken_params": {"string_length": Quantity(value=0.6, unit="m")},
        },
    )
    assert merged.target_concept == "red ball"
    assert set(merged.spoken_params) == {"drop_height", "string_length"}
    assert spec.target_concept == "orange ball"  # the original spec is untouched
