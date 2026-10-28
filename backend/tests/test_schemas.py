from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from framelab.schemas import (
    Claim,
    ExperimentSpec,
    Quantity,
    ReferenceObject,
    RunState,
    Stage,
    utcnow,
)


def test_quantity_to_si() -> None:
    assert Quantity(value=50, unit="cm").to_si() == pytest.approx(0.5)
    assert Quantity(value=1.5, unit="s").to_si() == 1.5
    assert Quantity(value=180, unit="deg").to_si() == pytest.approx(math.pi)
    with pytest.raises(ValueError):
        Quantity(value=1, unit="furlong").to_si()


def test_experiment_spec_roundtrip() -> None:
    spec = ExperimentSpec(
        experiment="pendulum",
        target_concept="metal nut",
        reference=ReferenceObject(concept="ruler", length_m=0.3, length_source="spoken"),
        spoken_params={"string_length": Quantity(value=50, unit="cm", source_turn=0)},
        analysis_window_s=(1.0, 16.0),
        motion_axis="x",
        hypothesis="saya kira periodenya satu setengah detik",
        hypothesis_prediction=Quantity(value=1.5, unit="s", source_turn=1),
        confidence=0.9,
    )
    again = ExperimentSpec.model_validate_json(spec.model_dump_json())
    assert again == spec


def test_spec_confidence_bounds() -> None:
    with pytest.raises(ValidationError):
        ExperimentSpec(
            experiment="free_fall",
            target_concept="ball",
            analysis_window_s=(0, 1),
            motion_axis="y",
            confidence=1.5,
        )


def test_claim_requires_evidence() -> None:
    with pytest.raises(ValidationError):
        Claim(id="c01", claim_type="measurement", evidence=[], statement_en="x", statement_id="x")


def test_run_state_json_roundtrip() -> None:
    now = utcnow()
    state = RunState(
        run_id="r_1",
        created_at=now,
        updated_at=now,
        original_filename="a.mp4",
        stage_history=[(Stage.CREATED, now)],
    )
    assert RunState.model_validate_json(state.model_dump_json()) == state
