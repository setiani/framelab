"""M5 offline pipeline: the Planner drives the run (fake Spark, fake Voice, SAM fakes from M2)."""

from __future__ import annotations

import json
import shutil

import pytest
from typer.testing import CliRunner

from framelab.agent import planner
from framelab.cli import app as cli_app
from framelab.schemas import ExperimentSpec, ReferenceObject
from tests.fakes import MEDIA, freefall_fake

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")


def planner_spec(**changes) -> ExperimentSpec:
    """What a Planner run returns for the free-fall fixture (no audio, so no spoken values)."""
    base = {
        "experiment": "free_fall",
        "target_concept": "orange ball",
        "target_alt_concepts": ["ball"],
        "reference": ReferenceObject(
            concept="ruler", length_m=1.0, length_source="spoken", frame_hint_s=0.5
        ),
        "analysis_window_s": (0.2, 2.0),
        "motion_axis": "y",
        "language": "en",
        "setup_description": "Orange ball dropped in front of a ruler.",
        "confidence": 0.9,
    }
    base.update(changes)
    return ExperimentSpec(**base)


def install_planner(monkeypatch, spec: ExperimentSpec, calls: list[dict]) -> None:
    async def fake_run(file_id, transcript, probe, lang, note, *, emit=None, ledger=None):
        calls.append(
            {"file_id": file_id, "lang": lang, "turns": len(transcript.turns), "note": note}
        )
        return spec

    monkeypatch.setattr(planner, "run", fake_run)


def segmentation_calls(fake) -> list[tuple[str, str]]:
    """SAM requests only (uploads are not segmentation)."""
    return [c for c in fake.calls if c[0] in ("video", "image")]


def _run(tmp_path, *extra: str):
    runs = tmp_path / "runs"
    result = CliRunner().invoke(
        cli_app,
        ["run", str(MEDIA / "freefall_60fps_5s.mp4"), "--runs-dir", str(runs), *extra],
    )
    run_dir = next(runs.glob("r_*"))
    return result, run_dir, json.loads((run_dir / "state.json").read_text())


@needs_ffmpeg
def test_planner_spec_drives_the_run_and_uploads_are_deleted(tmp_path, monkeypatch) -> None:
    fake = freefall_fake().install(monkeypatch)
    calls: list[dict] = []
    install_planner(monkeypatch, planner_spec(), calls)

    result, run_dir, state = _run(tmp_path, "--lang", "en")

    assert result.exit_code == 0, result.output
    assert calls == [{"file_id": "file-fake-1", "lang": "en", "turns": 0, "note": None}]
    assert ("upload", "video.mp4") in fake.calls  # the planner gets the normalized video
    assert ("video", "orange ball") in fake.calls
    assert "file-fake-1" in fake.deleted  # planner upload removed after use (M1 V10)
    spec = json.loads((run_dir / "spec.json").read_text())
    assert spec["target_concept"] == "orange ball" and spec["experiment"] == "free_fall"
    assert json.loads((run_dir / "transcript.json").read_text())["turns"] == []
    assert state["stage"] == "analyzing" and state["experiment"] == "free_fall"
    assert any("No audio track" in w for w in state["warnings"])


@needs_ffmpeg
def test_flags_override_the_planner_spec(tmp_path, monkeypatch) -> None:
    fake = freefall_fake().install(monkeypatch)
    calls: list[dict] = []
    install_planner(monkeypatch, planner_spec(target_concept="red ball"), calls)

    result, run_dir, _ = _run(tmp_path, "--concept", "orange ball")

    assert result.exit_code == 0, result.output
    assert ("video", "orange ball") in fake.calls and ("video", "red ball") not in fake.calls
    assert json.loads((run_dir / "spec.json").read_text())["target_concept"] == "orange ball"


@needs_ffmpeg
def test_unknown_experiment_stops_with_needs_rerecord(tmp_path, monkeypatch) -> None:
    fake = freefall_fake().install(monkeypatch)
    install_planner(monkeypatch, planner_spec(experiment="unknown"), [])

    result, _, state = _run(tmp_path)

    assert result.exit_code == 0, result.output
    assert state["verdict"] == "NEEDS_RERECORD" and state["stage"] == "done"
    assert "Verdict: NEEDS_RERECORD" in result.output
    assert segmentation_calls(fake) == []  # no segmentation after the stop (06 §2)


@needs_ffmpeg
def test_low_confidence_stops_with_needs_rerecord(tmp_path, monkeypatch) -> None:
    fake = freefall_fake().install(monkeypatch)
    install_planner(monkeypatch, planner_spec(confidence=0.3), [])

    _, _, state = _run(tmp_path)

    assert state["verdict"] == "NEEDS_RERECORD"
    assert segmentation_calls(fake) == []


@needs_ffmpeg
def test_melting_is_not_supported_yet(tmp_path, monkeypatch) -> None:
    freefall_fake().install(monkeypatch)
    install_planner(monkeypatch, planner_spec(experiment="melting"), [])

    result, _, state = _run(tmp_path)

    assert result.exit_code == 1
    assert state["stage"] == "failed" and "not supported yet" in state["error"]


@needs_ffmpeg
def test_no_planner_makes_no_api_calls(tmp_path, monkeypatch) -> None:
    fake = freefall_fake().install(monkeypatch)

    async def must_not_run(*args, **kwargs):
        raise AssertionError("Planner called with --no-planner")

    monkeypatch.setattr(planner, "run", must_not_run)

    result, _, state = _run(
        tmp_path, "--no-planner", "--concept", "orange ball", "--ref-concept", "ruler",
        "--ref-length-m", "1.0",
    )  # fmt: skip

    assert result.exit_code == 0, result.output
    assert ("video", "orange ball") in fake.calls and state["stage"] == "analyzing"
    assert ("upload", "video.mp4") not in fake.calls  # no planner upload was made
