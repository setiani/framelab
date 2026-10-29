"""End-to-end M2 pipeline (ingest → SAM → observations) on recorded SAM responses."""

from __future__ import annotations

import csv
import json
import shutil

import pytest
from typer.testing import CliRunner

from framelab.cli import app as cli_app
from framelab.cli import flag_fields, manual_spec
from framelab.config import Settings
from framelab.events import EventBus
from framelab.pipeline.run import execute_run
from framelab.schemas import Stage
from framelab.store import RunStore
from tests.fakes import MEDIA, freefall_fake

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")


def spec():
    fields = flag_fields(
        experiment="free_fall",
        concept="orange ball",
        alt_concepts=[],
        ref_concept="ruler",
        ref_length_m=1.0,
        ref_frame_s=None,
        window=None,
        ball_diameter_m=None,
        string_length_m=None,
        predict=None,
    )
    return manual_spec("id", fields)


async def test_execute_run_until_measuring(tmp_path, monkeypatch) -> None:
    fake = freefall_fake().install(monkeypatch)
    settings = Settings(runs_dir=tmp_path / "runs")
    store = RunStore(settings.runs_dir)
    state = store.create_run("freefall.mp4")
    shutil.copyfile(MEDIA / "freefall_60fps_5s.mp4", store.input_video_path(state.run_id))

    final = await execute_run(state.run_id, settings, spec_override=spec(), until=Stage.MEASURING)

    run_dir = store.run_dir(state.run_id)
    assert final.stage == Stage.MEASURING, (
        (run_dir / "error.json").read_text() if (run_dir / "error.json").exists() else ""
    )
    assert fake.calls == [("upload", "analysis.mp4"), ("video", "orange ball"), ("image", "ruler")]
    assert fake.deleted == ["file-fake-1"]  # uploaded clip deleted (M1 V10)

    rows = list(csv.DictReader((run_dir / "observations.csv").open()))
    ball = [r for r in rows if r["concept"] == "orange ball"]
    ruler = [r for r in rows if r["concept"] == "ruler"]
    assert len(ball) == 268 and int(ball[0]["frame_index"]) == 32
    assert float(ball[0]["t"]) == pytest.approx(32 / 60)
    # PCA "major axis" of a uniform bar is 4·sqrt(L²/12) ≈ 1.155·L (true length via extent, M3)
    assert len(ruler) == 1 and float(ruler[0]["major_axis_px"]) == pytest.approx(
        462 * 1.1547, rel=0.01
    )

    summary = json.loads((run_dir / "measurement.json").read_text())
    assert summary["target"]["first_mask_frame"] == 32
    assert summary["reference"]["concept"] == "ruler"
    assert (run_dir / "debug" / "centroids.png").stat().st_size > 1000
    for name in ("video", "observations", "debug_centroids"):
        assert name in final.artifacts

    types = [e["type"] for e in EventBus(settings.runs_dir).iter_events(state.run_id)]
    assert types[0] == "stage_started" and types[-1] == "done"
    assert "artifact" in types


def test_cli_run_m2(tmp_path, monkeypatch) -> None:
    freefall_fake().install(monkeypatch)
    video = tmp_path / "ball.mp4"
    shutil.copyfile(MEDIA / "freefall_60fps_5s.mp4", video)
    result = CliRunner().invoke(
        cli_app,
        ["run", str(video), "--no-planner", "--concept", "orange ball", "--ref-concept", "ruler",
         "--ref-length-m", "1.0", "--until", "measuring", "--runs-dir", str(tmp_path / "runs")],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert "Stopped after measuring" in result.output
    assert "observations" in result.output


async def test_missing_target_still_measures_reference(tmp_path, monkeypatch) -> None:
    fake = freefall_fake()
    fake.video = {}  # SAM finds nothing → alt concepts → box fallback also empty
    fake.install(monkeypatch)
    settings = Settings(runs_dir=tmp_path / "runs")
    store = RunStore(settings.runs_dir)
    state = store.create_run("x.mp4")
    shutil.copyfile(MEDIA / "freefall_60fps_5s.mp4", store.input_video_path(state.run_id))
    s = spec().model_copy(update={"target_alt_concepts": ["ball"]})
    final = await execute_run(state.run_id, settings, spec_override=s, until=Stage.MEASURING)
    assert final.stage == Stage.MEASURING
    assert ("video", "ball") in fake.calls
    assert any("No usable target track" in w for w in final.warnings)


async def test_failure_writes_error_json(tmp_path) -> None:
    settings = Settings(runs_dir=tmp_path / "runs")
    store = RunStore(settings.runs_dir)
    state = store.create_run("bad.mp4")
    store.input_video_path(state.run_id).write_bytes(b"not a video")
    final = await execute_run(state.run_id, settings, spec_override=spec(), until=Stage.MEASURING)
    assert final.stage == Stage.FAILED and "IngestError" in (final.error or "")
    assert (store.run_dir(state.run_id) / "error.json").is_file()
