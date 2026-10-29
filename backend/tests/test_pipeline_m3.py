"""M3 acceptance offline: recorded SAM responses (M1, live API) → g within 0.5 % of truth."""

from __future__ import annotations

import json
import shutil

import pytest
from typer.testing import CliRunner

from framelab.agent.ledger import Ledger
from framelab.cli import app as cli_app
from framelab.clients.sam import load_record
from framelab.physics.freefall import analyze_free_fall
from framelab.pipeline import measure
from tests.fakes import MEDIA, SAM_FIXTURES, freefall_fake

G_TRUE = 9.81  # synthetic videos were rendered with g = 9.81 m/s², 460 px = 1 m


@pytest.mark.parametrize(
    ("fixture", "fps"),
    [
        ("freefall_60fps_5s_orange_ball.sam.json.gz", 60),
        ("freefall_240fps_3s_orange_ball.sam.json.gz", 240),
        ("hold2s_60fps_orange_ball.sam.json.gz", 60),
    ],
)
async def test_recorded_sam_gives_true_g(fixture: str, fps: int) -> None:
    out = await load_record(SAM_FIXTURES / fixture)
    obs = measure.observations_from_result(
        out.result, "orange ball", offset_frame=0, effective_fps=fps
    )
    track = measure.clean_track(obs, effective_fps=fps, concept="orange ball", object_id="0")
    t, _x, y = measure.valid_arrays(track)
    r = analyze_free_fall(t, y, 1 / 460, 0.003 / 460, bootstrap_n=300)
    assert r.g == pytest.approx(G_TRUE, rel=0.005)
    assert r.ci95[0] <= G_TRUE <= r.ci95[1]
    assert r.fit.rms_px < 0.3


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")
def test_cli_run_reports_g_and_writes_evidence(tmp_path, monkeypatch) -> None:
    freefall_fake().install(monkeypatch)
    runs = tmp_path / "runs"
    result = CliRunner().invoke(
        cli_app,
        ["run", str(MEDIA / "freefall_60fps_5s.mp4"), "--concept", "orange ball",
         "--ref-concept", "ruler", "--ref-length-m", "1.0", "--ball-diameter-m", "0.10",
         "--runs-dir", str(runs)],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert "g = 9.8" in result.output and "Stopped after analyzing" in result.output
    run_dir = next(runs.glob("r_*"))
    state = json.loads((run_dir / "state.json").read_text())
    h = state["headline"]
    assert h["value"] == pytest.approx(G_TRUE, rel=0.005) and h["ci95"][0] <= G_TRUE <= h["ci95"][1]
    assert h["scale_method"] == "ruler" and h["scale_agreement_pct"] < 3.0
    ledger = Ledger(run_dir / "analysis" / "ledger.jsonl")
    ids = [e.id for e in ledger.evidence()]
    for expected in ("trk_01", "cal_01", "cal_02", "win_01", "fit_01", "ci_01", "der_01", "cmp_01"):
        assert expected in ids
    der = ledger.get(h["evidence"])
    assert der.data["value"] == h["value"]  # headline value comes from evidence, not elsewhere
    assert (run_dir / "tracks" / "orange_ball_0.csv").is_file()
    assert (run_dir / "calibration.json").is_file() and (
        run_dir / "analysis" / "freefall.json"
    ).is_file()


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")
def test_no_scale_reference_fails_with_clear_message(tmp_path, monkeypatch) -> None:
    freefall_fake().install(monkeypatch)
    runs = tmp_path / "runs"
    result = CliRunner().invoke(
        cli_app,
        [
            "run",
            str(MEDIA / "freefall_60fps_5s.mp4"),
            "--concept",
            "orange ball",
            "--runs-dir",
            str(runs),
        ],
    )
    assert result.exit_code == 1
    assert "No scale reference" in result.output
