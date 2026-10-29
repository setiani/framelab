"""M8 outputs: charts, data.csv, report HTML and PDF, recording feedback, overlay, replay.

Spec: docs/plan/04-pipeline-and-measurement.md §7–§10, docs/plan/09 M8.
The run folders are built from synthetic values (free fall with g = 9.8 m/s² by construction), so
the numbers can be checked exactly. The overlay test runs the real pipeline on the recorded SAM
fixtures, and replay is checked with every model call blocked."""

from __future__ import annotations

import csv
import json
import re
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image
from typer.testing import CliRunner

from framelab.agent.ledger import Ledger
from framelab.cli import app as cli_app
from framelab.clients import meta as meta_client
from framelab.clients import sam as sam_client
from framelab.pipeline import charts, report
from framelab.pipeline.outputs import render_outputs
from framelab.schemas import (
    IngestResult,
    ProbeInfo,
    ReportDraft,
    RunState,
    Stage,
    Verdict,
)
from framelab.schemas import (
    utcnow as now,
)
from framelab.store import write_json_atomic
from tests.fakes import MEDIA, freefall_fake

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")

M_PER_PX = 0.01
G = 9.8
FPS = 60.0
N = 120
C2 = G / (2 * M_PER_PX)  # y_px = C2·t²  →  y_m = G·t²/2


def make_run(tmp_path: Path, *, verdict: str = "PASS", rerecord: bool = False) -> Path:
    """A finished run folder with the files the renderers read, built through the Ledger."""
    run = tmp_path / "r_test"
    (run / "analysis").mkdir(parents=True)
    ledger = Ledger(run / "analysis" / "ledger.jsonl")
    cal = ledger.add_evidence("cal", "calibrate_scale",
        {"method": "ruler", "m_per_px": M_PER_PX, "sigma": 1e-4, "value": M_PER_PX * 1000,
         "unit": "mm/px"}, "ruler scale")  # fmt: skip
    trk = ledger.add_evidence("trk", "measure_track", {"track_id": "orange_ball_0"}, "track")
    window = {"t_start": 0.0, "t_end": 1.0, "method": "auto"}
    win = ledger.add_evidence("win", "set_fit_window", window, "window", [trk.id])
    fit = ledger.add_evidence("fit", "fit_model",
        {"model": "quadratic", "params": {"c0": 0.0, "c1": 0.0, "c2": C2}, "t_offset": 0.0,
         "r2": 1.0, "rms_px": 0.0, "durbin_watson": 2.0, "n": N},
        "quadratic fit", [trk.id, win.id])  # fmt: skip
    ci = ledger.add_evidence("ci", "bootstrap_ci",
        {"quantity": "g", "value": G, "lo": 9.7, "hi": 9.9, "unit": "m/s^2"},
        "g CI", [fit.id, cal.id])  # fmt: skip
    der = ledger.add_evidence("der", "derive",
        {"quantity": "g", "value": G, "unit": "m/s^2", "ci95": [9.7, 9.9]},
        "g", [ci.id, cal.id])  # fmt: skip
    cmp = ledger.add_evidence("cmp", "compare_reference",
        {"value": G, "lo": 9.7, "hi": 9.9, "rel_diff_pct": 0.0, "unit": "m/s^2", "quantity": "g"},
        "compared", [der.id])  # fmt: skip
    ledger.add_evidence("gate", "run_gates",
        {"gate": "G1", "name": "Track integrity", "status": "pass", "hard": True,
         "remedy": "none", "message_en": "Passed.", "message_id": "Lulus."},
        "G1 pass")  # fmt: skip
    if rerecord:
        ledger.add_evidence("gate", "run_gates",
            {"gate": "G2", "name": "Scale consistency", "status": "fail", "hard": True,
             "remedy": "rerecord", "diagnosis": "ruler_offplane",
             "message_en": "Ruler scale and ball scale disagree by 99.0%.",
             "message_id": "Skala penggaris dan skala bola berbeda 99.0%."},
            "G2 fail")  # fmt: skip
    ledger.add_claim("derived", der.id, "g = {value}", "g = {value}", quantity="g")
    ledger.add_claim("comparison", cmp.id, "Compared with the reference: {rel_diff}.",
                     "Dibandingkan referensi: {rel_diff}.", quantity="g")  # fmt: skip

    rows = []
    for frame in range(N):
        t = frame / FPS
        y_px = C2 * t * t
        bad = frame in (60, 61)
        rows.append([frame, f"{t:.6f}", 100.0, 500.0 if bad else y_px, 1500.0,
                     0 if bad else 1, "outlier" if bad else "ok"])  # fmt: skip
    (run / "tracks").mkdir()
    with (run / "tracks" / "orange_ball_0.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["frame", "t_s", "x_px", "y_px", "area_px", "valid", "flag"])
        writer.writerows(rows)

    probe = ProbeInfo(width=640, height=360, duration_s=N / FPS, avg_fps=FPS, r_fps=FPS,
                      nb_frames=N, is_vfr=False, has_audio=False, codec="h264")  # fmt: skip
    ingest = IngestResult(probe=probe, norm_width=640, norm_height=360, fps=FPS, fps_num=60,
                          fps_den=1, file_fps=FPS, effective_fps=FPS, n_frames=N,
                          video_path="ingest/video.mp4")  # fmt: skip
    write_json_atomic(run / "ingest" / "ingest.json", ingest)
    state = RunState(run_id="r_test", created_at=now(), updated_at=now(), stage=Stage.DONE,
                     original_filename="drop.mp4", lang="en", verdict=verdict)  # fmt: skip
    write_json_atomic(run / "state.json", state)
    draft = ReportDraft(
        lang="en", title="Free-fall measurement", aim="Measure g.", setup="A ball fell.",
        method="A quadratic fit.",
        results=[{"heading": "Result", "body": "The value is {{c01}}. Compared: {{c02}}."}],
        error_analysis="Small spread.", conclusion="See {{c01}}.", claims_used=["c01", "c02"],
    )  # fmt: skip
    write_json_atomic(run / "draft_report.json", draft)
    tips_extra = ["Ruler scale and ball scale disagree by 99.0%."] if rerecord else []
    write_json_atomic(run / "analysis" / "verdict.json", Verdict(
        status=verdict, auditor_model="muse-spark-1.3",
        rerecord_tips_en=["Film in bright light."] + tips_extra,
        rerecord_tips_id=["Rekam dengan cahaya terang."]))  # fmt: skip
    return run


def load(run: Path) -> tuple[RunState, Ledger]:
    return RunState.model_validate_json((run / "state.json").read_text(encoding="utf-8")), Ledger(
        run / "analysis" / "ledger.jsonl"
    )


def test_trajectory_data_follows_the_fit_and_uses_metres(tmp_path) -> None:
    run = make_run(tmp_path)
    _, ledger = load(run)
    data = charts.build_trajectory(run, ledger, "en")

    assert data["title"] == "g = 9.80 ± 0.05 m/s²"  # the headline claim, formatted by fmt
    assert len(data["valid"]) == N - 2 and len(data["invalid"]) == 2  # outliers are hollow
    assert data["fit"][0]["y"] == pytest.approx(0.0, abs=1e-9)
    t_mid = data["fit"][100]["t"]
    assert data["fit"][100]["y"] == pytest.approx(G * t_mid * t_mid / 2, rel=1e-6)
    # t is stored to 1 µs in the track file, which moves a point by ~0.01 mm at this scale
    assert data["residuals"] and max(abs(p["mm"]) for p in data["residuals"]) < 0.02


def test_trajectory_files_are_plotly_json_and_a_1600_by_900_png(tmp_path) -> None:
    run = make_run(tmp_path)
    _, ledger = load(run)
    paths = charts.write_trajectory(run, ledger, "en")

    plotly = json.loads(paths["chart_trajectory_json"].read_text(encoding="utf-8"))
    assert {"data", "layout"} <= set(plotly) and len(plotly["data"]) >= 3
    assert Image.open(paths["chart_trajectory_png"]).size == (1600, 900)


def test_series_chart_is_a_1600_by_900_png(tmp_path) -> None:
    series = tmp_path / "series" / "s1"
    (series / "analysis").mkdir(parents=True)
    result = {
        "points": [{"L": L, "T0": 2 * np.pi * np.sqrt(L / G), "sigma_T": 0.002, "sigma_L": 0.001}
                   for L in (0.3, 0.5, 0.8)],
        "linear": {"k": 4 * np.pi**2 / G, "b": 0.0, "sigma_k": 0.01, "sigma_b": 0.001},
    }  # fmt: skip
    (series / "analysis" / "series.json").write_text(json.dumps(result), encoding="utf-8")
    paths = charts.series_chart(series)
    assert Image.open(paths["chart_series_png"]).size == (1600, 900)


def test_data_csv_is_in_si_relative_to_the_first_valid_point(tmp_path) -> None:
    run = make_run(tmp_path)
    path = report.export_csv(run)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "# run_id: r_test"
    assert lines[1].startswith("# scale: 10.0000 mm/px") and lines[2] == "# fps: 60"
    rows = list(csv.DictReader(line for line in lines if not line.startswith("#")))
    assert len(rows) == N
    assert float(rows[0]["x_m"]) == 0.0 and float(rows[0]["y_m"]) == 0.0
    t = float(rows[30]["t_s"])
    assert float(rows[30]["y_m"]) == pytest.approx(G * t * t / 2, rel=1e-5)
    assert float(rows[0]["area_m2"]) == pytest.approx(1500.0 * M_PER_PX**2)


def test_report_html_is_self_contained_and_states_claims(tmp_path) -> None:
    run = make_run(tmp_path)
    state, ledger = load(run)
    draft = ReportDraft.model_validate_json((run / "draft_report.json").read_text(encoding="utf-8"))
    html = report.render_html(run, draft, state, ledger)

    assert "http://" not in html and "https://" not in html  # no network resource
    assert "{{" not in html and "[?]" not in html  # every placeholder rendered
    assert "g = 9.80 ± 0.05 m/s²" in html
    assert "g = 9.80 ± 0.05 m/s²" in html.split("<h2>Results</h2>")[1]


def test_result_text_is_the_claim_phrases_only(tmp_path) -> None:
    run = make_run(tmp_path)
    state, ledger = load(run)
    draft = ReportDraft.model_validate_json((run / "draft_report.json").read_text(encoding="utf-8"))
    body = report.render_text(draft.results[0].body, ledger, "en")
    assert body == ("The value is g = 9.80 ± 0.05 m/s². Compared: a difference of 0.00% "
                    "from the reference value.")  # fmt: skip


def test_report_pdf_is_an_a4_document(tmp_path) -> None:
    run = make_run(tmp_path)
    state, ledger = load(run)
    draft = ReportDraft.model_validate_json((run / "draft_report.json").read_text(encoding="utf-8"))
    pdf = report.html_to_pdf(report.render_html(run, draft, state, ledger), tmp_path / "r.pdf")
    data = pdf.read_bytes()
    assert data.startswith(b"%PDF") and len(data) > 2000
    box = re.search(rb"/MediaBox\s*\[\s*0 0 ([\d.]+) ([\d.]+)\s*\]", data)
    assert box is not None
    width, height = float(box.group(1)), float(box.group(2))
    assert abs(width - 595.28) < 2 and abs(height - 841.89) < 2  # A4 in points


def test_recording_feedback_lists_the_failed_gate_in_both_languages(tmp_path) -> None:
    run = make_run(tmp_path, verdict="NEEDS_RERECORD", rerecord=True)
    state, ledger = load(run)
    html = report.render_feedback(run, state, ledger)

    assert "Ruler scale and ball scale disagree by 99.0%." in html  # English
    assert "Skala penggaris dan skala bola berbeda 99.0%." in html  # Indonesian
    assert html.count("Ruler scale and ball scale disagree by 99.0%.") == 1  # not repeated as a tip
    assert "Film in bright light." in html and "Rekam dengan cahaya terang." in html
    assert "http://" not in html and "https://" not in html


def test_render_outputs_survives_a_failing_output(tmp_path) -> None:
    run = make_run(tmp_path)
    (run / "tracks" / "orange_ball_0.csv").unlink()  # the charts and the CSV cannot be read
    result = render_outputs(run, with_overlay=False)
    assert any("charts not rendered" in w for w in result.warnings)
    assert any("data not rendered" in w for w in result.warnings)
    assert "report_en" in result.paths  # the report does not need the track file


def test_replay_makes_no_model_calls(tmp_path, monkeypatch) -> None:
    run = make_run(tmp_path, verdict="NEEDS_RERECORD", rerecord=True)

    def blocked(*args, **kwargs):
        raise AssertionError("replay must not call a model or the Files API")

    monkeypatch.setattr(meta_client, "chat", blocked)
    monkeypatch.setattr(meta_client, "responses", blocked)
    monkeypatch.setattr(sam_client, "upload_video", blocked)
    result = CliRunner().invoke(cli_app, ["replay", str(run), "--no-overlay"])

    assert result.exit_code == 0, result.output
    for name in ("chart_trajectory.json", "chart_trajectory.png", "data.csv", "report_en.pdf",
                 "recording_feedback.pdf"):  # fmt: skip
        assert (run / "outputs" / name).exists(), name
    # the draft is in one language only, so there is no second report (04 §9)
    assert not (run / "outputs" / "report_id.pdf").exists()


@needs_ffmpeg
def test_overlay_on_the_real_pipeline_frames(tmp_path, monkeypatch) -> None:
    """A deterministic run on the recorded SAM fixtures, then replay with the overlay."""
    freefall_fake().install(monkeypatch)
    runs = tmp_path / "runs"
    result = CliRunner().invoke(cli_app, [
        "run", str(MEDIA / "freefall_60fps_5s.mp4"), "--no-planner", "--concept", "orange ball",
        "--ref-concept", "ruler", "--ref-length-m", "1.0", "--runs-dir", str(runs),
    ])  # fmt: skip
    assert result.exit_code == 0, result.output
    run = next(runs.glob("r_*"))

    replayed = CliRunner().invoke(cli_app, ["replay", str(run)])
    assert replayed.exit_code == 0, replayed.output
    overlay = run / "outputs" / "overlay.mp4"
    capture = cv2.VideoCapture(str(overlay))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    source = cv2.VideoCapture(str(run / "ingest" / "video.mp4"))
    video_frames = int(source.get(cv2.CAP_PROP_FRAME_COUNT))
    source.release()
    card = round(1.5 * 60)  # the end card holds the last frame for 1.5 s
    assert abs(frames - (video_frames + card)) <= 2
    assert overlay.stat().st_size > 10_000
