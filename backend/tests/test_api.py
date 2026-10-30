"""M9 HTTP API with the pipeline mocked (07 §1–§2, §5).

The API is created with a fake runner: it writes events and a final state the way the pipeline does,
without any model. Uploads use the recorded free-fall clip, so ffprobe really reads the file.
"""

from __future__ import annotations

import asyncio
import json
import time

import pytest
from fastapi.testclient import TestClient

from framelab.api import sse
from framelab.api.main import create_app
from framelab.config import Settings
from framelab.events import EventBus
from framelab.schemas import Stage
from framelab.store import RunStore
from tests.fakes import MEDIA

CLIP = MEDIA / "freefall_60fps_5s.mp4"


def fake_runner(*, delay: float = 0.0, fail: bool = False, track: dict | None = None):
    """A stand-in for execute_run: the events a real run emits, in order, with no model."""

    async def runner(run_id: str, settings: Settings) -> None:
        store = RunStore(settings.runs_dir)
        bus = EventBus(settings.runs_dir)
        state = store.load_state(run_id)
        if track is not None:
            track["running"] = track.get("running", 0) + 1
            track["max"] = max(track.get("max", 0), track["running"])
        store.set_stage(state, Stage.INGESTING)
        bus.emit(run_id, "stage_started", "ingesting")
        await asyncio.sleep(delay)
        bus.emit(run_id, "progress", "ingesting", message="working")
        if fail:
            store.set_stage(state, Stage.FAILED)
            bus.emit(run_id, "error", "ingesting", message="boom", exception_type="RuntimeError",
                     retryable=False)  # fmt: skip
        else:
            bus.emit(run_id, "stage_completed", "ingesting", duration_s=0.0)
            state.verdict = "PASS"
            store.set_stage(state, Stage.DONE)
            bus.emit(run_id, "done", "ingesting", verdict="PASS", headline=None, partial=False,
                     cost_usd=0.0)  # fmt: skip
        if track is not None:
            track["running"] -= 1

    return runner


def client_for(settings: Settings, runner=None) -> TestClient:
    return TestClient(create_app(settings, runner=runner or fake_runner()))


def upload(client: TestClient, data: bytes = CLIP.read_bytes(), **form) -> object:
    return client.post(
        "/api/runs",
        files={"video": ("drop.mp4", data, "video/mp4")},
        data={"lang": "en", **form},
    )


def wait_until(predicate, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("timed out waiting for the run")


def wait_done(client: TestClient, run_id: str) -> dict:
    wait_until(lambda: client.get(f"/api/runs/{run_id}").json()["stage"] in ("done", "failed"))
    return client.get(f"/api/runs/{run_id}").json()


def read_events(response) -> list[tuple[str, str]]:
    """(event, id) pairs from an SSE body, pings skipped."""
    pairs, event, eid = [], None, None
    for line in response.iter_lines():
        if line.startswith("event:"):
            event = line.split(":", 1)[1].strip()
        elif line.startswith("id:"):
            eid = line.split(":", 1)[1].strip()
        elif line == "" and event is not None:
            pairs.append((event, eid))
            event = eid = None
    return pairs


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(runs_dir=tmp_path / "runs")


# --- uploads -------------------------------------------------------------------------------


def test_upload_starts_a_run_and_it_completes(settings) -> None:
    with client_for(settings) as client:
        response = upload(client)
        assert response.status_code == 201
        run_id = response.json()["run_id"]
        body = wait_done(client, run_id)
    assert body["stage"] == "done" and body["verdict"] == "PASS"
    assert body["lang"] == "en"
    assert (settings.runs_dir / run_id / "input" / "original.mp4").stat().st_size > 1000


@pytest.mark.parametrize(
    ("name", "mime", "data", "status", "code"),
    [
        ("drop.txt", "video/mp4", CLIP.read_bytes(), 415, "unsupported_media_type"),
        ("drop.mp4", "image/png", CLIP.read_bytes(), 415, "unsupported_media_type"),
        ("drop.mp4", "video/mp4", b"", 422, "empty_upload"),
        ("drop.mp4", "video/mp4", b"not a video at all", 422, "invalid_video"),
    ],
)  # fmt: skip
def test_bad_uploads_are_refused_with_a_code(settings, name, mime, data, status, code) -> None:
    with client_for(settings) as client:
        response = client.post("/api/runs", files={"video": (name, data, mime)},
                               data={"lang": "en"})  # fmt: skip
    assert response.status_code == status
    assert response.json()["error"]["code"] == code
    assert not any(settings.runs_dir.glob("r_*"))  # nothing is left behind


def test_form_fields_are_checked(settings) -> None:
    with client_for(settings) as client:
        assert upload(client, lang="fr").status_code == 422
        assert upload(client, note="x" * 301).status_code == 422
        assert upload(client, capture_fps=0).status_code == 422
        response = upload(client, series_id="s_missing")
    assert response.status_code == 404 and response.json()["error"]["code"] == "not_found"


def test_upload_over_the_size_limit_is_refused(tmp_path) -> None:
    small = Settings(runs_dir=tmp_path / "runs", max_upload_mb=1)
    with client_for(small) as client:
        response = upload(client, data=b"\0" * (1 << 20) + b"\0" * 4096)
    assert response.status_code == 413 and response.json()["error"]["code"] == "too_large"


def test_demo_mode_refuses_uploads_unless_allowed(tmp_path) -> None:
    demo = Settings(runs_dir=tmp_path / "runs", demo=True)
    with client_for(demo) as client:
        response = upload(client)
    assert response.status_code == 403 and response.json()["error"]["code"] == "demo_mode"

    allowed = Settings(runs_dir=tmp_path / "runs2", demo=True, demo_allow_upload=True)
    with client_for(allowed) as client:
        assert upload(client).status_code == 201


# --- reading runs --------------------------------------------------------------------------


def test_runs_are_listed_newest_first(settings) -> None:
    with client_for(settings) as client:
        first = wait_done(client, upload(client).json()["run_id"])["run_id"]
        second = wait_done(client, upload(client).json()["run_id"])["run_id"]
        listed = client.get("/api/runs?limit=5").json()
    assert [r["run_id"] for r in listed] == [second, first]


def test_unknown_run_is_a_404_with_an_error_body(settings) -> None:
    with client_for(settings) as client:
        response = client.get("/api/runs/r_20000101_000000_000000")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


def test_run_detail_has_spec_gates_and_claims(settings) -> None:
    with client_for(settings) as client:
        run_id = wait_done(client, upload(client).json()["run_id"])["run_id"]
        body = client.get(f"/api/runs/{run_id}").json()
    assert body["spec"] is None and body["gates"] == [] and body["claims"] == []  # none yet
    assert body["verdict"] == "PASS"


# --- events (SSE) -------------------------------------------------------------------------


def test_events_stream_replays_the_run_and_closes_on_done(settings) -> None:
    with client_for(settings) as client:
        run_id = wait_done(client, upload(client).json()["run_id"])["run_id"]
        with client.stream("GET", f"/api/runs/{run_id}/events") as response:
            assert response.headers["content-type"].startswith("text/event-stream")
            pairs = read_events(response)
    names = [name for name, _ in pairs]
    assert names[0] == "stage_started" and names[-1] == "done"
    ids = [int(i) for _, i in pairs]
    assert ids == sorted(ids) and len(set(ids)) == len(ids)


def test_last_event_id_resumes_after_that_seq(settings) -> None:
    with client_for(settings) as client:
        run_id = wait_done(client, upload(client).json()["run_id"])["run_id"]
        with client.stream(
            "GET", f"/api/runs/{run_id}/events", headers={"Last-Event-ID": "2"}
        ) as response:
            pairs = read_events(response)  # fmt: skip
    assert pairs and int(pairs[0][1]) == 3


def test_an_error_event_closes_the_stream(settings) -> None:
    with client_for(settings, fake_runner(fail=True)) as client:
        run_id = wait_done(client, upload(client).json()["run_id"])["run_id"]
        with client.stream("GET", f"/api/runs/{run_id}/events") as response:
            pairs = read_events(response)
    assert pairs[-1][0] == "error"


def test_heartbeat_is_sent_while_the_run_is_quiet(settings, monkeypatch) -> None:
    monkeypatch.setattr(sse, "HEARTBEAT_S", 0.05)
    with client_for(settings, fake_runner(delay=0.4)) as client:
        run_id = upload(client).json()["run_id"]
        with client.stream("GET", f"/api/runs/{run_id}/events") as response:
            lines = [line for line in response.iter_lines()]
    assert ": ping" in lines
    assert any(line.startswith("event: done") for line in lines)


def test_live_events_reach_a_subscriber_started_before_the_run(settings) -> None:
    """The stream was opened while the run was still going: its events come live."""
    with client_for(settings, fake_runner(delay=0.3)) as client:
        run_id = upload(client).json()["run_id"]
        with client.stream("GET", f"/api/runs/{run_id}/events") as response:
            pairs = read_events(response)
    assert [name for name, _ in pairs][-1] == "done"


# --- concurrency ----------------------------------------------------------------------------


def test_runs_are_limited_by_the_concurrency_setting(tmp_path) -> None:
    limited = Settings(runs_dir=tmp_path / "runs", concurrency=1)
    track: dict = {}
    with client_for(limited, fake_runner(delay=0.15, track=track)) as client:
        ids = [upload(client).json()["run_id"] for _ in range(3)]
        for run_id in ids:
            wait_done(client, run_id)
    assert track["max"] == 1


# --- artifacts, ledger, track, rerun -------------------------------------------------------


def test_artifacts_are_limited_to_the_allow_list(settings) -> None:
    with client_for(settings) as client:
        run_id = wait_done(client, upload(client).json()["run_id"])["run_id"]
        outputs = settings.runs_dir / run_id / "outputs"
        (outputs / "data.csv").write_text("frame,t_s\n0,0.0\n", encoding="utf-8")
        ok = client.get(f"/api/runs/{run_id}/artifacts/data.csv")
        not_listed = client.get(f"/api/runs/{run_id}/artifacts/state.json")
        escape = client.get(f"/api/runs/{run_id}/artifacts/..%2Fstate.json")
        missing = client.get(f"/api/runs/{run_id}/artifacts/overlay.mp4")
    assert ok.status_code == 200 and ok.text.startswith("frame,t_s")
    assert not_listed.status_code == 404
    assert escape.status_code == 404
    assert missing.status_code == 404 and missing.json()["error"]["code"] == "not_found"


def test_rerun_creates_a_new_run_with_the_capture_fps(settings) -> None:
    with client_for(settings) as client:
        original = wait_done(client, upload(client).json()["run_id"])["run_id"]
        response = client.post(f"/api/runs/{original}/rerun", json={"capture_fps": 240})
        assert response.status_code == 201
        new_id = response.json()["run_id"]
        body = wait_done(client, new_id)
    assert new_id != original and body["capture_fps"] == 240
    assert body["stage"] == "done"


def test_ledger_and_track_views_read_the_saved_run(settings) -> None:
    from tests.test_outputs import make_run

    settings.runs_dir.mkdir(parents=True, exist_ok=True)
    run_dir = make_run(settings.runs_dir.parent)  # a finished run folder: ledger, track, ingest
    run_dir.rename(settings.runs_dir / "r_20261029_120000_abcdef")
    run_id = "r_20261029_120000_abcdef"
    with client_for(settings) as client:
        ledger = client.get(f"/api/runs/{run_id}/ledger").json()
        track = client.get(f"/api/runs/{run_id}/track").json()
    assert [c["id"] for c in ledger["claims"]] == ["c01", "c02"]
    assert ledger["evidence"] and ledger["evidence"][0]["id"] == "cal_01"
    assert track["fps"] == 60.0 and len(track["points"]) == 120
    assert track["fit_curve"] and track["fit_curve"][0]["t"] == pytest.approx(0.0)
    assert json.dumps(track)  # JSON-serialisable end to end


# --- series -----------------------------------------------------------------------------------


def test_series_endpoints_validate_their_input(settings) -> None:
    with client_for(settings) as client:
        assert client.post("/api/series", json={"experiment": "free_fall"}).status_code == 422
        created = client.post("/api/series", json={"experiment": "pendulum"})
        assert created.status_code == 201
        series_id = created.json()["series_id"]
        bad_run = wait_done(client, upload(client).json()["run_id"])["run_id"]
        refused = client.post(f"/api/series/{series_id}/runs", json={"run_id": bad_run})
        fetched = client.get(f"/api/series/{series_id}")
    assert refused.status_code == 422 and refused.json()["error"]["code"] == "series_error"
    assert fetched.status_code == 200 and fetched.json()["results"] is None


def test_health_still_answers(settings) -> None:
    with client_for(settings) as client:
        body = client.get("/api/health").json()
    assert body["ok"] is True and body["models"]["sam"] == "sam-3.1"


def test_run_detail_has_claim_sentences_transcript_and_series_list(tmp_path) -> None:
    """M10 additions: rendered claim text, transcript, verdict detail, next experiment, series."""
    from fastapi.testclient import TestClient

    from framelab.agent.ledger import Ledger
    from framelab.api.main import create_app
    from framelab.config import Settings
    from framelab.store import RunStore, write_json_atomic

    settings = Settings(runs_dir=tmp_path / "runs")
    store = RunStore(settings.runs_dir)
    state = store.create_run("x.mp4")
    run_dir = store.run_dir(state.run_id)
    ledger = Ledger(run_dir / "analysis" / "ledger.jsonl")
    ev = ledger.add_evidence("der", "derive", {"quantity": "g", "value": 9.8123, "unit": "m/s^2",
                                               "ci95": [9.75, 9.87]}, "g")  # fmt: skip
    ledger.add_claim("derived", ev.id, "g = {value}", "g = {value}", quantity="g")
    write_json_atomic(run_dir / "transcript.json", {"text": "satu meter", "turns": []})
    write_json_atomic(run_dir / "analysis" / "verdict.json", {"status": "PASS", "notes": "ok"})
    write_json_atomic(run_dir / "analysis" / "designer.json", {
        "title": "Next", "prediction_en": "From {{p1}}.", "prediction_id": "Dari {{p1}}.",
        "predicted_values": [{"id": "p1", "predicted": 0.3194, "unit_out": "s"}]})  # fmt: skip
    store.create_series([])
    client = TestClient(create_app(settings, runner=lambda *a: None))
    body = client.get(f"/api/runs/{state.run_id}").json()
    assert body["claims"][0]["text_en"] == "g = 9.81 ± 0.03 m/s²"
    assert body["claims"][0]["text_id"] == "g = 9,81 ± 0,03 m/s²"
    assert body["transcript"]["text"] == "satu meter"
    assert body["verdict_detail"]["notes"] == "ok"
    assert body["next_experiment"]["prediction_en_rendered"] == "From 0.319 s."
    assert len(client.get("/api/series").json()) == 1
    assert client.get(f"/api/runs/{state.run_id}/artifacts/video.mp4").status_code == 404
