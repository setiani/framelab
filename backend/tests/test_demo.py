"""M11: demo mode (cached samples, replays) and credit protection (07 §3, 10 §2)."""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from framelab.api import demo
from framelab.api.limits import RateLimiter, client_address, spent_today, upload_status
from framelab.api.main import create_app
from framelab.config import Settings
from framelab.events import EventBus
from framelab.schemas import RunState, Stage, utcnow
from framelab.store import RunStore, write_json_atomic
from tests.test_api import CLIP, fake_runner, upload, wait_until

ROOT = Path(__file__).resolve().parents[2]
RUN_ID = "r_20261101_093012_ab12cd"
T0 = "2026-11-01T09:30:12"

EVENTS = [  # (seconds after start, type, stage, payload)
    (0.00, "stage_started", "ingesting", {}),
    (0.02, "artifact", "ingesting", {"name": "video", "path": "ingest/video.mp4"}),
    (0.03, "stage_completed", "ingesting", {"duration_s": 0.03}),
    (0.04, "stage_started", "planning", {}),
    (0.05, "model_result", "planning", {"model": "muse-spark-1.3", "cost_usd": 0.01}),
    (0.06, "artifact", "planning", {"name": "spec", "path": "spec.json"}),
    (0.07, "stage_completed", "planning", {"duration_s": 0.03}),
    (0.08, "stage_started", "analyzing", {}),
    (0.10, "claim_added", "analyzing", {"claim_id": "c01"}),
    (0.12, "stage_completed", "analyzing", {"duration_s": 0.04}),
    (0.13, "stage_started", "auditing", {}),
    (0.14, "verdict", "auditing", {"status": "PASS", "final": True, "revise_rounds": 0}),
    (0.15, "stage_completed", "auditing", {"duration_s": 0.02}),
    (0.16, "stage_started", "rendering", {}),
    (0.18, "artifact", "rendering", {"name": "overlay", "path": "outputs/overlay.mp4"}),
    (0.19, "stage_completed", "rendering", {"duration_s": 0.03}),
    (0.20, "done", "done", {"verdict": "PASS", "headline": {"value": 9.8}, "partial": False,
                            "cost_usd": 0.01}),
]  # fmt: skip


def _ts(seconds: float) -> str:
    return f"{T0}.{int(seconds * 1000):03d}000+00:00" if seconds < 1 else f"{T0}+00:00"


def make_finished_run(root: Path, run_id: str = RUN_ID, *, stage: str = "done") -> Path:
    """A finished run folder as the pipeline leaves it, small and without any model."""
    run = root / run_id
    files = {
        "ingest/ingest.json": json.dumps({"file_fps": 60.0}),
        "ingest/video.mp4": "video",
        "transcript.json": json.dumps({"turns": []}),
        "spec.json": json.dumps({"experiment": "free_fall"}),
        "sam/summary.json": "{}",
        "sam/target_ball.sam.json": "raw masks",
        "debug/centroids.png": "png",
        "input/original.mp4": "original",
        "ingest/analysis.mp4": "analysis copy",
        "analysis/verdict.json": json.dumps({"status": "PASS"}),
        "outputs/overlay.mp4": "overlay",
        "outputs/report_en.pdf": "%PDF",
    }
    for rel, text in files.items():
        (run / rel).parent.mkdir(parents=True, exist_ok=True)
        (run / rel).write_text(text, encoding="utf-8")
    ledger = [
        {"kind": "evidence", "id": "trk_01", "created_at": f"{T0}.050000Z", "tool": "x",
         "inputs": [], "data": {}, "summary": "track"},
        {"kind": "evidence", "id": "fit_01", "created_at": f"{T0}.110000Z", "tool": "x",
         "inputs": ["trk_01"], "data": {}, "summary": "fit"},
    ]  # fmt: skip
    (run / "analysis" / "ledger.jsonl").write_text(
        "".join(json.dumps(e) + "\n" for e in ledger), encoding="utf-8"
    )
    with (run / "events.jsonl").open("w", encoding="utf-8") as fh:
        for seq, (sec, kind, stg, payload) in enumerate(EVENTS, start=1):
            event = {"ts": _ts(sec), "run_id": run_id, "seq": seq, "type": kind, "stage": stg}
            fh.write(json.dumps(event | payload) + "\n")
    now = utcnow()
    state = RunState(
        run_id=run_id, created_at=now - timedelta(days=1), updated_at=now,
        stage=Stage(stage), original_filename="drop_ball.mp4", lang="id",
        experiment="free_fall", verdict="PASS", headline={"value": 9.8}, cost_usd=0.01,
        artifacts={"video": "ingest/video.mp4", "spec": "spec.json",
                   "overlay": "outputs/overlay.mp4"},
    )  # fmt: skip
    write_json_atomic(run / "state.json", state)
    return run


def make_demo_dir(root: Path) -> Path:
    """What build_demo_cache.py writes: one cached sample plus the manifest."""
    run = make_finished_run(root)
    state = RunState.model_validate_json((run / "state.json").read_text(encoding="utf-8"))
    state.demo, state.sample = True, "free_fall"
    write_json_atomic(run / "state.json", state)
    demo.write_manifest(root, [{"key": "free_fall", "run_id": RUN_ID,
                                "title": {"en": "Free fall", "id": "Jatuh bebas"},
                                "blurb": {"en": "A ball.", "id": "Bola."}}])  # fmt: skip
    return root


@pytest.fixture
def demo_settings(tmp_path: Path) -> Settings:
    return Settings(runs_dir=tmp_path / "runs", demo_dir=make_demo_dir(tmp_path / "demo_runs"))


@pytest.fixture
def no_key(monkeypatch):
    monkeypatch.setattr(Settings, "model_api_key", property(lambda self: None))


# --- samples -------------------------------------------------------------------------------


def test_samples_are_listed_and_readable_but_not_in_recent_runs(demo_settings) -> None:
    with TestClient(create_app(demo_settings, runner=fake_runner())) as client:
        samples = client.get("/api/samples").json()
        run = client.get(f"/api/runs/{RUN_ID}").json()
        overlay = client.get(f"/api/runs/{RUN_ID}/artifacts/overlay.mp4")
        listed = client.get("/api/runs").json()
        health = client.get("/api/health").json()
    assert [s["key"] for s in samples] == ["free_fall"]
    assert samples[0]["title"]["id"] == "Jatuh bebas" and samples[0]["verdict"] == "PASS"
    assert run["demo"] is True and run["sample"] == "free_fall" and run["stage"] == "done"
    assert overlay.status_code == 200 and overlay.content == b"overlay"
    assert listed == [] and health["samples"] == 1


def test_samples_cannot_be_rerun_or_join_a_series(demo_settings) -> None:
    with TestClient(create_app(demo_settings, runner=fake_runner())) as client:
        rerun = client.post(f"/api/runs/{RUN_ID}/rerun", json={})
        series = client.post("/api/series", json={"experiment": "pendulum", "run_ids": [RUN_ID]})
    assert rerun.status_code == 409 and rerun.json()["error"]["code"] == "sample_run"
    assert series.status_code == 409


def test_unknown_sample_is_a_404(demo_settings) -> None:
    with TestClient(create_app(demo_settings, runner=fake_runner())) as client:
        response = client.post("/api/samples/nope/replay")
    assert response.status_code == 404


def test_replay_reemits_the_events_and_ends_like_the_original(demo_settings, no_key) -> None:
    """Works with no API key at all (the M11 acceptance)."""
    with TestClient(create_app(demo_settings)) as client:
        assert client.get("/api/health").json()["uploads"]["reason"] == "no_api_key"
        body = client.post("/api/samples/free_fall/replay").json()
        rid = body["run_id"]
        wait_until(lambda: client.get(f"/api/runs/{rid}").json()["stage"] == "done")
        run = client.get(f"/api/runs/{rid}").json()
        overlay = client.get(f"/api/runs/{rid}/artifacts/overlay.mp4")
        ledger = client.get(f"/api/runs/{rid}/ledger").json()
        listed = client.get("/api/runs").json()
    assert body["replay_of"] == RUN_ID and rid != RUN_ID
    assert run["replay_of"] == RUN_ID and run["demo"] is True and run["verdict"] == "PASS"
    assert run["cost_usd"] == pytest.approx(0.01) and run["headline"] == {"value": 9.8}
    assert overlay.content == b"overlay"
    assert [e["id"] for e in ledger["evidence"]] == ["trk_01", "fit_01"]
    original = [(e[1], e[2]) for e in EVENTS]
    replayed = [(e["type"], e["stage"]) for e in EventBus(demo_settings.runs_dir).iter_events(rid)]
    assert replayed == original
    assert listed == []  # replays are not recent runs


def test_replay_copies_files_when_their_event_is_emitted(tmp_path) -> None:
    store = RunStore(tmp_path / "runs", sample_root=make_demo_dir(tmp_path / "demo_runs"))
    bus = EventBus(store.root)
    state = demo.create_replay(store, demo.find_sample(store, "free_fall"))
    dst = store.run_dir(state.run_id)
    seen: dict[str, set[str]] = {}
    original_emit = bus.emit

    def emit(run_id, type, stage, **payload):
        files = {p.relative_to(dst).as_posix() for p in dst.rglob("*") if p.is_file()}
        seen[f"{type}:{stage}"] = files
        return original_emit(run_id, type, stage, **payload)

    bus.emit = emit  # type: ignore[method-assign]
    asyncio.run(demo.run_replay(store, bus, state.run_id, max_total_s=0.01))
    ledger = (dst / "analysis" / "ledger.jsonl").read_text(encoding="utf-8").splitlines()
    assert "spec.json" in seen["stage_completed:planning"]
    assert "outputs/overlay.mp4" not in seen["stage_completed:planning"]
    assert "analysis/ledger.jsonl" in seen["model_result:planning"]  # trk_01 at .05 s
    assert "outputs/overlay.mp4" in seen["artifact:rendering"]
    assert len(ledger) == 2
    assert not (dst / "events.jsonl").read_text().count(RUN_ID)  # re-emitted under the new id
    final = store.load_state(state.run_id)
    assert final.stage == Stage.DONE and final.artifacts["overlay"] == "outputs/overlay.mp4"


def test_replay_timing_is_compressed_and_capped() -> None:
    events = [{"ts": f"2026-11-01T09:30:{s:02d}+00:00"} for s in (0, 4, 8, 48)]
    assert demo.replay_delays(events) == pytest.approx([0.0, 1.0, 1.0, demo.MAX_GAP_S])
    capped = demo.replay_delays(events, max_total_s=2.0)
    assert sum(capped) == pytest.approx(2.0)


def test_old_replays_are_cleaned_up(tmp_path) -> None:
    store = RunStore(tmp_path / "runs", sample_root=make_demo_dir(tmp_path / "demo_runs"))
    sample = demo.find_sample(store, "free_fall")
    old, fresh = demo.create_replay(store, sample), demo.create_replay(store, sample)
    old.created_at = utcnow() - timedelta(hours=7)
    store.save_state(old)
    assert demo.cleanup_replays(store, 6.0, keep_running=set()) == 1
    assert not store.run_dir(old.run_id).exists() and store.run_dir(fresh.run_id).exists()
    assert (store.sample_root / RUN_ID).exists()  # the cached sample itself stays


# --- credit protection ---------------------------------------------------------------------


def test_uploads_are_rate_limited_per_address(tmp_path) -> None:
    settings = Settings(runs_dir=tmp_path / "runs", upload_limit_per_hour=2, forwarded_hops=1)
    with TestClient(create_app(settings, runner=fake_runner())) as client:
        a = {"X-Forwarded-For": "203.0.113.5"}
        codes = [upload_with(client, a).status_code for _ in range(3)]
        other = upload_with(client, {"X-Forwarded-For": "198.51.100.7"}).status_code
        limited = upload_with(client, a)
    assert codes == [201, 201, 429] and other == 201
    assert limited.json()["error"]["code"] == "rate_limited"
    assert int(limited.headers["retry-after"]) > 0


def upload_with(client: TestClient, headers: dict) -> object:
    return client.post("/api/runs", headers=headers, data={"lang": "en"},
                       files={"video": ("drop.mp4", CLIP.read_bytes(), "video/mp4")})  # fmt: skip


def test_daily_budget_pauses_live_uploads(tmp_path) -> None:
    settings = Settings(runs_dir=tmp_path / "runs", daily_budget_usd=1.0)
    store = RunStore(settings.runs_dir)
    for cost in (0.6, 0.5):
        state = store.create_run("x.mp4")
        state.cost_usd = cost
        store.save_state(state)
    assert spent_today(store) == pytest.approx(1.1)
    with TestClient(create_app(settings, runner=fake_runner())) as client:
        health = client.get("/api/health").json()
        response = upload(client)
    assert health["uploads"] == {"enabled": False, "reason": "daily_budget",
                                 "message": health["uploads"]["message"]}  # fmt: skip
    assert response.status_code == 503 and response.json()["error"]["code"] == "daily_budget"


def test_no_api_key_turns_live_uploads_off(tmp_path, no_key) -> None:
    status = upload_status(Settings(runs_dir=tmp_path), RunStore(tmp_path), needs_key=True)
    assert not status.enabled and status.reason == "no_api_key"
    assert upload_status(Settings(runs_dir=tmp_path), RunStore(tmp_path), needs_key=False).enabled


def test_rate_limiter_window_slides() -> None:
    now = [0.0]
    limiter = RateLimiter(clock=lambda: now[0])
    assert limiter.hit("u", "a", 2) is None and limiter.hit("u", "a", 2) is None
    assert limiter.hit("u", "a", 2) == pytest.approx(3600.0)
    now[0] = 3600.0
    assert limiter.hit("u", "a", 2) is None


def test_client_address_uses_the_proxy_entry_not_the_forgeable_one() -> None:
    class Req:
        headers = {"x-forwarded-for": "6.6.6.6, 203.0.113.5"}
        client = type("C", (), {"host": "10.0.0.1"})()

    assert client_address(Req(), 0) == "10.0.0.1"  # type: ignore[arg-type]
    assert client_address(Req(), 1) == "203.0.113.5"  # type: ignore[arg-type]


# --- build_demo_cache.py -------------------------------------------------------------------


def _cache_script():
    sys.path.insert(0, str(ROOT / "scripts"))
    import build_demo_cache  # noqa: PLC0415

    return build_demo_cache


def test_build_demo_cache_copies_a_trimmed_sample(tmp_path, no_key) -> None:
    script = _cache_script()
    runs = tmp_path / "runs"
    make_finished_run(runs)
    out = tmp_path / "demo_runs"
    assert script.main(["--runs-dir", str(runs), "--out", str(out),
                        "--sample", f"free_fall={RUN_ID}"]) == 0  # fmt: skip
    manifest = json.loads((out / "samples.json").read_text(encoding="utf-8"))
    cached = out / RUN_ID
    state = json.loads((cached / "state.json").read_text(encoding="utf-8"))
    assert manifest["samples"][0]["key"] == "free_fall"
    assert state["demo"] is True and state["sample"] == "free_fall"
    assert (cached / "outputs" / "overlay.mp4").is_file() and (cached / "events.jsonl").is_file()
    assert (cached / "sam" / "summary.json").is_file()
    for skipped in ("input", "debug", "sam/target_ball.sam.json", "ingest/analysis.mp4"):
        assert not (cached / skipped).exists(), skipped
    store = RunStore(tmp_path / "live", sample_root=out)
    assert [s.key for s in demo.load_samples(store)] == ["free_fall"]


def test_build_demo_cache_refuses_unfinished_runs_and_secrets(tmp_path, monkeypatch) -> None:
    script = _cache_script()
    runs = tmp_path / "runs"
    make_finished_run(runs, stage="analyzing")
    args = ["--runs-dir", str(runs), "--out", str(tmp_path / "out"),
            "--sample", f"free_fall={RUN_ID}"]  # fmt: skip
    assert script.main(args) == 1
    make_finished_run(runs)
    (runs / RUN_ID / "transcript.json").write_text('{"oops": "sk-secret-key-123"}')
    monkeypatch.setattr(Settings, "model_api_key", property(lambda self: "sk-secret-key-123"))
    assert script.main(args) == 1
    assert not (tmp_path / "out").exists()


def test_sample_series_is_listed_and_read_only(demo_settings) -> None:
    from framelab.schemas import SeriesState

    sid = "s_20261101_110000_aa11bb"
    series = SeriesState(series_id=sid, created_at=utcnow(), run_ids=[RUN_ID], status="done")
    write_json_atomic(demo_settings.demo_dir / "series" / sid / "series.json", series)
    with TestClient(create_app(demo_settings, runner=fake_runner())) as client:
        listed = client.get("/api/series").json()
        detail = client.get(f"/api/series/{sid}").json()
        analyze = client.post(f"/api/series/{sid}/analyze")
        add = client.post(f"/api/series/{sid}/runs", json={"run_id": RUN_ID})
    assert [(s["series_id"], s["sample"]) for s in listed] == [(sid, True)]
    assert detail["sample"] is True
    assert analyze.status_code == 409 and add.status_code == 409


def test_build_demo_cache_refuses_a_passing_run_as_the_trap(tmp_path, no_key) -> None:
    script = _cache_script()
    runs = tmp_path / "runs"
    make_finished_run(runs)  # verdict PASS
    args = ["--runs-dir", str(runs), "--out", str(tmp_path / "out"), "--sample", f"trap={RUN_ID}"]
    assert script.main(args) == 1
    assert not (tmp_path / "out").exists()


def test_tests_do_not_see_the_repository_demo_runs() -> None:
    from framelab.config import Settings

    assert "no_demo_runs" in str(Settings().demo_dir)
