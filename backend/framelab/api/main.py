"""FastAPI application.

Spec: docs/plan/07-backend-api-and-frontend.md §1–§3.

Runs execute as asyncio tasks, at most ``Settings.concurrency`` at a time. The pipeline is passed to
``create_app`` as ``runner`` (default: ``execute_run``), so tests can run the API without any model.
Errors use ``{"error": {"code": ..., "message": ...}}`` with the right HTTP status.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import shutil
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from framelab import __version__
from framelab.agent.ledger import Ledger
from framelab.api import sse
from framelab.config import (
    MODEL_IMAGE,
    MODEL_SAM,
    MODEL_SPARK,
    MODEL_VOICE,
    MODEL_WONDER,
    Settings,
    get_settings,
)
from framelab.events import EventBus
from framelab.pipeline import ingest, view
from framelab.pipeline import series as series_mod
from framelab.pipeline.rerun import create_rerun
from framelab.store import RunNotFoundError, RunStore, read_json

logger = logging.getLogger("framelab.api")
WEB_DIST = Path(__file__).resolve().parents[3] / "web" / "dist"
Runner = Callable[[str, Settings], Awaitable[Any]]
ALLOWED_EXTENSIONS = {".mp4", ".mov"}
ALLOWED_MIME = {"video/mp4", "video/quicktime"}
CHUNK_BYTES = 1 << 20
MAX_LIST_ITEMS = 50  # ledger evidence data longer than this is truncated (07 §1)
# The artifact allow-list (07 §1). Names only: no slashes, no dots before the extension.
ARTIFACT_NAME = re.compile(
    r"^(overlay\.mp4|report_(en|id)\.pdf|recording_feedback\.pdf|data\.csv"
    r"|chart_[a-z_]+\.(json|png)|keyframe\.png|next_experiment\.json)$"
)


class ApiError(Exception):
    """An error the client should see with its HTTP status and a stable code."""

    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


class RerunBody(BaseModel):
    capture_fps: int | None = None
    lang: str | None = None


class SeriesCreate(BaseModel):
    experiment: str = "pendulum"
    run_ids: list[str] = []


class SeriesRun(BaseModel):
    run_id: str


async def default_runner(run_id: str, settings: Settings) -> Any:
    from framelab.pipeline.run import execute_run

    return await execute_run(run_id, settings)


def _truncate(value: Any) -> Any:
    """Long lists in evidence data are cut so the ledger view stays readable (07 §1)."""
    if isinstance(value, dict):
        return {k: _truncate(v) for k, v in value.items()}
    if isinstance(value, list):
        if len(value) > MAX_LIST_ITEMS:
            return [_truncate(v) for v in value[:MAX_LIST_ITEMS]] + [
                f"… {len(value) - MAX_LIST_ITEMS} more"
            ]
        return [_truncate(v) for v in value]
    return value


def create_app(settings: Settings | None = None, runner: Runner | None = None) -> FastAPI:
    settings = settings or get_settings()
    app = FastAPI(title="FrameLab", version=__version__)
    app.state.settings = settings
    app.state.runner = runner or default_runner
    app.state.store = RunStore(settings.runs_dir)
    app.state.bus = EventBus(settings.runs_dir)
    app.state.tasks: dict[str, asyncio.Task] = {}
    app.state.semaphore: asyncio.Semaphore | None = None
    store: RunStore = app.state.store
    bus: EventBus = app.state.bus

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(ApiError)
    async def _api_error(_: Request, exc: ApiError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status, content={"error": {"code": exc.code, "message": exc.message}}
        )

    @app.exception_handler(RunNotFoundError)
    async def _not_found(_: Request, exc: RunNotFoundError) -> JSONResponse:
        return JSONResponse(
            status_code=404,
            content={"error": {"code": "not_found", "message": f"not found: {exc.args[0]}"}},
        )

    @app.exception_handler(RequestValidationError)
    async def _invalid(_: Request, exc: RequestValidationError) -> JSONResponse:
        problems = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
        return JSONResponse(
            status_code=422,
            content={"error": {"code": "invalid_request", "message": problems[:500]}},
        )

    # --- background execution ----------------------------------------------------------------
    def _semaphore() -> asyncio.Semaphore:
        if app.state.semaphore is None:
            app.state.semaphore = asyncio.Semaphore(settings.concurrency)
        return app.state.semaphore

    def _spawn(key: str, coro_factory: Callable[[], Awaitable[Any]]) -> None:
        async def job() -> None:
            async with _semaphore():
                await coro_factory()

        task = asyncio.create_task(job())
        app.state.tasks[key] = task
        task.add_done_callback(lambda t: _finished(key, t))

    def _finished(key: str, task: asyncio.Task) -> None:
        app.state.tasks.pop(key, None)
        if not task.cancelled() and task.exception() is not None:
            logger.error("background task %s failed", key, exc_info=task.exception())

    def _ledger(run_id: str) -> Ledger | None:
        path = store.run_dir(run_id) / "analysis" / "ledger.jsonl"
        return Ledger(path) if path.is_file() else None

    async def _save_upload(upload: UploadFile, target: Path, limit: int) -> int:
        total = 0
        with target.open("wb") as fh:
            while chunk := await upload.read(CHUNK_BYTES):
                total += len(chunk)
                if total > limit:
                    raise ApiError(413, "too_large", f"the video is larger than {limit >> 20} MB")
                fh.write(chunk)
        if total == 0:
            raise ApiError(422, "empty_upload", "the upload is empty")
        return total

    def _discard(run_id: str) -> None:
        shutil.rmtree(store.run_dir(run_id), ignore_errors=True)

    # --- health ------------------------------------------------------------------------------
    @app.get("/api/health")
    def health() -> dict:
        return {
            "ok": True,
            "version": __version__,
            "demo": settings.demo,
            "ffmpeg": shutil.which("ffmpeg") is not None,
            "api_key_configured": bool(settings.model_api_key),
            "models": {
                "spark": MODEL_SPARK,
                "wonder": MODEL_WONDER if settings.use_wonder else MODEL_SPARK,
                "sam": MODEL_SAM,
                "voice": MODEL_VOICE,
                "image": MODEL_IMAGE,
            },
        }

    # --- runs --------------------------------------------------------------------------------
    @app.post("/api/runs", status_code=201)
    async def create_run(
        video: UploadFile = File(...),
        lang: str = Form("id"),
        capture_fps: int | None = Form(None),
        series_id: str | None = Form(None),
        note: str | None = Form(None),
    ) -> dict:
        if settings.demo and not settings.demo_allow_upload:
            raise ApiError(403, "demo_mode", "uploads are disabled in demo mode")
        if lang not in ("id", "en"):
            raise ApiError(422, "invalid_lang", "lang must be id or en")
        if note is not None and len(note) > 300:
            raise ApiError(422, "note_too_long", "note is limited to 300 characters")
        if capture_fps is not None and not 1 <= capture_fps <= 1000:
            raise ApiError(422, "invalid_capture_fps", "capture_fps must be 1 to 1000")
        suffix = Path(video.filename or "").suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS or video.content_type not in ALLOWED_MIME:
            raise ApiError(415, "unsupported_media_type", "upload an .mp4 or .mov video")
        if series_id is not None:
            store.load_series(series_id)  # 404 through the handler if it does not exist
        state = store.create_run(
            video.filename or "upload.mp4",
            lang=lang,  # type: ignore[arg-type]
            capture_fps=capture_fps,
            note=note,
            series_id=series_id,
            demo=settings.demo,
        )
        target = store.input_video_path(state.run_id)
        try:
            await _save_upload(video, target, settings.max_upload_mb * 1024 * 1024)
            info = await asyncio.to_thread(ingest.probe, target)
            if not 1.0 <= info.duration_s <= 120.0:
                raise ApiError(422, "invalid_duration", "the video must be 1 to 120 seconds")
        except ApiError:
            _discard(state.run_id)
            raise
        except Exception as exc:  # noqa: BLE001 - ffprobe refused the file
            _discard(state.run_id)
            raise ApiError(422, "invalid_video", f"not a readable video: {exc}"[:300]) from exc
        _spawn(state.run_id, lambda: app.state.runner(state.run_id, settings))
        return {"run_id": state.run_id}

    @app.get("/api/runs")
    def list_runs(limit: int = 20) -> list[dict]:
        limit = max(1, min(limit, 200))
        return [s.model_dump(mode="json") for s in store.list_runs(limit)]

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str) -> dict:
        state = store.load_state(run_id)
        run_dir = store.run_dir(run_id)
        body: dict[str, Any] = state.model_dump(mode="json")
        spec = run_dir / "spec.json"
        body["spec"] = read_json(spec) if spec.is_file() else None
        calibration = run_dir / "calibration.json"
        body["calibration"] = read_json(calibration) if calibration.is_file() else None
        ledger = _ledger(run_id)
        latest: dict[str, Any] = {}
        claims: list[Any] = []
        if ledger is not None:
            for ev in ledger.evidence():
                if ev.id.startswith("gate_"):
                    latest[ev.data.get("gate", ev.id)] = ev.data
            claims = [c.model_dump(mode="json") for c in ledger.claims()]
        body["gates"] = [latest[k] for k in sorted(latest)]
        body["claims"] = claims
        return body

    @app.get("/api/runs/{run_id}/events")
    async def run_events(run_id: str, request: Request) -> StreamingResponse:
        store.load_state(run_id)  # 404 for an unknown run
        try:
            last = int(request.headers.get("last-event-id", "0"))
        except ValueError:
            last = 0
        return StreamingResponse(
            sse.event_stream(bus, run_id, last),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/runs/{run_id}/ledger")
    def run_ledger(run_id: str) -> dict:
        store.load_state(run_id)
        ledger = _ledger(run_id)
        if ledger is None:
            return {"evidence": [], "claims": []}
        evidence = [_truncate(ev.model_dump(mode="json")) for ev in ledger.evidence()]
        claims = [c.model_dump(mode="json") for c in ledger.claims()]
        return {"evidence": evidence, "claims": claims}

    @app.get("/api/runs/{run_id}/track")
    def run_track(run_id: str) -> dict:
        store.load_state(run_id)
        run_dir = store.run_dir(run_id)
        ledger = _ledger(run_id)
        if ledger is None or not any((run_dir / "tracks").glob("*.csv")):
            raise ApiError(404, "no_track", "this run has no track yet")
        ing = read_json(run_dir / "ingest" / "ingest.json")
        rows = view.track_rows(run_dir, ledger)
        points = [
            {"frame": r.frame, "t": r.t, "x_px": r.x_px, "y_px": r.y_px, "valid": r.valid,
             "flag": r.flag} for r in rows
        ]  # fmt: skip
        fit = view.fit_view(ledger, "quadratic")
        curve: list[dict[str, float]] = []
        if fit is not None and fit.t_start is not None and fit.t_end is not None:
            steps = 200
            for k in range(steps + 1):
                t = fit.t_start + (fit.t_end - fit.t_start) * k / steps
                curve.append({"t": t, "y_px": view.poly_px(fit.params, fit.t_offset, t)})
        return {"fps": float(ing["file_fps"]), "points": points, "fit_curve": curve}

    @app.get("/api/runs/{run_id}/artifacts/{name}")
    def run_artifact(run_id: str, name: str) -> FileResponse:
        store.load_state(run_id)
        if not ARTIFACT_NAME.match(name):
            raise ApiError(404, "not_found", f"no artifact {name!r}")
        run_dir = store.run_dir(run_id)
        if name == "keyframe.png":
            frames = sorted((run_dir / "ingest" / "keyframes").glob("frame_*.png"))
            path = frames[0] if frames else run_dir / "missing"
        elif name == "next_experiment.json":
            path = run_dir / "analysis" / "designer.json"
        else:
            path = run_dir / "outputs" / name
        path = store.artifact_path(run_id, str(path.relative_to(run_dir)))
        if not path.is_file():
            raise ApiError(404, "not_found", f"{name} is not rendered yet")
        return FileResponse(path)

    @app.post("/api/runs/{run_id}/rerun", status_code=201)
    async def rerun(run_id: str, body: RerunBody | None = None) -> dict:
        body = body or RerunBody()
        if body.lang is not None and body.lang not in ("id", "en"):
            raise ApiError(422, "invalid_lang", "lang must be id or en")
        if body.capture_fps is not None and not 1 <= body.capture_fps <= 1000:
            raise ApiError(422, "invalid_capture_fps", "capture_fps must be 1 to 1000")
        state = create_rerun(store, run_id, body.capture_fps, body.lang)
        _spawn(state.run_id, lambda: app.state.runner(state.run_id, settings))
        return {"run_id": state.run_id}

    # --- series ------------------------------------------------------------------------------
    @app.post("/api/series", status_code=201)
    def create_series(body: SeriesCreate) -> dict:
        if body.experiment != "pendulum":
            raise ApiError(422, "invalid_experiment", "series are for the pendulum")
        try:
            series = series_mod.create(store, body.run_ids)
        except series_mod.SeriesError as exc:
            raise ApiError(422, "series_error", str(exc)) from exc
        return series.model_dump(mode="json")

    @app.post("/api/series/{series_id}/runs")
    def add_series_run(series_id: str, body: SeriesRun) -> dict:
        try:
            series = series_mod.add_run(store, series_id, body.run_id)
        except series_mod.SeriesError as exc:
            raise ApiError(422, "series_error", str(exc)) from exc
        return series.model_dump(mode="json")

    @app.post("/api/series/{series_id}/analyze", status_code=202)
    async def analyze_series(series_id: str) -> dict:
        store.load_series(series_id)

        async def job() -> None:
            with contextlib.suppress(series_mod.SeriesError):  # the series keeps the reason
                await asyncio.to_thread(series_mod.analyze, store, series_id)

        _spawn(f"series:{series_id}", job)
        return {"series_id": series_id, "status": "analyzing"}

    @app.get("/api/series/{series_id}")
    def get_series(series_id: str) -> dict:
        series = store.load_series(series_id)
        body: dict[str, Any] = series.model_dump(mode="json")
        result = store.series_dir(series_id) / "analysis" / "series.json"
        body["results"] = read_json(result) if result.is_file() else None
        return body

    # --- static web app (M10) ----------------------------------------------------------------
    if WEB_DIST.is_dir():
        assets = WEB_DIST / "assets"
        if assets.is_dir():
            app.mount("/assets", StaticFiles(directory=assets), name="assets")

        @app.get("/{full_path:path}", include_in_schema=False)
        def spa(full_path: str) -> FileResponse:
            candidate = (WEB_DIST / full_path).resolve()
            if full_path and candidate.is_file() and WEB_DIST in candidate.parents:
                return FileResponse(candidate)
            return FileResponse(WEB_DIST / "index.html")

    return app


app = create_app()
