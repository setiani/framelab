"""FastAPI application.

Spec: docs/plan/07-backend-api-and-frontend.md §1. M0 implements ``/api/health`` and static
serving of ``web/dist``. All other endpoints are added in milestone M9.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from framelab import __version__
from framelab.config import (
    MODEL_IMAGE,
    MODEL_SAM,
    MODEL_SPARK,
    MODEL_VOICE,
    MODEL_WONDER,
    Settings,
    get_settings,
)

WEB_DIST = Path(__file__).resolve().parents[3] / "web" / "dist"


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    app = FastAPI(title="FrameLab", version=__version__)
    app.state.settings = settings

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

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

    # TODO(M9): /api/runs, /api/runs/{id}, /events (SSE), /ledger, /track, /artifacts,
    #           /rerun, /api/series… (07 §1–§2), demo mode (07 §3).

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
