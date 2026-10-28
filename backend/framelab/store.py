"""Run storage on the filesystem (no database).

Spec: docs/plan/01-architecture.md §5 (run folder layout). All JSON writes are atomic
(write to a temp file in the same directory, then ``os.replace``).
"""

from __future__ import annotations

import json
import os
import secrets
import tempfile
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from framelab.schemas import Lang, RunState, SeriesState, Stage, utcnow

RUN_SUBDIRS = ("input", "ingest", "ingest/keyframes", "sam", "tracks", "analysis", "outputs")


class RunNotFoundError(KeyError):
    """Raised when a run or series id does not exist."""


def new_run_id() -> str:
    """``r_YYYYMMDD_HHMMSS_<6 hex>`` (UTC)."""
    return f"r_{utcnow():%Y%m%d_%H%M%S}_{secrets.token_hex(3)}"


def new_series_id() -> str:
    """``s_YYYYMMDD_HHMMSS_<6 hex>`` (UTC)."""
    return f"s_{utcnow():%Y%m%d_%H%M%S}_{secrets.token_hex(3)}"


def write_json_atomic(path: Path, data: Any) -> None:
    """Atomically write JSON (pydantic models are dumped in JSON mode)."""
    if isinstance(data, BaseModel):
        payload = data.model_dump_json(indent=2)
    else:
        payload = json.dumps(data, indent=2, ensure_ascii=False, default=str)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(payload)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


class RunStore:
    """Creates, loads and saves runs and series under ``root``."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # --- runs ---------------------------------------------------------------------------
    def run_dir(self, run_id: str) -> Path:
        _check_id(run_id, "r_")
        return self.root / run_id

    def create_run(
        self,
        original_filename: str,
        *,
        lang: Lang = "id",
        capture_fps: int | None = None,
        note: str | None = None,
        series_id: str | None = None,
        demo: bool = False,
    ) -> RunState:
        run_id = new_run_id()
        run_dir = self.run_dir(run_id)
        for sub in RUN_SUBDIRS:
            (run_dir / sub).mkdir(parents=True, exist_ok=True)
        now = utcnow()
        state = RunState(
            run_id=run_id,
            created_at=now,
            updated_at=now,
            stage=Stage.CREATED,
            stage_history=[(Stage.CREATED, now)],
            original_filename=original_filename,
            lang=lang,
            capture_fps=capture_fps,
            note=note,
            series_id=series_id,
            demo=demo,
        )
        self.save_state(state)
        return state

    def input_video_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "input" / "original.mp4"

    def load_state(self, run_id: str) -> RunState:
        path = self.run_dir(run_id) / "state.json"
        if not path.is_file():
            raise RunNotFoundError(run_id)
        return RunState.model_validate_json(path.read_text(encoding="utf-8"))

    def save_state(self, state: RunState) -> None:
        state.updated_at = utcnow()
        write_json_atomic(self.run_dir(state.run_id) / "state.json", state)

    def set_stage(self, state: RunState, stage: Stage) -> RunState:
        state.stage = stage
        state.stage_history.append((stage, utcnow()))
        self.save_state(state)
        return state

    def list_runs(self, limit: int = 20) -> list[RunState]:
        states: list[RunState] = []
        for path in self.root.glob("r_*/state.json"):
            try:
                states.append(RunState.model_validate_json(path.read_text(encoding="utf-8")))
            except ValueError:
                continue
        states.sort(key=lambda s: s.created_at, reverse=True)
        return states[:limit]

    def artifact_path(self, run_id: str, relative: str) -> Path:
        """Resolve a path inside the run folder, refusing path traversal."""
        base = self.run_dir(run_id).resolve()
        target = (base / relative).resolve()
        if base != target and base not in target.parents:
            raise ValueError(f"Path escapes run directory: {relative!r}")
        return target

    # --- series -------------------------------------------------------------------------
    def series_dir(self, series_id: str) -> Path:
        _check_id(series_id, "s_")
        return self.root / "series" / series_id

    def create_series(self, run_ids: list[str] | None = None) -> SeriesState:
        series = SeriesState(
            series_id=new_series_id(), created_at=utcnow(), run_ids=list(run_ids or [])
        )
        self.save_series(series)
        return series

    def load_series(self, series_id: str) -> SeriesState:
        path = self.series_dir(series_id) / "series.json"
        if not path.is_file():
            raise RunNotFoundError(series_id)
        return SeriesState.model_validate_json(path.read_text(encoding="utf-8"))

    def save_series(self, series: SeriesState) -> None:
        write_json_atomic(self.series_dir(series.series_id) / "series.json", series)


def _check_id(value: str, prefix: str) -> None:
    """Reject ids that could be used for path traversal."""
    if not value.startswith(prefix) or not all(c.isalnum() or c == "_" for c in value):
        raise RunNotFoundError(value)
