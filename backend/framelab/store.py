"""Run storage on the filesystem (no database).

Spec: docs/plan/01-architecture.md §5 (run folder layout). All JSON writes are atomic
(write to a temp file in the same directory, then ``os.replace``).
"""

from __future__ import annotations

import json
import os
import secrets
import tempfile
import time
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


REPLACE_RETRY_DELAYS_S = (0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8)


def replace_with_retry(src: str | Path, dst: str | Path) -> None:
    """``os.replace`` that tolerates a destination held open for a moment by another process.

    On Windows, replacing a file fails with ``PermissionError`` (WinError 5/32) while anyone
    has it open, e.g. Defender or the Search indexer scanning a file we just wrote, or the API
    reading ``state.json``. Those locks last milliseconds, so retry with a short backoff
    (≈2 s in total) before giving up.
    """
    for delay in REPLACE_RETRY_DELAYS_S:
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            time.sleep(delay)
    os.replace(src, dst)


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
        replace_with_retry(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def read_text_with_retry(path: Path) -> str:
    """Read a file another thread may be replacing at this moment.

    On Windows, ``os.replace`` leaves the old file "delete pending" for a few milliseconds, and
    opening it then fails with ``PermissionError`` (Errno 13) instead of returning either
    version. A replay rewrites ``state.json`` many times a second while the API reads it, so
    readers retry briefly, like writers do in ``replace_with_retry``.
    """
    for delay in REPLACE_RETRY_DELAYS_S:
        try:
            return path.read_text(encoding="utf-8")
        except PermissionError:
            time.sleep(delay)
    return path.read_text(encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(read_text_with_retry(path))


class RunStore:
    """Creates, loads and saves runs and series under ``root``.

    ``sample_root`` (optional) holds cached sample runs and series for demo mode (07 §3). They are
    read through the same methods, but never written: new runs and series always go to ``root``.
    """

    def __init__(self, root: Path, sample_root: Path | None = None) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.sample_root = Path(sample_root) if sample_root is not None else None

    # --- runs ---------------------------------------------------------------------------
    def run_dir(self, run_id: str) -> Path:
        _check_id(run_id, "r_")
        own = self.root / run_id
        if not own.exists() and self.sample_root is not None:
            cached = self.sample_root / run_id
            if (cached / "state.json").is_file():
                return cached
        return own

    def is_sample(self, run_id: str) -> bool:
        """True for a cached sample run (read-only)."""
        try:
            path = self.run_dir(run_id)
        except RunNotFoundError:
            return False
        return self.sample_root is not None and path.parent == self.sample_root

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
        return RunState.model_validate_json(read_text_with_retry(path))

    def save_state(self, state: RunState) -> None:
        state.updated_at = utcnow()
        write_json_atomic(self.run_dir(state.run_id) / "state.json", state)

    def set_stage(self, state: RunState, stage: Stage) -> RunState:
        state.stage = stage
        state.stage_history.append((stage, utcnow()))
        self.save_state(state)
        return state

    def list_runs(self, limit: int = 20, *, include_replays: bool = False) -> list[RunState]:
        """Runs in ``root``, newest first. Cached samples are not listed (see ``list_samples``);
        replays only with ``include_replays``."""
        states = [s for s in _load_states(self.root) if include_replays or s.replay_of is None]
        states.sort(key=lambda s: s.created_at, reverse=True)
        return states[:limit]

    def list_sample_runs(self) -> list[RunState]:
        if self.sample_root is None or not self.sample_root.is_dir():
            return []
        return _load_states(self.sample_root)

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
        own = self.root / "series" / series_id
        if not own.exists() and self.sample_root is not None:
            cached = self.sample_root / "series" / series_id
            if (cached / "series.json").is_file():
                return cached
        return own

    def series_roots(self) -> list[Path]:
        """Folders that hold series: our own, then the cached samples."""
        roots = [self.root / "series"]
        if self.sample_root is not None:
            roots.append(self.sample_root / "series")
        return [r for r in roots if r.is_dir()]

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
        return SeriesState.model_validate_json(read_text_with_retry(path))

    def save_series(self, series: SeriesState) -> None:
        write_json_atomic(self.series_dir(series.series_id) / "series.json", series)


def _load_states(root: Path) -> list[RunState]:
    states: list[RunState] = []
    for path in root.glob("r_*/state.json"):
        try:
            states.append(RunState.model_validate_json(read_text_with_retry(path)))
        except (ValueError, OSError):  # half-written, or deleted since the glob (old replay)
            continue
    return states


def _check_id(value: str, prefix: str) -> None:
    """Reject ids that could be used for path traversal."""
    if not value.startswith(prefix) or not all(c.isalnum() or c == "_" for c in value):
        raise RunNotFoundError(value)
