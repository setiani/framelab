"""Demo mode: cached sample runs and their replays, without API calls.

Spec: docs/plan/07-backend-api-and-frontend.md §3.

``demo_runs/`` (built by ``scripts/build_demo_cache.py``) holds finished runs plus
``samples.json``, the manifest behind the "Try a sample" cards. A cached run can be opened
directly (it is finished). A *replay* is a new run whose events are re-emitted from the cached
``events.jsonl`` with the original relative timing compressed ×4 (25 s at most). Each file is
copied into the replay when the event that produced it is emitted, so the run view fills in the
same order as during the original run. Replays are labelled in the UI and deleted after
``Settings.replay_ttl_hours``.
"""

from __future__ import annotations

import asyncio
import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from framelab.events import EventBus
from framelab.schemas import RunState, Stage, utcnow
from framelab.store import (
    RunStore,
    read_json,
    read_text_with_retry,
    replace_with_retry,
    write_json_atomic,
)

MANIFEST = "samples.json"
COMPRESS = 4.0
MAX_TOTAL_S = 25.0
MAX_GAP_S = 6.0  # one long wait (e.g. SAM) never stalls the replay for long
_EVENT_KEYS = {"ts", "run_id", "seq", "type", "stage"}

# Files that become visible when a stage completes (01 §5 run folder layout). Files named by an
# ``artifact`` event are copied with that event; the rest at ``done``.
STAGE_FILES: dict[str, tuple[str, ...]] = {
    "ingesting": ("ingest",),
    "transcribing": ("transcript.json",),
    "planning": ("spec.json",),
    "segmenting": ("sam",),
    "measuring": ("measurement.json", "observations.csv", "calibration.json", "tracks"),
    "analyzing": ("calibration.json", "tracks", "draft_report.json", "analysis/figures"),
    "auditing": ("analysis",),
    "designing": ("analysis/designer.json",),
    "rendering": ("outputs",),
}
LEDGER = "analysis/ledger.jsonl"
NEVER_COPY = {"state.json", "events.jsonl"}
_STAGES = {s.value for s in Stage}


class SampleError(Exception):
    """The sample does not exist or its cached run is incomplete."""


@dataclass
class Sample:
    key: str
    run_id: str
    title: dict[str, str]
    blurb: dict[str, str]
    state: RunState

    def public(self) -> dict[str, Any]:
        s = self.state
        return {
            "key": self.key,
            "run_id": self.run_id,
            "title": self.title,
            "blurb": self.blurb,
            "experiment": s.experiment,
            "verdict": s.verdict,
            "headline": s.headline,
            "original_filename": s.original_filename,
            "recorded_at": s.created_at.isoformat(),
        }


def load_samples(store: RunStore) -> list[Sample]:
    """The manifest's samples whose cached run exists, in manifest order."""
    root = store.sample_root
    if root is None or not (root / MANIFEST).is_file():
        return []
    try:
        manifest = read_json(root / MANIFEST)
    except ValueError:
        return []
    samples: list[Sample] = []
    for item in manifest.get("samples", []):
        run_id = str(item.get("run_id", ""))
        # Read the cached copy itself: on the machine that recorded the samples the same run id
        # also exists in runs/ (the original), and that one must not hide the card.
        path = cached_run_dir(store, run_id)
        if path is None:
            continue
        try:
            state = RunState.model_validate_json(read_text_with_retry(path / "state.json"))
        except (ValueError, OSError):
            continue
        samples.append(
            Sample(
                key=str(item["key"]),
                run_id=run_id,
                title=dict(item.get("title") or {}),
                blurb=dict(item.get("blurb") or {}),
                state=state,
            )
        )
    return samples


def cached_run_dir(store: RunStore, run_id: str) -> Path | None:
    """The run's folder in the sample cache, or None (also for ids that are not run ids)."""
    root = store.sample_root
    if root is None or not run_id.startswith("r_") or not run_id.replace("_", "").isalnum():
        return None
    path = root / run_id
    return path if (path / "state.json").is_file() else None


def find_sample(store: RunStore, key: str) -> Sample:
    for sample in load_samples(store):
        if sample.key == key:
            return sample
    raise SampleError(f"no sample {key!r}")


# --- replay ---------------------------------------------------------------------------------


@dataclass
class _Copier:
    """Copies files from the cached run into the replay as their events are emitted."""

    src: Path
    dst: Path
    ledger_lines: list[tuple[datetime | None, str]] = field(default_factory=list)
    ledger_written: int = 0

    def __post_init__(self) -> None:
        path = self.src / LEDGER
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    ts = _parse_ts(json.loads(line).get("created_at"))
                except (ValueError, AttributeError):
                    ts = None
                self.ledger_lines.append((ts, line))

    def copy(self, relative: str) -> None:
        source = self.src / relative
        target = self.dst / relative
        if not source.exists() or relative in NEVER_COPY or relative == LEDGER:
            return
        if source.is_dir():
            for path in sorted(source.rglob("*")):
                rel = path.relative_to(self.src).as_posix()
                if path.is_file() and rel not in NEVER_COPY and rel != LEDGER:
                    self._copy_file(path, self.dst / rel)
        else:
            self._copy_file(source, target)

    @staticmethod
    def _copy_file(source: Path, target: Path) -> None:
        if target.is_file() and target.stat().st_size == source.stat().st_size:
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(f".{target.name}.part")
        shutil.copyfile(source, tmp)
        replace_with_retry(tmp, target)

    def ledger_until(self, ts: datetime | None) -> None:
        """Write the ledger entries created up to ``ts`` (all of them for ``None``)."""
        count = self.ledger_written
        for i in range(self.ledger_written, len(self.ledger_lines)):
            entry_ts = self.ledger_lines[i][0]
            if ts is not None and (entry_ts is None or entry_ts > ts):
                break
            count = i + 1
        if count == self.ledger_written:
            return
        target = self.dst / LEDGER
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as fh:
            for _, line in self.ledger_lines[self.ledger_written : count]:
                fh.write(line + "\n")
        self.ledger_written = count

    def everything(self) -> None:
        self.ledger_until(None)
        for path in sorted(self.src.iterdir()):
            self.copy(path.name)


def _read_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict) and event.get("type"):
                events.append(event)
    return events


def _parse_ts(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value).replace("Z", "+00:00")
    return datetime.fromisoformat(text)


def replay_delays(
    events: list[dict[str, Any]], compress: float = COMPRESS, max_total_s: float = MAX_TOTAL_S
) -> list[float]:
    """Seconds to wait before each event: original gaps ÷ ``compress``, each gap at most
    ``MAX_GAP_S``, and the whole replay scaled down to ``max_total_s``."""
    stamps = [_parse_ts(e.get("ts")) for e in events]
    gaps: list[float] = []
    prev: datetime | None = None
    for ts in stamps:
        gap = 0.0 if prev is None or ts is None else max((ts - prev).total_seconds(), 0.0)
        gaps.append(min(gap / compress, MAX_GAP_S))
        prev = ts or prev
    total = sum(gaps)
    if total > max_total_s > 0:
        gaps = [g * max_total_s / total for g in gaps]
    return gaps


def create_replay(store: RunStore, sample: Sample) -> RunState:
    """A new run in ``store.root`` that will replay ``sample``. Call ``run_replay`` next."""
    src = sample.state
    state = store.create_run(
        src.original_filename,
        lang=src.lang,
        capture_fps=src.capture_fps,
        note=src.note,
    )
    state.demo = True
    state.sample = sample.key
    state.replay_of = sample.run_id
    store.save_state(state)
    return state


async def run_replay(
    store: RunStore,
    bus: EventBus,
    run_id: str,
    *,
    compress: float = COMPRESS,
    max_total_s: float = MAX_TOTAL_S,
) -> None:
    """Re-emit the cached run's events into ``run_id`` and copy its files as they appear."""
    state = store.load_state(run_id)
    if state.replay_of is None:
        raise SampleError(f"{run_id} is not a replay")
    src_dir = cached_run_dir(store, state.replay_of) or store.run_dir(state.replay_of)
    source = RunState.model_validate_json(read_text_with_retry(src_dir / "state.json"))
    copier = _Copier(src_dir, store.run_dir(run_id))
    events = _read_events(src_dir / "events.jsonl")
    try:
        for event, delay in zip(events, replay_delays(events, compress, max_total_s), strict=True):
            if delay > 0:
                await asyncio.sleep(delay)
            await asyncio.to_thread(_apply, store, state, source, copier, event)
            payload = {k: v for k, v in event.items() if k not in _EVENT_KEYS}
            bus.emit(run_id, event["type"], event.get("stage", ""), **payload)
        if not events or events[-1]["type"] not in ("done", "error"):
            await asyncio.to_thread(_finish, store, state, source, copier)
            bus.emit(
                run_id,
                "done",
                "done",
                verdict=source.verdict,
                headline=source.headline,
                partial=False,
                cost_usd=source.cost_usd,
            )
    except Exception as exc:  # noqa: BLE001 - a broken cache must not leave a run hanging
        state.stage = Stage.FAILED
        state.error = f"replay failed: {exc}"[:300]
        store.save_state(state)
        bus.emit(run_id, "error", "failed", message=state.error)


def _apply(
    store: RunStore, state: RunState, source: RunState, copier: _Copier, event: dict[str, Any]
) -> None:
    """Side effects of one event, done before it is emitted so a refresh finds the files."""
    kind = event["type"]
    stage = event.get("stage", "")
    copier.ledger_until(_parse_ts(event.get("ts")))
    changed = False
    if kind == "stage_started" and stage in _STAGES:
        state.stage = Stage(stage)
        state.stage_history.append((state.stage, utcnow()))
        changed = True
    elif kind == "stage_completed":
        for rel in STAGE_FILES.get(stage, ()):
            copier.copy(rel)
        if stage == "planning":
            state.experiment = source.experiment
            changed = True
    elif kind == "artifact" and event.get("path"):
        copier.copy(str(event["path"]))
        if event.get("name"):
            state.artifacts[str(event["name"])] = str(event["path"])
            changed = True
    elif kind == "model_result" and event.get("cost_usd"):
        state.cost_usd = round(state.cost_usd + float(event["cost_usd"]), 6)
        changed = True
    elif kind == "verdict" and event.get("final"):
        state.verdict = source.verdict
        state.experiment = source.experiment
        changed = True
    if kind == "done":
        _finish(store, state, source, copier)
        return
    if kind == "error":
        copier.everything()
        state.stage = Stage.FAILED
        state.error = source.error
        state.stage_history.append((Stage.FAILED, utcnow()))
        changed = True
    if changed:
        store.save_state(state)


def _finish(store: RunStore, state: RunState, source: RunState, copier: _Copier) -> None:
    copier.everything()
    for name in (
        "experiment",
        "verdict",
        "headline",
        "revise_rounds",
        "warnings",
        "error",
        "artifacts",
        "cost_usd",
    ):
        setattr(state, name, getattr(source, name))
    state.stage = source.stage
    state.stage_history.append((source.stage, utcnow()))
    store.save_state(state)


def cleanup_replays(store: RunStore, ttl_hours: float, keep_running: set[str]) -> int:
    """Delete replays older than ``ttl_hours`` (never one still replaying). Returns the count."""
    cutoff = utcnow() - timedelta(hours=ttl_hours)
    removed = 0
    for state in store.list_runs(limit=10_000, include_replays=True):
        if state.replay_of and state.created_at < cutoff and state.run_id not in keep_running:
            shutil.rmtree(store.root / state.run_id, ignore_errors=True)
            removed += 1
    return removed


def write_manifest(root: Path, samples: list[dict[str, Any]]) -> None:
    write_json_atomic(root / MANIFEST, {"samples": samples})
