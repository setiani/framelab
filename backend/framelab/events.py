"""Append-only audit log (events.jsonl) + live fan-out to SSE subscribers.

Spec: docs/plan/03-data-contracts.md §5, docs/plan/07-backend-api-and-frontend.md §2.
Every event has ``ts``, ``run_id``, ``seq``, ``type``, ``stage`` plus flat type-specific keys.
"""

from __future__ import annotations

import asyncio
import json
import threading
from collections import defaultdict
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

from framelab.schemas import utcnow

TERMINAL_EVENTS = frozenset({"done", "error"})


class EventBus:
    """Writes events to ``<runs_dir>/<run_id>/events.jsonl`` and pushes them to subscribers.

    Thread-safe for writers (pipeline steps may run in ``asyncio.to_thread``). Subscribers are
    asyncio queues; events emitted from worker threads are delivered via ``call_soon_threadsafe``.
    """

    def __init__(self, runs_dir: Path) -> None:
        self.runs_dir = Path(runs_dir)
        self._lock = threading.Lock()
        self._seq: dict[str, int] = {}
        self._subscribers: dict[str, list[tuple[asyncio.AbstractEventLoop, asyncio.Queue]]] = (
            defaultdict(list)
        )

    def log_path(self, run_id: str) -> Path:
        return self.runs_dir / run_id / "events.jsonl"

    def _next_seq(self, run_id: str) -> int:
        if run_id not in self._seq:
            self._seq[run_id] = sum(1 for _ in self.iter_events(run_id))
        self._seq[run_id] += 1
        return self._seq[run_id]

    def emit(self, run_id: str, type: str, stage: str, **payload: Any) -> dict[str, Any]:
        """Append one event and notify subscribers. Returns the event dict."""
        with self._lock:
            event: dict[str, Any] = {
                "ts": utcnow().isoformat(),
                "run_id": run_id,
                "seq": self._next_seq(run_id),
                "type": type,
                "stage": stage,
                **payload,
            }
            path = self.log_path(run_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
            subscribers = list(self._subscribers.get(run_id, []))
        for loop, queue in subscribers:
            loop.call_soon_threadsafe(queue.put_nowait, event)
        return event

    def iter_events(self, run_id: str, after_seq: int = 0) -> Iterator[dict[str, Any]]:
        """Read events from disk (skips malformed lines)."""
        path = self.log_path(run_id)
        if not path.is_file():
            return
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("seq", 0) > after_seq:
                    yield event

    async def subscribe(self, run_id: str, after_seq: int = 0) -> AsyncIterator[dict[str, Any]]:
        """Replay past events after ``after_seq``, then stream live ones until a terminal event."""
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        with self._lock:
            self._subscribers[run_id].append((loop, queue))
            replay = list(self.iter_events(run_id, after_seq))
        try:
            last_seq = after_seq
            for event in replay:
                last_seq = event["seq"]
                yield event
                if event["type"] in TERMINAL_EVENTS:
                    return
            while True:
                event = await queue.get()
                if event["seq"] <= last_seq:
                    continue
                last_seq = event["seq"]
                yield event
                if event["type"] in TERMINAL_EVENTS:
                    return
        finally:
            with self._lock:
                subs = self._subscribers.get(run_id, [])
                if (loop, queue) in subs:
                    subs.remove((loop, queue))
