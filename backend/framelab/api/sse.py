"""Server-Sent Events helpers for /api/runs/{id}/events.

Spec: docs/plan/07-backend-api-and-frontend.md §2

Each event is one frame: ``id`` (the seq), ``event`` (the type), ``data`` (the event as JSON).
A ``: ping`` comment goes out every HEARTBEAT_S seconds of silence, so proxies keep the
connection open. The stream ends after the terminal event (``done`` or ``error``) has been sent.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

from framelab.events import EventBus

HEARTBEAT_S = 15.0
_END = object()


def format_sse(event: dict[str, Any]) -> str:
    """Format one event as an SSE frame: ``id``, ``event``, ``data`` lines."""
    data = json.dumps(event, ensure_ascii=False, default=str)
    return f"id: {event['seq']}\nevent: {event['type']}\ndata: {data}\n\n"


async def event_stream(bus: EventBus, run_id: str, last_event_id: int = 0) -> AsyncIterator[str]:
    """Replay the events after ``last_event_id``, then the live ones, with heartbeats.

    The subscription runs in its own task. A heartbeat timeout then never cancels the
    subscription, and a client that goes away cancels it through the ``finally`` below.
    """
    queue: asyncio.Queue = asyncio.Queue()

    async def pump() -> None:
        try:
            async for event in bus.subscribe(run_id, last_event_id):
                await queue.put(event)
        finally:
            await queue.put(_END)

    task = asyncio.create_task(pump())
    try:
        while True:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=HEARTBEAT_S)
            except TimeoutError:
                yield ": ping\n\n"
                continue
            if item is _END:
                return
            yield format_sse(item)
    finally:
        task.cancel()
