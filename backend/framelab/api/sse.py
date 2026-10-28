"""Server-Sent Events helpers for /api/runs/{id}/events.

Spec: docs/plan/07-backend-api-and-frontend.md §2
Status: skeleton (M0). Implement in milestone M9.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any


def format_sse(event: dict[str, Any]) -> str:
    """Format one event as an SSE frame: ``id``, ``event``, ``data`` lines."""
    raise NotImplementedError("Milestone M9: see docs/plan/09-milestones-and-tasks.md")


async def event_stream(run_id: str, last_event_id: int = 0) -> AsyncIterator[str]:
    """Replay + live events with a ``: ping`` heartbeat every 15 s; closes after done/error."""
    raise NotImplementedError("Milestone M9: see docs/plan/09-milestones-and-tasks.md")
    yield ""  # pragma: no cover
