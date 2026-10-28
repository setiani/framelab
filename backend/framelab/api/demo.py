"""Demo mode: replay cached runs from demo_runs/ without API calls.

Spec: docs/plan/07-backend-api-and-frontend.md §3
Status: skeleton (M0). Implement in milestone M11.
"""

from __future__ import annotations

from pathlib import Path


def list_demo_runs(demo_dir: Path) -> list[str]:
    """Return cached demo run ids."""
    raise NotImplementedError("Milestone M11: see docs/plan/09-milestones-and-tasks.md")


async def start_replay(demo_run_id: str) -> str:
    """Create a replay run whose events are re-emitted with compressed timing (≤25 s)."""
    raise NotImplementedError("Milestone M11: see docs/plan/09-milestones-and-tasks.md")
