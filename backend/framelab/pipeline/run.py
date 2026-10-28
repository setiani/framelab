"""Orchestrator: executes stages in order with the analyst/auditor revise loop.

Spec: docs/plan/04-pipeline-and-measurement.md §1, docs/plan/01-architecture.md §3
Status: skeleton (M0). Implement in milestone M2–M7.
"""

from __future__ import annotations

from framelab.config import Settings
from framelab.schemas import RunState


async def execute_run(run_id: str, settings: Settings) -> RunState:
    """INGESTING → … → DONE | FAILED. Never raises; failures end in FAILED with error.json."""
    raise NotImplementedError("Milestone M2–M7: see docs/plan/09-milestones-and-tasks.md")
