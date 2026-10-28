"""Planner: video + transcript → ExperimentSpec (Muse Spark 1.3, Responses API).

Spec: docs/plan/06-agents-and-verification.md §2, §9.1
Status: skeleton (M0). Implement in milestone M5.
"""

from __future__ import annotations

from framelab.schemas import ExperimentSpec, ProbeInfo, Transcript


async def run(
    file_id: str, transcript: Transcript, probe: ProbeInfo, lang: str, note: str | None
) -> ExperimentSpec:
    """Call Spark with the video, validate strictly, retry once with errors (06 §2)."""
    raise NotImplementedError("Milestone M5: see docs/plan/09-milestones-and-tasks.md")


def validate_spec(spec: ExperimentSpec, transcript: Transcript, duration_s: float) -> list[str]:
    """Return validation errors (concept length, window bounds, spoken values traceable)."""
    raise NotImplementedError("Milestone M5: see docs/plan/09-milestones-and-tasks.md")
