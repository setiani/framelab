"""Analyst agent: tool-calling loop on Chat Completions (Muse Spark 1.3).

Spec: docs/plan/06-agents-and-verification.md §3, §9.2
Status: skeleton (M0). Implement in milestone M6.
"""

from __future__ import annotations

from typing import Any


def run(ctx: Any, spec: Any, measurement: Any, round_: int) -> Any:
    """≤12 steps; returns ReportDraft or RerecordRequest; deterministic fallback on exhaustion."""
    raise NotImplementedError("Milestone M6: see docs/plan/09-milestones-and-tasks.md")
