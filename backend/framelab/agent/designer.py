"""Designer: next experiment (one variable changed), predictions computed by code.

Spec: docs/plan/06-agents-and-verification.md §6, §9.4
Status: skeleton (M0). Implement in milestone M7.
"""

from __future__ import annotations

from typing import Any

from framelab.schemas import DesignerProposal


def predict(
    experiment: str, changed_variable: str, values: list[float], g_measured: float
) -> list[dict[str, Any]]:
    """pendulum: T = 2π√(L/g); free fall: t = √(2h/g)."""
    raise NotImplementedError("Milestone M7: see docs/plan/09-milestones-and-tasks.md")


def run(ctx: Any, spec: Any, verdict: Any) -> DesignerProposal:
    """One tool (predict); output DesignerProposal with {{pN}} placeholders."""
    raise NotImplementedError("Milestone M7: see docs/plan/09-milestones-and-tasks.md")
