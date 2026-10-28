"""Auditor: independent review with Muse Wonder (fallback Spark).

Spec: docs/plan/06-agents-and-verification.md §5, §9.3
Status: skeleton (M0). Implement in milestone M7.
"""

from __future__ import annotations

from typing import Any

from framelab.schemas import GateResult, Verdict


def run(ctx: Any, draft: Any, round_: int) -> Verdict:
    """Deterministic pre-checks → model verdict → deterministic post-checks/overrides."""
    raise NotImplementedError("Milestone M7: see docs/plan/09-milestones-and-tasks.md")


def finalize_verdict(auditor: Verdict, gates: list[GateResult]) -> tuple[str, list[str]]:
    """Final status per 06 §8 (never REJECT at DONE) and extra warnings."""
    warnings: list[str] = []
    hard_failed = [g for g in gates if g.hard and g.status == "fail"]
    if hard_failed:
        return "NEEDS_RERECORD", warnings
    if auditor.status == "REJECT":
        warnings.append("auditor_unresolved")
        return "PASS_WITH_WARNINGS", warnings
    return auditor.status, warnings
