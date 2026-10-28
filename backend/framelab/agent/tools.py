"""Analyst tools: JSON schemas, dispatch, compact results, evidence records.

Spec: docs/plan/06-agents-and-verification.md §4
Status: skeleton (M0). Implement in milestone M6.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

TOOL_NAMES = [
    "get_run_summary",
    "choose_track",
    "resegment",
    "calibrate_scale",
    "set_fit_window",
    "fit_model",
    "bootstrap_ci",
    "derive",
    "compare_reference",
    "compare_hypothesis",
    "run_gates",
    "make_chart",
    "add_claim",
    "submit_report",
    "request_rerecord",
]


@dataclass
class ToolResult:
    ok: bool
    summary: str
    evidence_ids: list[str] = field(default_factory=list)
    data_excerpt: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    terminal: bool = False
    payload: Any = None

    def to_json(self) -> str:
        """Compact JSON for the model (≤2 KB, no large arrays)."""
        raise NotImplementedError("Milestone M6: see docs/plan/09-milestones-and-tasks.md")


def tool_schemas() -> list[dict[str, Any]]:
    """OpenAI function-calling schemas for all tools in TOOL_NAMES."""
    raise NotImplementedError("Milestone M6: see docs/plan/09-milestones-and-tasks.md")


def dispatch(ctx: Any, name: str, args: dict[str, Any]) -> ToolResult:
    """Validate args, run the tool, write evidence, emit tool_call/tool_result events."""
    raise NotImplementedError("Milestone M6: see docs/plan/09-milestones-and-tasks.md")
