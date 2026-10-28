"""Placeholder rendering ({{cNN}}, {{pN}}) and the G5 bare-number check.

Spec: docs/plan/06-agents-and-verification.md §7 (G5)
Status: skeleton (M0). Implement in milestone M6.
"""

from __future__ import annotations

import re

PLACEHOLDER_RE = re.compile(r"\{\{(c\d{2,}|p\d+)\}\}")


def find_placeholders(text: str) -> list[str]:
    return PLACEHOLDER_RE.findall(text)


def bare_numbers(text: str) -> list[str]:
    """Digits outside placeholders and the allow-list (06 §7 G5). Empty list = OK."""
    raise NotImplementedError("Milestone M6: see docs/plan/09-milestones-and-tasks.md")


def render(text: str, claims: dict[str, str]) -> str:
    """Replace placeholders with formatted claim strings; unknown id → KeyError."""
    raise NotImplementedError("Milestone M6: see docs/plan/09-milestones-and-tasks.md")
