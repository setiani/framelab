"""Muse Image client (stretch S-2): apparatus diagram from a key frame.

Spec: docs/plan/02-meta-api-reference.md §9
Status: skeleton (M0). Implement in milestone S-2.
"""

from __future__ import annotations

from pathlib import Path


def diagram_from_keyframe(keyframe_png: Path, description: str, out_path: Path) -> Path | None:
    """Return the saved diagram path, or None if unavailable (non-fatal). No text in image."""
    raise NotImplementedError("Milestone S-2: see docs/plan/09-milestones-and-tasks.md")
