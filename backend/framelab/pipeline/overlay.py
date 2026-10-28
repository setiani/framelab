"""Annotated overlay MP4: mask contours, trail, HUD, result card.

Spec: docs/plan/04-pipeline-and-measurement.md §8
Status: skeleton (M0). Implement in milestone M8.
"""

from __future__ import annotations

from pathlib import Path


def render(run_dir: Path, out_path: Path) -> Path:
    """Pipe BGR frames to ffmpeg (libx264 -preset veryfast). HUD text from ledger claims only."""
    raise NotImplementedError("Milestone M8: see docs/plan/09-milestones-and-tasks.md")
