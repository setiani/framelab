"""Charts: Plotly JSON for the web, matplotlib PNG for the PDF.

Spec: docs/plan/04-pipeline-and-measurement.md §7
Status: skeleton (M0). Implement in milestone M8.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def trajectory_chart(run_dir: Path, fit_id: str) -> dict[str, Any]:
    """Write outputs/chart_trajectory.json + .png; return paths."""
    raise NotImplementedError("Milestone M8: see docs/plan/09-milestones-and-tasks.md")


def series_chart(series_dir: Path) -> dict[str, Any]:
    """T0² vs L with error bars and regression band."""
    raise NotImplementedError("Milestone M8: see docs/plan/09-milestones-and-tasks.md")
