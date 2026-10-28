"""Segmentation stage: target video tracking, alt concepts, box fallback, reference image.

Spec: docs/plan/04-pipeline-and-measurement.md §4
Status: skeleton (M0). Implement in milestone M2.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from framelab.schemas import ExperimentSpec, SamRequestSummary


async def run(run_dir: Path, spec: ExperimentSpec, clip_file_id: str) -> list[SamRequestSummary]:
    """Issue SAM requests per 04 §4 and store raw + parsed outputs under sam/."""
    raise NotImplementedError("Milestone M2: see docs/plan/09-milestones-and-tasks.md")


def motion_blob_box(frames: list[np.ndarray]) -> tuple[int, int, int, int] | None:
    """Largest persistent moving blob → box prompt (04 §4.1)."""
    raise NotImplementedError("Milestone M2: see docs/plan/09-milestones-and-tasks.md")


def slug(concept: str) -> str:
    """File-safe slug for a concept, e.g. 'orange ball' → 'orange_ball'."""
    return "_".join("".join(c if c.isalnum() else " " for c in concept.lower()).split())


_ = Any  # keep import for stubs
