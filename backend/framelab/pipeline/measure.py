"""Masks → observations → cleaned tracks.

Spec: docs/plan/04-pipeline-and-measurement.md §5
Status: skeleton (M0). Implement in milestone M2–M3.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from framelab.schemas import ExperimentSpec, Observation, Track


def mask_to_observation(record: Any, concept: str, t: float) -> Observation | None:
    """Decode a SegmentationMaskRecord: sub-pixel centroid, area, PCA axes (bounds-scaled)."""
    raise NotImplementedError("Milestone M2–M3: see docs/plan/09-milestones-and-tasks.md")


def observations_from_result(
    result: Any, concept: str, offset_frame: int, effective_fps: float
) -> list[Observation]:
    """Keep the highest revision per (object_id, frame_index)."""
    raise NotImplementedError("Milestone M2–M3: see docs/plan/09-milestones-and-tasks.md")


def choose_target_track(observations: list[Observation], spec: ExperimentSpec) -> str:
    """Score = coverage × motion range; merge split ids (04 §5.1). Returns object_id."""
    raise NotImplementedError("Milestone M2–M3: see docs/plan/09-milestones-and-tasks.md")


def hampel(series: np.ndarray, window: int = 7, k: float = 3.5) -> np.ndarray:
    """Boolean outlier mask using rolling median and MAD-scaled sigma."""
    raise NotImplementedError("Milestone M2–M3: see docs/plan/09-milestones-and-tasks.md")


def clean_track(observations: list[Observation], spec: ExperimentSpec, n_frames: int) -> Track:
    """Grid, area/position outliers, gap interpolation (excluded from fits), metrics (04 §5.2)."""
    raise NotImplementedError("Milestone M2–M3: see docs/plan/09-milestones-and-tasks.md")
