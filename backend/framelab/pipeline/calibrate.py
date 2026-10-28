"""Scale calibration (ruler, object diameter, spoken length) and pendulum pivot.

Spec: docs/plan/04-pipeline-and-measurement.md §6
Status: skeleton (M0). Implement in milestone M3–M4.
"""

from __future__ import annotations

import numpy as np

from framelab.schemas import Calibration, Observation, ScaleEstimate


def ruler_scale(
    xs: np.ndarray, ys: np.ndarray, length_m: float, length_sigma_m: float = 0.001
) -> ScaleEstimate:
    """PCA major-axis length of the reference mask pixels → m/px with uncertainty."""
    raise NotImplementedError("Milestone M3–M4: see docs/plan/09-milestones-and-tasks.md")


def diameter_scale(
    observations: list[Observation], diameter_m: float, motion_axis: str
) -> ScaleEstimate:
    """Blur-robust diameter (minor axis when elongated) → m/px."""
    raise NotImplementedError("Milestone M3–M4: see docs/plan/09-milestones-and-tasks.md")


def fit_circle(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Kåsa algebraic circle fit → (cx, cy, r)."""
    raise NotImplementedError("Milestone M3–M4: see docs/plan/09-milestones-and-tasks.md")


def combine(estimates: list[ScaleEstimate]) -> Calibration:
    """Primary/secondary by priority ruler > object_diameter > spoken_length; agreement %."""
    raise NotImplementedError("Milestone M3–M4: see docs/plan/09-milestones-and-tasks.md")
