"""Experiment registry.

Spec: docs/plan/05-physics-and-uncertainty.md §1
Status: skeleton (M0). Implement in milestone M3.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from framelab.agent.gates import GateThresholds


@dataclass(frozen=True)
class ExperimentDef:
    name: str
    motion_axis: Literal["x", "y", "xy"]
    primary_model: str
    headline_quantity: str
    gates: GateThresholds
    min_valid_points: int
    plausible_range: tuple[float, float]
    required_scale: bool


REGISTRY: dict[str, ExperimentDef] = {
    "free_fall": ExperimentDef(
        "free_fall", "y", "quadratic", "g", GateThresholds(), 8, (9.0, 10.6), True
    ),
    "pendulum": ExperimentDef(
        "pendulum",
        "x",
        "damped_sine",
        "g",
        GateThresholds(g1_min_points=60, g2_max_scale_disagreement_pct=8.0, g3_min_r2=0.98),
        60,
        (9.0, 10.6),
        False,
    ),
}
