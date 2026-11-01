"""Experiment registry.

Spec: docs/plan/05-physics-and-uncertainty.md §1
Status: used since M3 (free fall) and M4 (pendulum: min_valid_points); gates use it in M7.
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
        GateThresholds(
            g1_min_points=60,
            g1_fail_coverage=0.75,  # 75–90 %: warn (blur at the bottom of the swing)
            g2_max_scale_disagreement_pct=8.0,
            g3_min_r2=0.98,
            g3_durbin_watson_soft=True,  # 11 decisions, 1 Nov: real swings, R² 0.9997, DW 0.03
            g3_max_rms_rel_amplitude=0.03,
        ),
        60,
        (9.0, 10.6),
        False,
    ),
}
