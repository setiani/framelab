"""Quality gates G1–G5 with per-experiment thresholds and EN/ID message templates.

Spec: docs/plan/06-agents-and-verification.md §7
Status: thresholds defined (M0); gate logic implemented in milestone M6.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from framelab.schemas import GateResult


class GateThresholds(BaseModel):
    g1_min_coverage: float = 0.90
    g1_max_jump_factor: float = 5.0
    g1_min_points: int = 8  # pendulum: 60
    g1_min_cycles: float = 3.0  # pendulum only
    g2_max_scale_disagreement_pct: float = 5.0  # pendulum L check: 8.0
    g3_min_r2: float = 0.995  # pendulum: 0.98
    g3_max_rms_px: float = 2.0
    g3_min_durbin_watson: float = 1.0  # below → warn; below 0.5 → fail
    g3_fail_durbin_watson: float = 0.5
    g4_range: tuple[float, float] = (9.0, 10.6)
    g4_slowmo_k: tuple[int, ...] = (2, 4, 8)


def run_gates(ctx: Any, thresholds: GateThresholds) -> list[GateResult]:
    """G1 track integrity, G2 scale consistency, G3 fit quality, G4 plausibility."""
    raise NotImplementedError("Milestone M6: see docs/plan/09-milestones-and-tasks.md")


def gate_g5(draft: Any, ledger: Any) -> GateResult:
    """Ledger completeness: bare numbers, unknown claims, missing hypothesis claim."""
    raise NotImplementedError("Milestone M6: see docs/plan/09-milestones-and-tasks.md")


def slowmo_factor(g_measured: float, g_ref: float, ks: tuple[int, ...] = (2, 4, 8)) -> int | None:
    """Return k if g·k² is within 5 % of g_ref (retimed slow-motion signature), else None."""
    for k in ks:
        if abs(g_measured * k * k - g_ref) / g_ref < 0.05:
            return k
    return None
