"""Event detection: free-fall release/impact window.

Spec: docs/plan/05-physics-and-uncertainty.md §2.1
Status: skeleton (M0). Implement in milestone M3.
"""

from __future__ import annotations

import numpy as np


def freefall_window(t: np.ndarray, y_m: np.ndarray) -> tuple[int, int, list[str]]:
    """Return (start_idx, end_idx, warnings) excluding contact frames."""
    raise NotImplementedError("Milestone M3: see docs/plan/09-milestones-and-tasks.md")
