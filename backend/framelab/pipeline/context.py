"""RunContext: paths, state, emit(), clients and ledger for one run.

Spec: docs/plan/04-pipeline-and-measurement.md §1
Status: partial (M0). ``fail``/``finish`` are implemented in milestone M2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from framelab.config import Settings
from framelab.events import EventBus
from framelab.schemas import RunState
from framelab.store import RunStore


@dataclass
class RunContext:
    """Everything a stage needs. Stages read/write files under ``run_dir`` and call ``emit``."""

    settings: Settings
    store: RunStore
    bus: EventBus
    state: RunState
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def run_dir(self) -> Path:
        return self.store.run_dir(self.state.run_id)

    def emit(self, type: str, **payload: Any) -> dict[str, Any]:
        """Emit an event for this run at the current stage."""
        return self.bus.emit(self.state.run_id, type, self.state.stage.value, **payload)

    def fail(self, exc: BaseException) -> None:
        """Write error.json, set stage FAILED, emit ``error``."""
        raise NotImplementedError("Milestone M2: see docs/plan/09-milestones-and-tasks.md")

    def finish(self, verdict: Any) -> None:
        """Set DONE, verdict and headline; emit ``done``."""
        raise NotImplementedError("Milestone M2: see docs/plan/09-milestones-and-tasks.md")
