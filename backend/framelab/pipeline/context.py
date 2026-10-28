"""RunContext: paths, state, emit(), stage bookkeeping for one run.

Spec: docs/plan/04-pipeline-and-measurement.md §1, docs/plan/03-data-contracts.md §2/§5.
"""

from __future__ import annotations

import time
import traceback
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from framelab.config import Settings
from framelab.events import EventBus
from framelab.schemas import RunState, Stage, utcnow
from framelab.store import RunStore, write_json_atomic


@dataclass
class RunContext:
    """Everything a stage needs. Stages read/write files under ``run_dir`` and call ``emit``."""

    settings: Settings
    store: RunStore
    bus: EventBus
    state: RunState
    extras: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, run_id: str, settings: Settings) -> RunContext:
        store = RunStore(settings.runs_dir)
        return cls(
            settings=settings,
            store=store,
            bus=EventBus(settings.runs_dir),
            state=store.load_state(run_id),
        )

    @property
    def run_dir(self) -> Path:
        return self.store.run_dir(self.state.run_id)

    def emit(self, type: str, **payload: Any) -> dict[str, Any]:
        """Emit an event at the current stage. ``model_result`` costs are added to the run."""
        if type == "model_result" and payload.get("cost_usd"):
            self.state.cost_usd = round(self.state.cost_usd + float(payload["cost_usd"]), 6)
        if type == "warning" and payload.get("message"):
            self.state.warnings.append(str(payload["message"]))
        return self.bus.emit(self.state.run_id, type, self.state.stage.value, **payload)

    def save(self) -> None:
        self.store.save_state(self.state)

    def add_artifact(self, name: str, path: Path, mime: str = "application/octet-stream") -> None:
        rel = str(path.relative_to(self.run_dir)).replace("\\", "/")
        self.state.artifacts[name] = rel
        self.emit("artifact", name=name, path=rel, mime=mime)

    @contextmanager
    def stage(self, stage: Stage) -> Iterator[None]:
        """Set the stage, emit started/completed, persist state. Errors propagate."""
        self.store.set_stage(self.state, stage)
        t0 = time.perf_counter()
        self.emit("stage_started")
        yield
        self.emit("stage_completed", duration_s=round(time.perf_counter() - t0, 3))
        self.save()

    def fail(self, exc: BaseException) -> None:
        """Write error.json, set stage FAILED, emit ``error``."""
        message = f"{type(exc).__name__}: {exc}"
        write_json_atomic(
            self.run_dir / "error.json",
            {
                "message": message,
                "stage": self.state.stage.value,
                "traceback": traceback.format_exception(type(exc), exc, exc.__traceback__)[-20:],
                "at": utcnow().isoformat(),
            },
        )
        failed_at = self.state.stage.value
        self.state.error = message
        self.store.set_stage(self.state, Stage.FAILED)
        self.bus.emit(
            self.state.run_id,
            "error",
            failed_at,
            message=message,
            exception_type=type(exc).__name__,
            retryable=False,
        )

    def finish(
        self, verdict: str | None, headline: dict[str, Any] | None = None, *, partial: bool = False
    ) -> None:
        """Set DONE (or keep the last stage for partial dev runs) and emit ``done``."""
        self.state.verdict = verdict  # type: ignore[assignment]
        self.state.headline = headline
        if not partial:
            self.store.set_stage(self.state, Stage.DONE)
        self.save()
        self.bus.emit(
            self.state.run_id,
            "done",
            self.state.stage.value,
            verdict=verdict,
            headline=headline,
            partial=partial,
            cost_usd=self.state.cost_usd,
        )
