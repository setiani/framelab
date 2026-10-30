"""All output files of a run, rendered from the saved files only (04 §7–§10, M8 replay).

The orchestrator's RENDERING stage and ``framelab replay`` both call ``render_outputs``. It makes
no model call, so the same run folder always gives the same outputs. A failing output is reported
as a warning and the others are still written.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from framelab.agent.ledger import Ledger
from framelab.pipeline import charts, overlay, report, view
from framelab.schemas import RunState
from framelab.store import read_json


@dataclass
class RenderResult:
    paths: dict[str, Path] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def render_outputs(run_dir: Path, *, with_overlay: bool = True) -> RenderResult:
    """Charts, data export, overlay video, reports and the recording feedback page."""
    state = RunState.model_validate(read_json(run_dir / "state.json"))
    ledger = Ledger(run_dir / "analysis" / "ledger.jsonl")
    result = RenderResult()
    (run_dir / "outputs").mkdir(parents=True, exist_ok=True)

    def attempt(name: str, fn) -> None:
        try:
            produced = fn()
        except Exception as exc:  # noqa: BLE001 - one failing output must not stop the others
            result.warnings.append(f"{name} not rendered: {type(exc).__name__}: {exc}")
            return
        result.paths.update(produced)

    measured = view.primary_fit(ledger) is not None
    if measured:  # a run stopped before any fit (no target, Planner stop) has only feedback
        attempt("charts", lambda: charts.write_trajectory(run_dir, ledger, state.lang))
        attempt("data", lambda: {"data_csv": report.export_csv(run_dir)})
    if with_overlay and measured:
        attempt(
            "overlay",
            lambda: {"overlay": overlay.render(run_dir, run_dir / "outputs" / "overlay.mp4")},
        )
    attempt("reports", lambda: report.write_reports(run_dir, ledger, state))
    return result
