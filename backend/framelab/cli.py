"""FrameLab command-line interface (Typer).

Spec: docs/plan/07-backend-api-and-frontend.md §5. ``doctor`` is implemented (M0); the other
commands are wired in later milestones (see docs/plan/09-milestones-and-tasks.md).
"""

from __future__ import annotations

import importlib.util
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer

from framelab import __version__
from framelab.config import META_BASE_URL, MODEL_SPARK, Settings, mask_secret

app = typer.Typer(help="FrameLab: every phone is a physics lab.", no_args_is_help=True)
series_app = typer.Typer(help="Multi-run (series) analysis.", no_args_is_help=True)
app.add_typer(series_app, name="series")

OK, WARN, FAIL = "[ OK ]", "[WARN]", "[FAIL]"


def _not_yet(milestone: str) -> None:
    typer.secho(
        f"Not implemented yet (milestone {milestone}). See docs/plan/09-milestones-and-tasks.md.",
        fg=typer.colors.YELLOW,
    )
    raise typer.Exit(code=2)


@app.command()
def version() -> None:
    """Print the FrameLab version."""
    typer.echo(__version__)


@app.command()
def doctor(
    ping: Annotated[
        bool, typer.Option(help="Also call Muse Spark once to verify the key (costs < $0.001).")
    ] = False,
    skip_browser: Annotated[bool, typer.Option(help="Skip launching Playwright Chromium.")] = False,
) -> None:
    """Check the local environment: Python, ffmpeg, Chromium, API key, runs folder."""
    settings = Settings()
    failures = 0

    def report(status: str, name: str, detail: str = "") -> None:
        nonlocal failures
        color = {OK: typer.colors.GREEN, WARN: typer.colors.YELLOW, FAIL: typer.colors.RED}[status]
        typer.secho(f"{status} {name:<26} {detail}", fg=color)
        if status == FAIL:
            failures += 1

    py = sys.version_info
    report(
        OK if py >= (3, 11) else FAIL,
        "python",
        f"{platform.python_version()} ({platform.system()})",
    )

    for binary in ("ffmpeg", "ffprobe"):
        path = shutil.which(binary)
        if not path:
            hint = "install: winget install Gyan.FFmpeg (Windows) / apt install ffmpeg"
            report(FAIL, binary, f"not found on PATH ({hint})")
            continue
        try:
            out = subprocess.run(
                [path, "-version"], capture_output=True, text=True, check=True, timeout=20
            )
            report(OK, binary, out.stdout.splitlines()[0][:70])
        except (subprocess.SubprocessError, OSError) as exc:
            report(FAIL, binary, str(exc))

    for module in ("meta_sam_parser", "openai", "cv2", "scipy", "playwright"):
        found = importlib.util.find_spec(module) is not None
        report(OK if found else FAIL, f"python pkg {module}", "" if found else "run: uv sync")

    if skip_browser:
        report(WARN, "playwright chromium", "skipped")
    else:
        try:
            from playwright.sync_api import sync_playwright

            with sync_playwright() as pw:
                browser = pw.chromium.launch()
                browser.close()
            report(OK, "playwright chromium", "launches (PDF reports will work)")
        except Exception as exc:  # noqa: BLE001
            msg = str(exc).splitlines()[0][:80]
            report(FAIL, "playwright chromium", f"{msg} → run: uv run playwright install chromium")

    runs_dir = Path(settings.runs_dir)
    try:
        runs_dir.mkdir(parents=True, exist_ok=True)
        probe = runs_dir / ".write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        report(OK, "runs dir", str(runs_dir.resolve()))
    except OSError as exc:
        report(FAIL, "runs dir", str(exc))

    key = settings.model_api_key
    if key:
        report(OK, "MODEL_API_KEY", mask_secret(key))
    else:
        report(WARN, "MODEL_API_KEY", "missing: set it in .env for live runs")

    if ping:
        if not key:
            report(FAIL, "Muse Spark ping", "no MODEL_API_KEY")
        else:
            try:
                from openai import OpenAI

                client = OpenAI(base_url=META_BASE_URL, api_key=key, timeout=60)
                resp = client.chat.completions.create(
                    model=MODEL_SPARK,
                    messages=[{"role": "user", "content": "Reply with the single word: ready"}],
                    reasoning_effort="minimal",
                )
                text = (resp.choices[0].message.content or "").strip()
                report(OK, "Muse Spark ping", f"{MODEL_SPARK} replied {text[:20]!r}")
            except Exception as exc:  # noqa: BLE001
                report(FAIL, "Muse Spark ping", str(exc).splitlines()[0][:100])

    typer.echo("")
    if failures:
        typer.secho(f"{failures} check(s) failed.", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    typer.secho("All required checks passed.", fg=typer.colors.GREEN)


EXPERIMENT_AXIS = {"free_fall": "y", "pendulum": "x", "melting": "xy"}
UNTIL_CHOICES = ("ingesting", "planning", "segmenting", "measuring")


def manual_spec(
    experiment: str,
    concept: str,
    alt_concepts: list[str],
    ref_concept: str | None,
    ref_length_m: float | None,
    ref_frame_s: float | None,
    window: str | None,
    lang: str,
):
    """Temporary ExperimentSpec from CLI flags until the Planner exists (M5)."""
    from framelab.schemas import ExperimentSpec, ReferenceObject

    if window:
        t0, t1 = (float(v) for v in window.split(","))
    else:
        t0, t1 = 0.0, 1e9  # clamped to the video length after ingest
    reference = (
        ReferenceObject(
            concept=ref_concept,
            length_m=ref_length_m,
            length_source="spoken" if ref_length_m else "unknown",
            frame_hint_s=ref_frame_s,
        )
        if ref_concept
        else None
    )
    return ExperimentSpec(
        experiment=experiment,  # type: ignore[arg-type]
        target_concept=concept,
        target_alt_concepts=alt_concepts,
        reference=reference,
        analysis_window_s=(t0, t1),
        motion_axis=EXPERIMENT_AXIS.get(experiment, "xy"),  # type: ignore[arg-type]
        language=lang,  # type: ignore[arg-type]
        setup_description="Manual spec from CLI flags (Planner arrives in M5).",
        confidence=1.0,
    )


@app.command()
def run(
    video: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="Input video")],
    concept: Annotated[
        str | None, typer.Option(help="SAM noun phrase for the moving object (required until M5)")
    ] = None,
    alt_concept: Annotated[
        list[str] | None, typer.Option("--alt-concept", help="Fallback noun phrase (repeat)")
    ] = None,
    ref_concept: Annotated[
        str | None, typer.Option(help="Static scale reference, e.g. ruler")
    ] = None,
    ref_length_m: Annotated[
        float | None, typer.Option(help="Real length of the reference (m)")
    ] = None,
    ref_frame_s: Annotated[
        float | None, typer.Option(help="Time (s) where the reference is fully visible")
    ] = None,
    window: Annotated[str | None, typer.Option(help="Analysis window 't0,t1' in seconds")] = None,
    experiment: Annotated[str, typer.Option(help="free_fall | pendulum")] = "free_fall",
    lang: Annotated[str, typer.Option(help="Report language: id | en")] = "id",
    capture_fps: Annotated[int | None, typer.Option(help="Real capture fps for slow-mo")] = None,
    until: Annotated[
        str, typer.Option(help=f"Stop after stage: {', '.join(UNTIL_CHOICES)}")
    ] = "measuring",
    runs_dir: Annotated[Path | None, typer.Option(help="Runs folder (default: settings)")] = None,
) -> None:
    """Run the pipeline on a video. M2: ingest → SAM → observations (stops at --until)."""
    import asyncio

    from framelab.pipeline.run import execute_run
    from framelab.schemas import Stage
    from framelab.store import RunStore

    if until not in UNTIL_CHOICES:
        raise typer.BadParameter(f"--until must be one of {UNTIL_CHOICES}")
    if not concept:
        raise typer.BadParameter("--concept is required until the Planner lands (M5)")
    settings = Settings(runs_dir=runs_dir) if runs_dir else Settings()
    store = RunStore(settings.runs_dir)
    state = store.create_run(video.name, lang=lang, capture_fps=capture_fps)  # type: ignore[arg-type]
    shutil.copyfile(video, store.input_video_path(state.run_id))
    spec = manual_spec(
        experiment, concept, alt_concept or [], ref_concept, ref_length_m, ref_frame_s, window, lang
    )
    typer.echo(f"Run {state.run_id} → {store.run_dir(state.run_id)}")
    final = asyncio.run(execute_run(state.run_id, settings, spec_override=spec, until=Stage(until)))
    run_dir = store.run_dir(state.run_id)
    if final.stage == Stage.FAILED:
        typer.secho(f"FAILED: {final.error}", fg=typer.colors.RED)
        typer.echo(f"Details: {run_dir / 'error.json'}")
        raise typer.Exit(code=1)
    typer.secho(
        f"Stopped after {final.stage.value}. Cost ≈ ${final.cost_usd:.4f}", fg=typer.colors.GREEN
    )
    for name, rel in final.artifacts.items():
        typer.echo(f"  {name:<16} {run_dir / rel}")
    for warning in final.warnings:
        typer.secho(f"  ! {warning}", fg=typer.colors.YELLOW)


@app.command()
def replay(run_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)]) -> None:
    """Re-render outputs of a finished run without API calls (M8)."""
    _not_yet("M8")


@series_app.command("create")
def series_create(run: Annotated[list[str], typer.Option("--run", help="Run id (repeat)")]) -> None:
    """Create a series from finished pendulum runs (M4)."""
    _not_yet("M4")


@series_app.command("analyze")
def series_analyze(series_id: str) -> None:
    """Analyse a series: T0² vs L regression (M4)."""
    _not_yet("M4")


if __name__ == "__main__":  # pragma: no cover
    app()
