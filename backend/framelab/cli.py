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
UNTIL_CHOICES = (
    "ingesting",
    "planning",
    "segmenting",
    "measuring",
    "analyzing",
    "auditing",
    "rendering",
    "designing",
)


def flag_fields(
    *,
    experiment: str | None,
    concept: str | None,
    alt_concepts: list[str],
    ref_concept: str | None,
    ref_length_m: float | None,
    ref_frame_s: float | None,
    window: str | None,
    ball_diameter_m: float | None,
    string_length_m: float | None,
    predict: str | None,
) -> dict:
    """ExperimentSpec fields for the flags the user actually gave (no defaults).

    Without ``--no-planner`` these override the Planner's spec. With it they make the spec.
    """
    from framelab.schemas import Quantity, ReferenceObject

    fields: dict = {}
    if experiment:
        fields["experiment"] = experiment
        fields["motion_axis"] = EXPERIMENT_AXIS.get(experiment, "xy")
    if concept:
        fields["target_concept"] = concept
    if alt_concepts:
        fields["target_alt_concepts"] = alt_concepts
    if ref_concept:
        fields["reference"] = ReferenceObject(
            concept=ref_concept,
            length_m=ref_length_m,
            length_source="spoken" if ref_length_m else "unknown",
            frame_hint_s=ref_frame_s,
        )
    if window:
        t0, t1 = (float(v) for v in window.split(","))
        fields["analysis_window_s"] = (t0, t1)
    if ball_diameter_m:
        fields["object_diameter_m"] = ball_diameter_m
    if string_length_m:
        fields["spoken_params"] = {"string_length": Quantity(value=string_length_m, unit="m")}
    if predict:
        fields["hypothesis"] = f"Prediction: {predict}"
        fields["hypothesis_prediction"] = parse_quantity(predict)
    return fields


def manual_spec(lang: str, fields: dict):
    """Spec for ``--no-planner``: the flags on top of safe defaults (no Voice or Spark calls)."""
    from framelab.agent.planner import apply_overrides
    from framelab.schemas import ExperimentSpec

    base = ExperimentSpec(
        experiment="free_fall",
        target_concept="",
        analysis_window_s=(0.0, 1e9),  # clamped to the video length after ingest
        motion_axis="y",
        language=lang,  # type: ignore[arg-type]
        setup_description="Manual spec from CLI flags (--no-planner).",
        confidence=1.0,
    )
    return apply_overrides(base, fields)


def parse_quantity(text: str):
    """'1.5 s' / '1,5s' / '9.8 m/s^2' → Quantity (unit must be known)."""
    import re

    from framelab.schemas import Quantity

    m = re.fullmatch(r"\s*([-+]?\d+(?:[.,]\d+)?)\s*([A-Za-z/^0-9]+)\s*", text)
    if not m:
        raise typer.BadParameter(f"cannot parse {text!r}; use e.g. '1.5 s' or '9.8 m/s^2'")
    q = Quantity(value=float(m.group(1).replace(",", ".")), unit=m.group(2))
    try:
        q.to_si()
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    return q


def print_headline(h: dict) -> None:
    if h.get("experiment") == "pendulum":
        label = "g" if h["quantity"] == "g" else "T"
        typer.secho(f"\n  {label} = {h['formatted_en']}", bold=True)
        unit = "m/s²" if h["quantity"] == "g" else "s"
        parts = [f"95% CI [{h['ci95'][0]:.4g}, {h['ci95'][1]:.4g}] {unit}"]
        if h.get("ref"):
            parts.append(f"reference {h['ref']} m/s² ({h['rel_diff_en']})")
        typer.echo("  " + " · ".join(parts))
        theta = h.get("theta_mean_deg")
        details = [
            f"T = {h['period_formatted_en']}",
            f"T0 = {h['period0']:.4f} s"
            + (f" (amplitude {theta:.1f}°)" if theta is not None else " (no amplitude correction)"),
            f"{h['n_cycles']:.1f} cycles",
            f"{h['fit_points']} points",
        ]
        if h.get("length_m"):
            details.append(f"L = {h['length_m']:.3f} m ({h['length_source']})")
        typer.echo("  " + " · ".join(details))
        if h.get("hypothesis_outcome"):
            typer.echo(f"  hypothesis: {h['hypothesis_outcome']}")
        typer.echo("")
        return
    agree = h.get("scale_agreement_pct")
    typer.secho(f"\n  g = {h['formatted_en']}", bold=True)
    typer.echo(
        f"  95% CI [{h['ci95'][0]:.3f}, {h['ci95'][1]:.3f}] m/s² · reference {h['ref']} m/s² "
        f"({h['rel_diff_en']}) · {h['fit_points']} points · scale: {h['scale_method']}"
        + (f" (2nd method agrees within {agree:.1f}%)" if agree is not None else "")
    )
    typer.echo("")


@app.command()
def run(
    video: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="Input video")],
    concept: Annotated[
        str | None,
        typer.Option(
            help="SAM noun phrase for the moving object (override; required with --no-planner)"
        ),
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
    ball_diameter_m: Annotated[
        float | None, typer.Option(help="Real diameter of the ball (m), second scale method")
    ] = None,
    string_length_m: Annotated[
        float | None,
        typer.Option(help="Pendulum length pivot → bob centre (m), as if spoken"),
    ] = None,
    predict: Annotated[
        str | None, typer.Option(help="Hypothesis prediction, e.g. '1.5 s' or '9.8 m/s^2'")
    ] = None,
    experiment: Annotated[
        str | None,
        typer.Option(help="free_fall | pendulum (default: the Planner decides)"),
    ] = None,
    no_planner: Annotated[
        bool,
        typer.Option(
            "--no-planner",
            help="Skip Voice and Spark: the spec comes from the flags only (needs --concept).",
        ),
    ] = False,
    lang: Annotated[str, typer.Option(help="Report language: id | en")] = "id",
    capture_fps: Annotated[int | None, typer.Option(help="Real capture fps for slow-mo")] = None,
    until: Annotated[
        str, typer.Option(help=f"Stop after stage: {', '.join(UNTIL_CHOICES)}")
    ] = "analyzing",
    runs_dir: Annotated[Path | None, typer.Option(help="Runs folder (default: settings)")] = None,
) -> None:
    """Run the pipeline: ingest → transcript → Planner → SAM → observations → analysis.

    The Planner (Muse Voice + Muse Spark) reads the video and the speech and proposes the
    experiment spec. Flags override its fields. ``--no-planner`` uses the flags only, for offline
    runs and replays. Analysis stays deterministic until the agents land (M6)."""
    import asyncio

    from framelab.pipeline.run import execute_run
    from framelab.schemas import Stage
    from framelab.store import RunStore

    if until not in UNTIL_CHOICES:
        raise typer.BadParameter(f"--until must be one of {UNTIL_CHOICES}")
    if no_planner and not concept:
        raise typer.BadParameter("--concept is required with --no-planner")
    if experiment and experiment not in ("free_fall", "pendulum"):
        raise typer.BadParameter("--experiment must be free_fall or pendulum")
    settings = Settings(runs_dir=runs_dir) if runs_dir else Settings()
    store = RunStore(settings.runs_dir)
    state = store.create_run(video.name, lang=lang, capture_fps=capture_fps)  # type: ignore[arg-type]
    shutil.copyfile(video, store.input_video_path(state.run_id))
    fields = flag_fields(
        experiment=experiment,
        concept=concept,
        alt_concepts=alt_concept or [],
        ref_concept=ref_concept,
        ref_length_m=ref_length_m,
        ref_frame_s=ref_frame_s,
        window=window,
        ball_diameter_m=ball_diameter_m,
        string_length_m=string_length_m,
        predict=predict,
    )
    typer.echo(f"Run {state.run_id} → {store.run_dir(state.run_id)}")
    if no_planner:
        run = execute_run(
            state.run_id,
            settings,
            spec_override=manual_spec(lang, fields),
            until=Stage(until),
        )
    else:
        run = execute_run(state.run_id, settings, overrides=fields, until=Stage(until))
    final = asyncio.run(run)
    run_dir = store.run_dir(state.run_id)
    if final.stage == Stage.FAILED:
        typer.secho(f"FAILED: {final.error}", fg=typer.colors.RED)
        typer.echo(f"Details: {run_dir / 'error.json'}")
        raise typer.Exit(code=1)
    typer.secho(
        f"Stopped after {final.stage.value}. Cost ≈ ${final.cost_usd:.4f}", fg=typer.colors.GREEN
    )
    if final.verdict:
        typer.secho(f"Verdict: {final.verdict}", fg=typer.colors.YELLOW)
    if final.headline:
        print_headline(final.headline)
    for name, rel in final.artifacts.items():
        typer.echo(f"  {name:<16} {run_dir / rel}")
    for warning in final.warnings:
        typer.secho(f"  ! {warning}", fg=typer.colors.YELLOW)


@app.command()
def replay(
    run_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    overlay: Annotated[bool, typer.Option(help="Also render outputs/overlay.mp4")] = True,
) -> None:
    """Re-render the outputs (charts, data.csv, overlay, PDFs) from a run's saved files.

    No model is called: everything is read from the run folder and the ledger."""
    from framelab.pipeline.outputs import render_outputs

    result = render_outputs(run_dir, with_overlay=overlay)
    for name, path in sorted(result.paths.items()):
        typer.echo(f"  {name:<22} {path}")
    for warning in result.warnings:
        typer.secho(f"  ! {warning}", fg=typer.colors.YELLOW)
    if not result.paths:
        raise typer.Exit(code=1)


def _store(runs_dir: Path | None):
    from framelab.store import RunStore

    settings = Settings(runs_dir=runs_dir) if runs_dir else Settings()
    return RunStore(settings.runs_dir)


@series_app.command("create")
def series_create(
    run: Annotated[list[str], typer.Option("--run", help="Pendulum run id (repeat, ≥ 3)")],
    runs_dir: Annotated[Path | None, typer.Option(help="Runs folder (default: settings)")] = None,
) -> None:
    """Create a series from finished pendulum runs with known string lengths."""
    from framelab.pipeline import series as series_mod

    store = _store(runs_dir)
    try:
        s = series_mod.create(store, run)
    except (series_mod.SeriesError, KeyError) as exc:
        typer.secho(f"Cannot create the series: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc
    typer.echo(s.series_id)
    typer.secho(
        f"Series {s.series_id} with {len(s.run_ids)} runs. "
        f"Next: framelab series analyze {s.series_id}",
        fg=typer.colors.GREEN,
        err=True,
    )


@series_app.command("add")
def series_add(
    series_id: str,
    run: Annotated[str, typer.Option("--run", help="Pendulum run id")],
    runs_dir: Annotated[Path | None, typer.Option(help="Runs folder (default: settings)")] = None,
) -> None:
    """Add one more finished pendulum run to a series."""
    from framelab.pipeline import series as series_mod

    store = _store(runs_dir)
    try:
        s = series_mod.add_run(store, series_id, run)
    except (series_mod.SeriesError, KeyError) as exc:
        typer.secho(f"Cannot add the run: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc
    typer.echo(f"{s.series_id}: {len(s.run_ids)} runs")


@series_app.command("analyze")
def series_analyze(
    series_id: str,
    runs_dir: Annotated[Path | None, typer.Option(help="Runs folder (default: settings)")] = None,
) -> None:
    """Analyse a series: weighted T0² vs L → g, and T ∝ L^n."""
    from framelab.pipeline import series as series_mod

    store = _store(runs_dir)
    try:
        s = series_mod.analyze(store, series_id)
    except (series_mod.SeriesError, KeyError) as exc:
        typer.secho(f"Series analysis failed: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc
    h = s.headline or {}
    typer.secho(f"\n  g = {h['formatted_en']}  (slope of T0² vs L, {h['n_runs']} runs)", bold=True)
    typer.echo(
        f"  95% CI [{h['ci95'][0]:.3f}, {h['ci95'][1]:.3f}] m/s² · reference {h['ref']} m/s² "
        f"({h['rel_diff_pct']:+.2f}%)"
    )
    typer.echo(f"  T ∝ L^n with n = {h['n_formatted']} → √L law {h['sqrt_law']}")
    for flag in h["flags"]:
        typer.secho(f"  ! {flag}", fg=typer.colors.YELLOW)
    typer.echo(f"\n  {store.series_dir(series_id) / 'analysis' / 'series.json'}")


if __name__ == "__main__":  # pragma: no cover
    app()
