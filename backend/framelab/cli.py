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


@app.command()
def run(
    video: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="Input video")],
    lang: Annotated[str, typer.Option(help="Report language: id | en")] = "id",
    capture_fps: Annotated[int | None, typer.Option(help="Real capture fps for slow-mo")] = None,
    resume: Annotated[str | None, typer.Option(help="Resume an existing run id")] = None,
    no_agents: Annotated[bool, typer.Option(help="Bypass agents (M3 dev mode)")] = False,
) -> None:
    """Run the full pipeline on a video (implemented from M2 onwards)."""
    _not_yet("M2–M8")


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
