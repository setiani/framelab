"""Copy selected finished runs into demo_runs/ for the "Try a sample" cards (milestone M11).

Spec: docs/plan/07-backend-api-and-frontend.md §3.

Usage (from the repository root)::

    uv run python scripts/build_demo_cache.py \\
        --sample free_fall=r_20261101_093012_ab12cd \\
        --sample pendulum=r_20261101_101544_ef34ab \\
        --sample trap=r_20261101_104210_cd56ef \\
        [--series s_20261101_110000_aa11bb]

Each run is copied without what the viewer never reads (raw SAM masks, the analysis copy of the
video, debug images and the uploaded original), marked as a sample, and listed in
``demo_runs/samples.json``. A series is copied with its member runs. The previous cache is
replaced. The script refuses runs that are not finished and caches that contain the API key, and
warns about absolute paths and about a total size over 50 MB (the limit for committing to git).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from framelab.config import Settings  # noqa: E402
from framelab.schemas import RunState, SeriesState  # noqa: E402
from framelab.store import RunNotFoundError, RunStore, read_json, write_json_atomic  # noqa: E402

GIT_LIMIT_MB = 50
SKIP_DIRS = {"debug", "input"}
SKIP_FILES = {"ingest/analysis.mp4"}
TEXT_SUFFIXES = {".json", ".jsonl", ".csv", ".txt", ".md", ".html"}

DEFAULT_TEXT: dict[str, dict[str, dict[str, str]]] = {
    "free_fall": {
        "title": {"en": "Free fall", "id": "Jatuh bebas"},
        "blurb": {
            "en": "A dropped ball, filmed in slow motion. The result is g with its uncertainty.",
            "id": "Bola dijatuhkan dan direkam slow motion. Hasilnya g beserta ketidakpastiannya.",
        },
    },
    "pendulum": {
        "title": {"en": "Pendulum", "id": "Bandul"},
        "blurb": {
            "en": "A pendulum with a spoken string length. The period gives g.",
            "id": "Bandul dengan panjang tali yang diucapkan. Periodenya menghasilkan g.",
        },
    },
    "trap": {
        "title": {"en": "Bad recording", "id": "Rekaman bermasalah"},
        "blurb": {
            "en": "The ruler is closer to the camera than the motion. The auditor asks for a "
            "new recording.",
            "id": "Penggaris lebih dekat ke kamera daripada gerakannya. Auditor meminta "
            "rekaman ulang.",
        },
    },
}


class CacheError(Exception):
    pass


def keep(relative: Path) -> bool:
    parts = relative.parts
    if not parts or parts[0] in SKIP_DIRS or relative.as_posix() in SKIP_FILES:
        return False
    if parts[0] == "sam" and relative.name.endswith(".sam.json"):
        return False  # raw masks: large and not read by the viewer
    return not relative.name.startswith(".")  # temp files of atomic writes


def copy_run(src: Path, dst: Path, sample_key: str | None) -> RunState:
    state = RunState.model_validate_json((src / "state.json").read_text(encoding="utf-8"))
    if state.stage.value not in ("done", "failed"):
        raise CacheError(f"{state.run_id} is not finished (stage {state.stage.value})")
    if state.replay_of or state.demo:
        raise CacheError(f"{state.run_id} is a sample or a replay; cache the original run")
    for path in sorted(src.rglob("*")):
        rel = path.relative_to(src)
        if path.is_file() and keep(rel):
            target = dst / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    state.demo = True
    state.sample = sample_key
    write_json_atomic(dst / "state.json", state)
    return state


def scan(out: Path, secrets: list[str], local_paths: list[str]) -> list[str]:
    """Abort on secrets; return warnings for absolute local paths."""
    warnings: list[str] = []
    for path in out.rglob("*"):
        if not path.is_file() or path.suffix not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for secret in secrets:
            if secret and secret in text:
                raise CacheError(f"{path.relative_to(out)} contains the API key; not cached")
        for local in local_paths:
            if local and local in text:
                warnings.append(f"{path.relative_to(out)} mentions the local path {local!r}")
                break
    return warnings


def build(runs_dir: Path, out: Path, samples: list[tuple[str, str]],
          series_ids: list[str]) -> dict:  # fmt: skip
    store = RunStore(runs_dir)
    staging = out.with_name(out.name + ".building")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    manifest: list[dict] = []
    copied: dict[str, RunState] = {}
    try:
        for key, run_id in samples:
            src = store.run_dir(run_id)
            if not (src / "state.json").is_file():
                raise CacheError(f"no run {run_id} in {runs_dir}")
            state = copy_run(src, staging / run_id, key)
            copied[run_id] = state
            text = DEFAULT_TEXT.get(key, {})
            manifest.append({
                "key": key,
                "run_id": run_id,
                "title": text.get("title", {"en": state.original_filename,
                                             "id": state.original_filename}),
                "blurb": text.get("blurb", {"en": "", "id": ""}),
            })  # fmt: skip
            if key == "trap" and state.verdict != "NEEDS_RERECORD":
                # The card says the auditor asked for a new recording; a passing run would
                # make that card untrue.
                raise CacheError(
                    f"trap sample {run_id} has verdict {state.verdict}; the trap card needs a "
                    "run that ended as NEEDS_RERECORD"
                )
        for series_id in series_ids:
            try:
                series = store.load_series(series_id)
            except RunNotFoundError as exc:
                raise CacheError(f"no series {series_id} in {runs_dir}") from exc
            if series.status != "done":
                raise CacheError(f"series {series_id} is not analysed (status {series.status})")
            for run_id in series.run_ids:
                if run_id not in copied:
                    copied[run_id] = copy_run(store.run_dir(run_id), staging / run_id, None)
            sdst = staging / "series" / series_id
            shutil.copytree(store.series_dir(series_id), sdst)
            SeriesState.model_validate(read_json(sdst / "series.json"))
        write_json_atomic(staging / "samples.json", {"samples": manifest})
        readme = out / "README.md"
        if readme.is_file():
            shutil.copyfile(readme, staging / "README.md")
        key = Settings().model_api_key
        local = [str(ROOT), str(Path.home())]
        local += [p.replace("\\", "\\\\") for p in local if "\\" in p]  # as escaped in JSON
        warnings = scan(staging, [key] if key else [], local)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    shutil.rmtree(out, ignore_errors=True)
    staging.rename(out)
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    return {"runs": len(copied), "samples": len(manifest), "series": len(series_ids),
            "bytes": size, "warnings": warnings}  # fmt: skip


def parse_sample(value: str) -> tuple[str, str]:
    key, sep, run_id = value.partition("=")
    if not sep or not key or not run_id.startswith("r_"):
        raise argparse.ArgumentTypeError("use KEY=RUN_ID, e.g. free_fall=r_20261101_093012_ab12cd")
    return key.strip(), run_id.strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--sample",
        action="append",
        type=parse_sample,
        default=[],
        metavar="KEY=RUN_ID",
        help="free_fall, pendulum or trap (repeat)",
    )
    parser.add_argument("--series", action="append", default=[], metavar="SERIES_ID")
    parser.add_argument("--runs-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=ROOT / "demo_runs")
    args = parser.parse_args(argv)  # fmt: skip
    if not args.sample:
        parser.error("give at least one --sample KEY=RUN_ID")
    runs_dir = args.runs_dir or Settings().runs_dir
    try:
        result = build(runs_dir, args.out, args.sample, args.series)
    except CacheError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    mb = result["bytes"] / 1e6
    print(json.dumps({k: v for k, v in result.items() if k != "warnings"}))
    print(f"cached {result['samples']} samples ({result['runs']} runs) in {args.out}: {mb:.1f} MB")
    for warning in result["warnings"]:
        print(f"warning: {warning}")
    if mb > GIT_LIMIT_MB:
        print(f"warning: over {GIT_LIMIT_MB} MB; shorten the clips or do not commit demo_runs/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
