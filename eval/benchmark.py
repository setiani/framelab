"""Accuracy benchmark on self-recorded videos (milestone M12, docs/plan/08 §5).

Reads the finished runs in ``runs/`` and the ground truth in ``samples/ground_truth.json``
(keyed by the uploaded file name) and writes ``eval/results.md`` and ``eval/results.json``.
Every number comes from the runs' own records (state, verdict, ledger); nothing is typed in.

For each video the latest finished run is used, so re-uploading a video after a fix updates the
benchmark. Earlier runs of the same video are kept as repeat analyses (reproducibility). No
model is called: the runs were made with the app.

Usage (from the repository root)::

    uv run python eval/benchmark.py [--runs-dir runs] [--series s_...]

Standard library only, so it also runs without the project environment.
"""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone  # noqa: UP017 - runs on Python 3.10 too
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
G_REF = 9.781  # Jakarta, as in framelab.config.G_REF_DEFAULT
SKIP_FILES = {"freefall_60fps_5s.mp4", "pendulum_60fps_12s.mp4"}  # synthetic test clips


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def latest_gates(run_dir: Path) -> dict[str, dict[str, Any]]:
    """The last result of each gate in the run's ledger."""
    gates: dict[str, dict[str, Any]] = {}
    ledger = run_dir / "analysis" / "ledger.jsonl"
    if ledger.is_file():
        for line in ledger.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            if str(entry.get("id", "")).startswith("gate_"):
                data = entry.get("data", {})
                gates[data.get("gate", entry["id"])] = data
    return gates


def run_rows(runs_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for state_path in sorted(runs_dir.glob("r_*/state.json")):
        state = load(state_path)
        if state.get("replay_of") or state.get("demo"):
            continue
        if state.get("stage") not in ("done", "failed"):
            continue
        name = state.get("original_filename", "")
        if name in SKIP_FILES:
            continue
        run_dir = state_path.parent
        gates = latest_gates(run_dir)
        verdict_path = run_dir / "analysis" / "verdict.json"
        verdict = load(verdict_path) if verdict_path.is_file() else {}
        ingest_path = run_dir / "ingest" / "ingest.json"
        ingest = load(ingest_path) if ingest_path.is_file() else {}
        designer_path = run_dir / "analysis" / "designer.json"
        designer = load(designer_path) if designer_path.is_file() else {}
        rows.append(
            {
                "run_id": state["run_id"],
                "file": name,
                "created_at": state["created_at"],
                "experiment": state.get("experiment"),
                "verdict": state.get("verdict") or state.get("stage"),
                "headline": state.get("headline"),
                "cost_usd": state.get("cost_usd", 0.0),
                "failed_gates": sorted(g for g, d in gates.items() if d.get("status") == "fail"),
                "warned_gates": sorted(g for g, d in gates.items() if d.get("status") == "warn"),
                "gate_diagnoses": {
                    g: d.get("diagnosis") for g, d in gates.items() if d.get("diagnosis")
                },
                "auditor_model": verdict.get("auditor_model"),
                "designer_models": designer.get("models") or [],
                "fps": ingest.get("effective_fps"),
            }
        )
    return rows


def pct(x: float | None, digits: int = 2) -> str:
    return "–" if x is None else f"{x:+.{digits}f} %"


def summarise(runs_dir: Path, truth: dict[str, Any], series_ids: list[str]) -> dict[str, Any]:
    rows = run_rows(runs_dir)
    by_file: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_file.setdefault(row["file"], []).append(row)
    entries = []
    for name, spec in truth.items():
        if name.startswith("_") or name not in by_file:
            continue
        runs = sorted(by_file[name], key=lambda r: r["created_at"])
        latest = runs[-1]
        hl = latest["headline"] or {}
        value = hl.get("value") if hl.get("quantity", "g") == "g" else None
        lo, hi = (hl.get("ci95") or [None, None])[:2]
        entry = {
            "file": name,
            **{k: v for k, v in spec.items() if k != "note"},
            "note": spec.get("note"),
            "run_id": latest["run_id"],
            "verdict": latest["verdict"],
            "g": value,
            "sigma_g": hl.get("sigma") if value is not None else None,
            "formatted": hl.get("formatted_en"),
            "error_pct": None if value is None else (value - G_REF) / G_REF * 100,
            "ref_in_ci95": None if value is None or lo is None else lo <= G_REF <= hi,
            "failed_gates": latest["failed_gates"],
            "warned_gates": latest["warned_gates"],
            "diagnoses": latest["gate_diagnoses"],
            "cost_usd": latest["cost_usd"],
            "fps": latest["fps"],
            "designer_models": latest["designer_models"],
            "earlier_runs": [
                {
                    "run_id": r["run_id"],
                    "verdict": r["verdict"],
                    "g": (r["headline"] or {}).get("value"),
                }
                for r in runs[:-1]
            ],
        }
        if spec.get("purpose") == "trap":
            expected = set(spec.get("expected_failed_gates") or [])
            entry["trap_caught"] = latest["verdict"] == "NEEDS_RERECORD" and expected <= set(
                latest["failed_gates"]
            )
        entries.append(entry)

    def measured(exp: str) -> list[dict[str, Any]]:
        return [e for e in entries if e["experiment"] == exp and e["purpose"] == "measure"]

    groups = {}
    for exp in ("pendulum", "free_fall"):
        group = measured(exp)
        passed = [e for e in group if e["verdict"] in ("PASS", "PASS_WITH_WARNINGS")]
        errors = [abs(e["error_pct"]) for e in passed if e["error_pct"] is not None]
        groups[exp] = {
            "videos": len(group),
            "passed": len(passed),
            "refused": len(group) - len(passed),
            "mean_abs_error_pct": statistics.fmean(errors) if errors else None,
            "max_abs_error_pct": max(errors) if errors else None,
            "ref_in_ci95": sum(1 for e in passed if e["ref_in_ci95"]),
        }
    traps = [e for e in entries if e["purpose"] == "trap"]
    series = []
    for sid in series_ids:
        sdir = runs_dir / "series" / sid
        if not (sdir / "series.json").is_file():
            continue
        state = load(sdir / "series.json")
        result_path = sdir / "analysis" / "series.json"
        result = load(result_path) if result_path.is_file() else {}
        hl = state.get("headline") or {}
        series.append(
            {
                "series_id": sid,
                "status": state.get("status"),
                "runs": state.get("run_ids"),
                "lengths_m": hl.get("lengths_m"),
                "n": result.get("n"),
                "sigma_n": result.get("sigma_n"),
                "g": result.get("g"),
                "sigma_g": result.get("sigma_g"),
                "ci_g": result.get("ci_g"),
                "sqrt_law_supported": result.get("sqrt_law_supported"),
            }
        )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),  # noqa: UP017
        "g_ref": G_REF,
        "groups": groups,
        "traps": {"total": len(traps), "caught": sum(1 for t in traps if t.get("trap_caught"))},
        "entries": entries,
        "series": series,
        "total_cost_usd": sum(e["cost_usd"] for e in entries),
    }


def markdown(result: dict[str, Any]) -> str:
    g = result["groups"]
    fps = sorted({round(e["fps"]) for e in result["entries"] if e.get("fps")})
    fps_text = ", ".join(str(f) for f in fps) if fps else "unknown"
    out = [
        "# Benchmark results",
        "",
        f"Generated by `eval/benchmark.py` on {result['generated_at']} from the finished runs in "
        "`runs/` and the ground truth in `samples/ground_truth.json`. Reference "
        f"g = {result['g_ref']} m/s² (Jakarta). All videos were recorded at home with one phone "
        f"by the author during the hackathon, at {fps_text} fps.",
        "",
        "## Summary",
        "",
        "| | Videos | Passed | Refused | Mean \\|error\\| | Max \\|error\\| "
        "| Reference inside 95 % CI |",
        "|---|---|---|---|---|---|---|",
    ]
    for exp, label in (("pendulum", "Pendulum"), ("free_fall", "Free fall")):
        s = g[exp]
        mean = "–" if s["mean_abs_error_pct"] is None else f"{s['mean_abs_error_pct']:.2f} %"
        mx = "–" if s["max_abs_error_pct"] is None else f"{s['max_abs_error_pct']:.2f} %"
        out.append(
            f"| {label} | {s['videos']} | {s['passed']} | {s['refused']} | {mean} | {mx} | "
            f"{s['ref_in_ci95']} of {s['passed']} |"
        )
    t = result["traps"]
    out += [
        f"| Trap recordings (should be refused) | {t['total']} | – | {t['caught']} caught | – | – "
        "| – |",
        "",
        "## Every video (latest run)",
        "",
        "| Video | fps | Purpose | Verdict | g | Error | Failed gates | Note |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for e in result["entries"]:
        purpose = (
            e["purpose"]
            if e["purpose"] != "trap"
            else ("trap, caught" if e.get("trap_caught") else "trap, missed")
        )
        failed = ", ".join(f"{gate} ({e['diagnoses'].get(gate, '')})" for gate in e["failed_gates"])
        out.append(
            f"| `{e['file']}` | {round(e['fps']) if e.get('fps') else '–'} | {purpose} | "
            f"{e['verdict']} | {e['formatted'] or '–'} | "
            f"{pct(e['error_pct'])} | {failed or '–'} | {e['note'] or ''} |"
        )
    repeats = [e for e in result["entries"] if e["earlier_runs"]]
    if repeats:
        out += ["", "## Repeat analyses of the same video", ""]
        for e in repeats:
            earlier = "; ".join(
                f"{r['verdict']}" + (f", g = {r['g']:.3f}" if r["g"] else "")
                for r in e["earlier_runs"]
            )
            latest = f"g = {e['g']:.3f}" if e["g"] is not None else e["verdict"]
            out.append(f"- `{e['file']}`: latest {latest}; earlier: {earlier}")
    if result["series"]:
        out += ["", "## Pendulum series", ""]
        for s in result["series"]:
            lengths = ", ".join(f"{x * 100:.0f} cm" for x in s["lengths_m"] or [])
            lo, hi = s["ci_g"] or [None, None]
            out.append(
                f"- `{s['series_id']}` ({lengths}): exponent n = {s['n']:.3f} ± {s['sigma_n']:.3f} "
                f"(theory 0.5; √L law {'supported' if s['sqrt_law_supported'] else 'not supported'}"
                f"); g from the slope = {s['g']:.2f} ± {s['sigma_g']:.2f} m/s², 95 % interval "
                f"{lo:.2f}–{hi:.2f}."
            )
    ff = [
        e
        for e in result["entries"]
        if e["experiment"] == "free_fall"
        and e["purpose"] == "measure"
        and e["error_pct"] is not None
    ]
    ff_errors = [e["error_pct"] for e in ff]
    if not ff:
        ff_range = "–"
    elif min(ff_errors) == max(ff_errors):
        ff_range = f"{ff_errors[0]:+.1f} %"
    else:
        ff_range = f"{min(ff_errors):+.1f} % to {max(ff_errors):+.1f} %"
    out += [
        "",
        "## Reading these results",
        "",
        "- The pendulum measures g from time (the period) and the spoken string length, so it does "
        "not depend on a spatial scale. It is the reliable experiment on an ordinary phone.",
        "- Free fall needs a spatial scale and a fast camera. The accepted free-fall runs were "
        f"off by {ff_range}; FrameLab flagged them with a plausibility warning (G4) but did not "
        "refuse them, because only one scale (the spoken ball diameter) was available.",
        "- G2 can only check the scale when two scales exist (a ruler of known length and the "
        "ball diameter). A trap without a ruler of known length cannot be caught by G2.",
        "- Runs marked as analysed before a fix were not re-run to save API credits.",
        f"- Model cost of the runs in this table: ${result['total_cost_usd']:.2f}.",
        "",
    ]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runs")
    parser.add_argument("--truth", type=Path, default=ROOT / "samples" / "ground_truth.json")
    parser.add_argument("--series", action="append", default=[], metavar="SERIES_ID")
    parser.add_argument("--out", type=Path, default=ROOT / "eval")
    args = parser.parse_args(argv)
    result = summarise(args.runs_dir, load(args.truth), args.series)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (args.out / "results.md").write_text(markdown(result), encoding="utf-8")
    print(markdown(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
