"""Pendulum series: several runs with different string lengths → T0² vs L → g and T ∝ L^n.

Spec: docs/plan/05-physics-and-uncertainty.md §4, 07 §2 (series endpoints), 03 §8. Used by
``framelab series create/analyze`` now and by the API in M8. Numbers go through the series'
own Claim Ledger (``series/<id>/analysis/ledger.jsonl``).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from framelab.agent.ledger import Ledger
from framelab.config import G_REF_DEFAULT
from framelab.physics.derive import compare_reference, pendulum_series
from framelab.physics.uncertainty import fmt
from framelab.schemas import SeriesState, Stage
from framelab.store import RunStore, write_json_atomic

BAD_VERDICTS = {"NEEDS_RERECORD", "REJECT"}


class SeriesError(ValueError):
    """A run cannot join the series, or the series cannot be analysed."""


def run_point(store: RunStore, run_id: str) -> dict[str, Any]:
    """(L, T0, σ_T, σ_L) of one finished pendulum run, or SeriesError with the reason."""
    state = store.load_state(run_id)
    if state.experiment != "pendulum":
        raise SeriesError(f"{run_id}: not a pendulum run ({state.experiment})")
    if state.stage == Stage.FAILED or state.verdict in BAD_VERDICTS:
        raise SeriesError(f"{run_id}: run did not pass ({state.verdict or state.stage.value})")
    path = store.run_dir(run_id) / "analysis" / "pendulum.json"
    h = state.headline or {}
    if path.is_file():  # deterministic path (--no-planner, Analyst fallback)
        data = json.loads(path.read_text(encoding="utf-8"))
        point = {"L": data.get("length_m"), "sigma_L": data.get("sigma_length_m"),
                 "T0": data.get("period0"), "sigma_T": data.get("sigma_period0"),
                 "length_source": data.get("length_source"),
                 "theta_mean_deg": data.get("theta_mean_deg")}  # fmt: skip
    elif h.get("experiment") == "pendulum":  # agent path: the headline holds the same values
        point = {"L": h.get("length_m"), "sigma_L": h.get("length_sigma_m"),
                 "T0": h.get("period0"), "sigma_T": h.get("period0_sigma"),
                 "length_source": h.get("length_source"),
                 "theta_mean_deg": h.get("theta_mean_deg")}  # fmt: skip
    else:
        raise SeriesError(f"{run_id}: no pendulum analysis yet")
    if not point["L"]:
        raise SeriesError(f"{run_id}: string length unknown (needed for the series)")
    return {
        "run_id": run_id,
        "L": float(point["L"]),
        "sigma_L": float(point["sigma_L"] or 0.0),
        "T0": float(point["T0"]),
        "sigma_T": float(point["sigma_T"]),
        "length_source": point["length_source"],
        "theta_mean_deg": point["theta_mean_deg"],
    }


def create(store: RunStore, run_ids: list[str]) -> SeriesState:
    for rid in run_ids:
        run_point(store, rid)  # validate early, with a clear message
    series = store.create_series(run_ids)
    for rid in run_ids:
        state = store.load_state(rid)
        state.series_id = series.series_id
        store.save_state(state)
    return series


def add_run(store: RunStore, series_id: str, run_id: str) -> SeriesState:
    series = store.load_series(series_id)
    run_point(store, run_id)
    if run_id not in series.run_ids:
        series.run_ids.append(run_id)
        state = store.load_state(run_id)
        state.series_id = series_id
        store.save_state(state)
    store.save_series(series)
    return series


def analyze(store: RunStore, series_id: str, *, n_boot: int = 2000) -> SeriesState:
    """Regression + ledger evidence + headline; status ``done`` (or ``failed`` with reason)."""
    series = store.load_series(series_id)
    sdir = store.series_dir(series_id)
    series.status = "analyzing"
    store.save_series(series)
    try:
        points = [run_point(store, rid) for rid in series.run_ids]
        result = pendulum_series(points, n_boot=n_boot)
    except (SeriesError, ValueError) as exc:
        series.status = "failed"
        series.headline = {"error": str(exc)}
        store.save_series(series)
        raise SeriesError(str(exc)) from exc
    result["runs"] = points
    out = sdir / "analysis" / "series.json"
    write_json_atomic(out, result)

    ledger_path = sdir / "analysis" / "ledger.jsonl"
    if ledger_path.exists():  # re-analysis replaces the previous evidence
        ledger_path.unlink()
    ledger = Ledger(ledger_path)
    trk = ledger.add_evidence(
        "trk",
        "series_points",
        {"points": points},
        f"{len(points)} pendulum runs, L = "
        + ", ".join(f"{p['L']:.3f}" for p in sorted(points, key=lambda p: p["L"]))
        + " m",
    )
    lin = result["linear"]
    fit = ledger.add_evidence(
        "fit",
        "fit_model",
        {"model": "weighted_linear", "y": "T0^2", "x": "L", **lin},
        f"T0² = {lin['k']:.4f}·L {lin['b']:+.4f} (χ²_red {lin['chi2_red']:.2f})",
        [trk.id],
    )
    pw = result["power_law"]
    fit_pw = ledger.add_evidence(
        "fit",
        "fit_model",
        {"model": "weighted_linear", "y": "ln T0", "x": "ln L", **pw},
        f"ln T0 = {pw['k']:.4f}·ln L {pw['b']:+.4f}",
        [trk.id],
    )
    ci = ledger.add_evidence(
        "ci",
        "pairs_bootstrap",
        {
            "quantity": "g",
            "lo": result["boot_ci_g"][0] if result["boot_ci_g"] else None,
            "hi": result["boot_ci_g"][1] if result["boot_ci_g"] else None,
            "n_lo": result["boot_ci_n"][0] if result["boot_ci_n"] else None,
            "n_hi": result["boot_ci_n"][1] if result["boot_ci_n"] else None,
            "n": result["n_boot_used"],
            "seed": 1234,
            "method": "pairs_bootstrap",
        },
        "pairs bootstrap over runs (g, n)",
        [fit.id, fit_pw.id],
    )
    der_g = ledger.add_evidence(
        "der",
        "derive",
        {
            "quantity": "g",
            "value": result["g"],
            "unit": "m/s^2",
            "sigma": result["sigma_g"],
            "ci95": result["ci_g"],
            "formula": "g = 4π²/k",
        },
        f"g = {fmt(result['g'], result['sigma_g'], 'm/s^2')} (slope of T0² vs L)",
        [fit.id, ci.id],
    )
    der_n = ledger.add_evidence(
        "der",
        "derive",
        {
            "quantity": "n",
            "value": result["n"],
            "unit": "",
            "sigma": result["sigma_n"],
            "ci95": result["ci_n"],
            "formula": "T0 ∝ L^n",
        },
        f"n = {fmt(result['n'], result['sigma_n'], '')} (T ∝ L^n)",
        [fit_pw.id, ci.id],
    )
    comparison = compare_reference(result["g"], *result["ci_g"], G_REF_DEFAULT)
    cmp = ledger.add_evidence(
        "cmp",
        "compare_reference",
        comparison,
        f"g differs from reference {G_REF_DEFAULT} m/s² by {comparison['rel_diff_pct']:+.2f}%",
        [der_g.id],
    )
    hyp = ledger.add_evidence(
        "cmp",
        "hypothesis_test",
        {
            "statement": "T ∝ √L",
            "p": 0.5,
            "m": result["n"],
            "lo": result["ci_n"][0],
            "hi": result["ci_n"][1],
            "outcome": "supported" if result["sqrt_law_supported"] else "not_supported",
            "rule": "|n − 0.5| ≤ 2σ_n",
        },
        f"T ∝ √L {'supported' if result['sqrt_law_supported'] else 'not supported'} "
        f"(n = {result['n']:.3f} ± {result['sigma_n']:.3f})",
        [der_n.id],
    )
    if "length_offset" in result["flags"]:
        ledger.add_evidence(
            "gate",
            "series_check",
            {
                "flag": "length_offset",
                "b": lin["b"],
                "sigma_b": lin["sigma_b"],
                "offset_m_estimate": -lin["b"] / lin["k"],
            },  # fmt: skip
            "intercept ≠ 0: lengths probably measured to the wrong point (top of the bob?)",
            [fit.id],
        )

    series.headline = {
        "quantity": "g",
        "value": result["g"],
        "sigma": result["sigma_g"],
        "ci95": result["ci_g"],
        "unit": "m/s^2",
        "evidence": der_g.id,
        "formatted_en": fmt(result["g"], result["sigma_g"], "m/s^2", "en"),
        "formatted_id": fmt(result["g"], result["sigma_g"], "m/s^2", "id"),
        "ref": G_REF_DEFAULT,
        "rel_diff_pct": comparison["rel_diff_pct"],
        "n": result["n"],
        "sigma_n": result["sigma_n"],
        "n_formatted": fmt(result["n"], result["sigma_n"], ""),
        "n_evidence": der_n.id,
        "sqrt_law": hyp.data["outcome"],
        "flags": result["flags"],
        "n_runs": result["n_runs"],
        "lengths_m": sorted(p["L"] for p in points),
        "comparison_evidence": cmp.id,
    }
    series.artifacts["series_json"] = str(Path("analysis") / "series.json")
    series.status = "done"
    store.save_series(series)
    if not math.isfinite(result["g"]):
        raise SeriesError("regression failed")
    return series
