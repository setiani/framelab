"""Deterministic analysis (no LLM): track → calibration → physics → ledger evidence.

Used by ``framelab run`` until the Analyst agent lands (M6), and afterwards as the Analyst's
deterministic fallback (docs/plan/06-agents-and-verification.md §3.4). Every number written
here goes through the Claim Ledger as evidence (06 §1).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from framelab.agent.ledger import Ledger
from framelab.clients.sam import SamOutput
from framelab.config import G_REF_DEFAULT
from framelab.physics.derive import hypothesis_test
from framelab.physics.experiments import REGISTRY
from framelab.physics.freefall import analyze_free_fall
from framelab.physics.pendulum import analyze_pendulum
from framelab.physics.uncertainty import fmt, fmt_pct
from framelab.pipeline import calibrate, measure
from framelab.pipeline.context import RunContext
from framelab.pipeline.segment import SegmentationOutput
from framelab.schemas import (
    Calibration,
    ExperimentSpec,
    IngestResult,
    Observation,
    ScaleEstimate,
    Track,
)
from framelab.store import write_json_atomic


class AnalysisError(RuntimeError):
    """The data cannot support a measurement (becomes NEEDS_RERECORD from M7)."""


def reference_record(out: SamOutput | None) -> Any | None:
    if out is None:
        return None
    masks = measure.latest_masks(out.result)
    if not masks:
        return None
    # The longest mask is the reference (ruler, tape, paper).
    return max(
        masks,
        key=lambda r: (r.bounds.right - r.bounds.left) ** 2 + (r.bounds.bottom - r.bounds.top) ** 2,
    )


def calibrate_run(
    ctx: RunContext,
    spec: ExperimentSpec,
    ing: IngestResult,
    seg: SegmentationOutput,
    target_obs: list[Observation],
    ledger: Ledger,
    extra: list[ScaleEstimate] | None = None,
    inputs: list[str] | None = None,
) -> tuple[Calibration | None, dict[str, str]]:
    """Ruler (primary) and object-diameter / spoken-length (secondary) scales → ``cal_``."""
    estimates: list[ScaleEstimate] = list(extra or [])
    ids: dict[str, str] = {}
    ref = reference_record(seg.reference)
    if ref is not None and spec.reference and spec.reference.length_m:
        est = calibrate.ruler_scale(
            ref,
            spec.reference.length_m,
            frame_long_edge_px=max(ing.norm_width, ing.norm_height),
        )
        estimates.append(est)
        if est.details.get("warning"):
            ctx.emit(
                "warning", message="Scale reference is short (<15% of the frame): lower precision"
            )
    elif spec.reference and not spec.reference.length_m:
        ctx.emit(
            "warning", message="Reference length unknown: say or type it to use the ruler scale"
        )
    diameter = spec.object_diameter_m or (
        spec.spoken_params["ball_diameter"].to_si()
        if "ball_diameter" in spec.spoken_params
        else None
    )
    if diameter and target_obs:
        estimates.append(
            calibrate.diameter_scale(
                target_obs, diameter, dilation_px=ctx.settings.mask_dilation_px
            )
        )
    if not estimates:
        return None, ids
    cal = calibrate.combine(estimates)
    for est in [cal.primary, *([cal.secondary] if cal.secondary else [])]:
        ev = ledger.add_evidence(
            "cal",
            "calibrate_scale",
            {
                "method": est.method,
                "m_per_px": est.m_per_px,
                "sigma": est.sigma_m_per_px,
                "rel_sigma": est.sigma_m_per_px / est.m_per_px,
                "details": est.details,
            },
            f"{est.method} scale {est.m_per_px * 1000:.4f} mm/px "
            f"(±{est.sigma_m_per_px / est.m_per_px * 100:.2f}%)",
            inputs if est.method == "spoken_length" else None,
        )
        ids[est.method] = ev.id
    write_json_atomic(ctx.run_dir / "calibration.json", cal)
    return cal, ids


def run(
    ctx: RunContext,
    spec: ExperimentSpec,
    ing: IngestResult,
    seg: SegmentationOutput,
    observations: list[Observation],
    summary: dict[str, Any],
) -> dict[str, Any] | None:
    """Deterministic analysis end to end. Returns the headline (or None if no measurement)."""
    if spec.experiment == "free_fall":
        return run_free_fall(ctx, spec, ing, seg, observations, summary)
    if spec.experiment == "pendulum":
        return run_pendulum(ctx, spec, ing, seg, observations, summary)
    raise NotImplementedError(f"{spec.experiment} analysis is not implemented (stretch goal)")


def build_track(
    ctx: RunContext,
    ing: IngestResult,
    seg: SegmentationOutput,
    observations: list[Observation],
    summary: dict[str, Any],
    ledger: Ledger,
) -> tuple[Track, list[Observation], str, dict[str, Any]]:
    """Clean the target track, write tracks/<id>.csv and the ``trk_`` evidence."""
    target = summary.get("target") or {}
    obj = target.get("object_id")
    if not obj or seg.target is None:
        raise AnalysisError(
            "No usable target track (re-record: brighter light, contrasting object)"
        )
    target_obs = [o for o in observations if o.concept == seg.target.concept and o.object_id == obj]
    track = measure.clean_track(
        target_obs,
        effective_fps=ing.effective_fps,
        concept=seg.target.concept,
        object_id=obj,
        id_switches=len(target.get("merged_ids") or []),
    )
    track_csv = ctx.run_dir / "tracks" / f"{track.track_id}.csv"
    measure.write_track_csv(track_csv, track)
    ctx.add_artifact("track", track_csv, "text/csv")
    trk = ledger.add_evidence(
        "trk",
        "measure_track",
        {
            "track_id": track.track_id,
            "n_points": len(track.points),
            "n_valid": sum(p.valid for p in track.points),
            "coverage": track.coverage,
            "n_outliers": track.n_outliers,
            "n_interpolated": track.n_interpolated,
            "max_gap_frames": track.max_gap_frames,
            "id_switches": track.id_switches_suspected,
            "first_mask_frame": target.get("first_mask_frame"),
            "fps": track.fps,
        },
        f"track {track.track_id}: {sum(p.valid for p in track.points)} valid points, "
        f"{track.n_outliers} outliers",
    )
    return track, target_obs, trk.id, target


def window_coverage(track: Track, t_fit: np.ndarray, fps: float) -> dict[str, Any]:
    first_frame = round(float(t_fit[0]) * fps)
    last_frame = round(float(t_fit[-1]) * fps)
    valid_frames = {round(p.t * fps) for p in track.points if p.valid}
    n = last_frame - first_frame + 1
    covered = sum(first_frame <= f <= last_frame for f in valid_frames)
    return {"first_frame": first_frame, "last_frame": last_frame, "coverage": covered / n}


def run_free_fall(
    ctx: RunContext,
    spec: ExperimentSpec,
    ing: IngestResult,
    seg: SegmentationOutput,
    observations: list[Observation],
    summary: dict[str, Any],
) -> dict[str, Any] | None:
    """Free-fall analysis end to end (M3)."""
    ledger = Ledger(ctx.run_dir / "analysis" / "ledger.jsonl")
    track, target_obs, trk_id, target = build_track(ctx, ing, seg, observations, summary, ledger)

    cal, cal_ids = calibrate_run(ctx, spec, ing, seg, target_obs, ledger)
    if cal is None:
        raise AnalysisError(
            "No scale reference: put a ruler in the plane of motion or give the ball diameter"
        )
    if cal.agreement_pct is not None:
        ctx.emit("progress", message=f"scale methods agree within {cal.agreement_pct:.1f}%")

    t, _x, y = measure.valid_arrays(track)
    result = analyze_free_fall(
        t,
        y,
        cal.primary.m_per_px,
        cal.primary.sigma_m_per_px,
        bootstrap_n=ctx.settings.bootstrap_n,
        is_vfr=ing.probe.is_vfr,
        g_ref=G_REF_DEFAULT,
    )
    data = result.to_dict()
    cov = window_coverage(track, result.t_fit, ing.effective_fps)
    first_frame, last_frame = cov["first_frame"], cov["last_frame"]
    data["window"].update(
        {**cov, "release_frame_estimate": (target.get("first_mask_frame") or first_frame) - 1}
    )
    write_json_atomic(ctx.run_dir / "analysis" / "freefall.json", data)

    primary_cal = cal_ids[cal.primary.method]
    win = ledger.add_evidence(
        "win",
        "set_fit_window",
        {**data["window"], "method": "auto"},
        f"fit window frames {first_frame}–{last_frame} ({result.fit.n} valid points)",
        [trk_id],
    )
    fit = ledger.add_evidence(
        "fit",
        "fit_model",
        data["fit"],
        f"quadratic fit R²={result.fit.r2:.5f}, rms {result.fit.rms_px:.2f} px, "
        f"DW {result.fit.durbin_watson:.2f}",
        [trk_id, win.id],
    )
    if result.cubic is not None:
        ledger.add_evidence(
            "fit",
            "fit_model",
            {**data["cubic"], "systematic_curvature": result.systematic_curvature},
            f"cubic control fit (systematic curvature: {result.systematic_curvature})",
            [trk_id, win.id],
        )
    ci = ledger.add_evidence(
        "ci",
        "bootstrap_ci",
        {
            "quantity": "g",
            "value": result.g,
            "lo": result.ci95[0],
            "hi": result.ci95[1],
            "stat_lo": result.ci_stat[0],
            "stat_hi": result.ci_stat[1],
            "sigma_boot": result.sigma_boot,
            "sigma_analytic": result.sigma_analytic,
            "sigma_window": result.sigma_window,
            "sigma_stat": result.sigma_stat,
            "sigma_scale": result.sigma_scale,
            "sigma_vfr": result.sigma_vfr,
            "sigma_total": result.sigma_total,
            "n": result.bootstrap_n,
            "seed": 1234,
            "method": "block_bootstrap+analytic+window+propagation",
            "unit": "m/s^2",
        },
        f"g 95% CI [{result.ci95[0]:.3f}, {result.ci95[1]:.3f}] m/s²",
        [fit.id, primary_cal],
    )
    der = ledger.add_evidence(
        "der",
        "derive",
        {
            "quantity": "g",
            "value": result.g,
            "unit": "m/s^2",
            "sigma": result.sigma_total,
            "ci95": list(result.ci95),
            "formula": "g = 2·c2·s",
        },
        f"g = {fmt(result.g, result.sigma_total, 'm/s^2')}",
        [fit.id, ci.id, primary_cal],
    )
    ledger.add_evidence(
        "der",
        "derive",
        {
            "quantity": "v0",
            "value": result.v0,
            "unit": "m/s",
            "formula": "v0 = c1·s (velocity at the first fitted frame)",
        },
        f"v0 = {result.v0:.3f} m/s",
        [fit.id, primary_cal],
    )
    cmp = ledger.add_evidence(
        "cmp",
        "compare_reference",
        result.comparison,
        f"g differs from reference {G_REF_DEFAULT} m/s² by "
        f"{result.comparison['rel_diff_pct']:+.2f}% (reference "
        f"{'inside' if result.comparison['within_ci'] else 'outside'} the 95% CI)",
        [der.id],
    )
    for w in result.warnings:
        ctx.emit("warning", message=f"free-fall analysis: {w}")

    headline = {
        "experiment": "free_fall",
        "quantity": "g",
        "value": result.g,
        "sigma": result.sigma_total,
        "ci95": list(result.ci95),
        "unit": "m/s^2",
        "evidence": der.id,
        "comparison_evidence": cmp.id,
        "formatted_en": fmt(result.g, result.sigma_total, "m/s^2", "en"),
        "formatted_id": fmt(result.g, result.sigma_total, "m/s^2", "id"),
        "ref": G_REF_DEFAULT,
        "rel_diff_pct": result.comparison["rel_diff_pct"],
        "rel_diff_en": fmt_pct(result.comparison["rel_diff_pct"], "en", 2),
        "scale_method": cal.primary.method,
        "scale_agreement_pct": cal.agreement_pct,
        "fit_points": result.fit.n,
    }
    ctx.state.headline = headline
    ctx.emit("progress", message=f"g = {headline['formatted_en']}")
    return headline


# --------------------------------------------------------------------------------------------
# Pendulum (M4)
# --------------------------------------------------------------------------------------------
SPOKEN_LENGTH_SIGMA_M = 0.005


def _string_top(seg: SegmentationOutput) -> tuple[float, float] | None:
    rec = reference_record(seg.string)
    return calibrate.string_top_point(rec) if rec is not None else None


def run_pendulum(
    ctx: RunContext,
    spec: ExperimentSpec,
    ing: IngestResult,
    seg: SegmentationOutput,
    observations: list[Observation],
    summary: dict[str, Any],
) -> dict[str, Any] | None:
    """Pendulum analysis end to end (05 §3, 04 §6.3): period, amplitude correction, g."""
    ledger = Ledger(ctx.run_dir / "analysis" / "ledger.jsonl")
    track, target_obs, trk_id, _target = build_track(ctx, ing, seg, observations, summary, ledger)
    t, x, y = measure.valid_arrays(track)
    if t.size < 12:
        raise AnalysisError(f"Only {t.size} valid points: the bob was not tracked (re-record)")

    # --- pivot, length, scale ----------------------------------------------------------------
    pivot = calibrate.pivot_from_track(x, y, string_top=_string_top(seg))
    for w in pivot["warnings"]:
        ctx.emit("warning", message=f"pivot: {w}")
    piv = ledger.add_evidence(
        "cal",
        "calibrate_pivot",
        {k: v for k, v in pivot.items() if k != "warnings"} | {"warnings": pivot["warnings"]},
        (
            f"pivot ({pivot['cx']:.1f}, {pivot['cy']:.1f}) px, radius {pivot['r_px']:.1f} ± "
            f"{pivot['sigma_r_px']:.1f} px, swing {pivot['span_deg']:.0f}°"
            if pivot.get("r_px")
            else "pivot not found"
        ),
        [trk_id],
    )
    reliable = bool(pivot.get("reliable"))
    spoken = spec.spoken_params.get("string_length")
    spoken_l = spoken.to_si() if spoken else None
    extra = []
    if spoken_l and reliable:
        extra.append(
            calibrate.spoken_length_scale(
                spoken_l, pivot["r_px"], pivot["sigma_r_px"], length_sigma_m=SPOKEN_LENGTH_SIGMA_M
            )
        )
    cal, cal_ids = calibrate_run(ctx, spec, ing, seg, target_obs, ledger, extra, [piv.id])

    length_m = sigma_l = None
    length_source = None
    length_inputs: list[str] = []
    if spoken_l:
        length_m, sigma_l, length_source = spoken_l, SPOKEN_LENGTH_SIGMA_M, "spoken"
    elif reliable and cal is not None and cal.primary.method != "spoken_length":
        s = cal.primary.m_per_px
        length_m = pivot["r_px"] * s
        sigma_l = math.hypot(pivot["sigma_r_px"] * s, pivot["r_px"] * cal.primary.sigma_m_per_px)
        length_source = "measured"
        length_inputs = [piv.id, cal_ids[cal.primary.method]]
    if length_m is None:
        ctx.emit(
            "warning",
            message="String length unknown: reporting the period only (say or type the length)",
        )
    m_per_px = cal.primary.m_per_px if cal is not None else None

    result = analyze_pendulum(
        t,
        x,
        length_m=length_m,
        sigma_length_m=sigma_l,
        length_source=length_source,
        radius_px=pivot["r_px"] if reliable else None,
        m_per_px=m_per_px,
        bootstrap_n=min(ctx.settings.bootstrap_n, 300),
        is_vfr=ing.probe.is_vfr,
        g_ref=G_REF_DEFAULT,
    )
    data = result.to_dict()
    cov = window_coverage(track, result.t_fit, ing.effective_fps)
    data["window"].update(cov)
    data["pivot"] = pivot
    write_json_atomic(ctx.run_dir / "analysis" / "pendulum.json", data)

    min_points = REGISTRY["pendulum"].min_valid_points
    win = ledger.add_evidence(
        "win",
        "set_fit_window",
        {**data["window"], "method": "auto", "min_valid_points": min_points},
        f"fit window frames {cov['first_frame']}–{cov['last_frame']} "
        f"({result.fit.n} points, {result.n_cycles:.1f} cycles)",
        [trk_id],
    )
    fit = ledger.add_evidence(
        "fit",
        "fit_model",
        data["fit"],
        f"damped sine R²={result.fit.r2:.5f}, rms {result.fit.rms_px:.2f} px, "
        f"DW {result.fit.durbin_watson:.2f}, drift {result.drift_px_s:+.2f} px/s",
        [trk_id, win.id],
    )
    ci = ledger.add_evidence(
        "ci",
        "bootstrap_ci",
        {
            "quantity": "period0",
            "value": result.period0,
            "lo": result.ci_period0[0],
            "hi": result.ci_period0[1],
            "period": result.period,
            "sigma_boot": result.sigma_period_boot,
            "sigma_analytic": result.sigma_period_analytic,
            "sigma_stat": result.sigma_period_stat,
            "sigma_vfr": result.sigma_period_vfr,
            "sigma_period": result.sigma_period,
            "sigma_corr_rel": result.sigma_corr_rel,
            "sigma_total": result.sigma_period0,
            "n": result.bootstrap_n,
            "seed": 1234,
            "method": "block_bootstrap+analytic+propagation",
            "unit": "s",
        },
        f"T0 95% CI [{result.ci_period0[0]:.4f}, {result.ci_period0[1]:.4f}] s",
        [fit.id],
    )
    theta_inputs = [piv.id] if result.theta_source == "circle_fit" else length_inputs
    der_t = ledger.add_evidence(
        "der",
        "derive",
        {
            "quantity": "period",
            "value": result.period,
            "unit": "s",
            "sigma": result.sigma_period,
            "ci95": [
                result.period - 1.96 * result.sigma_period,
                result.period + 1.96 * result.sigma_period,
            ],
            "period_damped": result.period_damped,
            "n_cycles": result.n_cycles,
            "formula": "T = 2π/√(ω_d² + γ²)",
        },
        f"T = {fmt(result.period, result.sigma_period, 's')} ({result.n_cycles:.1f} cycles)",
        [fit.id, ci.id],
    )
    der_t0 = ledger.add_evidence(
        "der",
        "derive",
        {
            "quantity": "period0",
            "value": result.period0,
            "unit": "s",
            "sigma": result.sigma_period0,
            "ci95": list(result.ci_period0),
            "theta_mean_deg": result.theta_mean_deg,
            "theta_source": result.theta_source,
            "amp_factor": result.amp_factor,
            "formula": "T0 = T / mean(1 + θ²/16 + 11θ⁴/3072)",
        },
        f"T0 = {fmt(result.period0, result.sigma_period0, 's')} (amplitude "
        + (f"{result.theta_mean_deg:.1f}°)" if result.theta_mean_deg is not None else "unknown)"),
        [der_t.id, ci.id, *theta_inputs],
    )
    headline_ev = der_t
    cmp = None
    if result.g is not None:
        len_ev = ledger.add_evidence(
            "der",
            "derive",
            {
                "quantity": "length",
                "value": result.length_m,
                "unit": "m",
                "sigma": result.sigma_length_m,
                "source": result.length_source,
                "formula": "spoken (pivot → bob centre)"
                if result.length_source == "spoken"
                else "L = r_px · s",
            },
            f"L = {fmt(result.length_m, result.sigma_length_m, 'm')} ({result.length_source})",
            length_inputs,
        )
        headline_ev = ledger.add_evidence(
            "der",
            "derive",
            {
                "quantity": "g",
                "value": result.g,
                "unit": "m/s^2",
                "sigma": result.sigma_g,
                "ci95": list(result.ci_g),
                "g_uncorrected": result.g_uncorrected,
                "formula": "g = 4π²·L / T0²",
            },
            f"g = {fmt(result.g, result.sigma_g, 'm/s^2')}",
            [der_t0.id, len_ev.id],
        )
        cmp = ledger.add_evidence(
            "cmp",
            "compare_reference",
            result.comparison,
            f"g differs from reference {G_REF_DEFAULT} m/s² by "
            f"{result.comparison['rel_diff_pct']:+.2f}% (reference "
            f"{'inside' if result.comparison['within_ci'] else 'outside'} the 95% CI)",
            [headline_ev.id],
        )

    hyp = None
    pred = spec.hypothesis_prediction
    if pred is not None:
        unit = pred.si_unit()
        target_ev = der_t if unit == "s" else headline_ev if unit == "m/s^2" else None
        if target_ev is not None and (unit == "s" or result.g is not None):
            d = target_ev.data
            h = hypothesis_test(pred.to_si(), d["value"], d["ci95"][0], d["ci95"][1])
            hyp = ledger.add_evidence(
                "cmp",
                "hypothesis_test",
                {**h, "quantity": d["quantity"], "unit": unit, "statement": spec.hypothesis},
                f"prediction {pred.value:g} {pred.unit} vs measured {d['value']:.4g} {unit}: "
                f"{h['outcome']}",
                [target_ev.id],
            )

    for w in result.warnings:
        ctx.emit("warning", message=f"pendulum analysis: {w}")

    if result.g is not None:
        value, sigma, ci95, unit, quantity = (
            result.g, result.sigma_g, list(result.ci_g), "m/s^2", "g"
        )  # fmt: skip
    else:
        value, sigma, unit, quantity = result.period, result.sigma_period, "s", "period"
        ci95 = list(der_t.data["ci95"])
    headline = {
        "experiment": "pendulum",
        "quantity": quantity,
        "value": value,
        "sigma": sigma,
        "ci95": ci95,
        "unit": unit,
        "evidence": headline_ev.id,
        "comparison_evidence": cmp.id if cmp else None,
        "hypothesis_evidence": hyp.id if hyp else None,
        "hypothesis_outcome": hyp.data["outcome"] if hyp else None,
        "formatted_en": fmt(value, sigma, unit, "en"),
        "formatted_id": fmt(value, sigma, unit, "id"),
        "ref": G_REF_DEFAULT if quantity == "g" else None,
        "rel_diff_pct": result.comparison["rel_diff_pct"] if result.comparison else None,
        "rel_diff_en": fmt_pct(result.comparison["rel_diff_pct"], "en", 2)
        if result.comparison
        else None,
        "period": result.period,
        "period_formatted_en": fmt(result.period, result.sigma_period, "s", "en"),
        "period0": result.period0,
        "period0_sigma": result.sigma_period0,
        "length_m": result.length_m,
        "length_sigma_m": result.sigma_length_m,
        "length_source": result.length_source,
        "theta_mean_deg": result.theta_mean_deg,
        "n_cycles": result.n_cycles,
        "scale_method": cal.primary.method if cal else None,
        "scale_agreement_pct": cal.agreement_pct if cal else None,
        "fit_points": result.fit.n,
    }
    ctx.state.headline = headline
    ctx.emit("progress", message=f"{quantity} = {headline['formatted_en']}")
    return headline
