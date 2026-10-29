"""Deterministic analysis (no LLM): track → calibration → physics → ledger evidence.

Used by ``framelab run`` until the Analyst agent lands (M6), and afterwards as the Analyst's
deterministic fallback (docs/plan/06-agents-and-verification.md §3.4). Every number written
here goes through the Claim Ledger as evidence (06 §1).
"""

from __future__ import annotations

from typing import Any

from framelab.agent.ledger import Ledger
from framelab.clients.sam import SamOutput
from framelab.config import G_REF_DEFAULT
from framelab.physics.freefall import analyze_free_fall
from framelab.physics.uncertainty import fmt, fmt_pct
from framelab.pipeline import calibrate, measure
from framelab.pipeline.context import RunContext
from framelab.pipeline.segment import SegmentationOutput
from framelab.schemas import Calibration, ExperimentSpec, IngestResult, Observation, ScaleEstimate
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
) -> tuple[Calibration | None, dict[str, str]]:
    """Ruler (primary) and object-diameter (secondary) scales → ``cal_`` evidence."""
    estimates: list[ScaleEstimate] = []
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
    """Free-fall analysis end to end. Returns the headline (or None if no measurement)."""
    if spec.experiment != "free_fall":
        raise NotImplementedError(f"{spec.experiment} analysis arrives in M4")
    ledger = Ledger(ctx.run_dir / "analysis" / "ledger.jsonl")
    target = summary.get("target") or {}
    obj = target.get("object_id")
    if not obj:
        raise AnalysisError("No usable target track (re-record: brighter light, contrasting ball)")
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
    fps = ing.effective_fps
    first_frame = round(result.t_fit[0] * fps)
    last_frame = round(result.t_fit[-1] * fps)
    valid_frames = {round(p.t * fps) for p in track.points if p.valid}
    window_frames = last_frame - first_frame + 1
    coverage_window = sum(first_frame <= f <= last_frame for f in valid_frames) / window_frames
    data["window"].update(
        {
            "first_frame": first_frame,
            "last_frame": last_frame,
            "coverage": coverage_window,
            "release_frame_estimate": (target.get("first_mask_frame") or first_frame) - 1,
        }
    )
    write_json_atomic(ctx.run_dir / "analysis" / "freefall.json", data)

    primary_cal = cal_ids[cal.primary.method]
    win = ledger.add_evidence(
        "win",
        "set_fit_window",
        {**data["window"], "method": "auto"},
        f"fit window frames {first_frame}–{last_frame} ({result.fit.n} valid points)",
        [trk.id],
    )
    fit = ledger.add_evidence(
        "fit",
        "fit_model",
        data["fit"],
        f"quadratic fit R²={result.fit.r2:.5f}, rms {result.fit.rms_px:.2f} px, "
        f"DW {result.fit.durbin_watson:.2f}",
        [trk.id, win.id],
    )
    if result.cubic is not None:
        ledger.add_evidence(
            "fit",
            "fit_model",
            {**data["cubic"], "systematic_curvature": result.systematic_curvature},
            f"cubic control fit (systematic curvature: {result.systematic_curvature})",
            [trk.id, win.id],
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
