"""AnalysisSession: the state the Analyst's tools compute on (06 §3–§4).

Track and calibration are computed by code before the Analyst starts and written as ``trk_`` and
``cal_`` evidence (06 §3.2, the pre-computed evidence list). The tools compute everything else:
fit window, fit, bootstrap CI, derived value, comparisons, charts, and a re-segmentation request.
Every number is written to the ledger here, never returned from the model's text.

Free fall (05 §2) and pendulum (05 §3) share the session; the experiment decides which models,
quantities and length sources the tools accept. The pendulum's pivot (04 §6.3) and its length
options (spoken, or measured = pivot radius × scale) are computed in ``prepare``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from framelab.agent.ledger import Ledger
from framelab.clients import meta as meta_client
from framelab.clients import sam as sam_client
from framelab.config import G_REF_DEFAULT
from framelab.physics import derive
from framelab.physics.events import freefall_window
from framelab.physics.experiments import REGISTRY
from framelab.physics.freefall import (
    FreeFallAnalysis,
    analyze_free_fall,
    fall_time,
    trim_contact_points,
)
from framelab.physics.models import FitResult, fit_cubic, fit_damped_sine, fit_quadratic
from framelab.physics.pendulum import PendulumAnalysis, analyze_pendulum, g_with_length, trim_ends
from framelab.physics.uncertainty import fmt, fmt_pct
from framelab.pipeline import analysis_auto, calibrate, measure
from framelab.pipeline import ingest as ingest_mod
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


class ToolError(ValueError):
    """A tool was called in a way the analysis cannot honour. Returned to the model as a failure."""


@dataclass
class AnalysisSession:
    ctx: RunContext
    spec: ExperimentSpec
    ing: IngestResult
    seg: SegmentationOutput
    observations: list[Observation]  # as measured; never modified (the fallback reads them)
    summary: dict[str, Any]
    ledger: Ledger
    track: Track
    track_obs: list[Observation]
    track_evidence_id: str
    calibration: Calibration | None  # pendulum: may be None (period only)
    cal_ids: dict[str, str]
    scale: ScaleEstimate | None  # the active scale; calibrate_scale can change it
    scale_id: str | None
    min_points: int
    round: int = 0
    windows: dict[str, tuple[int, int]] = field(default_factory=dict)  # win id → valid indices
    fits: dict[str, dict[str, Any]] = field(default_factory=dict)  # fit id → result + window
    analysis: FreeFallAnalysis | PendulumAnalysis | None = None
    analysis_fit_id: str | None = None
    analysis_ci_id: str | None = None
    g_der_id: str | None = None
    g_cmp_id: str | None = None
    period_der_id: str | None = None
    sam_requests: int = 0
    # pendulum (04 §6.3): pivot evidence and the length sources the Analyst may choose from
    pivot: dict[str, Any] | None = None
    pivot_id: str | None = None
    lengths: dict[str, tuple[float, float, list[str]]] = field(default_factory=dict)

    @property
    def experiment(self) -> str:
        return self.spec.experiment

    @property
    def pendulum(self) -> bool:
        return self.spec.experiment == "pendulum"

    # --- reading -------------------------------------------------------------------------------
    def valid_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return measure.valid_arrays(self.track)

    @property
    def g_value(self) -> float | None:
        """The derived g (free fall: from the bootstrap; pendulum: once derive(g) ran)."""
        if self.g_der_id is not None:
            return float(self.ledger.get(self.g_der_id).data["value"])
        if isinstance(self.analysis, FreeFallAnalysis):
            return self.analysis.g
        return None

    @property
    def n_cycles(self) -> float | None:
        return self.analysis.n_cycles if isinstance(self.analysis, PendulumAnalysis) else None

    @property
    def drift_px_s(self) -> float | None:
        return self.analysis.drift_px_s if isinstance(self.analysis, PendulumAnalysis) else None

    @property
    def latest_fit_summary(self) -> dict[str, Any] | None:
        if self.analysis is not None:
            return self.analysis.fit.summary()
        if not self.fits:
            return None
        return next(reversed(self.fits.values()))["result"].summary()

    def window_span(self) -> tuple[tuple[float, float] | None, list[str]]:
        """Time span and warnings of the window behind the latest analysis (or latest window)."""
        win_id = None
        if self.analysis_fit_id and self.analysis_fit_id in self.fits:
            win_id = self.fits[self.analysis_fit_id]["window"]
        elif self.windows:
            win_id = next(reversed(self.windows))
        if win_id is None:
            return None, []
        data = self.ledger.get(win_id).data
        return (float(data["t_start"]), float(data["t_end"])), list(data.get("warnings") or [])

    def track_summary(self) -> dict[str, Any]:
        valid = sum(p.valid for p in self.track.points)
        return {
            "track_id": self.track.track_id,
            "n_valid": valid,
            "coverage": round(self.track.coverage, 3),
            "n_outliers": self.track.n_outliers,
            "candidates": sorted(
                {o.object_id for o in self.observations if o.concept == self.track.concept}
            ),  # fmt: skip
        }

    def calibration_summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {"active": None, "scale_id": None, "methods": [],
                               "agreement_pct": None}  # fmt: skip
        if self.calibration is not None and self.scale is not None:
            methods = [self.calibration.primary, *([self.calibration.secondary]
                       if self.calibration.secondary else [])]  # fmt: skip
            out = {
                "active": self.scale.method,
                "scale_id": self.scale_id,
                "methods": [
                    {"method": m.method, "mm_per_px": round(m.m_per_px * 1000, 4)} for m in methods
                ],  # fmt: skip
                "agreement_pct": self.calibration.agreement_pct,
            }
        if self.pendulum:
            out["pivot_id"] = self.pivot_id
            out["pivot_reliable"] = bool(self.pivot and self.pivot.get("reliable"))
            out["length_sources"] = sorted(self.lengths)
        return out

    # --- tools: track and scale ---------------------------------------------------------------
    def choose_track(self, track_id: str) -> Track:
        """Re-clean the track for another candidate object (writes a new ``trk_``)."""
        object_id = next(
            (o.object_id for o in self.observations if f"{self._slug()}_{o.object_id}" == track_id),
            None,
        )
        if object_id is None:
            raise ToolError(
                f"unknown track_id {track_id!r}; candidates are "
                f"{self.track_summary()['candidates']}"
            )
        obs = [o for o in self.observations if o.object_id == object_id]
        self._adopt_track(obs, object_id, reason="chosen by the Analyst")
        return self.track

    def _slug(self) -> str:
        return measure._slug(self.track.concept)

    def _adopt_track(self, obs: list[Observation], object_id: str, *, reason: str) -> None:
        track = measure.clean_track(
            obs,
            effective_fps=self.ing.effective_fps,
            concept=self.track.concept,
            object_id=object_id,
        )
        measure.write_track_csv(self.ctx.run_dir / "tracks" / f"{track.track_id}.csv", track)
        ev = self.ledger.add_evidence(
            "trk",
            "choose_track",
            {
                "track_id": track.track_id,
                "n_valid": sum(p.valid for p in track.points),
                "coverage": track.coverage,
                "reason": reason,
            },  # fmt: skip
            f"track {track.track_id} ({reason})",
        )
        self.track, self.track_obs, self.track_evidence_id = track, obs, ev.id
        # Windows and fits belong to the old track: they must be recomputed.
        self.windows.clear()
        self.fits.clear()
        self.analysis = None

    def estimate_scale(self, method: str) -> ScaleEstimate:
        """A scale by one method (06 §4 ``calibrate_scale``). Free fall: ruler or diameter."""
        if method == "ruler":
            ref = analysis_auto.reference_record(self.seg.reference)
            if ref is None or self.spec.reference is None or not self.spec.reference.length_m:
                raise ToolError("no ruler reference with a known length was found")
            return calibrate.ruler_scale(
                ref,
                self.spec.reference.length_m,
                frame_long_edge_px=max(self.ing.norm_width, self.ing.norm_height),
            )
        if method == "object_diameter":
            diameter = self.spec.object_diameter_m
            if diameter is None and "ball_diameter" in self.spec.spoken_params:
                diameter = self.spec.spoken_params["ball_diameter"].to_si()
            if not diameter:
                raise ToolError("no ball diameter is known (spoken or given)")
            return calibrate.diameter_scale(
                self.track_obs, diameter, dilation_px=self.ctx.settings.mask_dilation_px
            )
        if method == "spoken_length" and self.pendulum:
            spoken = self.spec.spoken_params.get("string_length")
            if spoken is None or not (self.pivot and self.pivot.get("reliable")):
                raise ToolError("needs a spoken string length and a reliable pivot")
            return calibrate.spoken_length_scale(
                spoken.to_si(), self.pivot["r_px"], self.pivot["sigma_r_px"]
            )
        raise ToolError(f"method {method!r} is not available for {self.experiment}")

    def use_scale(self, method: str) -> dict[str, Any]:
        est = self.estimate_scale(method)
        ev = self.ledger.add_evidence(
            "cal",
            "calibrate_scale",
            {
                "method": est.method,
                "m_per_px": est.m_per_px,
                "sigma": est.sigma_m_per_px,
                "rel_sigma": est.sigma_m_per_px / est.m_per_px,
                "details": est.details,
                "value": est.m_per_px * 1000.0,  # claims copy {value} in mm/px (06 §4)
                "unit": "mm/px",
            },  # fmt: skip
            f"{est.method} scale {est.m_per_px * 1000:.4f} mm/px "
            f"(±{est.sigma_m_per_px / est.m_per_px * 100:.2f}%)",
            [self.track_evidence_id] if est.method == "object_diameter"
            else [self.pivot_id] if est.method == "spoken_length" and self.pivot_id else None,
        )  # fmt: skip
        self.scale, self.scale_id = est, ev.id
        self.analysis = None  # the CI was computed with the previous scale
        self._refresh_lengths()
        return {"scale_id": ev.id, "method": est.method,
                "mm_per_px": round(est.m_per_px * 1000, 4)}  # fmt: skip

    # --- tools: window, fit, bootstrap, derive, compare -----------------------------------------
    def set_window(
        self, mode: str, t_start: float | None, t_end: float | None
    ) -> tuple[str, list[str]]:
        t, x, y = self.valid_arrays()
        warnings: list[str] = []
        if mode == "auto" and self.pendulum:
            if t.size < 12:
                raise ToolError(f"only {t.size} valid points: too few for a pendulum")
            start, end, _fit = trim_ends(t, x)  # hand at release / catching the bob
            release = impact = None
        elif mode == "auto":
            win = freefall_window(t, y)
            warnings = list(win.warnings)
            start, end = trim_contact_points(t, y, win.start, win.end)
            release, impact = win.release_index, win.impact_index
        else:
            if t_start is None or t_end is None or t_end <= t_start:
                raise ToolError("a manual window needs t_start < t_end")
            if t_start < t[0] - 1e-9 or t_end > t[-1] + 1e-9:
                raise ToolError(
                    f"manual window must lie inside the track [{t[0]:.3f}, {t[-1]:.3f}] s"
                )
            idx = np.where((t >= t_start) & (t <= t_end))[0]
            if idx.size < max(self.min_points, 3):
                raise ToolError(f"window holds {idx.size} valid points; {self.min_points} needed")
            start, end = int(idx[0]), int(idx[-1])
            release = impact = None
        ev = self.ledger.add_evidence(
            "win",
            "set_fit_window",
            {"mode": mode, "t_start": float(t[start]), "t_end": float(t[end]),
             "start": start, "end": end, "n": end - start + 1,
             "release_index": release, "impact_index": impact, "warnings": warnings},  # fmt: skip
            f"fit window {float(t[start]):.3f}–{float(t[end]):.3f} s "
            f"({end - start + 1} points, {mode})",
            [self.track_evidence_id],
        )  # fmt: skip
        self.windows[ev.id] = (start, end)
        return ev.id, warnings

    def fit(self, window_id: str, model: str) -> tuple[str, FitResult]:
        if window_id not in self.windows:
            raise ToolError(f"unknown window_id {window_id!r}; call set_fit_window first")
        allowed = ("damped_sine",) if self.pendulum else ("quadratic", "cubic")
        if model not in allowed:
            raise ToolError(f"{self.experiment} fits use {' or '.join(allowed)}")
        start, end = self.windows[window_id]
        t, x, y = self.valid_arrays()
        if model == "damped_sine":
            try:
                result = fit_damped_sine(t[start : end + 1], x[start : end + 1])
            except ValueError as exc:
                raise ToolError(str(exc)) from None
        else:
            result = (fit_quadratic if model == "quadratic" else fit_cubic)(
                t[start : end + 1], y[start : end + 1]
            )
        extra = f", drift {result.params['d']:+.2f} px/s" if model == "damped_sine" else ""
        ev = self.ledger.add_evidence(
            "fit",
            "fit_model",
            {
                **result.summary(),
                "window": window_id,
                "round": self.round,
                "range": [start, end],
            },  # fmt: skip
            f"{model} fit R²={result.r2:.5f}, rms {result.rms_px:.2f} px, "
            f"DW {result.durbin_watson:.2f}{extra}",
            [window_id],
        )
        self.fits[ev.id] = {"result": result, "window": window_id, "range": (start, end),
                            "round": self.round, "model": model}  # fmt: skip
        return ev.id, result

    def bootstrap(self, fit_id: str, quantity: str) -> dict[str, Any]:
        rec = self.fits.get(fit_id)
        if rec is None:
            raise ToolError(f"unknown fit_id {fit_id!r}; call fit_model first")
        if rec["round"] != self.round:
            raise ToolError("this fit is from an earlier round; call fit_model again")
        if self.pendulum:
            if quantity != "period" or rec["model"] != "damped_sine":
                raise ToolError("the pendulum CI is computed for the period from a damped_sine fit")
            return self._bootstrap_pendulum(fit_id, rec)
        if quantity != "g" or rec["model"] != "quadratic":
            raise ToolError("the CI is computed for g from a quadratic fit")
        if self.scale is None or self.scale_id is None:
            raise ToolError("no scale: free fall needs a ruler or the ball diameter")
        t, _x, y = self.valid_arrays()
        start, end = rec["range"]
        res = analyze_free_fall(
            t,
            y,
            self.scale.m_per_px,
            self.scale.sigma_m_per_px,
            bootstrap_n=self.ctx.settings.bootstrap_n,
            is_vfr=self.ing.probe.is_vfr,
            g_ref=G_REF_DEFAULT,
            window=(start, end),
        )
        self.analysis, self.analysis_fit_id = res, fit_id
        ev = self.ledger.add_evidence(
            "ci",
            "bootstrap_ci",
            {"quantity": "g", "value": res.g, "lo": res.ci95[0], "hi": res.ci95[1],
             "stat_lo": res.ci_stat[0], "stat_hi": res.ci_stat[1],
             "sigma_boot": res.sigma_boot, "sigma_analytic": res.sigma_analytic,
             "sigma_window": res.sigma_window, "sigma_stat": res.sigma_stat,
             "sigma_scale": res.sigma_scale, "sigma_vfr": res.sigma_vfr,
             "sigma_total": res.sigma_total, "n": res.bootstrap_n, "seed": 1234,
             "method": "block_bootstrap+analytic+window+propagation", "unit": "m/s^2"},  # fmt: skip
            f"g 95% CI [{res.ci95[0]:.3f}, {res.ci95[1]:.3f}] m/s²",
            [fit_id, self.scale_id],
        )  # fmt: skip
        self.analysis_ci_id = ev.id
        return {"ci_id": ev.id, "value": res.g, "ci95": list(res.ci95), "sigma": res.sigma_total}

    def _bootstrap_pendulum(self, fit_id: str, rec: dict[str, Any]) -> dict[str, Any]:
        t, x, _y = self.valid_arrays()
        start, end = rec["range"]
        reliable = bool(self.pivot and self.pivot.get("reliable"))
        best = self.best_length()
        res = analyze_pendulum(
            t[start : end + 1],
            x[start : end + 1],
            length_m=best[0] if best else None,
            sigma_length_m=best[1] if best else None,
            radius_px=self.pivot["r_px"] if reliable and self.pivot else None,
            m_per_px=self.scale.m_per_px if self.scale is not None else None,
            bootstrap_n=min(self.ctx.settings.bootstrap_n, 300),
            is_vfr=self.ing.probe.is_vfr,
            g_ref=G_REF_DEFAULT,
        )
        self.analysis, self.analysis_fit_id = res, fit_id
        inputs = [fit_id] + ([self.pivot_id] if res.theta_source == "circle_fit" and self.pivot_id
                             else [])  # fmt: skip
        ev = self.ledger.add_evidence(
            "ci",
            "bootstrap_ci",
            {"quantity": "period0", "value": res.period0, "lo": res.ci_period0[0],
             "hi": res.ci_period0[1], "period": res.period,
             "sigma_boot": res.sigma_period_boot, "sigma_analytic": res.sigma_period_analytic,
             "sigma_stat": res.sigma_period_stat, "sigma_vfr": res.sigma_period_vfr,
             "sigma_period": res.sigma_period, "sigma_corr_rel": res.sigma_corr_rel,
             "sigma_total": res.sigma_period0, "theta_mean_deg": res.theta_mean_deg,
             "theta_source": res.theta_source, "n_cycles": res.n_cycles, "n": res.bootstrap_n,
             "seed": 1234, "method": "block_bootstrap+analytic+propagation", "unit": "s"},
            f"T0 95% CI [{res.ci_period0[0]:.4f}, {res.ci_period0[1]:.4f}] s "
            f"({res.n_cycles:.1f} cycles)",
            inputs,
        )  # fmt: skip
        self.analysis_ci_id = ev.id
        return {"ci_id": ev.id, "value": res.period0, "ci95": list(res.ci_period0),
                "sigma": res.sigma_period0, "n_cycles": round(res.n_cycles, 2),
                "warnings": list(res.warnings)}  # fmt: skip

    def best_length(self) -> tuple[float, float, list[str]] | None:
        for source in ("spoken", "measured"):
            if source in self.lengths:
                return self.lengths[source]
        return None

    def _refresh_lengths(self) -> None:
        """Length sources for the pendulum (05 §3.3): spoken, or the pivot radius × scale."""
        if not self.pendulum:
            return
        self.lengths.pop("measured", None)
        reliable = bool(self.pivot and self.pivot.get("reliable"))
        if reliable and self.scale is not None and self.scale.method != "spoken_length":
            s = self.scale.m_per_px
            r, sr = self.pivot["r_px"], self.pivot["sigma_r_px"]
            sigma = float(np.hypot(sr * s, r * self.scale.sigma_m_per_px))
            self.lengths["measured"] = (r * s, sigma, [i for i in (self.pivot_id, self.scale_id)
                                                       if i])  # fmt: skip

    def derive(
        self, quantity: str, fit_id: str, scale_id: str | None, length_source: str | None = None
    ) -> str:
        res = self.analysis
        if res is None or self.analysis_fit_id != fit_id or self.analysis_ci_id is None:
            raise ToolError("run bootstrap_ci for this fit first")
        if isinstance(res, PendulumAnalysis):
            return self._derive_pendulum(res, quantity, fit_id, length_source)
        if quantity == "fall_time":
            return self._derive_fall_time(res, fit_id)
        if scale_id is None:
            raise ToolError("derive for free fall needs scale_id")
        if scale_id != self.scale_id:
            raise ToolError("scale_id is not the active scale; bootstrap_ci uses the active scale")
        if quantity == "g":
            data = {"quantity": "g", "value": res.g, "unit": "m/s^2", "sigma": res.sigma_total,
                    "ci95": list(res.ci95), "formula": "g = 2·c2·s"}  # fmt: skip
            inputs = [self.analysis_ci_id, scale_id]
        elif quantity == "v0":
            data = {"quantity": "v0", "value": res.v0, "unit": "m/s",
                    "formula": "v0 = c1·s (velocity at the first fitted frame)"}  # fmt: skip
            inputs = [fit_id, scale_id]
        else:
            raise ToolError("free fall derives g, v0 or fall_time")
        ev = self.ledger.add_evidence(
            "der", "derive", data, f"{quantity} = {fmt(res.g, res.sigma_total, 'm/s^2')}"
            if quantity == "g" else f"v0 = {res.v0:.3f} m/s",
            inputs,
        )  # fmt: skip
        if quantity == "g":
            self.g_der_id = ev.id
        return ev.id

    def _derive_fall_time(self, res: FreeFallAnalysis, fit_id: str) -> str:
        """Release (vertex) to impact time, for a student's prediction in seconds (05 §3.4)."""
        rec = self.fits[fit_id]
        win = self.ledger.get(rec["window"]).data
        impact = win.get("impact_index")
        if impact is None:
            raise ToolError("no impact was detected in this window (set_fit_window auto)")
        _t, _x, y = self.valid_arrays()
        sigma_c2_rel = res.sigma_stat / res.g if res.g else 0.0
        try:
            value, sigma = fall_time(res.fit, sigma_c2_rel, float(y[int(impact)]),
                                     max(1.0, res.fit.rms_px))  # fmt: skip
        except ValueError as exc:
            raise ToolError(str(exc)) from None
        ci = [value - 1.96 * sigma, value + 1.96 * sigma]
        ev = self.ledger.add_evidence(
            "der", "derive",
            {"quantity": "fall_time", "value": value, "unit": "s", "sigma": sigma, "ci95": ci,
             "formula": "t = √((y_impact − y_vertex)/c2), release at v = 0"},
            f"fall time = {fmt(value, sigma, 's')}",
            [fit_id, rec["window"], self.analysis_ci_id],
        )  # fmt: skip
        return ev.id

    def _derive_pendulum(
        self, res: PendulumAnalysis, quantity: str, fit_id: str, length_source: str | None
    ) -> str:
        ci_id = self.analysis_ci_id
        if quantity == "period":
            ci = [res.period - 1.96 * res.sigma_period, res.period + 1.96 * res.sigma_period]
            ev = self.ledger.add_evidence(
                "der", "derive",
                {"quantity": "period", "value": res.period, "unit": "s",
                 "sigma": res.sigma_period, "ci95": ci, "period_damped": res.period_damped,
                 "n_cycles": res.n_cycles, "formula": "T = 2π/√(ω_d² + γ²)"},
                f"T = {fmt(res.period, res.sigma_period, 's')} ({res.n_cycles:.1f} cycles)",
                [fit_id, ci_id],
            )  # fmt: skip
            self.period_der_id = ev.id
            return ev.id
        if quantity == "period0":
            ev = self.ledger.add_evidence(
                "der", "derive",
                {"quantity": "period0", "value": res.period0, "unit": "s",
                 "sigma": res.sigma_period0, "ci95": list(res.ci_period0),
                 "theta_mean_deg": res.theta_mean_deg, "theta_source": res.theta_source,
                 "amp_factor": res.amp_factor,
                 "formula": "T0 = T / mean(1 + θ²/16 + 11θ⁴/3072)"},
                f"T0 = {fmt(res.period0, res.sigma_period0, 's')}",
                [fit_id, ci_id],
            )  # fmt: skip
            return ev.id
        if quantity != "g":
            raise ToolError("the pendulum derives period, period0 or g")
        if length_source is None:
            raise ToolError(f"derive g needs length_source; available: {sorted(self.lengths)}")
        if length_source not in self.lengths:
            raise ToolError(
                f"length_source {length_source!r} is not available; "
                f"available: {sorted(self.lengths) or 'none (report the period only)'}"
            )
        length, sigma_l, inputs = self.lengths[length_source]
        len_ev = self.ledger.add_evidence(
            "der", "derive",
            {"quantity": "length", "value": length, "unit": "m", "sigma": sigma_l,
             "ci95": [length - 1.96 * sigma_l, length + 1.96 * sigma_l], "source": length_source,
             "formula": "spoken (pivot → bob centre)" if length_source == "spoken"
             else "L = r_px · s"},
            f"L = {fmt(length, sigma_l, 'm')} ({length_source})",
            inputs,
        )  # fmt: skip
        g = g_with_length(res, length, sigma_l)
        ev = self.ledger.add_evidence(
            "der", "derive",
            {"quantity": "g", "value": g["g"], "unit": "m/s^2", "sigma": g["sigma"],
             "ci95": list(g["ci95"]), "g_uncorrected": g["g_uncorrected"],
             "length_source": length_source, "formula": "g = 4π²·L / T0²"},
            f"g = {fmt(g['g'], g['sigma'], 'm/s^2')} (L {length_source})",
            [fit_id, ci_id, len_ev.id],
        )  # fmt: skip
        self.g_der_id = ev.id
        return ev.id

    def _derived(self, derived_id: str, quantity: str | None = None) -> dict[str, Any]:
        try:
            ev = self.ledger.get(derived_id)
        except KeyError:
            raise ToolError(f"unknown derived_id {derived_id!r}") from None
        if ev.tool != "derive" or (quantity and ev.data.get("quantity") != quantity):
            raise ToolError(f"{derived_id} is not a derived {quantity or 'value'}")
        return ev.data

    def compare_reference(self, derived_id: str) -> str:
        d = self._derived(derived_id, "g")
        comparison = derive.compare_reference(d["value"], *d["ci95"], G_REF_DEFAULT)
        ev = self.ledger.add_evidence(
            "cmp",
            "compare_reference",
            {**comparison, "unit": d["unit"], "quantity": "g"},
            f"g differs from reference {G_REF_DEFAULT} m/s² by "
            f"{comparison['rel_diff_pct']:+.2f}% (reference "
            f"{'inside' if comparison['within_ci'] else 'outside'} the 95% CI)",
            [derived_id],
        )
        self.g_cmp_id = ev.id
        return ev.id

    def compare_hypothesis(self, derived_id: str) -> str:
        pred = self.spec.hypothesis_prediction
        if pred is None:
            raise ToolError("the student stated no numeric prediction")
        d = self._derived(derived_id)
        if pred.si_unit() != d["unit"]:
            raise ToolError(
                f"prediction unit {pred.unit} does not match {d['quantity']} ({d['unit']})"
            )
        test = derive.hypothesis_test(pred.to_si(), d["value"], d["ci95"][0], d["ci95"][1])
        ev = self.ledger.add_evidence(
            "cmp",
            "hypothesis_test",
            {**test, "quantity": d["quantity"], "unit": d["unit"],
             "statement": self.spec.hypothesis},  # fmt: skip
            f"prediction {pred.value:g} {pred.unit} vs measured {d['value']:.4g} {d['unit']}: "
            f"{test['outcome']}",
            [derived_id],
        )  # fmt: skip
        return ev.id

    # --- tools: charts and resegmentation ----------------------------------------------------
    def chart(self, kind: str, fit_id: str) -> str:
        rec = self.fits.get(fit_id)
        if rec is None:
            raise ToolError(f"unknown fit_id {fit_id!r}")
        t, x, y = self.valid_arrays()
        start, end = rec["range"]
        result: FitResult = rec["result"]
        coord, key = (x, "x_px") if rec["model"] == "damped_sine" else (y, "y_px")
        if kind == "trajectory":
            body = {"t": t[start : end + 1].tolist(), key: coord[start : end + 1].tolist(),
                    "fit_px": (coord[start : end + 1] - result.residuals).tolist()}  # fmt: skip
        elif kind == "residuals":
            body = {"t": t[start : end + 1].tolist(), "residuals_px": result.residuals.tolist()}
        else:
            raise ToolError("chart kind must be trajectory or residuals")
        path = self.ctx.run_dir / "analysis" / "figures" / f"{kind}_{fit_id}.json"
        write_json_atomic(path, {"kind": kind, "fit_id": fit_id, **body})
        rel = str(path.relative_to(self.ctx.run_dir)).replace("\\", "/")
        ev = self.ledger.add_evidence(
            "fig", "make_chart", {"kind": kind, "path": rel, "n": end - start + 1},
            f"{kind} chart ({end - start + 1} points)", [fit_id],
        )  # fmt: skip
        return ev.id

    async def resegment(self, concept: str, reason: str) -> dict[str, Any]:
        """A new SAM request on the analysis clip (06 §4 ``resegment``, max 2 per run)."""
        budget = self.ctx.settings.max_extra_sam_requests
        if self.sam_requests >= budget:
            raise ToolError(f"SAM request budget used ({budget} extra requests per run)")
        if not self.ing.clip_path:
            raise ToolError("no analysis clip for this run")
        clip = self.ctx.run_dir / self.ing.clip_path
        self.sam_requests += 1
        file_id = await sam_client.upload_video(clip, emit=self.ctx.emit)
        try:
            record = self.ctx.run_dir / "sam" / f"resegment_{self.sam_requests:02d}.sam.json"
            out = await sam_client.segment_video(file_id, concept, record, emit=self.ctx.emit)
        finally:
            await meta_client.delete_file(file_id, emit=self.ctx.emit)
        obs = measure.observations_from_result(
            out.result,
            concept,
            offset_frame=self.ing.clip_offset_frame,
            effective_fps=self.ing.effective_fps,
        )
        clip_frames = ingest_mod.count_frames(clip)
        best, merged = measure.choose_target_track(obs, self.spec, clip_frames)
        seg_ev = self.ledger.add_evidence(
            "seg", "resegment",
            {"concept": concept, "status": out.status, "n_objects": out.n_objects,
             "frames_with_mask": len(out.frames_with_mask()), "reason": reason},  # fmt: skip
            f"SAM '{concept}': {out.n_objects} object(s), {len(out.frames_with_mask())} frames",
        )  # fmt: skip
        if best is None:
            raise ToolError(f"SAM found no usable '{concept}' track")
        if merged:
            obs = [
                o.model_copy(update={"object_id": best}) if o.object_id in merged else o
                for o in obs
            ]
        self._adopt_track(
            [o for o in obs if o.object_id == best], best, reason=f"resegment {concept}"
        )
        return {"seg_id": seg_ev.id, "track_id": self.track.track_id}

    # --- headline ----------------------------------------------------------------------------
    def headline(self) -> dict[str, Any] | None:
        """The headline dict the CLI and UI show (same keys as the deterministic path)."""
        if isinstance(self.analysis, PendulumAnalysis):
            return self._pendulum_headline(self.analysis)
        if self.analysis is None or self.g_der_id is None or self.g_cmp_id is None:
            return None
        res = self.analysis
        cmp = self.ledger.get(self.g_cmp_id).data
        return {
            "experiment": "free_fall",
            "quantity": "g",
            "value": res.g,
            "sigma": res.sigma_total,
            "ci95": list(res.ci95),
            "unit": "m/s^2",
            "evidence": self.g_der_id,
            "comparison_evidence": self.g_cmp_id,
            "formatted_en": fmt(res.g, res.sigma_total, "m/s^2", "en"),
            "formatted_id": fmt(res.g, res.sigma_total, "m/s^2", "id"),
            "ref": G_REF_DEFAULT,
            "rel_diff_pct": cmp["rel_diff_pct"],
            "rel_diff_en": fmt_pct(cmp["rel_diff_pct"], "en", 2),
            "scale_method": self.scale.method if self.scale else None,
            "scale_agreement_pct": self.calibration.agreement_pct if self.calibration else None,
            "fit_points": res.fit.n,
        }

    def _pendulum_headline(self, res: PendulumAnalysis) -> dict[str, Any] | None:
        """g when it was derived with a length, else the measured period (05 §3.3)."""
        if self.g_der_id is not None:
            d = self.ledger.get(self.g_der_id).data
            quantity, value, sigma, unit, evidence = (
                "g",
                d["value"],
                d["sigma"],
                "m/s^2",
                self.g_der_id,
            )
            ci95 = list(d["ci95"])
            length_ev = next(
                (
                    self.ledger.get(i)
                    for i in self.ledger.get(self.g_der_id).inputs
                    if self.ledger.get(i).data.get("quantity") == "length"
                ),
                None,
            )
        elif self.period_der_id is not None:
            d = self.ledger.get(self.period_der_id).data
            quantity, value, sigma, unit, evidence = (
                "period",
                d["value"],
                d["sigma"],
                "s",
                self.period_der_id,
            )
            ci95, length_ev = list(d["ci95"]), None
        else:
            return None
        cmp = self.ledger.get(self.g_cmp_id).data if self.g_cmp_id else None
        return {
            "experiment": "pendulum",
            "quantity": quantity,
            "value": value,
            "sigma": sigma,
            "ci95": ci95,
            "unit": unit,
            "evidence": evidence,
            "comparison_evidence": self.g_cmp_id,
            "formatted_en": fmt(value, sigma, unit, "en"),
            "formatted_id": fmt(value, sigma, unit, "id"),
            "ref": G_REF_DEFAULT if quantity == "g" else None,
            "rel_diff_pct": cmp["rel_diff_pct"] if cmp else None,
            "rel_diff_en": fmt_pct(cmp["rel_diff_pct"], "en", 2) if cmp else None,
            "period": res.period,
            "period_formatted_en": fmt(res.period, res.sigma_period, "s", "en"),
            "period0": res.period0,
            "period0_sigma": res.sigma_period0,
            "length_m": length_ev.data["value"] if length_ev else None,
            "length_sigma_m": length_ev.data["sigma"] if length_ev else None,
            "length_source": length_ev.data["source"] if length_ev else None,
            "theta_mean_deg": res.theta_mean_deg,
            "n_cycles": res.n_cycles,
            "scale_method": self.scale.method if self.scale else None,
            "scale_agreement_pct": self.calibration.agreement_pct if self.calibration else None,
            "fit_points": res.fit.n,
        }


def prepare(
    ctx: RunContext,
    spec: ExperimentSpec,
    ing: IngestResult,
    seg: SegmentationOutput,
    observations: list[Observation],
    summary: dict[str, Any],
) -> AnalysisSession:
    """Track and calibration by code (``trk_``, ``cal_``), then the session for the Analyst."""
    ledger = Ledger(ctx.run_dir / "analysis" / "ledger.jsonl")
    track, target_obs, trk_id, _target = analysis_auto.build_track(
        ctx, ing, seg, observations, summary, ledger
    )
    pivot = pivot_id = None
    extra: list[ScaleEstimate] = []
    if spec.experiment == "pendulum":
        pivot, pivot_id, extra = analysis_auto.pendulum_pivot(ctx, spec, seg, track, ledger, trk_id)
    cal, cal_ids = analysis_auto.calibrate_run(
        ctx, spec, ing, seg, target_obs, ledger, extra, [pivot_id] if pivot_id else None
    )
    if cal is None and spec.experiment != "pendulum":
        raise analysis_auto.NoScaleError(
            "No scale reference: put a ruler in the plane of motion or give the ball diameter"
        )
    session = AnalysisSession(
        ctx=ctx,
        spec=spec,
        ing=ing,
        seg=seg,
        observations=observations,
        summary=summary,
        ledger=ledger,
        track=track,
        track_obs=target_obs,
        track_evidence_id=trk_id,
        calibration=cal,
        cal_ids=cal_ids,
        scale=cal.primary if cal else None,
        scale_id=cal_ids[cal.primary.method] if cal else None,
        min_points=REGISTRY[spec.experiment].min_valid_points,
        pivot=pivot,
        pivot_id=pivot_id,
    )
    if spec.experiment == "pendulum":
        spoken = spec.spoken_params.get("string_length")
        if spoken is not None:
            spk = [e.id for e in ledger.evidence()
                   if e.id.startswith("spk_") and e.data.get("key") == "string_length"]  # fmt: skip
            session.lengths["spoken"] = (
                spoken.to_si(),
                analysis_auto.SPOKEN_LENGTH_SIGMA_M,
                spk[-1:],
            )
        session._refresh_lengths()
    return session
