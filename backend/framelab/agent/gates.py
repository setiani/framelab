"""Quality gates G1–G5 with per-experiment thresholds and EN/ID message templates.

Spec: docs/plan/06-agents-and-verification.md §7
G1 track integrity, G2 scale consistency, G3 fit quality, G4 plausibility (soft, never alters
data), G5 ledger completeness of the report draft. Each gate result is written to the ledger as
``gate_`` evidence (06 §4 ``run_gates``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from pydantic import BaseModel

from framelab.agent.narrative import PLACEHOLDER_RE, bare_numbers
from framelab.config import G_REF_DEFAULT
from framelab.schemas import Calibration, GateResult, ReportDraft, Track

if TYPE_CHECKING:
    from framelab.agent.ledger import Ledger
    from framelab.agent.session import AnalysisSession


class GateThresholds(BaseModel):
    g1_min_coverage: float = 0.90
    g1_max_jump_factor: float = 5.0
    g1_min_points: int = 8  # pendulum: 60
    g1_min_cycles: float = 3.0  # pendulum only
    g2_max_scale_disagreement_pct: float = 5.0  # pendulum L check: 8.0
    g3_min_r2: float = 0.995  # pendulum: 0.98
    g3_max_rms_px: float = 2.0
    g3_min_durbin_watson: float = 1.0  # below → warn; below 0.5 → fail
    g3_fail_durbin_watson: float = 0.5
    g4_range: tuple[float, float] = (9.0, 10.6)
    g4_slowmo_k: tuple[int, ...] = (2, 4, 8)


# Message templates (EN, ID). Filled with metrics that code formatted; never with model text.
TEMPLATES: dict[str, tuple[str, str]] = {
    "pass": ("Passed.", "Lulus."),
    "too_few_points": (
        "Only {n} valid points; at least {min} are needed. Re-record with a plain, brightly lit "
        "background and the object in focus.",
        "Hanya {n} titik valid; minimal {min} diperlukan. Rekam ulang dengan latar polos, cahaya "
        "terang, dan objek fokus.",
    ),
    "low_coverage_blur": (
        "Track coverage is {coverage}: the object was not found in some frames, usually from "
        "motion "
        "blur. Use slow motion or more light.",
        "Cakupan lintasan {coverage}: objek tidak terdeteksi di sebagian frame, biasanya karena "
        "blur "
        "gerak. Gunakan slow motion atau tambah cahaya.",
    ),
    "id_switch": (
        "The tracked object jumps ({jump}× the median step) or changes identity. Check that only "
        "one object of this colour is in view.",
        "Objek yang dilacak melompat ({jump}× langkah median) atau berganti identitas. Pastikan "
        "hanya "
        "satu objek berwarna ini yang terlihat.",
    ),
    "too_few_cycles": (
        "Only {cycles} full swings were tracked; at least {min} are needed. Record a longer swing.",
        "Hanya {cycles} ayunan penuh yang terlacak; minimal {min} diperlukan. Rekam ayunan yang "
        "lebih "
        "panjang.",
    ),
    "no_scale": (
        "No scale reference was found. Put a ruler in the plane of motion or give the ball "
        "diameter.",
        "Tidak ada referensi skala. Letakkan penggaris pada bidang gerak atau berikan diameter "
        "bola.",
    ),
    "single_scale": (
        "Only one scale method is available, so scale consistency cannot be checked.",
        "Hanya satu metode skala yang tersedia, sehingga konsistensi skala tidak bisa diperiksa.",
    ),
    "ruler_offplane": (
        "Ruler scale ({s1}) and ball-diameter scale ({s2}) disagree by {pct}. The ruler is "
        "probably "
        "closer to or farther from the camera than the ball's path. Tape the ruler to the wall "
        "directly behind the falling ball.",
        "Skala penggaris ({s1}) dan skala diameter bola ({s2}) berbeda {pct}. Penggaris "
        "kemungkinan "
        "tidak sejajar dengan lintasan bola. Tempel penggaris di dinding tepat di belakang "
        "lintasan "
        "bola.",
    ),
    "spoken_length_mismatch": (
        "The spoken length and the measured pivot-to-ball length disagree by {pct}. Re-measure the "
        "string length.",
        "Panjang yang diucapkan dan panjang pivot-ke-bola yang terukur berbeda {pct}. Ukur ulang "
        "panjang tali.",
    ),
    "wrong_window": (
        "The fit explains the track poorly (R² {r2}, RMS {rms} px). The fit window may include "
        "contact or hands.",
        "Fit menjelaskan lintasan dengan buruk (R² {r2}, RMS {rms} px). Jendela fit mungkin memuat "
        "kontak atau tangan.",
    ),
    "systematic_residuals": (
        "The residuals are correlated (Durbin–Watson {dw}). Check the fit window and whether the "
        "camera moved.",
        "Residu berkorelasi (Durbin–Watson {dw}). Periksa jendela fit dan apakah kamera bergerak.",
    ),
    "out_of_range": (
        "The headline value {value} is outside the plausible range {lo}–{hi}. The data are "
        "unchanged; check the scale and the fit window.",
        "Nilai utama {value} berada di luar rentang wajar {lo}–{hi}. Data tidak diubah; periksa "
        "skala dan jendela fit.",
    ),
    "slowmo_retimed": (
        "This looks like a slow-motion video saved at normal playback speed. Re-upload and set "
        "capture fps = {fps}.",
        "Ini tampak seperti video slow motion yang disimpan dengan kecepatan normal. Unggah ulang "
        "dan atur capture fps = {fps}.",
    ),
    "bare_number": (
        "The report text contains numbers that did not come from a claim: {numbers}. Refer to "
        "claims by id (for example c01) in placeholders instead.",
        "Teks laporan memuat angka yang bukan berasal dari claim: {numbers}. Rujuk claim dengan id "
        "(misalnya c01) dalam placeholder.",
    ),
    "unknown_claim": (
        "The report refers to claims that do not exist: {ids}.",
        "Laporan merujuk ke claim yang tidak ada: {ids}.",
    ),
    "claims_mismatch": (
        "claims_used does not match the placeholders in the text. Missing: {missing}. "
        "Extra: {extra}.",
        "claims_used tidak cocok dengan placeholder di teks. Kurang: {missing}. Lebih: {extra}.",
    ),
    "missing_hypothesis_claim": (
        "The student stated a prediction, but no hypothesis_test claim exists.",
        "Siswa menyatakan prediksi, tetapi belum ada claim hypothesis_test.",
    ),
    "no_analysis": (
        "The fit has not been computed yet.",
        "Fit belum dihitung.",
    ),
}


def _gate(
    gate: str,
    name: str,
    status: str,
    *,
    hard: bool,
    diagnosis: str | None = None,
    remedy: str = "none",
    metrics: dict[str, Any] | None = None,
    thresholds: dict[str, Any] | None = None,
    fill: dict[str, Any] | None = None,
) -> GateResult:
    key = diagnosis if diagnosis and status != "pass" else "pass"
    en, idn = TEMPLATES.get(key, TEMPLATES["pass"])
    values = fill or {}
    return GateResult(
        gate=gate,  # type: ignore[arg-type]
        name=name,
        status=status,  # type: ignore[arg-type]
        hard=hard,
        metrics=metrics or {},
        thresholds=thresholds or {},
        message_en=en.format(**values) if status != "pass" else en,
        message_id=idn.format(**values) if status != "pass" else idn,
        remedy=remedy,  # type: ignore[arg-type]
        diagnosis=diagnosis if status != "pass" else None,
    )


# --------------------------------------------------------------------------------------------
# G1 track integrity
# --------------------------------------------------------------------------------------------


def jump_factor(track: Track, *, half_window: int = 5, moving_px: float = 0.5) -> float:
    """Largest centroid step divided by the median of its neighbours (06 §7 G1 ``max_jump``).

    A smooth fall has steps that grow slowly, so each step is compared with its local median,
    not the global one (a track holds still before release, which makes the global median zero).
    Steps where the neighbours are nearly still (< ``moving_px`` px per frame) are skipped.
    """
    valid = [p for p in track.points if p.valid]
    if len(valid) < 3:
        return 0.0
    xy = np.array([[p.x_px, p.y_px] for p in valid], dtype=float)
    frames = np.array([round(p.t * track.fps) for p in valid], dtype=float)
    # step per frame: a gap in the track (interpolated frames) must not look like a jump
    steps = np.hypot(*np.diff(xy, axis=0).T) / np.maximum(np.diff(frames), 1.0)
    ratios = []
    for i, step in enumerate(steps):
        neighbours = np.concatenate([steps[max(0, i - half_window) : i],
                                     steps[i + 1 : i + 1 + half_window]])  # fmt: skip
        reference = float(np.median(neighbours)) if neighbours.size else 0.0
        if reference > moving_px:
            ratios.append(float(step) / reference)
    return max(ratios, default=0.0)


def gate_track(
    track: Track,
    thresholds: GateThresholds,
    *,
    min_points: int,
    cycles: float | None = None,
) -> GateResult:
    valid = [p for p in track.points if p.valid]
    n_valid = len(valid)
    jump = jump_factor(track)
    metrics: dict[str, Any] = {
        "coverage": round(track.coverage, 4),
        "n_valid": n_valid,
        "jump_factor": round(jump, 3),
        "id_switches": track.id_switches_suspected,
    }
    thr: dict[str, Any] = {
        "min_coverage": thresholds.g1_min_coverage,
        "min_points": min_points,
        "max_jump_factor": thresholds.g1_max_jump_factor,
    }
    fill = {
        "n": n_valid,
        "min": min_points,
        "coverage": f"{track.coverage:.0%}",
        "missing": len(track.points) - n_valid,
        "jump": f"{jump:.1f}",
        "cycles": f"{cycles:.1f}" if cycles is not None else "",
    }
    if cycles is not None:
        metrics["cycles"] = round(cycles, 2)
        thr["min_cycles"] = thresholds.g1_min_cycles
        fill["min"] = thresholds.g1_min_cycles
    name = "Track integrity"
    if track.coverage < thresholds.g1_min_coverage:
        return _gate("G1", name, "fail", hard=True, diagnosis="low_coverage_blur",
                     remedy="rerecord", metrics=metrics, thresholds=thr, fill=fill)  # fmt: skip
    if n_valid < min_points:
        return _gate("G1", name, "fail", hard=True, diagnosis="too_few_points",
                     remedy="rerecord", metrics=metrics, thresholds=thr, fill=fill)  # fmt: skip
    if track.id_switches_suspected or jump > thresholds.g1_max_jump_factor:
        return _gate("G1", name, "fail", hard=True, diagnosis="id_switch",
                     remedy="reanalyze", metrics=metrics, thresholds=thr, fill=fill)  # fmt: skip
    if cycles is not None and cycles < thresholds.g1_min_cycles:
        return _gate("G1", name, "fail", hard=True, diagnosis="too_few_cycles",
                     remedy="rerecord", metrics=metrics, thresholds=thr, fill=fill)  # fmt: skip
    return _gate("G1", name, "pass", hard=True, metrics=metrics, thresholds=thr)


# --------------------------------------------------------------------------------------------
# G2 scale consistency
# --------------------------------------------------------------------------------------------


def gate_scale(
    calibration: Calibration | None, thresholds: GateThresholds, *, pendulum: bool = False
) -> GateResult:
    name = "Scale consistency"
    if calibration is None:
        return _gate("G2", name, "fail", hard=True, diagnosis="no_scale", remedy="rerecord")
    if calibration.secondary is None or calibration.agreement_pct is None:
        return _gate("G2", name, "warn", hard=False, diagnosis="single_scale", remedy="none",
                     metrics={"methods": 1})  # fmt: skip
    pct = abs(calibration.agreement_pct)
    limit = thresholds.g2_max_scale_disagreement_pct
    metrics = {"agreement_pct": round(pct, 3), "methods": 2}
    thr = {"max_disagreement_pct": limit}
    if pct <= limit:
        return _gate("G2", name, "pass", hard=True, metrics=metrics, thresholds=thr)
    primary, secondary = calibration.primary, calibration.secondary
    diagnosis = "spoken_length_mismatch" if pendulum else "ruler_offplane"
    fill = {
        "s1": f"{primary.m_per_px * 1000:.3f} mm/px",
        "s2": f"{secondary.m_per_px * 1000:.3f} mm/px",
        "pct": f"{pct:.1f}%",
    }
    return _gate("G2", name, "fail", hard=True, diagnosis=diagnosis, remedy="rerecord",
                 metrics=metrics, thresholds=thr, fill=fill)  # fmt: skip


# --------------------------------------------------------------------------------------------
# G3 fit quality
# --------------------------------------------------------------------------------------------


def gate_fit(fit: dict[str, Any], thresholds: GateThresholds) -> GateResult:
    r2, rms, dw = float(fit["r2"]), float(fit["rms_px"]), float(fit["durbin_watson"])
    metrics = {"r2": round(r2, 6), "rms_px": round(rms, 3), "durbin_watson": round(dw, 3)}
    thr = {
        "min_r2": thresholds.g3_min_r2,
        "max_rms_px": thresholds.g3_max_rms_px,
        "min_durbin_watson": thresholds.g3_min_durbin_watson,
    }
    name = "Fit quality"
    fill = {"r2": f"{r2:.4f}", "rms": f"{rms:.2f}", "dw": f"{dw:.2f}"}
    if r2 < thresholds.g3_min_r2 or rms > thresholds.g3_max_rms_px:
        return _gate("G3", name, "fail", hard=True, diagnosis="wrong_window", remedy="reanalyze",
                     metrics=metrics, thresholds=thr, fill=fill)  # fmt: skip
    if dw < thresholds.g3_fail_durbin_watson:
        return _gate("G3", name, "fail", hard=True, diagnosis="systematic_residuals",
                     remedy="reanalyze", metrics=metrics, thresholds=thr, fill=fill)  # fmt: skip
    if dw < thresholds.g3_min_durbin_watson:
        return _gate("G3", name, "warn", hard=False, diagnosis="systematic_residuals",
                     remedy="reanalyze", metrics=metrics, thresholds=thr, fill=fill)  # fmt: skip
    return _gate("G3", name, "pass", hard=True, metrics=metrics, thresholds=thr)


# --------------------------------------------------------------------------------------------
# G4 plausibility (soft: diagnosis only, never changes data)
# --------------------------------------------------------------------------------------------


def slowmo_factor(g_measured: float, g_ref: float, ks: tuple[int, ...] = (2, 4, 8)) -> int | None:
    """Return k if g·k² is within 5 % of g_ref (retimed slow-motion signature), else None."""
    for k in ks:
        if abs(g_measured * k * k - g_ref) / g_ref < 0.05:
            return k
    return None


def gate_plausibility(
    value: float, thresholds: GateThresholds, *, g_ref: float = G_REF_DEFAULT
) -> GateResult:
    lo, hi = thresholds.g4_range
    name = "Physical plausibility"
    metrics = {"value": round(value, 4)}
    thr = {"range_lo": lo, "range_hi": hi}
    k = slowmo_factor(value, g_ref, thresholds.g4_slowmo_k)
    if k is not None:
        metrics["slowmo_k"] = k
        return _gate("G4", name, "warn", hard=False, diagnosis="slowmo_retimed", remedy="none",
                     metrics=metrics, thresholds=thr, fill={"fps": 30 * k})  # fmt: skip
    if not lo <= value <= hi:
        return _gate("G4", name, "warn", hard=False, diagnosis="out_of_range", remedy="none",
                     metrics=metrics, thresholds=thr,
                     fill={"value": f"{value:.2f} m/s²", "lo": lo, "hi": hi})  # fmt: skip
    return _gate("G4", name, "pass", hard=False, metrics=metrics, thresholds=thr)


# --------------------------------------------------------------------------------------------
# G5 ledger completeness (report draft)
# --------------------------------------------------------------------------------------------


def draft_texts(draft: ReportDraft | dict[str, Any]) -> list[str]:
    """Every text field of the draft that a reader sees."""
    d = draft.model_dump() if isinstance(draft, ReportDraft) else draft
    texts = [d.get("title", ""), d.get("aim", ""), d.get("setup", ""), d.get("method", "")]
    for section in d.get("results", []):
        texts += [section.get("heading", ""), section.get("body", "")]
    texts += [
        d.get("hypothesis_evaluation") or "",
        d.get("error_analysis", ""),
        d.get("conclusion", ""),
    ]
    return texts


def gate_g5(
    draft: ReportDraft | dict[str, Any], ledger: Ledger, *, hypothesis: bool = False
) -> GateResult:
    """Bare numbers, unknown claim ids, claims_used mismatch, missing hypothesis claim (06 §7)."""
    texts = draft_texts(draft)
    d = draft.model_dump() if isinstance(draft, ReportDraft) else draft
    violations: dict[str, list[str]] = {}

    numbers = [n for text in texts for n in bare_numbers(text)]
    if numbers:
        violations["bare_number"] = sorted(set(numbers))
    found = {p for text in texts for p in PLACEHOLDER_RE.findall(text) if p.startswith("c")}
    known = {c.id for c in ledger.claims()}
    unknown = sorted(found - known)
    if unknown:
        violations["unknown_claim"] = unknown
    used = set(d.get("claims_used", []))
    if used != found:
        violations["claims_mismatch"] = [sorted(found - used), sorted(used - found)]
    if hypothesis and not any(c.claim_type == "hypothesis_test" for c in ledger.claims()):
        violations["missing_hypothesis_claim"] = []

    name = "Ledger completeness"
    if not violations:
        return _gate("G5", name, "pass", hard=True, metrics={"claims": len(found)})
    first = next(iter(violations))
    fill = {
        "numbers": ", ".join(violations.get("bare_number", [])),
        "ids": ", ".join(violations.get("unknown_claim", [])),
        "missing": ", ".join(violations.get("claims_mismatch", [[], []])[0]),
        "extra": ", ".join(violations.get("claims_mismatch", [[], []])[1]),
    }
    result = _gate("G5", name, "fail", hard=True, diagnosis=first, remedy="reanalyze",
                   metrics={"violations": len(violations)}, fill=fill)  # fmt: skip
    return result.model_copy(update={"metrics": {"violations": "; ".join(violations)}})


# --------------------------------------------------------------------------------------------
# Running the gates on a session
# --------------------------------------------------------------------------------------------


def record(ledger: Ledger, gate: GateResult, emit: Any) -> GateResult:
    """Write the gate as ``gate_`` evidence and emit ``gate_result``."""
    ev = ledger.add_evidence(
        "gate",
        "run_gates",
        gate.model_dump(mode="json"),
        f"{gate.gate} {gate.status}" + (f": {gate.diagnosis}" if gate.diagnosis else ""),
    )
    if emit:
        emit("gate_result", gate=gate.gate, status=gate.status, diagnosis=gate.diagnosis,
             remedy=gate.remedy, evidence_id=ev.id)  # fmt: skip
    return gate.model_copy(update={"evidence_id": ev.id})


def run_gates(session: AnalysisSession, thresholds: GateThresholds) -> list[GateResult]:
    """G1–G4 on the current state. Gates that need a fit or an analysis are ``skipped`` until then.

    G5 runs at ``submit_report`` (06 §4), not here.
    """
    pendulum = session.spec.experiment == "pendulum"
    gates = [
        gate_track(session.track, thresholds, min_points=session.min_points),
        gate_scale(session.calibration, thresholds, pendulum=pendulum),
    ]
    if session.latest_fit_summary is not None:
        gates.append(gate_fit(session.latest_fit_summary, thresholds))
    else:
        gates.append(_gate("G3", "Fit quality", "skipped", hard=False, diagnosis="no_analysis"))
    if session.g_value is not None:
        gates.append(gate_plausibility(session.g_value, thresholds))
    else:
        gates.append(_gate("G4", "Physical plausibility", "skipped", hard=False,
                           diagnosis="no_analysis"))  # fmt: skip
    return [record(session.ledger, g, session.ctx.emit) for g in gates]
