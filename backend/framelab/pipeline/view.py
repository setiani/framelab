"""Read-only views of a finished run for the renderers (charts, overlay, report, replay).

Spec: docs/plan/04-pipeline-and-measurement.md §7–§10.

Everything a reader sees comes from the run's files. Headline numbers are ledger claims formatted
with ``physics.uncertainty.fmt``. Per-frame values (time, position) are observations from the track
evidence (``trk_``), not claims, and they are formatted with the same function. Nothing here calls a
model, so ``framelab replay`` works offline.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

from framelab.agent.ledger import Ledger
from framelab.agent.narrative import render_statement
from framelab.physics.uncertainty import fmt, sigma_from_ci
from framelab.schemas import Claim, Evidence

HEADLINE_SYMBOL = {"g": "g", "period": "T"}


@dataclass(frozen=True)
class TrackRow:
    frame: int
    t: float
    x_px: float
    y_px: float
    area_px: float
    valid: bool
    flag: str


@dataclass(frozen=True)
class FitView:
    fit: Evidence
    model: str
    params: dict[str, float]
    t_offset: float
    t_start: float | None
    t_end: float | None


def track_rows(run_dir: Path, ledger: Ledger) -> list[TrackRow]:
    """The rows of the track the latest ``trk_`` evidence names (or the only track file)."""
    tracks = [e for e in ledger.evidence() if e.id.startswith("trk_")]
    name = tracks[-1].data.get("track_id") if tracks else None
    path = run_dir / "tracks" / f"{name}.csv" if name else None
    if path is None or not path.is_file():
        candidates = sorted((run_dir / "tracks").glob("*.csv"))
        if not candidates:
            raise FileNotFoundError("no track file in tracks/")
        path = candidates[0]
    rows = []
    with path.open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            rows.append(
                TrackRow(
                    frame=int(r["frame"]),
                    t=float(r["t_s"]),
                    x_px=float(r["x_px"]),
                    y_px=float(r["y_px"]),
                    area_px=float(r["area_px"]),
                    valid=r["valid"] in ("1", "True", "true"),
                    flag=r["flag"],
                )
            )
    return rows


def latest_evidence(ledger: Ledger, prefix: str, predicate=None) -> Evidence | None:
    for ev in reversed(ledger.evidence()):
        if ev.id.startswith(prefix) and (predicate is None or predicate(ev)):
            return ev
    return None


def scale_m_per_px(ledger: Ledger) -> float | None:
    """The scale the headline was computed with: the ``cal_`` input of the latest g derivation."""
    der = latest_evidence(ledger, "der_", lambda e: e.data.get("quantity") == "g")
    if der is not None:
        for source in der.inputs:
            if source.startswith("cal_"):
                return float(ledger.get(source).data["m_per_px"])
    cal = latest_evidence(ledger, "cal_", lambda e: "m_per_px" in e.data)
    return float(cal.data["m_per_px"]) if cal is not None else None


def fit_view(ledger: Ledger, model: str = "quadratic") -> FitView | None:
    """The latest fit of ``model`` and the time window it was made on (from its ``win_`` input)."""
    fit = latest_evidence(
        ledger, "fit_", lambda e: e.data.get("model") == model and "params" in e.data
    )
    if fit is None:
        return None
    t_start = t_end = None
    for source in fit.inputs:
        if source.startswith("win_"):
            win = ledger.get(source).data
            t_start, t_end = win.get("t_start"), win.get("t_end")
    return FitView(
        fit=fit,
        model=model,
        params={k: float(v) for k, v in fit.data["params"].items()},
        t_offset=float(fit.data.get("t_offset", 0.0)),
        t_start=float(t_start) if t_start is not None else None,
        t_end=float(t_end) if t_end is not None else None,
    )


def _claim_quantity(ledger: Ledger, claim: Claim) -> str | None:
    """The quantity of the ``der_`` evidence a claim copies its value from."""
    try:
        ev = ledger.get(claim.evidence[0])
    except (KeyError, IndexError):
        return None
    return ev.data.get("quantity") if ev.tool == "derive" else None


def headline_claim(ledger: Ledger) -> Claim | None:
    """The latest claim of the headline quantity: g, else (pendulum without a length) the
    measured period. The quantity is read from the claim's ``der_`` evidence, not its label."""
    for quantity in ("g", "period"):
        for claim in reversed(ledger.claims()):
            if claim.value is not None and claim.claim_type in ("derived", "measurement") and (
                _claim_quantity(ledger, claim) == quantity
            ):  # fmt: skip
                return claim
    for claim in reversed(ledger.claims()):  # a claim labelled g on other evidence (older runs)
        if claim.quantity == "g" and claim.value is not None and claim.claim_type in (
            "derived", "measurement",
        ):  # fmt: skip
            return claim
    return None


def headline_text(ledger: Ledger, lang: str) -> str:
    """'g = 9.82 ± 0.03 m/s²' (or 'T = 1.559 ± 0.002 s') from the headline claim, via fmt."""
    claim = headline_claim(ledger)
    if claim is None or claim.value is None:
        return ""
    sigma = sigma_from_ci(*claim.ci95) if claim.ci95 else None
    symbol = HEADLINE_SYMBOL.get(_claim_quantity(ledger, claim) or "", claim.quantity or "")
    return f"{symbol} = {fmt(claim.value, sigma, claim.unit or '', lang)}"


def primary_fit(ledger: Ledger) -> FitView | None:
    """The latest headline-model fit: quadratic (free fall) or damped_sine (pendulum)."""
    fits = [fit_view(ledger, m) for m in ("quadratic", "damped_sine")]
    fits = [f for f in fits if f is not None]
    return max(fits, key=lambda f: int(f.fit.id.split("_")[1])) if fits else None


def predict_px(fit: FitView, t: float) -> float:
    """The fitted position in pixels at time t (y for free fall, x for the pendulum)."""
    if fit.model == "damped_sine":
        p = fit.params
        tau = t - fit.t_offset
        return (p["A"] * math.exp(-p["gamma"] * tau) * math.cos(p["omega"] * tau + p["phi"])
                + p["c"] + p["d"] * tau)  # fmt: skip
    return poly_px(fit.params, fit.t_offset, t)


def pivot(ledger: Ledger) -> dict | None:
    """The pendulum pivot (``calibrate_pivot`` evidence) if it is reliable."""
    ev = latest_evidence(ledger, "cal_", lambda e: e.tool == "calibrate_pivot")
    if ev is None or not ev.data.get("reliable"):
        return None
    return ev.data


def claim_sentence(ledger: Ledger, claim: Claim, lang: str) -> str:
    return render_statement(claim, ledger.get(claim.evidence[0]), lang)


def relative_metres(
    rows: list[TrackRow], m_per_px: float, coord: str = "y"
) -> tuple[float, list[float]]:
    """Reference point (first valid y or x, px) and that coordinate of every row in metres
    (y downward positive). With ``m_per_px`` = 1 the values stay in pixels."""
    valid = [r for r in rows if r.valid]
    if not valid:
        raise ValueError("the track has no valid points")
    attr = "x_px" if coord == "x" else "y_px"
    v0 = getattr(valid[0], attr)
    return v0, [(getattr(r, attr) - v0) * m_per_px for r in rows]


def poly_px(params: dict[str, float], t_offset: float, t: float) -> float:
    """c0 + c1·(t−t0) + c2·(t−t0)² (quadratic, 05 §2.2) in pixels."""
    tau = t - t_offset
    return params.get("c0", 0.0) + params.get("c1", 0.0) * tau + params.get("c2", 0.0) * tau * tau


def fmt_time(t: float, lang: str) -> str:
    return fmt(t, None, "s", lang)
