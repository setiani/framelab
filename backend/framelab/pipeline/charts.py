"""Charts: Plotly JSON for the web, matplotlib PNG (1600×900, 200 dpi) for the PDF.

Spec: docs/plan/04-pipeline-and-measurement.md §7
Free fall: y(t) in metres (downward positive) with the fit, residuals in mm. Pendulum: x(t) in
metres with the damped-sine fit, residuals in mm (pixels when the run has no scale). Series: T₀²
against L with error bars, the regression line and an approximate 95 % band.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # no display needed; the charts go to files
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from framelab.agent.ledger import Ledger  # noqa: E402
from framelab.pipeline import view  # noqa: E402
from framelab.store import write_json_atomic  # noqa: E402

DPI = 200
SIZE_IN = (8.0, 4.5)  # 8 × 200 = 1600 px, 4.5 × 200 = 900 px
COLOR_VALID = "#0B7A99"
COLOR_INVALID = "#9AA0A6"
COLOR_OUTLIER = "#D93025"
COLOR_FIT = "#111827"
LABELS = {
    "en": {"title": "Free fall: height against time", "x": "time (s)",
           "y": "y (m, downward)", "res": "residual (mm)",
           "valid": "tracked", "invalid": "not used", "fit": "fit"},  # fmt: skip
    "id": {"title": "Jatuh bebas: posisi terhadap waktu", "x": "waktu (s)",
           "y": "y (m, ke bawah)", "res": "sisa (mm)",
           "valid": "terlacak", "invalid": "tidak dipakai", "fit": "fit"},  # fmt: skip
}  # fmt: skip
PENDULUM_LABELS = {
    "en": {"title": "Pendulum: horizontal position against time", "y": "x (m)",
           "y_px": "x (px)", "res_px": "residual (px)"},  # fmt: skip
    "id": {"title": "Bandul: posisi horizontal terhadap waktu", "y": "x (m)",
           "y_px": "x (px)", "res_px": "sisa (px)"},  # fmt: skip
}  # fmt: skip


def _labels(data: dict[str, Any]) -> dict[str, str]:
    lab = dict(LABELS[data["lang"]])
    if data.get("coord") == "x":
        pend = PENDULUM_LABELS[data["lang"]]
        lab["title"] = pend["title"]
        lab["y"] = pend["y"] if data.get("metres", True) else pend["y_px"]
        if not data.get("metres", True):
            lab["res"] = pend["res_px"]
    return lab


def build_trajectory(run_dir: Path, ledger: Ledger, lang: str) -> dict[str, Any]:
    """The arrays behind both charts. Metres are relative to the first valid point."""
    fit = view.primary_fit(ledger)
    coord = "x" if fit is not None and fit.model == "damped_sine" else "y"
    m = view.scale_m_per_px(ledger)
    metres = m is not None
    if m is None and coord == "y":
        raise ValueError("no scale in the ledger: the trajectory has no metres")
    m = m if m is not None else 1.0  # pendulum without a scale: pixels
    res_factor = 1000.0 if metres else 1.0
    rows = view.track_rows(run_dir, ledger)
    y0, y_m = view.relative_metres(rows, m, coord)
    data: dict[str, Any] = {
        "lang": lang,
        "coord": coord,
        "metres": metres,
        "title": view.headline_text(ledger, lang),
        "valid": [
            {"t": r.t, "y": y, "frame": r.frame} for r, y in zip(rows, y_m, strict=True) if r.valid
        ],  # fmt: skip
        "invalid": [
            {"t": r.t, "y": y, "frame": r.frame, "flag": r.flag}
            for r, y in zip(rows, y_m, strict=True)
            if not r.valid
        ],  # fmt: skip
        "fit": [],
        "residuals": [],
    }
    if fit is not None and fit.t_start is not None and fit.t_end is not None:
        grid = np.linspace(fit.t_start, fit.t_end, 200)
        n_grid = 200 if coord == "y" else 2000
        grid = np.linspace(fit.t_start, fit.t_end, n_grid)
        curve = [(view.predict_px(fit, t) - y0) * m for t in grid]
        data["fit"] = [{"t": float(t), "y": float(y)} for t, y in zip(grid, curve, strict=True)]
        for r in rows:
            if r.valid and fit.t_start - 1e-9 <= r.t <= fit.t_end + 1e-9:
                value = r.x_px if coord == "x" else r.y_px
                resid_px = value - view.predict_px(fit, r.t)
                data["residuals"].append({"t": r.t, "mm": resid_px * m * res_factor})
    return data


def _plotly(data: dict[str, Any]) -> dict[str, Any]:
    lab = _labels(data)
    traces: list[dict[str, Any]] = [
        {
            "type": "scatter",
            "mode": "markers",
            "name": lab["valid"],
            "xaxis": "x",
            "yaxis": "y",
            "x": [p["t"] for p in data["valid"]],
            "y": [p["y"] for p in data["valid"]],
            "customdata": [p["frame"] for p in data["valid"]],
            "hovertemplate": (
                "frame %{customdata}<br>t = %{x:.3f} s<br>"
                + data.get("coord", "y")
                + (" = %{y:.3f} m" if data.get("metres", True) else " = %{y:.1f} px")
                + "<extra></extra>"
            ),
            "marker": {"color": COLOR_VALID, "size": 6},
        },  # fmt: skip
        {
            "type": "scatter",
            "mode": "markers",
            "name": lab["invalid"],
            "xaxis": "x",
            "yaxis": "y",
            "x": [p["t"] for p in data["invalid"]],
            "y": [p["y"] for p in data["invalid"]],
            "customdata": [p["frame"] for p in data["invalid"]],
            "marker": {"color": COLOR_INVALID, "size": 6, "symbol": "circle-open"},
        },  # fmt: skip
    ]
    if data["fit"]:
        traces.append({"type": "scatter", "mode": "lines", "name": lab["fit"], "xaxis": "x",
                       "yaxis": "y", "x": [p["t"] for p in data["fit"]],
                       "y": [p["y"] for p in data["fit"]],
                       "line": {"color": COLOR_FIT, "width": 2}})  # fmt: skip
    if data["residuals"]:
        traces.append({"type": "scatter", "mode": "markers", "name": lab["res"], "xaxis": "x",
                       "yaxis": "y2", "x": [p["t"] for p in data["residuals"]],
                       "y": [p["mm"] for p in data["residuals"]],
                       "marker": {"color": COLOR_VALID, "size": 4}})  # fmt: skip
    layout = {
        "title": {"text": f"{lab['title']} · {data['title']}".strip(" ·")},
        "xaxis": {"title": lab["x"], "anchor": "y2"},
        "yaxis": {
            "title": lab["y"],
            "domain": [0.32, 1.0],
            **({"autorange": "reversed"} if data.get("coord", "y") == "y" else {}),
        },
        "yaxis2": {"title": lab["res"], "domain": [0.0, 0.24]},
        "hovermode": "closest",
        "showlegend": True,
        "template": "plotly_white",
    }
    return {"data": traces, "layout": layout}


def _png(data: dict[str, Any], path: Path) -> None:
    lab = _labels(data)
    fig, (ax, rx) = plt.subplots(
        2, 1, sharex=True, figsize=SIZE_IN, gridspec_kw={"height_ratios": [3, 1]}
    )
    ax.scatter([p["t"] for p in data["valid"]], [p["y"] for p in data["valid"]],
               s=10, color=COLOR_VALID, label=lab["valid"])  # fmt: skip
    if data["invalid"]:
        ax.scatter([p["t"] for p in data["invalid"]], [p["y"] for p in data["invalid"]],
                   s=14, facecolors="none", edgecolors=COLOR_INVALID,
                   label=lab["invalid"])  # fmt: skip
    if data["fit"]:
        ax.plot([p["t"] for p in data["fit"]], [p["y"] for p in data["fit"]],
                color=COLOR_FIT, lw=1.5, label=lab["fit"])  # fmt: skip
    if data.get("coord", "y") == "y":
        ax.invert_yaxis()  # y downward, as in the video
    ax.set_ylabel(lab["y"])
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=8, frameon=False)
    ax.set_title(f"{lab['title']} · {data['title']}".strip(" ·"), fontsize=11, loc="left")
    if data["residuals"]:
        rx.axhline(0, color=COLOR_INVALID, lw=0.8)
        rx.scatter([p["t"] for p in data["residuals"]], [p["mm"] for p in data["residuals"]],
                   s=8, color=COLOR_VALID)  # fmt: skip
    rx.set_ylabel(lab["res"])
    rx.set_xlabel(lab["x"])
    rx.grid(alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def write_trajectory(run_dir: Path, ledger: Ledger, lang: str) -> dict[str, Path]:
    """outputs/chart_trajectory.json (Plotly) and outputs/chart_trajectory.png (matplotlib)."""
    data = build_trajectory(run_dir, ledger, lang)
    out = run_dir / "outputs"
    json_path = out / "chart_trajectory.json"
    png_path = out / "chart_trajectory.png"
    write_json_atomic(json_path, _plotly(data))
    _png(data, png_path)
    return {"chart_trajectory_json": json_path, "chart_trajectory_png": png_path}


def series_chart(series_dir: Path) -> dict[str, Path]:
    """T₀² against L with 2·T·σ_T error bars, the regression line and an approximate 95 % band.

    The band ignores the covariance of the intercept and the slope, so it is a little wide at
    the ends; the regression itself is exact (see pipeline/series.py).
    """
    result = json.loads((series_dir / "analysis" / "series.json").read_text(encoding="utf-8"))
    pts = result["points"]
    L = np.array([p["L"] for p in pts])
    T2 = np.array([p["T0"] ** 2 for p in pts])
    err = np.array([2 * p["T0"] * p["sigma_T"] for p in pts])
    lin = result["linear"]
    grid = np.linspace(0.0, float(L.max()) * 1.05, 100)
    fit = lin["k"] * grid + lin["b"]
    band = 1.96 * np.sqrt(lin["sigma_b"] ** 2 + (grid * lin["sigma_k"]) ** 2)
    fig, ax = plt.subplots(figsize=SIZE_IN)
    ax.errorbar(L, T2, yerr=err, fmt="o", color=COLOR_VALID, ms=4, capsize=3, label="runs")
    ax.plot(grid, fit, color=COLOR_FIT, lw=1.5, label="fit")
    ax.fill_between(grid, fit - band, fit + band, color=COLOR_VALID, alpha=0.15, label="95 %")
    ax.set_xlabel("L (m)")
    ax.set_ylabel("T₀² (s²)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    out = series_dir / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    png = out / "chart_series.png"
    fig.savefig(png, dpi=DPI)
    plt.close(fig)
    write_json_atomic(out / "chart_series.json", {
        "points": pts, "linear": lin, "band": "approximate 95 %"})  # fmt: skip
    return {"chart_series_png": png, "chart_series_json": out / "chart_series.json"}
