"""Lab report: Jinja2 HTML → PDF (Playwright Chromium), ID and EN; NEEDS_RERECORD feedback page.

Spec: docs/plan/04-pipeline-and-measurement.md §9–§10

The HTML is self-contained: CSS inline, images as data URIs, no network resource. Every number in
the text is a claim placeholder rendered from the ledger (G5 already enforced this when the draft
was accepted). The export ``outputs/data.csv`` is in SI units.
"""

from __future__ import annotations

import base64
import csv
import json
import re
from pathlib import Path
from typing import Any

import cv2
from jinja2 import Environment, FileSystemLoader, select_autoescape

from framelab.agent import narrative
from framelab.agent.designer import render_text as render_prediction
from framelab.agent.ledger import Ledger
from framelab.physics.uncertainty import fmt
from framelab.pipeline import overlay, view
from framelab.schemas import IngestResult, ReportDraft, RunState
from framelab.store import read_json

TEMPLATES = Path(__file__).resolve().parent.parent / "templates"
PLACEHOLDER = re.compile(r"\{\{(c\d{2,})\}\}")
LABELS = {
    "en": {
        "title_fallback": "Free-fall measurement", "run": "Run", "date": "Date",
        "verdict": "Verdict",
        "aim": "Aim", "setup": "Setup", "method": "Method", "results": "Results",
        "hypothesis": "Hypothesis evaluation", "errors": "Error analysis",
        "conclusion": "Conclusion",
        "next": "Next experiment", "models": "Model", "role": "Role in this run",
        "gates": "Quality gates",
        "appendix_a": "Appendix A: Claim Ledger", "appendix_b": "Appendix B: Gate table",
        "appendix_c": "Appendix C: Data sample", "figure": "Trajectory",
        "keyframe": "Key frame", "claim": "Claim", "evidence": "Evidence", "value": "Value",
        "gate": "Gate", "status": "Status", "diagnosis": "Diagnosis", "message": "Message",
        "time": "t (s)", "x": "x (m)", "y": "y (m)", "valid": "valid", "flag": "flag",
        "protocol": "Protocol", "predictions": "Predictions", "safety": "Safety",
        "variable": "Changed variable", "controlled": "Controlled variables",
        "feedback": "Recording feedback", "tips": "Tips", "failed": "What failed",
        "rerecord": "Record the video again", "page_note": "Generated from the run's saved files.",
        "built": "Built with Meta SAM 3.1 · Muse Spark · Muse Wonder",
    },
    "id": {
        "title_fallback": "Pengukuran jatuh bebas", "run": "Run", "date": "Tanggal",
        "verdict": "Verdict", "aim": "Tujuan", "setup": "Persiapan", "method": "Metode",
        "results": "Hasil", "hypothesis": "Evaluasi hipotesis", "errors": "Analisis galat",
        "conclusion": "Kesimpulan", "next": "Eksperimen berikutnya",
        "models": "Model", "role": "Peran dalam run ini", "gates": "Gerbang kualitas",
        "appendix_a": "Lampiran A: Claim Ledger", "appendix_b": "Lampiran B: Tabel gerbang",
        "appendix_c": "Lampiran C: Contoh data", "figure": "Lintasan",
        "keyframe": "Frame kunci", "claim": "Claim", "evidence": "Evidence", "value": "Nilai",
        "gate": "Gerbang", "status": "Status", "diagnosis": "Diagnosis", "message": "Pesan",
        "time": "t (s)", "x": "x (m)", "y": "y (m)", "valid": "valid", "flag": "flag",
        "protocol": "Protokol", "predictions": "Prediksi", "safety": "Keselamatan",
        "variable": "Variabel yang diubah", "controlled": "Variabel terkendali",
        "feedback": "Umpan balik rekaman", "tips": "Saran", "failed": "Yang gagal",
        "rerecord": "Rekam ulang video", "page_note": "Dibuat dari berkas run yang tersimpan.",
        "built": "Dibangun dengan Meta SAM 3.1 · Muse Spark · Muse Wonder",
    },
}  # fmt: skip
VERDICT_LABEL = {
    "PASS": {"en": "PASS", "id": "LULUS"},
    "PASS_WITH_WARNINGS": {"en": "PASS WITH WARNINGS", "id": "LULUS DENGAN CATATAN"},
    "NEEDS_RERECORD": {"en": "RECORD AGAIN", "id": "REKAM ULANG"},
    "REJECT": {"en": "REJECTED", "id": "DITOLAK"},
}  # fmt: skip


# --------------------------------------------------------------------------------------------
# Text and data helpers
# --------------------------------------------------------------------------------------------


def render_text(text: str, ledger: Ledger, lang: str) -> str:
    """Claim placeholders replaced by their phrase ("g = 9.82 ± 0.03 m/s²"). Jinja escapes it."""
    phrases = {c.id: narrative.phrase(c, ledger.get(c.evidence[0]), lang) for c in ledger.claims()}
    return PLACEHOLDER.sub(lambda m: phrases.get(m.group(1), "[?]"), text)


def data_uri(path: Path, mime: str) -> str:
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def keyframe_uri(run_dir: Path) -> str | None:
    frames = sorted((run_dir / "ingest" / "keyframes").glob("frame_*.png"))
    return data_uri(frames[0], "image/png") if frames else None


def claims_table(ledger: Ledger, lang: str) -> list[dict[str, Any]]:
    rows = []
    for claim in ledger.claims():
        rows.append({
            "id": claim.id,
            "type": claim.claim_type,
            "text": view.claim_sentence(ledger, claim, lang),
            "evidence": ", ".join(claim.evidence),
        })  # fmt: skip
    return rows


def gates_table(ledger: Ledger, lang: str) -> list[dict[str, Any]]:
    """The latest result of each gate, with its code-written message in the report language."""
    latest: dict[str, dict[str, Any]] = {}
    for ev in ledger.evidence():
        if ev.tool in ("run_gates", "submit_report") and ev.id.startswith("gate_"):
            latest[ev.data.get("gate", "")] = ev.data
    key = "message_en" if lang == "en" else "message_id"
    return [{
        "gate": g, "name": d.get("name", ""), "status": d.get("status", ""),
        "hard": "hard" if d.get("hard") else "soft", "diagnosis": d.get("diagnosis") or "",
        "message": d.get(key, ""),
    } for g, d in sorted(latest.items())]  # fmt: skip


def data_sample(run_dir: Path, lang: str, rows: int = 25) -> list[dict[str, str]]:
    path = run_dir / "outputs" / "data.csv"
    if not path.is_file():
        return []
    body = [line for line in path.read_text(encoding="utf-8").splitlines()
            if line and not line.startswith("#")]  # fmt: skip
    reader = list(csv.DictReader(body))
    step = max(1, len(reader) // rows)
    return [{"t": r["t_s"], "x": r["x_m"], "y": r["y_m"], "valid": r["valid"], "flag": r["flag"]}
            for r in reader[::step]]  # fmt: skip


def models_used(state: RunState, verdict: dict[str, Any] | None) -> list[dict[str, str]]:
    """Which model did what in this run, from the run's own records."""
    auditor = (verdict or {}).get("auditor_model") or "—"
    return [
        {"role": "Muse Voice Transcribe", "use": "speech → transcript"},
        {"role": "Muse Spark 1.3", "use": "Planner, Analyst, Auditor, Designer, Translator"},
        {"role": "SAM 3.1", "use": "object tracking on the video and the ruler"},
        {
            "role": "Muse Wonder 1.0",
            "use": "not used in this run" if auditor != "muse-wonder-1.0" else "Auditor",
        },  # fmt: skip
    ]


def next_experiment(run_dir: Path, lang: str) -> dict[str, Any] | None:
    path = run_dir / "analysis" / "designer.json"
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    preds = data.get("predicted_values", [])
    pred_text = "prediction_en" if lang == "en" else "prediction_id"
    steps = "protocol_steps_en" if lang == "en" else "protocol_steps_id"
    safety = "safety_notes_en" if lang == "en" else "safety_notes_id"
    return {
        "title": data["title"],
        "variable": data["changed_variable"],
        "controlled": data.get("controlled_variables", []),
        "prediction": render_prediction(data.get(pred_text, ""), preds, lang),
        "predictions": [
            f"{fmt(p['input'], None, p['unit_in'], lang)} → {fmt(p['predicted'], None, p['unit_out'], lang)}"  # noqa: E501
            for p in preds
        ],
        "steps": data.get(steps, []),
        "safety": data.get(safety, []),
    }  # fmt: skip


# --------------------------------------------------------------------------------------------
# HTML and PDF
# --------------------------------------------------------------------------------------------


def _env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(TEMPLATES)),
        autoescape=select_autoescape(["html", "j2"]),
    )


def render_html(run_dir: Path, draft: ReportDraft, state: RunState, ledger: Ledger) -> str:
    """Fill templates/report.html.j2 for one language."""
    lang = draft.lang
    verdict = read_json(run_dir / "analysis" / "verdict.json") if (
        run_dir / "analysis" / "verdict.json"
    ).is_file() else None  # fmt: skip
    chart = run_dir / "outputs" / "chart_trajectory.png"
    figure = data_uri(chart, "image/png") if chart.is_file() else None
    status = state.verdict or ""
    context = {
        "lang": lang,
        "t": LABELS[lang],
        "title": render_text(draft.title, ledger, lang),
        "run_id": state.run_id,
        "date": state.created_at.date().isoformat(),
        "verdict": status,
        "verdict_label": VERDICT_LABEL.get(status, {}).get(lang, status),
        "headline": view.headline_text(ledger, lang),
        "aim": render_text(draft.aim, ledger, lang),
        "setup": render_text(draft.setup, ledger, lang),
        "method": render_text(draft.method, ledger, lang),
        "results": [
            {"heading": s.heading, "body": render_text(s.body, ledger, lang)} for s in draft.results
        ],  # fmt: skip
        "hypothesis": render_text(draft.hypothesis_evaluation, ledger, lang)
        if draft.hypothesis_evaluation
        else None,  # fmt: skip
        "errors": render_text(draft.error_analysis, ledger, lang),
        "conclusion": render_text(draft.conclusion, ledger, lang),
        "figure": figure,
        "keyframe": keyframe_uri(run_dir),
        "models": models_used(state, verdict),
        "gates": gates_table(ledger, lang),
        "claims": claims_table(ledger, lang),
        "sample": data_sample(run_dir, lang),
        "next": next_experiment(run_dir, lang),
        "auditor_model": (verdict or {}).get("auditor_model", ""),
        "built": LABELS[lang]["built"],
    }
    return _env().get_template("report.html.j2").render(**context)


def html_to_pdf(html: str, out_pdf: Path) -> Path:
    """Playwright Chromium, A4, print backgrounds. Sync API: call it from a worker thread."""
    from playwright.sync_api import sync_playwright

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        try:
            page = browser.new_page()
            page.set_content(html, wait_until="load")
            page.pdf(path=str(out_pdf), format="A4", print_background=True,
                     margin={"top": "0", "bottom": "0", "left": "0", "right": "0"})  # fmt: skip
        finally:
            browser.close()
    return out_pdf


def write_reports(run_dir: Path, ledger: Ledger, state: RunState) -> dict[str, Path]:
    """report_<lang>.html/.pdf for each language that has a draft; feedback page if re-record."""
    out = run_dir / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    drafts = {}
    main = run_dir / "draft_report.json"
    if main.is_file():
        d = ReportDraft.model_validate(read_json(main))
        drafts[d.lang] = d
    for lang in ("en", "id"):
        other = run_dir / f"draft_{lang}.json"
        if other.is_file() and lang not in drafts:
            drafts[lang] = ReportDraft.model_validate(read_json(other))
    for lang, draft in drafts.items():
        html = render_html(run_dir, draft, state, ledger)
        pdf = html_to_pdf(html, out / f"report_{lang}.pdf")
        written[f"report_{lang}"] = pdf
    if state.verdict == "NEEDS_RERECORD":
        written["feedback"] = html_to_pdf(render_feedback(run_dir, state, ledger),
                                          out / "recording_feedback.pdf")  # fmt: skip
    return written


# --------------------------------------------------------------------------------------------
# Recording feedback (NEEDS_RERECORD)
# --------------------------------------------------------------------------------------------


def annotated_keyframe(run_dir: Path) -> str | None:
    """A frame where the tracked object is visible, with the reference outline (amber, on a dark
    underlay so it shows on the yellow ruler) and the target outline (cyan) drawn on it."""
    try:
        targets, reference = overlay.load_masks(run_dir)
    except (OSError, ValueError):
        return None
    ing = IngestResult.model_validate(read_json(run_dir / "ingest" / "ingest.json"))
    index = min(targets) if targets else 0
    cap = cv2.VideoCapture(str(run_dir / ing.video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, image = cap.read()
    cap.release()
    if not ok:
        return None
    for poly in reference:
        cv2.polylines(image, [poly], True, (0, 0, 0), 5)
        cv2.polylines(image, [poly], True, overlay.REFERENCE_BGR, 2)
    for poly in targets.get(index, []):
        cv2.polylines(image, [poly], True, (0, 0, 0), 4)
        cv2.polylines(image, [poly], True, overlay.TARGET_BGR, 2)
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        return None
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def render_feedback(run_dir: Path, state: RunState, ledger: Ledger) -> str:
    """One A4 page: what failed (gate messages), the annotated key frame, and the tips."""
    verdict = read_json(run_dir / "analysis" / "verdict.json") if (
        run_dir / "analysis" / "verdict.json"
    ).is_file() else {}  # fmt: skip
    gates_en = gates_table(ledger, "en")
    gates_id = gates_table(ledger, "id")
    failed = [
        {"gate": g["gate"], "en": g["message"], "id": h["message"]}
        for g, h in zip(gates_en, gates_id, strict=True)
        if g["status"] == "fail"
    ]
    context = {
        "run_id": state.run_id,
        "date": state.created_at.date().isoformat(),
        "verdict_label": VERDICT_LABEL["NEEDS_RERECORD"],
        "failed": failed,
        # the gate messages are already listed under "what failed": keep the other tips only
        "tips_en": [t for t in verdict.get("rerecord_tips_en", [])
                    if t not in {f["en"] for f in failed}],  # fmt: skip
        "tips_id": [t for t in verdict.get("rerecord_tips_id", [])
                    if t not in {f["id"] for f in failed}],  # fmt: skip

        "keyframe": annotated_keyframe(run_dir),
        "t_en": LABELS["en"], "t_id": LABELS["id"],
    }  # fmt: skip
    return _env().get_template("feedback.html.j2").render(**context)


# --------------------------------------------------------------------------------------------
# Data export (04 §10)
# --------------------------------------------------------------------------------------------


def export_csv(run_dir: Path) -> Path:
    """outputs/data.csv: frame, t_s, x_m, y_m, area_m2, valid, flag. Relative to the first valid
    point; y downward positive; header comments with the run id, the scale and the fps."""
    state = RunState.model_validate(read_json(run_dir / "state.json"))
    ledger = Ledger(run_dir / "analysis" / "ledger.jsonl")
    m = view.scale_m_per_px(ledger)
    if m is None:
        raise ValueError("no scale in the ledger: data.csv has no metres")
    rows = view.track_rows(run_dir, ledger)
    valid = [r for r in rows if r.valid]
    if not valid:
        raise ValueError("the track has no valid points")
    x0, y0 = valid[0].x_px, valid[0].y_px
    fps = IngestResult.model_validate(read_json(run_dir / "ingest" / "ingest.json")).file_fps
    method = view.latest_evidence(ledger, "cal_", lambda e: "m_per_px" in e.data)
    name = method.data.get("method", "?") if method is not None else "?"
    path = run_dir / "outputs" / "data.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        fh.write(f"# run_id: {state.run_id}\n")
        fh.write(f"# scale: {m * 1000:.4f} mm/px ({name}); y downward positive\n")
        fh.write(f"# fps: {fps:g}\n")
        writer = csv.writer(fh)
        writer.writerow(["frame", "t_s", "x_m", "y_m", "area_m2", "valid", "flag"])
        for r in rows:
            writer.writerow([
                r.frame, f"{r.t:.6f}", f"{(r.x_px - x0) * m:.6f}", f"{(r.y_px - y0) * m:.6f}",
                f"{r.area_px * m * m:.8f}", int(r.valid), r.flag,
            ])  # fmt: skip
    return path
