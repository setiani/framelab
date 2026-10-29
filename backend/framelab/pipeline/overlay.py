"""Annotated overlay MP4: mask contours, trail, HUD, result card.

Spec: docs/plan/04-pipeline-and-measurement.md §8
Target contour #00E0FF, reference contour #FFC400 (faint on every frame), last 30 valid centroids
as a fading trail, outliers as red crosses, HUD top-left, and a 1.5 s end card on the last frame.
Frames go to ffmpeg through stdin, so nothing is written to disk until the MP4 is done.
"""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np
from meta_sam_parser import decode_mask_to_raster
from PIL import Image, ImageDraw, ImageFont

from framelab.agent.ledger import Ledger
from framelab.clients import sam as sam_client
from framelab.physics.uncertainty import fmt
from framelab.pipeline import ingest as ingest_mod
from framelab.pipeline import measure, view
from framelab.pipeline.view import TrackRow
from framelab.schemas import IngestResult, RunState
from framelab.store import read_json

# BGR for OpenCV: #00E0FF → (255, 224, 0), #FFC400 → (0, 196, 255)
TARGET_BGR = (255, 224, 0)
REFERENCE_BGR = (0, 196, 255)
OUTLIER_BGR = (40, 40, 220)
TRAIL_LEN = 30
END_CARD_S = 1.5
FONT_PATH = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"
BUILT_WITH = {
    "en": "Built with Meta SAM 3.1 · Muse Spark · Muse Wonder",
    "id": "Dibangun dengan Meta SAM 3.1 · Muse Spark · Muse Wonder",
}


def contours_of(record: Any, offset: int = 0) -> list[np.ndarray]:
    """Outline polygons (image pixels) of one mask record. Raster pixels are mapped with
    the same formula as measure.mask_pixels (02 §5.2)."""
    width, height = record.mask.width, record.mask.height
    raster = np.frombuffer(decode_mask_to_raster(record.mask), dtype=np.uint8).reshape(
        height, width
    )
    sx = (record.bounds.right - record.bounds.left) / width
    sy = (record.bounds.bottom - record.bounds.top) / height
    found, _ = cv2.findContours(raster, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for cnt in found:
        pts = cnt.reshape(-1, 2).astype(float)
        xs = record.bounds.left + (pts[:, 0] + 0.5) * sx
        ys = record.bounds.top + (pts[:, 1] + 0.5) * sy
        polys.append(np.stack([xs, ys], axis=1).round().astype(np.int32))
    return polys


def load_masks(run_dir: Path) -> tuple[dict[int, list[np.ndarray]], list[np.ndarray]]:
    """Target contours by global frame index, and the reference contours (all frames)."""
    targets: dict[int, list[np.ndarray]] = {}
    reference: list[np.ndarray] = []
    offset = IngestResult.model_validate_json(
        (run_dir / "ingest" / "ingest.json").read_text(encoding="utf-8")
    ).clip_offset_frame
    for path in sorted((run_dir / "sam").glob("target_*.sam.json*")):
        out = asyncio.run(sam_client.load_record(path))
        for rec in measure.latest_masks(out.result):
            frame = rec.frame.frame_index if rec.frame is not None else None
            if frame is None:
                continue
            targets.setdefault(offset + frame, []).extend(contours_of(rec))
    for path in sorted((run_dir / "sam").glob("reference_*.sam.json*")):
        out = asyncio.run(sam_client.load_record(path))
        for rec in measure.latest_masks(out.result):
            reference.extend(contours_of(rec))
    return targets, reference


def _font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_PATH), size)


def _text_block(frame: np.ndarray, lines: list[str], *, x: int, y: int, size: int) -> np.ndarray:
    """A translucent box with text lines (PIL: the lines hold ± and ², which OpenCV cannot draw)."""
    font = _font(size)
    pad = size // 2
    widths = [font.getlength(line) for line in lines]
    box_w = int(max(widths) + 2 * pad)
    box_h = int(len(lines) * size * 1.35 + 2 * pad)
    x2, y2 = min(x + box_w, frame.shape[1]), min(y + box_h, frame.shape[0])
    frame[y:y2, x:x2] = (frame[y:y2, x:x2] * 0.35).astype(np.uint8)
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    for i, line in enumerate(lines):
        draw.text((x + pad, y + pad + i * size * 1.35), line, font=font, fill=(255, 255, 255))
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def _trail(frame: np.ndarray, rows: list[TrackRow], m_frame: int) -> None:
    valid = [r for r in rows if r.valid and r.frame <= m_frame][-TRAIL_LEN:]
    for age, r in enumerate(reversed(valid)):
        fade = max(0.15, 1.0 - age / TRAIL_LEN)
        color = tuple(int(c * fade) for c in TARGET_BGR)
        cv2.circle(frame, (round(r.x_px), round(r.y_px)), 3, color, -1)
    for r in rows:
        if not r.valid and r.flag == "outlier" and r.frame <= m_frame:
            x, y = round(r.x_px), round(r.y_px)
            cv2.line(frame, (x - 5, y - 5), (x + 5, y + 5), OUTLIER_BGR, 2)
            cv2.line(frame, (x - 5, y + 5), (x + 5, y - 5), OUTLIER_BGR, 2)


def _card(frame: np.ndarray, headline: str, verdict: str, lang: str) -> np.ndarray:
    dark = (frame * 0.3).astype(np.uint8)
    h, w = frame.shape[:2]
    size = max(18, h // 14)
    image = Image.fromarray(cv2.cvtColor(dark, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    lines = [(headline, size * 2, (255, 255, 255)), (verdict, size, (255, 224, 0)),
             (BUILT_WITH[lang], int(size * 0.7), (200, 200, 200))]  # fmt: skip
    y = int(h * 0.35)
    for text, s, color in lines:
        f = _font(s)
        draw.text(((w - f.getlength(text)) / 2, y), text, font=f, fill=color)
        y += int(s * 1.8)
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def render(run_dir: Path, out_path: Path) -> Path:
    """Pipe BGR frames to ffmpeg (libx264 -preset veryfast). HUD text from ledger claims only."""
    state = RunState.model_validate(read_json(run_dir / "state.json"))
    ing = IngestResult.model_validate(read_json(run_dir / "ingest" / "ingest.json"))
    ledger = Ledger(run_dir / "analysis" / "ledger.jsonl")
    lang = state.lang
    rows = view.track_rows(run_dir, ledger)
    m = view.scale_m_per_px(ledger)
    if m is None:
        raise ValueError("no scale in the ledger: the HUD has no metres")
    y0, y_m = view.relative_metres(rows, m)
    by_frame = {r.frame: (r, ym) for r, ym in zip(rows, y_m, strict=True)}
    fit = view.fit_view(ledger, "quadratic")
    t_lo = fit.t_start if fit and fit.t_start is not None else -np.inf
    t_hi = fit.t_end if fit and fit.t_end is not None else np.inf
    targets, reference = load_masks(run_dir)
    headline = view.headline_text(ledger, lang)
    verdict = state.verdict or ""

    video = run_dir / ing.video_path
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video}")
    fps = ing.file_fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ingest_mod._bin("ffmpeg"), "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}", "-r", f"{ing.fps_num}/{ing.fps_den}", "-i", "-",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(out_path),
    ]  # fmt: skip
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)  # noqa: S603 (list args, no shell)
    last = None
    try:
        frame_index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            idx = frame_index  # global frame index of the frame just read
            frame_index += 1
            for poly in reference:
                cv2.polylines(frame, [poly], True, REFERENCE_BGR, 2)
            hit = by_frame.get(idx)
            if hit is not None and t_lo - 1e-9 <= hit[0].t <= t_hi + 1e-9:
                for poly in targets.get(idx, []):
                    cv2.polylines(frame, [poly], True, TARGET_BGR, 2)
                _trail(frame, rows, idx)
                t_s, y_s = hit[0].t, hit[1]
                frame = _text_block(
                    frame,
                    [f"t = {view.fmt_time(t_s, lang)}", f"y = {_fmt_m(y_s, lang)}", headline],
                    x=12, y=12, size=max(14, height // 28),
                )  # fmt: skip
            proc.stdin.write(frame.tobytes())
            last = frame
        card_frames = round(END_CARD_S * fps)
        card = _card(last, headline, verdict, lang) if last is not None else None
        for _ in range(card_frames if card is not None else 0):
            proc.stdin.write(card.tobytes())
    finally:
        cap.release()
        proc.stdin.close()
        code = proc.wait()
    if code != 0:
        raise RuntimeError(f"ffmpeg exited with code {code}")
    return out_path


def _fmt_m(value: float, lang: str) -> str:
    return fmt(value, None, "m", lang)
