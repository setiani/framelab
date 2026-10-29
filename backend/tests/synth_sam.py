"""Synthetic SAM 3.1 output text (same wire format as the live API) for offline fixtures.

Used where no live recording exists yet (pendulum, M4). Masks are rasterised from ground
truth with the ~1 px dilation SAM shows (M1 V11). Regenerate the fixtures with
``uv run python -m tests.synth_sam`` from ``backend/``.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
from meta_sam_parser._mask_codec import _encode_raster

FIXTURES = Path(__file__).parent / "fixtures"


def _mask_line(prefix: str, obj: str, raster: np.ndarray, left: int, top: int, w: int, h: int):
    rh, rw = raster.shape
    payload = _encode_raster(bytes(raster.astype(np.uint8).ravel()), rw, rh)
    box = f"<|box;x1={left};y1={top};x2={left + rw - 1};y2={top + rh - 1};w={w};h={h}|>"
    return f"{prefix}{obj}{box}<|mask;x=0;y=0;data={rh},{rw},{payload}|>"  # data=H,W


def disk_raster(cx: float, cy: float, radius: float) -> tuple[np.ndarray, int, int]:
    left, top = int(np.floor(cx - radius - 1)), int(np.floor(cy - radius - 1))
    size = int(np.ceil(2 * radius + 3))
    yy, xx = np.mgrid[0:size, 0:size] + 0.5
    raster = (xx + left - cx) ** 2 + (yy + top - cy) ** 2 <= radius**2
    return _crop(raster, left, top)


def line_raster(p0, p1, half_width: float) -> tuple[np.ndarray, int, int]:
    x0, y0 = p0
    x1, y1 = p1
    left, top = int(np.floor(min(x0, x1) - half_width - 1)), int(np.floor(min(y0, y1) - 1))
    w = int(np.ceil(abs(x1 - x0) + 2 * half_width + 3))
    h = int(np.ceil(abs(y1 - y0) + 3))
    yy, xx = np.mgrid[0:h, 0:w] + 0.5
    px, py = xx + left, yy + top
    d = np.array([x1 - x0, y1 - y0])
    u = ((px - x0) * d[0] + (py - y0) * d[1]) / (d @ d)
    dist = np.hypot(px - (x0 + u * d[0]), py - (y0 + u * d[1]))
    return _crop((dist <= half_width) & (u >= 0) & (u <= 1), left, top)


def _crop(raster: np.ndarray, left: int, top: int) -> tuple[np.ndarray, int, int]:
    ys, xs = np.nonzero(raster)
    return (
        raster[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1],
        left + int(xs.min()),
        top + int(ys.min()),
    )


def video_text(tracks: dict[str, list[tuple[int, float, float]]], radius: float, w: int, h: int):
    """``tracks``: object id → [(frame, cx, cy)]. One line per (frame, object)."""
    lines = []
    events = sorted((f, obj, cx, cy) for obj, pts in tracks.items() for f, cx, cy in pts)
    for f, obj, cx, cy in events:
        raster, left, top = disk_raster(cx, cy, radius)
        lines.append(_mask_line(f"<{f}f>", obj, raster, left, top, w, h))
    return "\n".join(lines) + "\n"


def write_record(path: Path, concept: str, media: str, text: str, frame_index=None) -> None:
    record = {"model": "sam-3.1", "concept": concept, "media": media, "prompt_kind": "text",
              "frame_index": frame_index, "status": "completed", "usage": {}, "latency_s": 0.0,
              "cost_usd": 0.0, "text": text, "synthetic": True}  # fmt: skip
    path.write_bytes(gzip.compress(json.dumps(record).encode("utf-8"), mtime=0))


def make_pendulum_fixtures() -> None:
    gt = json.loads((FIXTURES / "media" / "pendulum_60fps_12s.gt.json").read_text("utf-8"))
    w, h, fps = gt["width"], gt["height"], gt["fps"]
    hold_frames = int(round(gt["hold_s"] * fps))
    rng = np.random.default_rng(7)
    pts = []
    for fr in gt["frames"]:
        k = fr["frame"]
        if k <= hold_frames:  # SAM video reports an object only once it moves (M1 V12)
            continue
        if k in (200, 201, 455):  # a few dropped frames, like the live API
            continue
        pts.append((k, fr["cx"] + rng.normal(0, 0.15), fr["cy"] + rng.normal(0, 0.15)))
    radius = gt["bob_radius_px"] + 1.0  # SAM dilation
    text = video_text({"0": pts}, radius, w, h)
    write_record(FIXTURES / "sam" / "pendulum_60fps_12s_red_ball.sam.json.gz", "red ball",
                 "video", text)  # fmt: skip

    mid = gt["n_frames"] // 2
    fr = gt["frames"][mid]
    raster, left, top = line_raster(gt["pivot_px"], (fr["cx"], fr["cy"] - gt["bob_radius_px"]), 1.5)
    text = _mask_line("<0f>", "0", raster, left, top, w, h) + "\n"
    write_record(FIXTURES / "sam" / "pendulum_string.sam.json.gz", "string", "image", text, mid)
    print("pendulum fixtures written")


if __name__ == "__main__":
    make_pendulum_fixtures()
