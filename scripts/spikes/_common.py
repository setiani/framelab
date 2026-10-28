"""Shared helpers for the M1 API verification spikes.

Request patterns follow docs/plan/02-meta-api-reference.md exactly. The SAM streaming adapter
is adapted from meta-model-cookbook/07_sam/01_api_basics/python/sam_api.py.

Every spike writes ``scripts/spikes/out/<spike>.json`` (findings) and appends a section to
``scripts/spikes/out/SUMMARY.md``. The folder is git-ignored.
"""

from __future__ import annotations

import base64
import contextlib
import json
import mimetypes
import os
import shutil
import subprocess
import sys
import time
import traceback
from collections.abc import AsyncIterable, AsyncIterator, Callable
from contextlib import aclosing
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "scripts" / "spikes" / "out"
MEDIA = OUT / "media"
RAW = OUT / "raw"

# Windows consoles / redirected output may not be UTF-8; never crash on ✓/✗ symbols.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

# Make `import framelab` work when run as a plain script.
sys.path.insert(0, str(ROOT / "backend"))

from framelab.config import META_BASE_URL, get_settings  # noqa: E402
from framelab.costs import sam_cost, token_cost  # noqa: E402

for _d in (OUT, MEDIA, RAW):
    _d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------------------------
# Clients
# --------------------------------------------------------------------------------------------


def api_key() -> str:
    key = get_settings().model_api_key
    if not key:
        sys.exit("MODEL_API_KEY is not set. Put it in .env (see .env.example).")
    return key


# Override only for offline smoke tests of the spike scripts themselves.
BASE_URL = os.environ.get("FRAMELAB_SPIKE_BASE_URL", META_BASE_URL)


def sync_client(timeout: float = 300.0):
    from openai import OpenAI

    return OpenAI(base_url=BASE_URL, api_key=api_key(), timeout=timeout, max_retries=1)


def async_client(timeout: float = 600.0):
    from openai import AsyncOpenAI

    return AsyncOpenAI(base_url=BASE_URL, api_key=api_key(), timeout=timeout, max_retries=1)


def describe_error(exc: BaseException) -> dict[str, Any]:
    """Compact, log-safe error description (status code + body excerpt)."""
    info: dict[str, Any] = {"type": type(exc).__name__, "message": str(exc)[:600]}
    status = getattr(exc, "status_code", None)
    if status is not None:
        info["status"] = status
    response = getattr(exc, "response", None)
    if response is not None:
        with contextlib.suppress(Exception):
            info["body"] = response.text[:800]
        req_id = getattr(response, "headers", {}).get("x-request-id") if response else None
        if req_id:
            info["request_id"] = req_id
    return info


def usage_dict(usage: Any) -> dict[str, Any]:
    if usage is None:
        return {}
    try:
        return usage.model_dump()
    except AttributeError:
        return dict(usage)


def estimate_token_cost(model: str, usage: dict[str, Any]) -> float:
    inp = usage.get("input_tokens") or usage.get("prompt_tokens") or 0
    out = usage.get("output_tokens") or usage.get("completion_tokens") or 0
    details = usage.get("input_tokens_details") or usage.get("prompt_tokens_details") or {}
    cached = (details or {}).get("cached_tokens") or 0
    return token_cost(model, int(inp), int(out), int(cached))


# --------------------------------------------------------------------------------------------
# Media helpers
# --------------------------------------------------------------------------------------------


def ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        sys.exit("ffmpeg not found on PATH")
    return path


def ffprobe() -> str:
    path = shutil.which("ffprobe")
    if not path:
        sys.exit("ffprobe not found on PATH")
    return path


def count_frames(video: Path) -> int:
    out = subprocess.run(
        [
            ffprobe(),
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "csv=p=0",
            str(video),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return int(out.stdout.strip().split(",")[0])


def probe(video: Path) -> dict[str, Any]:
    out = subprocess.run(
        [ffprobe(), "-v", "error", "-print_format", "json", "-show_streams", str(video)],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(out.stdout)


def data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def extract_frame(video: Path, frame_index: int, out_png: Path) -> Path:
    subprocess.run(
        [
            ffmpeg(),
            "-y",
            "-v",
            "error",
            "-i",
            str(video),
            "-vf",
            f"select=eq(n\\,{frame_index})",
            "-vframes",
            "1",
            str(out_png),
        ],
        check=True,
        capture_output=True,
    )
    return out_png


def load_ground_truth(video: Path) -> dict[str, Any] | None:
    gt = video.with_suffix(".gt.json")
    return json.loads(gt.read_text(encoding="utf-8")) if gt.is_file() else None


# --------------------------------------------------------------------------------------------
# SAM 3.1 (pattern adapted from meta-model-cookbook/07_sam/01_api_basics/python/sam_api.py)
# --------------------------------------------------------------------------------------------


def _sam_input(concept: str, media: dict[str, str]) -> list[dict[str, Any]]:
    return [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": concept.strip()}, media],
        }
    ]


async def _events(stream: AsyncIterable[Any], raw_log: Path) -> AsyncIterator[dict[str, Any]]:
    """SDK events → plain dicts; synthesize output_text.done (the API does not send it).

    Every event is also appended to ``raw_log`` (JSONL) for later golden replay.
    """
    done: set[tuple[Any, Any, Any]] = set()
    with raw_log.open("w", encoding="utf-8") as fh:
        async for event in stream:
            data = event.model_dump() if hasattr(event, "model_dump") else dict(event)
            fh.write(json.dumps(data, default=str) + "\n")
            lane = (data.get("item_id"), data.get("output_index"), data.get("content_index"))
            if data.get("type") == "response.output_text.done":
                if lane not in done:
                    done.add(lane)
                    yield data
                continue
            yield data
            part = data.get("part")
            if (
                data.get("type") == "response.content_part.done"
                and lane not in done
                and isinstance(part, dict)
                and part.get("type") == "output_text"
                and isinstance(part.get("text"), str)
            ):
                done.add(lane)
                yield {
                    "type": "response.output_text.done",
                    "item_id": data.get("item_id"),
                    "output_index": data.get("output_index"),
                    "content_index": data.get("content_index"),
                    "text": part["text"],
                }


async def upload_file(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "video/mp4"
    async with async_client() as api:
        uploaded = await api.files.create(
            file=(path.name, path.read_bytes(), mime), purpose="user_data"
        )
    return uploaded.id


async def sam_segment_video(file_id: str, concept: str, raw_log: Path) -> Any:
    from meta_sam_parser import parse_responses_stream, video_segmentation_format

    async with async_client() as api:
        stream = await api.responses.create(
            model="sam-3.1",
            stream=True,
            input=_sam_input(concept, {"type": "input_video", "file_id": file_id}),
        )
        async with stream, aclosing(_events(stream, raw_log)) as events:
            parsed = parse_responses_stream(events, video_segmentation_format())
            async with parsed:
                async for _snapshot in parsed:
                    pass
                return await parsed.final_result()


async def sam_segment_image(image: Path, concept: str, raw_log: Path) -> Any:
    from meta_sam_parser import image_segmentation_format, parse_responses_stream

    async with async_client() as api:
        stream = await api.responses.create(
            model="sam-3.1",
            stream=True,
            input=_sam_input(concept, {"type": "input_image", "image_url": data_url(image)}),
        )
        async with stream, aclosing(_events(stream, raw_log)) as events:
            return await parse_responses_stream(events, image_segmentation_format()).final_result()


def summarize_sam(result: Any) -> dict[str, Any]:
    """Outcome, diagnostics, and per-mask geometry (centroid via bounds mapping, 02 §5.2)."""
    from meta_sam_parser import decode_mask_to_raster

    masks: dict[tuple[str, int | None], Any] = {}
    n_boxes = 0
    for rec in result.records:
        if rec.kind == "box":
            n_boxes += 1
        if rec.kind != "mask":
            continue
        frame = rec.frame.frame_index if rec.frame is not None else None
        key = (rec.object_id, frame)
        if key not in masks or rec.revision > masks[key].revision:
            masks[key] = rec
    geometry = []
    size_equal = 0
    for (obj, frame), rec in sorted(masks.items(), key=lambda kv: (kv[0][0], kv[0][1] or -1)):
        raster = np.frombuffer(decode_mask_to_raster(rec.mask), dtype=np.uint8).reshape(
            rec.mask.height, rec.mask.width
        )
        rows, cols = np.nonzero(raster)
        bw = rec.bounds.right - rec.bounds.left
        bh = rec.bounds.bottom - rec.bounds.top
        if abs(bw - rec.mask.width) <= 1 and abs(bh - rec.mask.height) <= 1:
            size_equal += 1
        if rows.size == 0:
            continue
        sx, sy = bw / rec.mask.width, bh / rec.mask.height
        cx = rec.bounds.left + float((cols + 0.5).mean()) * sx
        cy = rec.bounds.top + float((rows + 0.5).mean()) * sy
        geometry.append(
            {
                "object_id": obj,
                "frame": frame,
                "revision": rec.revision,
                "cx": round(cx, 3),
                "cy": round(cy, 3),
                "area_px": round(rows.size * sx * sy, 1),
                "mask_wh": [rec.mask.width, rec.mask.height],
                "bounds": [
                    round(rec.bounds.left, 2),
                    round(rec.bounds.top, 2),
                    round(rec.bounds.right, 2),
                    round(rec.bounds.bottom, 2),
                ],
                "confidence": rec.confidence,
            }
        )
    return {
        "outcome": result.outcome.status,
        "outcome_reason": getattr(result.outcome, "reason", None),
        "outcome_detail": getattr(result.outcome, "detail", None),
        "n_records": len(result.records),
        "n_boxes": n_boxes,
        "n_masks": len(masks),
        "mask_size_equals_bounds": f"{size_equal}/{len(masks)}",
        "diagnostics": [
            {"severity": d.severity, "code": d.code, "message": d.message[:200]}
            for d in result.diagnostics
        ][:10],
        "geometry": geometry,
    }


# --------------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------------


class Report:
    """Collects findings for one spike and writes JSON + a SUMMARY.md section."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.findings: dict[str, Any] = {"spike": name, "started": time.strftime("%F %T")}
        self.cost_usd = 0.0
        self.lines: list[str] = []

    def add(self, key: str, value: Any) -> None:
        self.findings[key] = value

    def note(self, line: str) -> None:
        print(line, flush=True)
        self.lines.append(line)

    def cost(self, usd: float) -> None:
        self.cost_usd += usd

    def attempt(self, key: str, fn: Callable[[], Any]) -> Any:
        """Run fn, record ok/error under key, never raise."""
        t0 = time.perf_counter()
        try:
            value = fn()
            self.findings[key] = {"ok": True, "latency_s": round(time.perf_counter() - t0, 2)}
            return value
        except Exception as exc:  # noqa: BLE001
            self.findings[key] = {
                "ok": False,
                "latency_s": round(time.perf_counter() - t0, 2),
                "error": describe_error(exc),
            }
            self.note(f"  ✗ {key}: {describe_error(exc)}")
            return None

    def save(self) -> Path:
        self.findings["estimated_cost_usd"] = round(self.cost_usd, 4)
        self.findings["finished"] = time.strftime("%F %T")
        path = OUT / f"{self.name}.json"
        path.write_text(json.dumps(self.findings, indent=2, default=str), encoding="utf-8")
        with (OUT / "SUMMARY.md").open("a", encoding="utf-8") as fh:
            fh.write(f"\n## {self.name} ({self.findings['finished']})\n\n")
            for line in self.lines:
                fh.write(f"- {line.strip()}\n")
            fh.write(f"- estimated cost: ${self.cost_usd:.4f}\n")
        print(f"→ wrote {path.relative_to(ROOT)}  (est. cost ${self.cost_usd:.4f})")
        return path


def fail_safe_main(main: Callable[[], None]) -> None:
    try:
        main()
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        sys.exit(1)


__all__ = [
    "MEDIA",
    "OUT",
    "RAW",
    "ROOT",
    "Report",
    "async_client",
    "count_frames",
    "data_url",
    "describe_error",
    "estimate_token_cost",
    "extract_frame",
    "fail_safe_main",
    "ffmpeg",
    "load_ground_truth",
    "probe",
    "sam_cost",
    "sam_segment_image",
    "sam_segment_video",
    "summarize_sam",
    "sync_client",
    "upload_file",
    "usage_dict",
]
