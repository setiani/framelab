"""SAM 3.1 client: video tracking, image segmentation, box-prompt fallback, record & replay.

Spec: docs/plan/02-meta-api-reference.md §4–§5 (pattern adapted from
meta-model-cookbook/07_sam/01_api_basics/python/sam_api.py).

Every request writes a compact record ``sam/<name>.sam.json`` with the raw model output text
(the segmentation grammar), status and usage. ``load_record()`` re-parses it with the official
``meta_sam_parser`` — this powers ``--resume``, ``framelab replay`` and golden tests.

M1 findings (docs/plan/11 §1): frame indices are 0-based and 1:1 with decoded frames; static
objects get NO masks in video mode until they move (segment static things in image mode).
"""

from __future__ import annotations

import base64
import json
import mimetypes
import time
from collections.abc import AsyncIterable, AsyncIterator, Iterable
from contextlib import aclosing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from meta_sam_parser import (
    FramePrompt,
    PromptBox,
    PromptObject,
    PromptPoint,
    build_object_prompt,
    image_segmentation_format,
    parse_responses_stream,
    video_segmentation_format,
)

from framelab.clients.meta import (
    CallTracker,
    Emit,
    MetaClientError,
    async_client,
    call_with_retries,
    upload_file,
)
from framelab.config import MODEL_SAM
from framelab.costs import sam_cost

Media = Literal["video", "image"]


class SamError(RuntimeError):
    """SAM returned an incomplete/failed response or the parser reported errors."""


@dataclass
class SamOutput:
    """One SAM request: parsed result plus everything needed to replay it."""

    concept: str
    media: Media
    prompt_kind: Literal["text", "box"]
    status: Literal["completed", "incomplete"]
    result: Any  # ImageSegmentationResult | VideoSegmentationResult
    text: str
    usage: dict[str, Any] = field(default_factory=dict)
    frame_index: int | None = None  # image requests: global frame index of the key frame
    latency_s: float = 0.0
    cost_usd: float = 0.0
    record_path: str | None = None

    @property
    def mask_records(self) -> list[Any]:
        return [r for r in self.result.records if r.kind == "mask"]

    @property
    def n_objects(self) -> int:
        return len({r.object_id for r in self.mask_records})

    def frames_with_mask(self) -> dict[str, int]:
        frames: dict[str, set[int]] = {}
        for rec in self.mask_records:
            idx = rec.frame.frame_index if rec.frame is not None else -1
            frames.setdefault(rec.object_id, set()).add(idx)
        return {obj: len(v) for obj, v in frames.items()}


# --------------------------------------------------------------------------------------------
# Event adapters
# --------------------------------------------------------------------------------------------


def _input(prompt: str, media: dict[str, str]) -> list[dict[str, Any]]:
    text = prompt.strip()
    if not text:
        raise ValueError("SAM prompt must be a non-empty noun phrase or object prompt")
    return [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": text}, media],
        }
    ]


async def _adapt(stream: AsyncIterable[Any], sink: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
    """SDK events → plain dicts; synthesize ``output_text.done`` (the API does not send it).

    Collects the output text and the final usage into ``sink``.
    """
    done: set[tuple[Any, Any, Any]] = set()
    async for event in stream:
        data = event.model_dump() if hasattr(event, "model_dump") else dict(event)
        etype = data.get("type")
        if etype == "response.output_text.delta" and isinstance(data.get("delta"), str):
            sink["text"] = sink.get("text", "") + data["delta"]
        if etype in ("response.completed", "response.incomplete"):
            response = data.get("response") or {}
            sink["usage"] = response.get("usage") or {}
            sink["status"] = "completed" if etype == "response.completed" else "incomplete"
        lane = (data.get("item_id"), data.get("output_index"), data.get("content_index"))
        if etype == "response.output_text.done":
            if lane not in done:
                done.add(lane)
                yield data
            continue
        yield data
        part = data.get("part")
        if (
            etype == "response.content_part.done"
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


def events_from_text(text: str, status: str = "completed") -> Iterable[dict[str, Any]]:
    """Minimal Responses event sequence that reproduces a recorded output text."""
    lane = {"item_id": "msg_replay", "output_index": 0, "content_index": 0}
    chunk = 65536
    for i in range(0, max(len(text), 1), chunk):
        yield {"type": "response.output_text.delta", **lane, "delta": text[i : i + chunk]}
    yield {"type": "response.output_text.done", **lane, "text": text}
    if status == "completed":
        yield {"type": "response.completed", "response": {"status": "completed"}}
    else:
        yield {
            "type": "response.incomplete",
            "response": {"status": "incomplete", "incomplete_details": {"reason": "replay"}},
        }


async def _aiter(items: Iterable[dict[str, Any]]) -> AsyncIterator[dict[str, Any]]:
    for item in items:
        yield item


async def parse_text(text: str, media: Media, status: str = "completed") -> Any:
    """Parse a recorded SAM output text with the official parser."""
    fmt = video_segmentation_format() if media == "video" else image_segmentation_format()
    return await parse_responses_stream(_aiter(events_from_text(text, status)), fmt).final_result()


def validate(result: Any) -> None:
    """Raise if the outcome is not completed or the parser reported errors.

    A completed result with zero masks is a valid EMPTY match, not an error.
    """
    if result.outcome.status != "completed":
        reason = getattr(result.outcome, "reason", "unknown")
        detail = getattr(result.outcome, "detail", None)
        raise SamError(f"SAM segmentation incomplete: {reason} ({detail})")
    errors = [d for d in result.diagnostics if d.severity == "error"]
    if errors:
        raise SamError("; ".join(f"line {d.line}: {d.message}" for d in errors))


# --------------------------------------------------------------------------------------------
# Requests
# --------------------------------------------------------------------------------------------


def data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


async def _segment(
    *,
    media: Media,
    media_part: dict[str, str],
    prompt: str,
    concept: str,
    prompt_kind: Literal["text", "box"],
    record_path: Path,
    emit: Emit | None,
    frames_hint: int = 0,
    frame_index: int | None = None,
) -> SamOutput:
    tracker = CallTracker(emit, MODEL_SAM, "responses", f"sam_{media}", concept=concept)
    fmt = video_segmentation_format() if media == "video" else image_segmentation_format()
    t0 = time.perf_counter()

    async def attempt() -> tuple[Any, dict[str, Any]]:
        sink: dict[str, Any] = {}
        async with async_client(timeout=600) as api:
            stream = await api.responses.create(
                model=MODEL_SAM, stream=True, input=_input(prompt, media_part)
            )
            async with stream, aclosing(_adapt(stream, sink)) as events:
                parsed = parse_responses_stream(events, fmt)
                async with parsed:
                    async for _snapshot in parsed:
                        pass
                    return await parsed.final_result(), sink

    try:
        result, sink = await call_with_retries(attempt)
    except MetaClientError as exc:
        tracker.done(ok=False, error=str(exc), cost_usd=0.0)
        raise
    usage = sink.get("usage") or {}
    frames = int(usage.get("video_frames_processed") or frames_hint or 0)
    cost = sam_cost(frames=frames) if media == "video" else sam_cost(images=1)
    tracker.done(
        ok=True,
        usage=usage,
        units={"frames": frames} if media == "video" else {"images": 1},
        cost_usd=cost,
    )
    status = "completed" if result.outcome.status == "completed" else "incomplete"
    out = SamOutput(
        concept=concept,
        media=media,
        prompt_kind=prompt_kind,
        status=status,
        result=result,
        text=sink.get("text", ""),
        usage=usage,
        frame_index=frame_index,
        latency_s=round(time.perf_counter() - t0, 3),
        cost_usd=cost,
        record_path=str(record_path),
    )
    save_record(out, record_path)
    return out


async def segment_video(
    file_id: str,
    concept: str,
    record_path: Path,
    *,
    emit: Emit | None = None,
    frames_hint: int = 0,
    prompt: str | None = None,
) -> SamOutput:
    """Track ``concept`` through an uploaded video (Files API id). ``prompt`` overrides the
    text sent (used for box prompts built with ``build_box_prompt``)."""
    if not file_id.startswith("file-"):
        raise ValueError("Expected a Files API id beginning with 'file-'")
    return await _segment(
        media="video",
        media_part={"type": "input_video", "file_id": file_id},
        prompt=prompt or concept,
        concept=concept,
        prompt_kind="box" if prompt else "text",
        record_path=record_path,
        emit=emit,
        frames_hint=frames_hint,
    )


async def segment_image(
    image_path: Path,
    concept: str,
    record_path: Path,
    *,
    emit: Emit | None = None,
    frame_index: int | None = None,
) -> SamOutput:
    """Segment ``concept`` in one image (a key frame)."""
    return await _segment(
        media="image",
        media_part={"type": "input_image", "image_url": data_url(image_path)},
        prompt=concept,
        concept=concept,
        prompt_kind="text",
        record_path=record_path,
        emit=emit,
        frame_index=frame_index,
    )


def build_box_prompt(
    size: tuple[int, int],
    frame: int,
    box: tuple[int, int, int, int],
    point: tuple[int, int] | None = None,
) -> str:
    """Object prompt text (02 §5.3). ``size`` = (W, H) of the uploaded video; box half-open."""
    left, top, right, bottom = (int(v) for v in box)
    points = [PromptPoint(int(point[0]), int(point[1]))] if point else []
    return build_object_prompt(
        size=(int(size[0]), int(size[1])),
        objects=[
            PromptObject(
                id=1,
                prompts=[
                    FramePrompt(frame=frame, box=PromptBox(left, top, right, bottom), points=points)
                ],
            )
        ],
    )


async def upload_video(path: Path, emit: Emit | None = None) -> str:
    return await upload_file(path, "video/mp4", emit=emit)


# --------------------------------------------------------------------------------------------
# Record & replay
# --------------------------------------------------------------------------------------------


def save_record(out: SamOutput, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "model": MODEL_SAM,
        "concept": out.concept,
        "media": out.media,
        "prompt_kind": out.prompt_kind,
        "frame_index": out.frame_index,
        "status": out.status,
        "usage": out.usage,
        "latency_s": out.latency_s,
        "cost_usd": out.cost_usd,
        "text": out.text,
    }
    path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")


async def load_record(path: Path) -> SamOutput:
    """Re-parse a saved ``.sam.json`` (or ``.sam.json.gz``) record without calling the API."""
    if path.suffix == ".gz":
        import gzip

        record = json.loads(gzip.decompress(path.read_bytes()).decode("utf-8"))
    else:
        record = json.loads(path.read_text(encoding="utf-8"))
    result = await parse_text(record["text"], record["media"], record.get("status", "completed"))
    return SamOutput(
        concept=record["concept"],
        media=record["media"],
        prompt_kind=record.get("prompt_kind", "text"),
        status=record.get("status", "completed"),
        result=result,
        text=record["text"],
        usage=record.get("usage") or {},
        frame_index=record.get("frame_index"),
        latency_s=record.get("latency_s", 0.0),
        cost_usd=record.get("cost_usd", 0.0),
        record_path=str(path),
    )
