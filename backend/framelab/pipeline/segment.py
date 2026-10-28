"""Segmentation stage: target video tracking, alt concepts, box fallback, static references.

Spec: docs/plan/04-pipeline-and-measurement.md §4 and the M1 findings (docs/plan/11 §1):
- SAM video reports an object only from the frame after it first moves (V12), so static
  things (ruler, pivot string) are segmented in IMAGE mode on key frames.
- Uploaded files never expire (V10): the clip is deleted at the end of the stage.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from framelab.clients import meta as meta_client
from framelab.clients import sam as sam_client
from framelab.pipeline import ingest as ingest_mod
from framelab.pipeline.measure import object_stats, observations_from_result
from framelab.schemas import ExperimentSpec, IngestResult, SamRequestSummary

Emit = Callable[..., Any]
MIN_TRACK_FRAMES = 10
MIN_TRACK_FRACTION = 0.25


def slug(concept: str) -> str:
    """File-safe slug for a concept, e.g. 'orange ball' → 'orange_ball'."""
    return "_".join("".join(c if c.isalnum() else " " for c in concept.lower()).split())


@dataclass
class SegmentationOutput:
    target: sam_client.SamOutput | None = None
    reference: sam_client.SamOutput | None = None
    string: sam_client.SamOutput | None = None
    target_concept_used: str | None = None
    attempts: list[SamRequestSummary] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def summarize(out: sam_client.SamOutput) -> SamRequestSummary:
    n = out.n_objects
    return SamRequestSummary(
        concept=out.concept,
        media=out.media,
        frame_index=out.frame_index,
        status="empty" if out.status == "completed" and n == 0 else out.status,
        n_objects=n,
        frames_with_mask=out.frames_with_mask(),
        raw_log_path=out.record_path or "",
        cost_usd=out.cost_usd,
    )


def track_is_usable(
    out: sam_client.SamOutput, ingest: IngestResult, clip_frames: int, axis: str
) -> bool:
    """At least MIN_TRACK_FRAMES frames and MIN_TRACK_FRACTION of the clip for the best object."""
    if out.status != "completed":
        return False
    obs = observations_from_result(
        out.result,
        out.concept,
        offset_frame=ingest.clip_offset_frame,
        effective_fps=ingest.effective_fps,
    )
    stats = object_stats(obs, clip_frames, axis)
    if not stats:
        return False
    best = max(stats.values(), key=lambda s: s["coverage"] * max(s["range_px"], 1.0))
    return best["n"] >= MIN_TRACK_FRAMES and best["coverage"] >= MIN_TRACK_FRACTION


def motion_blob_box(
    frames: list[np.ndarray], step: int = 3
) -> tuple[tuple[int, int, int, int], int] | None:
    """Largest persistent moving blob → (box, frame index of the box) (04 §4.1)."""
    import cv2

    if len(frames) <= step:
        return None
    gray = [cv2.GaussianBlur(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (5, 5), 0) for f in frames]
    best: tuple[int, tuple[int, int, int, int], int] | None = None
    for i in range(len(gray) - step):
        diff = cv2.absdiff(gray[i], gray[i + step])
        _, mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        for label in range(1, n):
            x, y, w, h, area = stats[label]
            if area < 30:
                continue
            if best is None or area > best[0]:
                best = (int(area), (int(x), int(y), int(x + w), int(y + h)), i)
        if best is not None and i >= 5:
            break
    if best is None:
        return None
    return best[1], best[2]


def _read_frames(video: Path, n: int = 30) -> list[np.ndarray]:
    import cv2

    cap = cv2.VideoCapture(str(video))
    frames = []
    while len(frames) < n:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


async def run(
    run_dir: Path,
    spec: ExperimentSpec,
    ingest: IngestResult,
    *,
    emit: Emit | None = None,
    max_extra_requests: int = 2,
) -> SegmentationOutput:
    """Issue the SAM requests of 04 §4 and store records under ``sam/``."""
    sam_dir = run_dir / "sam"
    sam_dir.mkdir(parents=True, exist_ok=True)
    out = SegmentationOutput()
    if not ingest.clip_path:
        raise ValueError("make_analysis_clip() must run before segmentation")
    clip = run_dir / ingest.clip_path
    clip_frames = ingest_mod.count_frames(clip)

    def note(message: str) -> None:
        out.warnings.append(message)
        if emit:
            emit("warning", message=message)

    file_id = await sam_client.upload_video(clip, emit=emit)
    try:
        # 1) target, text prompt (+ alternatives)
        concepts = [spec.target_concept, *spec.target_alt_concepts][: 1 + max_extra_requests]
        for concept in concepts:
            if emit:
                emit("progress", message=f"SAM tracking '{concept}'")
            result = await sam_client.segment_video(
                file_id,
                concept,
                sam_dir / f"target_{slug(concept)}.sam.json",
                emit=emit,
                frames_hint=clip_frames,
            )
            out.attempts.append(summarize(result))
            if track_is_usable(result, ingest, clip_frames, spec.motion_axis):
                out.target, out.target_concept_used = result, concept
                break
            note(f"SAM found no usable track for '{concept}'")

        # 2) box-prompt fallback from motion (04 §4.1)
        if out.target is None:
            found = motion_blob_box(_read_frames(clip))
            if found is not None:
                box, frame = found
                prompt = sam_client.build_box_prompt(
                    (ingest.norm_width, ingest.norm_height), frame, box
                )
                result = await sam_client.segment_video(
                    file_id,
                    spec.target_concept,
                    sam_dir / "target_box_prompt.sam.json",
                    emit=emit,
                    frames_hint=clip_frames,
                    prompt=prompt,
                )
                out.attempts.append(summarize(result))
                if track_is_usable(result, ingest, clip_frames, spec.motion_axis):
                    out.target, out.target_concept_used = result, f"box:{spec.target_concept}"
                else:
                    note("Box-prompt fallback did not produce a usable track")
            else:
                note("No moving object found for the box-prompt fallback")
    finally:
        await meta_client.delete_file(file_id, emit=emit)

    # 3) static reference in IMAGE mode (M1 V12)
    if spec.reference is not None:
        hint = spec.reference.frame_hint_s
        frame = int(round(hint * ingest.file_fps)) if hint is not None else ingest.clip_offset_frame
        frame = min(max(frame, 0), ingest.n_frames - 1)
        png = ingest_mod.extract_keyframe(
            run_dir / ingest.video_path,
            frame,
            run_dir / "ingest" / "keyframes" / f"frame_{frame:06d}.png",
        )
        result = await sam_client.segment_image(
            png,
            spec.reference.concept,
            sam_dir / f"reference_{slug(spec.reference.concept)}.sam.json",
            emit=emit,
            frame_index=frame,
        )
        out.attempts.append(summarize(result))
        if result.n_objects:
            out.reference = result
        else:
            note(f"Reference '{spec.reference.concept}' not found on frame {frame}")

    # 4) pendulum string (optional cross-check for the pivot, 04 §6.3)
    if spec.experiment == "pendulum":
        mid = ingest.clip_offset_frame + clip_frames // 2
        png = ingest_mod.extract_keyframe(
            run_dir / ingest.video_path,
            mid,
            run_dir / "ingest" / "keyframes" / f"frame_{mid:06d}.png",
        )
        try:
            result = await sam_client.segment_image(
                png, "string", sam_dir / "string.sam.json", emit=emit, frame_index=mid
            )
            out.attempts.append(summarize(result))
            out.string = result if result.n_objects else None
        except Exception as exc:  # noqa: BLE001 - optional cross-check only
            note(f"String segmentation skipped: {exc}")
    return out
