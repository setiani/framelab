"""Spike: SAM 3.1 video segmentation behaviour (VERIFY V1, V2, V3, V4, V10).

Checks
  V1  frame_index is 0-based and 1:1 with decoded frames (no sampling)  → best offset vs ground truth
  V2  coordinates are in the uploaded video's pixel space              → centroid error vs ground truth
  V3  240 fps and 20 s clips are accepted; frames returned             → per-clip outcome
  V4  an MP4 with an audio track is accepted                           → *_audio.mp4 outcome
  V10 Files API retrieve/delete behaviour                              → after segmentation

Usage
  uv run python scripts/spikes/spike_sam_video.py [--concept "orange ball"] [--skip-long] [--video X.mp4]
Approx. cost with defaults: ~2,520 frames × $0.20/1000 ≈ $0.50
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

from _common import (
    MEDIA,
    RAW,
    Report,
    async_client,
    count_frames,
    describe_error,
    fail_safe_main,
    load_ground_truth,
    sam_cost,
    sam_segment_video,
    summarize_sam,
    upload_file,
)

DEFAULT_VIDEOS = [
    "freefall_60fps_5s.mp4",
    "freefall_240fps_3s.mp4",
    "freefall_60fps_5s_audio.mp4",
    "freefall_60fps_20s.mp4",
]


def analyse(geometry: list[dict], gt: dict | None, n_frames: int) -> dict:
    """Frame-index mapping and coordinate checks against synthetic ground truth."""
    if not geometry:
        return {"note": "no masks returned"}
    by_obj: dict[str, list[dict]] = defaultdict(list)
    for g in geometry:
        by_obj[g["object_id"]].append(g)
    main_obj, track = max(by_obj.items(), key=lambda kv: len(kv[1]))
    frames = sorted(g["frame"] for g in track if g["frame"] is not None)
    diffs = Counter(b - a for a, b in zip(frames, frames[1:], strict=False))
    result: dict = {
        "n_objects": len(by_obj),
        "main_object_id": main_obj,
        "main_object_frames": len(frames),
        "frame_index_min": frames[0] if frames else None,
        "frame_index_max": frames[-1] if frames else None,
        "ffprobe_frames": n_frames,
        "max_index_plus_1_equals_ffprobe": bool(frames) and frames[-1] + 1 == n_frames,
        "coverage": round(len(set(frames)) / n_frames, 4),
        "frame_step_histogram": dict(diffs.most_common(5)),
    }
    if gt:
        gt_cy = {f["frame"]: f["cy"] for f in gt["frames"]}
        # Only frames where the ball moves discriminate between offsets (the ball rests at first).
        moving = {
            f
            for f in gt_cy
            if f - 1 in gt_cy and f + 1 in gt_cy and abs(gt_cy[f + 1] - gt_cy[f - 1]) > 2
        }
        best = None
        for k in (0, -1, 1, -2, 2, -3, 3):  # prefer 0 on ties
            errs = [abs(g["cy"] - gt_cy[g["frame"] + k]) for g in track if g["frame"] + k in moving]
            if errs:
                mean = statistics.fmean(errs)
                if best is None or mean < best[1] - 1e-9:
                    best = (k, mean)
        cx_err = [abs(g["cx"] - gt["frames"][0]["cx"]) for g in track]
        diam = [g["bounds"][2] - g["bounds"][0] for g in track]
        result.update(
            {
                "best_frame_offset": best[0] if best else None,
                "mean_abs_cy_error_px_moving_frames": round(best[1], 3) if best else None,
                "median_abs_cx_error_px": round(statistics.median(cx_err), 3),
                "median_mask_width_px": round(statistics.median(diam), 2),
                "true_ball_diameter_px": round(2 * gt["ball_radius_px"], 2),
            }
        )
    return result


async def files_lifecycle(file_id: str) -> dict:
    info: dict = {}
    async with async_client() as api:
        try:
            meta = await api.files.retrieve(file_id)
            info["retrieve"] = meta.model_dump() if hasattr(meta, "model_dump") else str(meta)
        except Exception as exc:  # noqa: BLE001
            info["retrieve_error"] = describe_error(exc)
        try:
            deleted = await api.files.delete(file_id)
            info["delete"] = (
                deleted.model_dump() if hasattr(deleted, "model_dump") else str(deleted)
            )
        except Exception as exc:  # noqa: BLE001
            info["delete_error"] = describe_error(exc)
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--concept", default="orange ball")
    parser.add_argument("--video", action="append", help="video path(s); default: synthetic set")
    parser.add_argument("--skip-long", action="store_true", help="skip the 20 s clip")
    args = parser.parse_args()

    report = Report("spike_sam_video")
    videos = [Path(v) for v in args.video] if args.video else [MEDIA / v for v in DEFAULT_VIDEOS]
    if args.skip_long:
        videos = [v for v in videos if "20s" not in v.name]
    report.add("concept", args.concept)
    results = {}
    for video in videos:
        if not video.is_file():
            report.note(f"{video.name}: missing (run make_test_media.py first)")
            continue
        n_frames = count_frames(video)
        report.note(f"{video.name}: {n_frames} frames, uploading…")
        entry: dict = {"ffprobe_frames": n_frames}
        t0 = time.perf_counter()
        try:
            file_id = asyncio.run(upload_file(video))
            entry["upload"] = {
                "ok": True,
                "file_id": file_id,
                "s": round(time.perf_counter() - t0, 2),
            }
        except Exception as exc:  # noqa: BLE001
            entry["upload"] = {"ok": False, "error": describe_error(exc)}
            report.note(f"  ✗ upload failed: {describe_error(exc)}")
            results[video.name] = entry
            continue
        t1 = time.perf_counter()
        raw_log = RAW / f"sam_{video.stem}_{args.concept.replace(' ', '_')}.jsonl"
        try:
            result = asyncio.run(sam_segment_video(file_id, args.concept, raw_log))
            summary = summarize_sam(result)
            entry["segment"] = {
                "ok": True,
                "latency_s": round(time.perf_counter() - t1, 2),
                **{k: v for k, v in summary.items() if k != "geometry"},
            }
            entry["analysis"] = analyse(summary["geometry"], load_ground_truth(video), n_frames)
            entry["geometry_sample"] = summary["geometry"][:5]
            report.cost(sam_cost(frames=n_frames))
            a = entry["analysis"]
            report.note(
                f"  ✓ {summary['outcome']}: objects={a.get('n_objects')} "
                f"coverage={a.get('coverage')} idx={a.get('frame_index_min')}..{a.get('frame_index_max')} "
                f"(ffprobe {n_frames}) offset={a.get('best_frame_offset')} "
                f"cy_err={a.get('mean_abs_cy_error_px_moving_frames')}px "
                f"width={a.get('median_mask_width_px')}/{a.get('true_ball_diameter_px')}px "
                f"steps={a.get('frame_step_histogram')} latency={entry['segment']['latency_s']}s"
            )
        except Exception as exc:  # noqa: BLE001
            entry["segment"] = {
                "ok": False,
                "latency_s": round(time.perf_counter() - t1, 2),
                "error": describe_error(exc),
            }
            report.note(f"  ✗ segmentation failed: {describe_error(exc)}")
        if video is videos[0]:
            entry["files_lifecycle"] = asyncio.run(files_lifecycle(file_id))
            report.note(f"  files retrieve/delete: {list(entry['files_lifecycle'])}")
        results[video.name] = entry
    report.add("results", results)
    report.save()


if __name__ == "__main__":
    fail_safe_main(main)
