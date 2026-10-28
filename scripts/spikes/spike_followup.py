"""Follow-up spike after the first M1 run (VERIFY V12, V7b).

V12  SAM video returned no masks for the first ~0.533 s of every clip (frames 0-31 @60 fps,
     0-127 @240 fps) although the ball is visible and static there. Two hypotheses:
       (H-time)   SAM skips a fixed warm-up of ~0.53 s at the start of every clip;
       (H-motion) SAM only starts reporting the object once it moves.
     Test clips:
       a) hold_2s:      ball held still for 2.0 s, then dropped   → H-time: first≈32, H-motion: first≈121
       b) moving_start: clip starts mid-fall (ball moving at frame 0) → H-time: first≈32, else ≈0
V7b  Muse Wonder ignored the JSON schema in non-strict mode (extra keys, wrong content).
     Retry with a strict-compatible schema (additionalProperties=false, all keys required)
     and strict=true on Chat Completions and the Responses API; Spark strict as control.

Usage
  uv run python scripts/spikes/spike_followup.py
Approx. cost: SAM ~510 frames ≈ $0.10 + 3 text calls ≈ $0.08
"""

from __future__ import annotations

import asyncio
import copy
import json
import subprocess
import time
from typing import Any

import make_test_media as mtm
from _common import (
    MEDIA,
    RAW,
    Report,
    async_client,
    count_frames,
    describe_error,
    estimate_token_cost,
    fail_safe_main,
    ffmpeg,
    sam_cost,
    sam_segment_video,
    summarize_sam,
    sync_client,
    upload_file,
    usage_dict,
)
from spike_structured_output import PROMPT, MiniSpec


def make_clips() -> dict[str, Any]:
    clips = {}
    hold = MEDIA / "followup_hold2s_60fps.mp4"
    if not hold.is_file():
        old = mtm.HOLD_S
        mtm.HOLD_S = 2.0
        try:
            mtm.render(hold, 60, 4.0, 4.0)
        finally:
            mtm.HOLD_S = old
    clips["a_hold_2s"] = {"path": hold, "motion_starts_at_frame": 121}

    moving = MEDIA / "followup_moving_start_60fps.mp4"
    src = MEDIA / "freefall_60fps_5s.mp4"
    if not moving.is_file():
        subprocess.run(
            [ffmpeg(), "-y", "-v", "error", "-i", str(src),
             "-vf", "select='gte(n\\,33)',setpts=N/(60*TB)", "-an",
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
             "-r", "60", str(moving)],
            check=True,
        )  # fmt: skip
    clips["b_moving_start"] = {"path": moving, "motion_starts_at_frame": 0}
    return clips


def strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Make a JSON schema strict-compatible: every object closed and all keys required."""
    schema = copy.deepcopy(schema)

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("type") == "object" and "properties" in node:
                node["additionalProperties"] = False
                node["required"] = list(node["properties"])
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)

    visit(schema)
    return schema


STRICT = strict_schema(MiniSpec.model_json_schema())


def check_semantics(text: str) -> dict[str, Any]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        return {"json": False, "error": str(exc)}
    extra = sorted(set(data) - set(MiniSpec.model_fields))
    return {
        "json": True,
        "extra_keys": extra,
        "experiment_ok": data.get("experiment") == "pendulum",
        "length_ok": data.get("spoken_length_cm") in (50, 50.0),
        "target": data.get("target_concept"),
    }


def main() -> None:
    report = Report("spike_followup")

    # ---- V12: SAM start gap ---------------------------------------------------------------
    sam = {}
    for name, clip in make_clips().items():
        path = clip["path"]
        n = count_frames(path)
        t0 = time.perf_counter()
        try:
            file_id = asyncio.run(upload_file(path))
            result = asyncio.run(
                sam_segment_video(file_id, "orange ball", RAW / f"sam_{path.stem}.jsonl")
            )
            summary = summarize_sam(result)
            frames = sorted({g["frame"] for g in summary["geometry"]})
            sam[name] = {
                "frames_in_clip": n,
                "first_mask_frame": frames[0] if frames else None,
                "last_mask_frame": frames[-1] if frames else None,
                "n_mask_frames": len(frames),
                "motion_starts_at_frame": clip["motion_starts_at_frame"],
                "latency_s": round(time.perf_counter() - t0, 2),
            }
            report.cost(sam_cost(frames=n))
            report.note(
                f"V12 {name}: first mask frame {sam[name]['first_mask_frame']} "
                f"(motion starts at {clip['motion_starts_at_frame']}), "
                f"{len(frames)}/{n} frames with masks"
            )
        except Exception as exc:  # noqa: BLE001
            sam[name] = {"error": describe_error(exc)}
            report.note(f"V12 {name}: ✗ {describe_error(exc)}")
    report.add("v12_sam_start_gap", sam)

    # ---- V7b: strict structured output --------------------------------------------------
    def chat_strict(model: str):
        return sync_client().chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": PROMPT}],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "mini_spec", "schema": STRICT, "strict": True},
            },
        )

    async def responses_strict(model: str):
        async with async_client() as api:
            return await api.responses.create(
                model=model,
                input=PROMPT,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "mini_spec",
                        "schema": STRICT,
                        "strict": True,
                    }
                },
            )

    variants = {
        "W1_wonder_chat_strict": ("muse-wonder-1.0", lambda: chat_strict("muse-wonder-1.0"), "chat"),
        "W2_wonder_responses_strict": (
            "muse-wonder-1.0", lambda: asyncio.run(responses_strict("muse-wonder-1.0")), "responses"),
        "W3_spark_responses_strict": (
            "muse-spark-1.3", lambda: asyncio.run(responses_strict("muse-spark-1.3")), "responses"),
    }  # fmt: skip
    out = {}
    for name, (model, call, api) in variants.items():
        t0 = time.perf_counter()
        try:
            resp = call()
            text = resp.output_text if api == "responses" else resp.choices[0].message.content
            usage = usage_dict(resp.usage)
            report.cost(estimate_token_cost(model, usage))
            sem = check_semantics(text or "")
            out[name] = {
                "ok": True,
                "latency_s": round(time.perf_counter() - t0, 2),
                "semantics": sem,
                "text": (text or "")[:600],
                "usage": usage,
            }
            report.note(f"V7b {name}: ✓ {sem} {(text or '')[:140]!r}")
        except Exception as exc:  # noqa: BLE001
            out[name] = {"ok": False, "error": describe_error(exc)}
            report.note(f"V7b {name}: ✗ {describe_error(exc)}")
    report.add("v7b_strict_structured_output", out)
    report.save()


if __name__ == "__main__":
    fail_safe_main(main)
