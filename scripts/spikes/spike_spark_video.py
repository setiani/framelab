"""Spike: Muse Spark 1.3 video input (VERIFY V5, V9).

Variants
  A  Responses API, content part {"type": "input_file",  "file_id": ...}
  B  Responses API, content part {"type": "input_video", "file_id": ...}
  C  Chat Completions, content part {"type": "video_url", "video_url": {"url": <data URL>}}
For each: does it work, does the model read the frames (frame counter) and the audio, token usage.

Usage
  uv run python scripts/spikes/spike_spark_video.py [--video X.mp4]
Approx. cost: 3 calls × ~50k input tokens ≈ $0.20 (actual usage is recorded)
"""

from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path

from _common import (
    MEDIA,
    Report,
    async_client,
    data_url,
    describe_error,
    estimate_token_cost,
    fail_safe_main,
    sync_client,
    upload_file,
    usage_dict,
)

MODEL = "muse-spark-1.3"
PROMPT = (
    "This is a short test video for a physics app. Answer briefly in English:\n"
    "1. Describe the motion of the orange object.\n"
    "2. What number does the counter at the top-left show in the FIRST and in the LAST frame?\n"
    "3. Is there a ruler? Where is it and what range of markings does it show?\n"
    "4. Audio: transcribe any speech verbatim (it may mix Indonesian and English). "
    "If there is no speech, describe the sound or say 'silent'."
)


async def responses_variant(file_id: str, part_type: str):
    async with async_client() as api:
        return await api.responses.create(
            model=MODEL,
            reasoning={"effort": "low"},
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": PROMPT},
                        {"type": part_type, "file_id": file_id},
                    ],
                }
            ],
        )


def chat_variant(video: Path):
    return sync_client().chat.completions.create(
        model=MODEL,
        reasoning_effort="low",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {"type": "video_url", "video_url": {"url": data_url(video)}},
                ],
            }
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=MEDIA / "freefall_60fps_5s_audio.mp4")
    args = parser.parse_args()
    if not args.video.is_file():
        raise SystemExit(f"{args.video} missing (run make_test_media.py first)")

    report = Report("spike_spark_video")
    report.add("video", args.video.name)
    report.add("has_speech", (MEDIA / "speech.wav").is_file())
    try:
        file_id = asyncio.run(upload_file(args.video))
        report.note(f"uploaded {args.video.name} → {file_id}")
    except Exception as exc:  # noqa: BLE001
        report.note(f"✗ upload failed: {describe_error(exc)}")
        file_id = None

    variants = {}
    for name, part in (
        ("A_responses_input_file", "input_file"),
        ("B_responses_input_video", "input_video"),
    ):
        if not file_id:
            break
        t0 = time.perf_counter()
        try:
            resp = asyncio.run(responses_variant(file_id, part))
            usage = usage_dict(resp.usage)
            cost = estimate_token_cost(MODEL, usage)
            report.cost(cost)
            variants[name] = {
                "ok": True,
                "latency_s": round(time.perf_counter() - t0, 2),
                "usage": usage,
                "est_cost_usd": round(cost, 4),
                "answer": (resp.output_text or "")[:1500],
            }
            report.note(
                f"{name}: ✓ {variants[name]['latency_s']}s tokens_in="
                f"{usage.get('input_tokens')} out={usage.get('output_tokens')} "
                f"answer: {(resp.output_text or '')[:220]!r}"
            )
        except Exception as exc:  # noqa: BLE001
            variants[name] = {"ok": False, "error": describe_error(exc)}
            report.note(f"{name}: ✗ {describe_error(exc)}")

    t0 = time.perf_counter()
    try:
        resp = chat_variant(args.video)
        usage = usage_dict(resp.usage)
        cost = estimate_token_cost(MODEL, usage)
        report.cost(cost)
        text = resp.choices[0].message.content or ""
        variants["C_chat_video_url"] = {
            "ok": True,
            "latency_s": round(time.perf_counter() - t0, 2),
            "usage": usage,
            "est_cost_usd": round(cost, 4),
            "answer": text[:1500],
        }
        report.note(
            f"C_chat_video_url: ✓ tokens_in={usage.get('prompt_tokens')} answer: {text[:220]!r}"
        )
    except Exception as exc:  # noqa: BLE001
        variants["C_chat_video_url"] = {"ok": False, "error": describe_error(exc)}
        report.note(f"C_chat_video_url: ✗ {describe_error(exc)}")

    report.add("variants", variants)
    report.save()


if __name__ == "__main__":
    fail_safe_main(main)
