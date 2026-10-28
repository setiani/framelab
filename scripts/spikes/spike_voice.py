"""Spike: Muse Voice Transcribe with Indonesian/English speech (VERIFY V8).

Variants (file endpoint POST /v1/asr/transcribe, pattern from meta-model-cookbook/06_muse_voice)
  V_a  ENDPOINTING, no languageBias (auto-detect + code-switching)
  V_b  ENDPOINTING, languageBias ["indonesian"]
  V_c  ENDPOINTING, languageBias ["id"]
  V_d  PUSH_TO_TALK, no languageBias
All with FrameLab keywords. Records status, transcript, turns (startMs) and errors.

Usage
  uv run python scripts/spikes/spike_voice.py [--audio my_recording.m4a]
Default audio: scripts/spikes/out/media/speech.wav (from make_test_media.py).
Approx. cost: 4 × ~10 s audio ≈ $0.002
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import httpx
from _common import MEDIA, Report, api_key, fail_safe_main, ffmpeg

URL = os.environ.get(
    "FRAMELAB_SPIKE_ASR_URL", "https://api.meta.ai/v1/asr/transcribe"
)  # override = offline smoke test only
KEYWORDS = ["bandul", "pendulum", "gravitasi", "sentimeter", "detik", "diameter", "hipotesis"]


def to_wav16k(src: Path) -> Path:
    if src.suffix.lower() == ".wav" and src.name == "speech.wav":
        return src
    out = MEDIA / f"{src.stem}_16k.wav"
    subprocess.run(
        [ffmpeg(), "-y", "-v", "error", "-i", str(src), "-vn", "-ac", "1", "-ar", "16000",
         "-sample_fmt", "s16", str(out)],
        check=True,
    )  # fmt: skip
    return out


def transcribe(wav: Path, mode: str, bias: list[str] | None) -> httpx.Response:
    request = {"mode": mode, "model": "muse-voice-transcribe-1.0", "audioEncoding": "WAV"}
    request["keywords"] = KEYWORDS
    if bias:
        request["languageBias"] = bias
    with wav.open("rb") as audio:
        return httpx.post(
            URL,
            headers={"Authorization": f"Bearer {api_key()}"},
            files={
                "request": (None, json.dumps(request), "application/json"),
                "audio": (wav.name, audio, "audio/wav"),
            },
            timeout=300,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, default=MEDIA / "speech.wav")
    args = parser.parse_args()
    if not args.audio.is_file():
        raise SystemExit(
            f"{args.audio} missing. Record a short voice note (mix Indonesian + English) and run:\n"
            "  uv run python scripts/spikes/spike_voice.py --audio <file>"
        )
    wav = to_wav16k(args.audio)
    report = Report("spike_voice")
    report.add("audio", wav.name)
    variants = {
        "V_a_endpointing_auto": ("ENDPOINTING", None),
        "V_b_endpointing_indonesian": ("ENDPOINTING", ["indonesian"]),
        "V_c_endpointing_id": ("ENDPOINTING", ["id"]),
        "V_d_push_to_talk_auto": ("PUSH_TO_TALK", None),
    }
    out = {}
    for name, (mode, bias) in variants.items():
        t0 = time.perf_counter()
        try:
            r = transcribe(wav, mode, bias)
            entry: dict = {"status": r.status_code, "latency_s": round(time.perf_counter() - t0, 2)}
            if r.status_code == 200:
                body = r.json()
                turns = body.get("turns") or []
                entry.update(
                    {
                        "transcript": body.get("transcript", "")[:800],
                        "audio_ms": body.get("audioDurationMs"),
                        "n_turns": len(turns),
                        "turns": [
                            {k: t.get(k) for k in ("startMs", "endMs", "transcript", "speaker")}
                            for t in turns[:10]
                        ],
                    }
                )
                report.cost((body.get("audioDurationMs") or 0) / 1000 * 0.18 / 3600)
                report.note(f"{name}: ✓ turns={len(turns)} {body.get('transcript', '')[:200]!r}")
            else:
                entry["body"] = r.text[:600]
                report.note(f"{name}: ✗ HTTP {r.status_code} {r.text[:200]!r}")
        except Exception as exc:  # noqa: BLE001
            entry = {"error": f"{type(exc).__name__}: {exc}"[:400]}
            report.note(f"{name}: ✗ {entry['error']}")
        out[name] = entry
    report.add("variants", out)
    report.save()


if __name__ == "__main__":
    fail_safe_main(main)
