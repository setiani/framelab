"""Muse Voice Transcribe client (file endpoint).

Spec: docs/plan/02-meta-api-reference.md §8 (pattern adapted from meta-model-cookbook/06_muse_voice)
Status: skeleton (M0). Implement in milestone M5.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

DEFAULT_KEYWORDS = [
    "bandul",
    "pendulum",
    "gravitasi",
    "gravity",
    "sentimeter",
    "centimeter",
    "meter",
    "detik",
    "second",
    "periode",
    "period",
    "penggaris",
    "ruler",
    "bola",
    "ball",
    "hipotesis",
    "hypothesis",
    "diameter",
]


def transcribe(
    path: Path,
    *,
    mode: str = "ENDPOINTING",
    keywords: list[str] | None = None,
    language_bias: list[str] | None = None,
) -> dict[str, Any]:
    """POST WAV (mono s16, 16 kHz) to /v1/asr/transcribe; return the parsed JSON."""
    raise NotImplementedError("Milestone M5: see docs/plan/09-milestones-and-tasks.md")
