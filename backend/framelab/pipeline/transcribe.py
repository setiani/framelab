"""Transcription stage (Muse Voice Transcribe, ENDPOINTING mode).

Spec: docs/plan/04-pipeline-and-measurement.md §3
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from framelab.clients import voice
from framelab.clients.meta import Emit, MetaClientError
from framelab.schemas import Transcript, TranscriptTurn


async def run(audio_path: Path | None, *, emit: Emit | None = None) -> Transcript:
    """Return a Transcript; empty (with a warning event) if there is no audio or the call fails.

    The run continues without a transcript: the Planner then relies on the video's own audio
    and the warning is shown to the user.
    """
    if audio_path is None:
        _warn(emit, "No audio track: transcript skipped, the Planner reads the video only")
        return Transcript()
    try:
        raw = await voice.transcribe(audio_path, keywords=voice.DEFAULT_KEYWORDS, emit=emit)
    except MetaClientError as exc:
        _warn(emit, f"Transcription failed, continuing without transcript: {exc}")
        return Transcript()
    return to_transcript(raw)


def to_transcript(raw: dict[str, Any]) -> Transcript:
    """Map the Voice response to a Transcript. Blank turns are dropped and indices renumbered.

    ``TranscriptTurn.index`` is the position in ``turns``, so ``source_turn`` in the spec can
    point straight at it.
    """
    turns: list[TranscriptTurn] = []
    for item in raw.get("turns") or []:
        text = (item.get("transcript") or "").strip()
        if not text:
            continue
        turns.append(
            TranscriptTurn(
                index=len(turns),
                text=text,
                start_ms=item.get("startMs"),
                end_ms=item.get("endMs"),
                speaker=item.get("speaker"),
            )
        )
    return Transcript(
        text=(raw.get("transcript") or "").strip(),
        turns=turns,
        audio_duration_ms=raw.get("audioDurationMs"),
        session_id=raw.get("sessionId"),
    )


def _warn(emit: Emit | None, message: str) -> None:
    if emit:
        emit("warning", message=message)
