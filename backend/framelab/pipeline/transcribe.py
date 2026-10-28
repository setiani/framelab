"""Transcription stage (Muse Voice Transcribe, ENDPOINTING mode).

Spec: docs/plan/04-pipeline-and-measurement.md §3
Status: skeleton (M0). Implement in milestone M5.
"""

from __future__ import annotations

from pathlib import Path

from framelab.schemas import Transcript


def run(audio_path: Path | None) -> Transcript:
    """Return a Transcript; empty (with a warning event) if there is no audio or the call fails."""
    raise NotImplementedError("Milestone M5: see docs/plan/09-milestones-and-tasks.md")
