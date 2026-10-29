"""M5 transcription offline tests: Voice client retries and the transcript stage."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from framelab.clients import meta as meta_client
from framelab.clients import voice
from framelab.clients.meta import MetaClientError
from framelab.pipeline import transcribe as transcribe_stage
from tests.fakes import FakeVoice

RAW = {
    "sessionId": "sess-1",
    "transcript": "Tinggi 0.5 meter. Hipotesis 1,5 detik.",
    "audioDurationMs": 3500,
    "turns": [
        {"transcript": "Tinggi 0.5 meter.", "startMs": 0, "endMs": 1800},
        {"transcript": "   ", "startMs": 1800, "endMs": 1900},
        {"transcript": "Hipotesis 1,5 detik.", "startMs": 1900, "endMs": 3500, "speaker": "A"},
    ],
}


def test_to_transcript_drops_blank_turns_and_renumbers() -> None:
    transcript = transcribe_stage.to_transcript(RAW)

    assert [t.index for t in transcript.turns] == [0, 1]
    assert transcript.turns[1].text == "Hipotesis 1,5 detik."
    assert transcript.turns[1].start_ms == 1900 and transcript.turns[1].speaker == "A"
    assert transcript.audio_duration_ms == 3500 and transcript.session_id == "sess-1"


async def test_no_audio_gives_empty_transcript_with_warning() -> None:
    messages: list[str] = []

    def emit(type: str, **payload) -> None:
        messages.append(payload["message"])

    transcript = await transcribe_stage.run(None, emit=emit)

    assert transcript.turns == []
    assert any("No audio track" in m for m in messages)


async def test_voice_failure_continues_without_transcript(tmp_path: Path, monkeypatch) -> None:
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"RIFF")
    FakeVoice(error=MetaClientError(413, "too large")).install(monkeypatch)
    messages: list[str] = []

    transcript = await transcribe_stage.run(
        audio, emit=lambda t, **p: messages.append(p["message"])
    )

    assert transcript.turns == []
    assert any("continuing without transcript" in m for m in messages)


async def test_stage_returns_mapped_transcript(tmp_path: Path, monkeypatch) -> None:
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"RIFF")
    fake = FakeVoice(RAW).install(monkeypatch)

    transcript = await transcribe_stage.run(audio)

    assert fake.calls == ["audio.wav"]
    assert len(transcript.turns) == 2


def _voice_client(handler, monkeypatch) -> None:
    """Route the Voice client through an in-process transport; no network."""
    monkeypatch.setenv("MODEL_API_KEY", "test-key")
    monkeypatch.setattr(meta_client, "backoff_delay", lambda attempt, rng=None: 0.0)
    monkeypatch.setattr(
        voice, "_http", lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )


async def test_voice_retries_server_errors_then_succeeds(tmp_path: Path, monkeypatch) -> None:
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"RIFF")
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(503, text="busy")
        assert request.headers["authorization"] == "Bearer test-key"
        assert b"ENDPOINTING" in request.read()  # default mode
        return httpx.Response(200, json=RAW)

    _voice_client(handler, monkeypatch)

    body = await voice.transcribe(audio, keywords=voice.DEFAULT_KEYWORDS)

    assert len(calls) == 2
    assert body["sessionId"] == "sess-1"


async def test_voice_does_not_retry_client_errors(tmp_path: Path, monkeypatch) -> None:
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"RIFF")
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(400, text=json.dumps({"error": "malformed"}))

    _voice_client(handler, monkeypatch)

    with pytest.raises(MetaClientError) as info:
        await voice.transcribe(audio)

    assert len(calls) == 1
    assert info.value.status == 400
