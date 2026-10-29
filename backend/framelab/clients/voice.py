"""Muse Voice Transcribe client (file endpoint).

Spec: docs/plan/02-meta-api-reference.md §8 (pattern adapted from meta-model-cookbook/06_muse_voice)
Plain HTTP, because the SDK has no ASR call. Retries, ``model_call`` events and cost follow
02 §11 and the Meta client.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx

from framelab.clients.meta import CallTracker, Emit, MetaClientError, call_with_retries
from framelab.config import META_TRANSCRIBE_URL, MODEL_VOICE, get_settings
from framelab.costs import voice_cost

VOICE_TIMEOUT_S = 300.0

# Keywords from 02 §8: they bias the recogniser towards the words this app needs.
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


def _http() -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=VOICE_TIMEOUT_S)


async def transcribe(
    path: Path,
    *,
    mode: str = "ENDPOINTING",
    keywords: list[str] | None = None,
    language_bias: list[str] | None = None,
    emit: Emit | None = None,
) -> dict[str, Any]:
    """POST a WAV (mono s16, 16 kHz) to /v1/asr/transcribe and return the parsed JSON.

    Retries 429/5xx and network errors. Any other non-2xx response raises ``MetaClientError``.
    ``language_bias`` stays ``None`` (M1 V8): digits come back as digits, not words.
    """
    key = get_settings().model_api_key
    if not key:
        raise MetaClientError(None, "MODEL_API_KEY is not set (see .env.example)")
    request: dict[str, Any] = {"mode": mode, "model": MODEL_VOICE, "audioEncoding": "WAV"}
    if keywords:
        request["keywords"] = keywords
    if language_bias:
        request["languageBias"] = language_bias
    audio = path.read_bytes()
    tracker = CallTracker(
        emit, MODEL_VOICE, "voice", "transcribe", audio_bytes=len(audio), mode=mode
    )

    async def post() -> dict[str, Any]:
        async with _http() as http:
            resp = await http.post(
                META_TRANSCRIBE_URL,
                headers={"Authorization": f"Bearer {key}"},
                files={
                    "request": (None, json.dumps(request), "application/json"),
                    "audio": (path.name, audio, "audio/wav"),
                },
            )
        if resp.status_code != 200:
            raise MetaClientError(resp.status_code, resp.text, resp.headers.get("x-request-id"))
        return resp.json()

    try:
        body = await call_with_retries(post)
    except MetaClientError as exc:
        tracker.done(ok=False, error=str(exc), request_id=exc.request_id)
        raise
    audio_ms = int(body.get("audioDurationMs") or 0)
    tracker.done(
        ok=True,
        units={"audio_ms": audio_ms},
        cost_usd=voice_cost(audio_ms / 1000),
        request_id=body.get("sessionId"),
    )
    return body
