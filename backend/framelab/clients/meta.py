"""Meta Model API client wrapper (OpenAI SDK pointed at api.meta.ai).

Spec: docs/plan/02-meta-api-reference.md §1, §4, §6, §7, §11.

- Client factories (the SDK does not read MODEL_API_KEY itself).
- Retry with exponential backoff + full jitter on 429/5xx/connection errors only.
- ``model_call`` / ``model_result`` events with usage and estimated cost via an ``emit`` callback.
- Optional recording of raw responses (``FRAMELAB_RECORD_DIR``) for golden fixtures.
- ``strict_schema()`` for ``strict: true`` structured output (M1 finding V6).
"""

from __future__ import annotations

import asyncio
import copy
import json
import random
import time
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, TypeVar

import httpx
from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI, OpenAI
from pydantic import BaseModel

from framelab.config import META_BASE_URL, get_settings
from framelab.costs import token_cost

T = TypeVar("T")
Emit = Callable[..., Any]

RETRY_STATUS = frozenset({429, 500, 502, 503, 504})
MAX_ATTEMPTS = 5
BASE_DELAY_S = 1.0
MAX_DELAY_S = 30.0


class MetaClientError(RuntimeError):
    """Non-retryable API error (or retries exhausted)."""

    def __init__(self, status: int | None, body_excerpt: str, request_id: str | None = None):
        super().__init__(f"Meta API error {status}: {body_excerpt[:300]}")
        self.status = status
        self.body_excerpt = body_excerpt
        self.request_id = request_id


# --------------------------------------------------------------------------------------------
# Clients
# --------------------------------------------------------------------------------------------


def _api_key() -> str:
    key = get_settings().model_api_key
    if not key:
        raise MetaClientError(None, "MODEL_API_KEY is not set (see .env.example)")
    return key


def sync_client(timeout: float = 180.0) -> OpenAI:
    """OpenAI SDK client for the Meta Model API. Retries are handled by ``call_with_retries``."""
    return OpenAI(base_url=META_BASE_URL, api_key=_api_key(), timeout=timeout, max_retries=0)


def async_client(timeout: float = 180.0) -> AsyncOpenAI:
    return AsyncOpenAI(base_url=META_BASE_URL, api_key=_api_key(), timeout=timeout, max_retries=0)


# --------------------------------------------------------------------------------------------
# Retry helpers
# --------------------------------------------------------------------------------------------


def is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, APIConnectionError | APITimeoutError | httpx.TransportError):
        return True
    if isinstance(exc, MetaClientError):  # raw httpx clients (Voice) raise this on non-2xx
        return exc.status in RETRY_STATUS
    status = getattr(exc, "status_code", None)
    return status in RETRY_STATUS


def backoff_delay(attempt: int, rng: random.Random | None = None) -> float:
    """Full-jitter exponential backoff: uniform(0, min(max, base·2^attempt))."""
    rng = rng or random
    return rng.uniform(0, min(MAX_DELAY_S, BASE_DELAY_S * 2**attempt))


def to_client_error(exc: BaseException) -> MetaClientError:
    if isinstance(exc, MetaClientError):
        return exc
    status = getattr(exc, "status_code", None)
    body = ""
    request_id = None
    if isinstance(exc, APIStatusError):
        try:
            body = exc.response.text[:800]
        except Exception:  # noqa: BLE001
            body = str(exc)
        request_id = exc.response.headers.get("x-request-id")
    return MetaClientError(status, body or str(exc), request_id)


async def call_with_retries(
    fn: Callable[[], Awaitable[T]],
    *,
    attempts: int = MAX_ATTEMPTS,
    sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
) -> T:
    """Await ``fn()``; retry retryable errors with jittered backoff; wrap others."""
    for attempt in range(attempts):
        try:
            return await fn()
        except Exception as exc:  # noqa: BLE001
            if attempt + 1 >= attempts or not is_retryable(exc):
                raise to_client_error(exc) from exc
            await sleep(backoff_delay(attempt))
    raise AssertionError("unreachable")


# --------------------------------------------------------------------------------------------
# Usage, cost, events, recording
# --------------------------------------------------------------------------------------------


def usage_dict(usage: Any) -> dict[str, Any]:
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    return dict(usage)


def usage_summary(usage: dict[str, Any]) -> dict[str, int]:
    """Normalise Chat and Responses usage into input/cached/output/reasoning token counts."""
    details_in = usage.get("input_tokens_details") or usage.get("prompt_tokens_details") or {}
    details_out = usage.get("output_tokens_details") or usage.get("completion_tokens_details") or {}
    return {
        "input": int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0),
        "cached": int((details_in or {}).get("cached_tokens") or 0),
        "output": int(usage.get("output_tokens") or usage.get("completion_tokens") or 0),
        "reasoning": int((details_out or {}).get("reasoning_tokens") or 0),
    }


def record_response(purpose: str, payload: Any) -> None:
    """Save a raw response under FRAMELAB_RECORD_DIR (golden fixtures). No-op if unset."""
    record_dir = get_settings().record_dir
    if not record_dir:
        return
    path = (
        Path(record_dir) / f"{time.strftime('%Y%m%d_%H%M%S')}_{purpose}_{uuid.uuid4().hex[:6]}.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    data = payload.model_dump() if hasattr(payload, "model_dump") else payload
    path.write_text(json.dumps(data, default=str, indent=1), encoding="utf-8")


class CallTracker:
    """Emits ``model_call`` at start and ``model_result`` at the end of one API call."""

    def __init__(self, emit: Emit | None, model: str, api: str, purpose: str, **summary: Any):
        self.emit = emit
        self.call_id = f"call_{uuid.uuid4().hex[:10]}"
        self.model = model
        self.api = api
        self.purpose = purpose
        self.t0 = time.perf_counter()
        if emit:
            emit(
                "model_call",
                call_id=self.call_id,
                model=model,
                api=api,
                purpose=purpose,
                input_summary=summary,
            )

    def done(
        self,
        *,
        ok: bool,
        usage: dict[str, Any] | None = None,
        units: dict[str, Any] | None = None,
        cost_usd: float | None = None,
        request_id: str | None = None,
        error: str | None = None,
    ) -> dict[str, Any]:
        tokens = usage_summary(usage or {})
        if cost_usd is None:
            cost_usd = token_cost(self.model, tokens["input"], tokens["output"], tokens["cached"])
        result = {
            "call_id": self.call_id,
            "model": self.model,
            "purpose": self.purpose,
            "latency_s": round(time.perf_counter() - self.t0, 3),
            "usage": tokens,
            "units": units or {},
            "cost_usd": round(cost_usd, 6),
            "request_id": request_id,
            "ok": ok,
        }
        if error:
            result["error"] = error[:500]
        if self.emit:
            self.emit("model_result", **result)
        return result


# --------------------------------------------------------------------------------------------
# High-level calls
# --------------------------------------------------------------------------------------------


async def chat(
    *, model: str, messages: list[dict[str, Any]], purpose: str, emit: Emit | None = None, **kw: Any
) -> Any:
    """Chat Completions with retries, events, cost and optional recording."""
    tracker = CallTracker(emit, model, "chat", purpose, n_messages=len(messages))
    try:
        async with async_client() as api:
            resp = await call_with_retries(
                lambda: api.chat.completions.create(model=model, messages=messages, **kw)
            )
    except MetaClientError as exc:
        tracker.done(ok=False, error=str(exc), request_id=exc.request_id)
        raise
    tracker.done(ok=True, usage=usage_dict(resp.usage), request_id=getattr(resp, "id", None))
    record_response(purpose, resp)
    return resp


async def responses(*, model: str, purpose: str, emit: Emit | None = None, **kw: Any) -> Any:
    """Responses API (non-streaming) with retries, events, cost and optional recording."""
    tracker = CallTracker(emit, model, "responses", purpose)
    try:
        async with async_client() as api:
            resp = await call_with_retries(lambda: api.responses.create(model=model, **kw))
    except MetaClientError as exc:
        tracker.done(ok=False, error=str(exc), request_id=exc.request_id)
        raise
    tracker.done(ok=True, usage=usage_dict(resp.usage), request_id=getattr(resp, "id", None))
    record_response(purpose, resp)
    return resp


async def upload_file(path: Path, mime: str = "video/mp4", emit: Emit | None = None) -> str:
    """Files API upload (purpose=user_data). Returns the ``file-…`` id."""
    tracker = CallTracker(emit, "files", "files", "upload", bytes=path.stat().st_size)
    data = path.read_bytes()
    try:
        async with async_client(timeout=300) as api:
            uploaded = await call_with_retries(
                lambda: api.files.create(file=(path.name, data, mime), purpose="user_data")
            )
    except MetaClientError as exc:
        tracker.done(ok=False, error=str(exc), cost_usd=0.0)
        raise
    tracker.done(ok=True, cost_usd=0.0, units={"bytes": len(data)})
    if not str(uploaded.id).startswith("file-"):
        raise MetaClientError(None, f"unexpected file id {uploaded.id!r}")
    return uploaded.id


async def delete_file(file_id: str, emit: Emit | None = None) -> bool:
    """Delete an uploaded file (files never expire on their own, M1 V10). Never raises."""
    try:
        async with async_client(timeout=60) as api:
            deleted = await api.files.delete(file_id)
        return bool(getattr(deleted, "deleted", False))
    except Exception as exc:  # noqa: BLE001
        if emit:
            emit("warning", message=f"could not delete uploaded file {file_id}: {exc}")
        return False


# --------------------------------------------------------------------------------------------
# Structured output
# --------------------------------------------------------------------------------------------


def strict_schema(model_or_schema: type[BaseModel] | dict[str, Any]) -> dict[str, Any]:
    """Strict-compatible JSON schema: every object closed and every property required.

    Required for ``strict: true`` (M1 finding V6: 'additionalProperties' must be false).
    Optional fields must be nullable types in the Pydantic model (``x: float | None``).
    """
    if isinstance(model_or_schema, dict):
        schema = copy.deepcopy(model_or_schema)
    else:
        schema = model_or_schema.model_json_schema()

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("type") == "object" or "properties" in node:
                props = node.get("properties", {})
                node["additionalProperties"] = False
                node["required"] = list(props)
                for prop in props.values():
                    if isinstance(prop, dict):
                        prop.pop("default", None)
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)

    visit(schema)
    return schema
