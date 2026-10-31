"""Muse Spark Open via Baseten (stretch S-3, open-weight bonus).

Spec: docs/plan/02-meta-api-reference.md §10.

Baseten serves Meta's open-weight Muse Spark Open behind an OpenAI-compatible Chat Completions
endpoint. The hackathon key only works with the slug ``meta-models/muse-spark-open-1026``.
Differences from the Meta API that matter here (Baseten Model APIs overview, 31 Oct 2026):

- no structured outputs or JSON mode: the Designer submits its proposal through a tool call;
- the model always reasons first; ``reasoning_effort`` (low…xhigh, default xhigh) sets how long.
  If the endpoint refuses the parameter, the call is repeated once without it;
- ``n`` > 1 is refused; ``temperature``/``seed`` do not make output reproducible.

Errors are ``BasetenError`` (a ``MetaClientError``), so callers handle both providers alike.
"""

from __future__ import annotations

from typing import Any

from openai import AsyncOpenAI

from framelab.clients.meta import (
    CallTracker,
    Emit,
    MetaClientError,
    call_with_retries,
    record_response,
    usage_dict,
)
from framelab.config import BASETEN_BASE_URL, MODEL_SPARK_OPEN_BASETEN, get_settings

TIMEOUT_S = 120.0
LABEL = "Muse Spark Open"


class BasetenError(MetaClientError):
    """Non-retryable Baseten error (or retries exhausted)."""

    def __str__(self) -> str:
        return f"Baseten error {self.status}: {self.body_excerpt[:300]}"


def _api_key() -> str:
    key = get_settings().baseten_api_key
    if not key:
        raise BasetenError(None, "BASETEN_API_KEY is not set (see .env.example)")
    return key


def async_client(timeout: float = TIMEOUT_S) -> AsyncOpenAI:
    return AsyncOpenAI(
        base_url=BASETEN_BASE_URL, api_key=_api_key(), timeout=timeout, max_retries=0
    )


def _as_baseten(exc: MetaClientError) -> BasetenError:
    if isinstance(exc, BasetenError):
        return exc
    return BasetenError(exc.status, exc.body_excerpt, exc.request_id)


def _refuses_reasoning_effort(exc: MetaClientError, kw: dict[str, Any]) -> bool:
    return (
        "reasoning_effort" in kw
        and exc.status in (400, 422)
        and "reasoning" in (exc.body_excerpt or "").lower()
    )


async def chat(
    *,
    messages: list[dict[str, Any]],
    purpose: str,
    emit: Emit | None = None,
    model: str = MODEL_SPARK_OPEN_BASETEN,
    **kw: Any,
) -> Any:
    """Chat Completions on Muse Spark Open, with the same events, retries and cost tracking as
    ``meta.chat``. ``model`` is fixed by the hackathon key; the argument keeps the signature."""
    tracker = CallTracker(emit, model, "chat", purpose, n_messages=len(messages), host="baseten")
    try:
        try:
            async with async_client() as api:
                resp = await call_with_retries(
                    lambda: api.chat.completions.create(model=model, messages=messages, **kw)
                )
        except MetaClientError as exc:
            if not _refuses_reasoning_effort(exc, kw):
                raise
            kw = {k: v for k, v in kw.items() if k != "reasoning_effort"}
            async with async_client() as api:
                resp = await call_with_retries(
                    lambda: api.chat.completions.create(model=model, messages=messages, **kw)
                )
    except MetaClientError as exc:
        err = _as_baseten(exc)
        tracker.done(ok=False, error=str(err), request_id=err.request_id)
        raise err from exc
    tracker.done(ok=True, usage=usage_dict(resp.usage), request_id=getattr(resp, "id", None))
    record_response(purpose, resp)
    return resp


def reasoning_text(message: Any) -> str | None:
    """The model's reasoning, if the endpoint returns it (``reasoning_content`` or
    ``reasoning``). Never shown as an answer; only summarised in the trace."""
    for name in ("reasoning_content", "reasoning"):
        value = getattr(message, name, None)
        if value is None and hasattr(message, "model_extra"):
            value = (message.model_extra or {}).get(name)
        if isinstance(value, str) and value.strip():
            return value
    return None
