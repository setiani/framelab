"""Meta Model API client wrapper (OpenAI SDK pointed at api.meta.ai).

Spec: docs/plan/02-meta-api-reference.md §1, §6, §7, §11
Status: client factories implemented (M0); retry/cost/event wrapper in milestone M2.
"""

from __future__ import annotations

from typing import Any

from openai import AsyncOpenAI, OpenAI

from framelab.config import META_BASE_URL, get_settings

RETRY_STATUS = frozenset({429, 500, 502, 503, 504})


class MetaClientError(RuntimeError):
    """Non-retryable API error (or retries exhausted)."""

    def __init__(self, status: int | None, body_excerpt: str, request_id: str | None = None):
        super().__init__(f"Meta API error {status}: {body_excerpt[:300]}")
        self.status = status
        self.body_excerpt = body_excerpt
        self.request_id = request_id


def _api_key() -> str:
    key = get_settings().model_api_key
    if not key:
        raise MetaClientError(None, "MODEL_API_KEY is not set (see .env.example)")
    return key


def sync_client(timeout: float = 180.0) -> OpenAI:
    """OpenAI SDK client for the Meta Model API. The SDK does not read MODEL_API_KEY itself."""
    return OpenAI(base_url=META_BASE_URL, api_key=_api_key(), timeout=timeout)


def async_client(timeout: float = 180.0) -> AsyncOpenAI:
    return AsyncOpenAI(base_url=META_BASE_URL, api_key=_api_key(), timeout=timeout)


def chat(*, model: str, messages: list[dict[str, Any]], purpose: str, **kwargs: Any) -> Any:
    """Chat Completions with retry (exp. backoff + full jitter, max 5), cost and events (02 §11)."""
    raise NotImplementedError("Milestone M2: see docs/plan/09-milestones-and-tasks.md")


async def responses(*, model: str, purpose: str, **kwargs: Any) -> Any:
    """Responses API call with the same retry/cost/event handling."""
    raise NotImplementedError("Milestone M2: see docs/plan/09-milestones-and-tasks.md")
