"""Protecting API credits on a public deployment (docs/plan/10-deployment-demo-submission.md §2).

Three layers: the per-run cost guard (06 §10, in the pipeline), a per-address rate limit on
uploads and replays (here), and a daily budget that pauses live uploads while the samples keep
working (here).
"""

from __future__ import annotations

import math
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass

from fastapi import Request

from framelab.config import Settings
from framelab.schemas import utcnow
from framelab.store import RunStore

WINDOW_S = 3600.0


class RateLimiter:
    """Sliding one-hour window per (bucket, client address). In memory: one app instance."""

    def __init__(self, clock=time.monotonic) -> None:
        self._clock = clock
        self._hits: dict[tuple[str, str], deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def hit(self, bucket: str, client: str, limit: int) -> float | None:
        """Count one request. Returns ``None`` if allowed, else seconds until the next slot."""
        if limit <= 0:
            return None
        now = self._clock()
        with self._lock:
            hits = self._hits[(bucket, client)]
            while hits and now - hits[0] >= WINDOW_S:
                hits.popleft()
            if len(hits) >= limit:
                return max(WINDOW_S - (now - hits[0]), 1.0)
            hits.append(now)
            return None


def client_address(request: Request, forwarded_hops: int) -> str:
    """The caller's address. Behind ``forwarded_hops`` proxies, the entry they appended to
    ``X-Forwarded-For`` (entries further left are written by the client and can be forged)."""
    if forwarded_hops > 0:
        parts = [p.strip() for p in request.headers.get("x-forwarded-for", "").split(",")]
        parts = [p for p in parts if p]
        if len(parts) >= forwarded_hops:
            return parts[-forwarded_hops]
    return request.client.host if request.client else "unknown"


def spent_today(store: RunStore) -> float:
    """Model cost of today's live runs (UTC day). Samples and replays cost nothing."""
    today = utcnow().date()
    return sum(
        s.cost_usd
        for s in store.list_runs(limit=100_000)
        if not s.demo and s.created_at.date() == today
    )


@dataclass
class UploadStatus:
    enabled: bool
    reason: str | None = None  # demo_mode | no_api_key | daily_budget
    message: str | None = None

    def public(self) -> dict:
        return {"enabled": self.enabled, "reason": self.reason, "message": self.message}


def upload_status(settings: Settings, store: RunStore, *, needs_key: bool) -> UploadStatus:
    """Whether live uploads (and re-runs) are accepted right now, and why not."""
    if settings.demo and not settings.demo_allow_upload:
        return UploadStatus(False, "demo_mode", "Live uploads are off in this demo. Try a sample.")
    if needs_key and not settings.model_api_key:
        return UploadStatus(
            False, "no_api_key", "Live uploads are off: no model API key is set. Try a sample."
        )
    if settings.daily_budget_usd > 0 and spent_today(store) >= settings.daily_budget_usd:
        return UploadStatus(
            False,
            "daily_budget",
            "Live uploads are paused to protect API credits. Try the samples.",
        )
    return UploadStatus(True)


def retry_minutes(seconds: float) -> int:
    return max(1, math.ceil(seconds / 60))
