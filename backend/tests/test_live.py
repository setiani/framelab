"""Live smoke tests against the real Meta Model API (cost a fraction of a cent).

Run with: FRAMELAB_LIVE_TESTS=1 uv run pytest -m live
"""

from __future__ import annotations

import pytest

from framelab.clients.meta import sync_client
from framelab.config import MODEL_SPARK


@pytest.mark.live
def test_live_spark_ping() -> None:
    resp = sync_client().chat.completions.create(
        model=MODEL_SPARK,
        messages=[{"role": "user", "content": "Reply with the single word: ready"}],
    )
    assert (resp.choices[0].message.content or "").strip()
