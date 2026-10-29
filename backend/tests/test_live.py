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


@pytest.mark.live
async def test_live_planner_strict_output() -> None:
    """Spark returns a spec that passes the strict schema and validation (M5 wire format)."""
    from framelab.agent import planner
    from framelab.clients import meta as meta_client
    from framelab.clients import sam as sam_client
    from framelab.pipeline import ingest
    from framelab.schemas import Transcript, TranscriptTurn
    from tests.fakes import MEDIA

    video = MEDIA / "freefall_60fps_5s.mp4"  # no audio: the transcript is supplied by hand
    line = "Ini bola oranye, saya jatuhkan dari setinggi 0.5 meter."
    transcript = Transcript(text=line, turns=[TranscriptTurn(index=0, text=line)])
    file_id = await sam_client.upload_video(video)
    try:
        spec = await planner.run(file_id, transcript, ingest.probe(video), "id", None)
    finally:
        await meta_client.delete_file(file_id)
    assert spec.experiment in ("free_fall", "unknown")
    assert spec.target_concept
    assert planner.validate_spec(spec, transcript, ingest.probe(video).duration_s) == []
