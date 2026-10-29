"""Re-run a run's video with the capture fps it needs (06 §7 slow-motion diagnosis).

The Analyst's G4 check finds a slow-motion clip saved at normal speed. The fix is the same video
with its real capture rate, so the run is copied rather than re-recorded.
"""

from __future__ import annotations

import shutil

from framelab.schemas import RunState
from framelab.store import RunStore


def create_rerun(
    store: RunStore, run_id: str, capture_fps: int | None = None, lang: str | None = None
) -> RunState:
    """A new run with the same input video. ``capture_fps`` and ``lang`` default to the
    original's. The original run is kept."""
    original = store.load_state(run_id)
    fps = capture_fps if capture_fps is not None else original.capture_fps
    new = store.create_run(
        original.original_filename,
        lang=lang or original.lang,  # type: ignore[arg-type]
        capture_fps=fps,
    )
    shutil.copyfile(store.input_video_path(run_id), store.input_video_path(new.run_id))
    new.note = f"re-run of {run_id} with capture fps {fps}"
    store.save_state(new)
    return new
