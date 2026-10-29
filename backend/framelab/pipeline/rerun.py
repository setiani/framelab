"""Re-run a run's video with the capture fps it needs (06 §7 slow-motion diagnosis).

The Analyst's G4 check finds a slow-motion clip saved at normal speed. The fix is the same video
with its real capture rate, so the run is copied rather than re-recorded.
"""

from __future__ import annotations

import shutil

from framelab.schemas import RunState
from framelab.store import RunStore


def create_rerun(store: RunStore, run_id: str, capture_fps: int) -> RunState:
    """A new run with the same input video and ``capture_fps``. The original is kept."""
    original = store.load_state(run_id)
    new = store.create_run(original.original_filename, lang=original.lang, capture_fps=capture_fps)
    shutil.copyfile(store.input_video_path(run_id), store.input_video_path(new.run_id))
    new.note = f"re-run of {run_id} with capture fps {capture_fps}"
    store.save_state(new)
    return new
