"""M7 slow-motion re-run (06 §7 G4): the same video, with the capture fps the diagnosis gives."""

from __future__ import annotations

from framelab.pipeline.rerun import create_rerun
from framelab.store import RunStore


def test_rerun_copies_the_input_and_sets_the_capture_fps(tmp_path) -> None:
    store = RunStore(tmp_path / "runs")
    original = store.create_run("slowmo.mp4", lang="en")
    store.input_video_path(original.run_id).write_bytes(b"video bytes")

    rerun = create_rerun(store, original.run_id, capture_fps=240)

    assert rerun.run_id != original.run_id
    assert rerun.capture_fps == 240 and rerun.lang == "en"
    assert store.input_video_path(rerun.run_id).read_bytes() == b"video bytes"
    assert store.load_state(original.run_id).capture_fps is None  # the original is untouched
    assert original.run_id in store.load_state(rerun.run_id).note
