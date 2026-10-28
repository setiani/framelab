from __future__ import annotations

import json

import pytest

from framelab.schemas import Stage
from framelab.store import RunNotFoundError, RunStore, write_json_atomic


def test_create_and_load_run(store: RunStore) -> None:
    state = store.create_run("ball.mp4", lang="en", capture_fps=240)
    assert state.run_id.startswith("r_")
    run_dir = store.run_dir(state.run_id)
    for sub in ("input", "ingest/keyframes", "sam", "tracks", "analysis", "outputs"):
        assert (run_dir / sub).is_dir()
    loaded = store.load_state(state.run_id)
    assert loaded.original_filename == "ball.mp4"
    assert loaded.capture_fps == 240
    assert loaded.stage == Stage.CREATED


def test_set_stage_records_history(store: RunStore) -> None:
    state = store.create_run("a.mp4")
    store.set_stage(state, Stage.INGESTING)
    loaded = store.load_state(state.run_id)
    assert loaded.stage == Stage.INGESTING
    assert [s for s, _ in loaded.stage_history] == [Stage.CREATED, Stage.INGESTING]


def test_list_runs_newest_first(store: RunStore) -> None:
    ids = [store.create_run(f"{i}.mp4").run_id for i in range(3)]
    listed = [s.run_id for s in store.list_runs()]
    assert set(listed) == set(ids)
    created = [s.created_at for s in store.list_runs()]
    assert created == sorted(created, reverse=True)


def test_unknown_and_malicious_ids(store: RunStore) -> None:
    with pytest.raises(RunNotFoundError):
        store.load_state("r_20260101_000000_abcdef")
    with pytest.raises(RunNotFoundError):
        store.run_dir("../etc")
    with pytest.raises(RunNotFoundError):
        store.run_dir("r_../../x")


def test_artifact_path_blocks_traversal(store: RunStore) -> None:
    state = store.create_run("a.mp4")
    ok = store.artifact_path(state.run_id, "outputs/overlay.mp4")
    assert ok.name == "overlay.mp4"
    with pytest.raises(ValueError):
        store.artifact_path(state.run_id, "../../secret.txt")


def test_write_json_atomic_leaves_no_temp(tmp_path) -> None:
    target = tmp_path / "x" / "data.json"
    write_json_atomic(target, {"a": 1})
    write_json_atomic(target, {"a": 2})
    assert json.loads(target.read_text()) == {"a": 2}
    assert [p.name for p in target.parent.iterdir()] == ["data.json"]


def test_series_roundtrip(store: RunStore) -> None:
    series = store.create_series(["r_1", "r_2"])
    loaded = store.load_series(series.series_id)
    assert loaded.run_ids == ["r_1", "r_2"]
    assert loaded.status == "collecting"
