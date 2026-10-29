"""Settings parsing that the live runs depend on."""

from __future__ import annotations

from pathlib import Path

from framelab.config import Settings


def test_empty_record_dir_means_recording_is_off(monkeypatch) -> None:
    # .env.example ships an empty FRAMELAB_RECORD_DIR=; that must not mean "the working directory"
    monkeypatch.setenv("FRAMELAB_RECORD_DIR", "")
    assert Settings().record_dir is None


def test_record_dir_is_used_when_set(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("FRAMELAB_RECORD_DIR", str(tmp_path))
    assert Settings().record_dir == tmp_path
