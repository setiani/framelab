"""Shared fixtures. Live tests (calling paid APIs) are skipped unless FRAMELAB_LIVE_TESTS=1."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from framelab.config import Settings
from framelab.events import EventBus
from framelab.store import RunStore


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if os.environ.get("FRAMELAB_LIVE_TESTS") == "1":
        return
    skip_live = pytest.mark.skip(reason="live API test: set FRAMELAB_LIVE_TESTS=1 to run")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)


@pytest.fixture(autouse=True)
def _no_baseten_key(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Offline tests never reach Baseten, even when the developer's .env has a key: with
    ``designer_backend="auto"`` a key would send the Designer to Muse Spark Open (S-3)."""
    if "live" not in request.keywords:
        monkeypatch.setattr(Settings, "baseten_api_key", property(lambda self: None))


@pytest.fixture(autouse=True)
def _no_demo_samples(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests never see the repository's real ``demo_runs/`` (cached samples and series): the
    default ``demo_dir`` points at an empty folder. Tests of demo mode pass their own."""
    monkeypatch.setenv("FRAMELAB_DEMO_DIR", str(tmp_path / "no_demo_runs"))


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(runs_dir=tmp_path / "runs")


@pytest.fixture
def store(settings: Settings) -> RunStore:
    return RunStore(settings.runs_dir)


@pytest.fixture
def bus(settings: Settings) -> EventBus:
    return EventBus(settings.runs_dir)
