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


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(runs_dir=tmp_path / "runs")


@pytest.fixture
def store(settings: Settings) -> RunStore:
    return RunStore(settings.runs_dir)


@pytest.fixture
def bus(settings: Settings) -> EventBus:
    return EventBus(settings.runs_dir)
