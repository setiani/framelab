from __future__ import annotations

from fastapi.testclient import TestClient
from typer.testing import CliRunner

from framelab import __version__
from framelab.api.main import create_app
from framelab.cli import app as cli_app


def test_health(settings) -> None:
    client = TestClient(create_app(settings))
    body = client.get("/api/health").json()
    assert body["ok"] is True
    assert body["version"] == __version__
    assert body["models"]["sam"] == "sam-3.1"


def test_cli_version() -> None:
    result = CliRunner().invoke(cli_app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_cli_run_not_implemented_yet(tmp_path) -> None:
    video = tmp_path / "v.mp4"
    video.write_bytes(b"\x00")
    result = CliRunner().invoke(cli_app, ["run", str(video)])
    assert result.exit_code == 2
    assert "Not implemented yet" in result.output
