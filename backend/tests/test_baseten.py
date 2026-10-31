"""S-3: the Baseten client for Muse Spark Open (02 §10), without network."""

from __future__ import annotations

from types import SimpleNamespace

import httpx
import openai
import pytest

from framelab.clients import baseten
from framelab.config import MODEL_SPARK_OPEN_BASETEN, Settings
from framelab.costs import model_family, token_cost


def bad_request(text: str) -> openai.BadRequestError:
    response = httpx.Response(400, request=httpx.Request("POST", "https://x/v1"), text=text)
    return openai.BadRequestError(text, response=response, body=None)


class FakeApi:
    """Stands in for AsyncOpenAI: records each create() and plays scripted outcomes."""

    def __init__(self, outcomes: list) -> None:
        self.outcomes = outcomes
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    async def create(self, **kw):
        self.calls.append(kw)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


def reply(text: str = "ok"):
    message = SimpleNamespace(
        content=text, tool_calls=None, model_extra={"reasoning_content": "hm"}
    )
    usage = SimpleNamespace(model_dump=lambda: {"prompt_tokens": 1000, "completion_tokens": 200})
    return SimpleNamespace(id="r1", choices=[SimpleNamespace(message=message)], usage=usage)


@pytest.fixture
def key(monkeypatch):
    monkeypatch.setattr(Settings, "baseten_api_key", property(lambda self: "bt-test"))


async def test_chat_uses_the_hackathon_model_and_reports_cost(monkeypatch, key) -> None:
    api = FakeApi([reply()])
    monkeypatch.setattr(baseten, "async_client", lambda timeout=0: api)
    events: list[tuple] = []

    resp = await baseten.chat(
        messages=[{"role": "user", "content": "hi"}],
        purpose="designer",
        emit=lambda t, **p: events.append((t, p)),
        reasoning_effort="low",
    )

    assert resp.choices[0].message.content == "ok"
    assert api.calls[0]["model"] == "meta-models/muse-spark-open-1026"
    result = next(p for t, p in events if t == "model_result")
    assert result["ok"] and result["cost_usd"] == pytest.approx(token_cost(
        MODEL_SPARK_OPEN_BASETEN, 1000, 200))  # fmt: skip
    assert model_family(MODEL_SPARK_OPEN_BASETEN) == "Muse Spark Open"


async def test_refused_reasoning_effort_is_dropped_once(monkeypatch, key) -> None:
    api = FakeApi([bad_request('{"error": "unknown parameter: reasoning_effort"}'), reply()])
    monkeypatch.setattr(baseten, "async_client", lambda timeout=0: api)

    await baseten.chat(messages=[], purpose="designer", reasoning_effort="low")

    assert "reasoning_effort" in api.calls[0] and "reasoning_effort" not in api.calls[1]


async def test_other_errors_are_baseten_errors(monkeypatch, key) -> None:
    api = FakeApi([bad_request('{"error": "bad tools"}')])
    monkeypatch.setattr(baseten, "async_client", lambda timeout=0: api)

    with pytest.raises(baseten.BasetenError, match="Baseten error 400"):
        await baseten.chat(messages=[], purpose="designer")


async def test_missing_key_is_a_clear_error() -> None:
    with pytest.raises(baseten.BasetenError, match="BASETEN_API_KEY"):
        await baseten.chat(messages=[], purpose="designer")


def test_reasoning_text_is_read_from_either_field() -> None:
    assert baseten.reasoning_text(reply().choices[0].message) == "hm"
    assert baseten.reasoning_text(SimpleNamespace(reasoning="why", model_extra={})) == "why"
    assert baseten.reasoning_text(SimpleNamespace(content="x", model_extra={})) is None


def test_health_reports_the_designer_model(tmp_path) -> None:
    from fastapi.testclient import TestClient

    from framelab.api.main import create_app
    from tests.test_api import fake_runner

    on = Settings(runs_dir=tmp_path / "a", designer_backend="baseten")
    off = Settings(runs_dir=tmp_path / "b", designer_backend="auto")  # no key in tests
    with TestClient(create_app(on, runner=fake_runner())) as client:
        health_on = client.get("/api/health").json()
    with TestClient(create_app(off, runner=fake_runner())) as client:
        health_off = client.get("/api/health").json()
    assert health_on["open_weights"] is True
    assert health_on["models"]["designer"] == MODEL_SPARK_OPEN_BASETEN
    assert health_off["open_weights"] is False
