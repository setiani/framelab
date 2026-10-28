from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel

from framelab.clients import meta as meta_client
from framelab.clients import sam as sam_client
from tests.fakes import SAM_FIXTURES


async def test_replay_recorded_video_fixture() -> None:
    out = await sam_client.load_record(SAM_FIXTURES / "freefall_60fps_5s_orange_ball.sam.json.gz")
    sam_client.validate(out.result)
    assert out.status == "completed"
    assert out.n_objects == 1
    frames = sorted(r.frame.frame_index for r in out.mask_records)
    # M1 finding V12: static ball (frames 0-31) is not reported; tracking starts at motion onset
    assert frames[0] == 32 and frames[-1] == 299 and len(frames) == 268
    assert out.usage.get("video_frames_processed") == 300


async def test_replay_image_fixtures() -> None:
    for name in ("keyframe_000_ruler", "keyframe_000_orange_ball"):
        out = await sam_client.load_record(SAM_FIXTURES / f"{name}.sam.json.gz")
        assert out.media == "image" and out.n_objects == 1


async def test_empty_text_is_valid_empty_result() -> None:
    result = await sam_client.parse_text("", "video")
    sam_client.validate(result)
    assert [r for r in result.records if r.kind == "mask"] == []


async def test_save_and_load_roundtrip(tmp_path) -> None:
    out = await sam_client.load_record(SAM_FIXTURES / "keyframe_000_ruler.sam.json.gz")
    path = tmp_path / "x.sam.json"
    sam_client.save_record(out, path)
    again = await sam_client.load_record(path)
    assert again.text == out.text and again.concept == "ruler"


def test_build_box_prompt() -> None:
    text = sam_client.build_box_prompt((960, 540), 5, (440, 50, 500, 110), (470, 80))
    assert "box;x1=440;y1=50;x2=499;y2=109;w=960;h=540" in text
    assert "point;x=470;y=80" in text


def test_input_rejects_empty_prompt() -> None:
    with pytest.raises(ValueError):
        sam_client._input("  ", {"type": "input_video", "file_id": "file-1"})


class Inner(BaseModel):
    a: int
    b: str | None = None


class Outer(BaseModel):
    inner: Inner
    tags: list[str]
    note: str | None = None


def test_strict_schema_closes_objects_and_requires_all() -> None:
    schema = meta_client.strict_schema(Outer)
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"inner", "tags", "note"}
    inner = schema["$defs"]["Inner"]
    assert inner["additionalProperties"] is False and set(inner["required"]) == {"a", "b"}
    assert "default" not in schema["properties"]["note"]


class FakeStatus(Exception):
    def __init__(self, status: int) -> None:
        super().__init__(f"status {status}")
        self.status_code = status


def test_retries_on_429_then_succeeds() -> None:
    calls = {"n": 0}

    async def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise FakeStatus(429)
        return "ok"

    async def no_sleep(_):
        return None

    assert asyncio.run(meta_client.call_with_retries(flaky, sleep=no_sleep)) == "ok"
    assert calls["n"] == 3


def test_no_retry_on_400() -> None:
    calls = {"n": 0}

    async def bad():
        calls["n"] += 1
        raise FakeStatus(400)

    async def no_sleep(_):
        return None

    with pytest.raises(meta_client.MetaClientError) as err:
        asyncio.run(meta_client.call_with_retries(bad, sleep=no_sleep))
    assert calls["n"] == 1 and err.value.status == 400


def test_usage_summary_handles_both_apis() -> None:
    chat = {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "prompt_tokens_details": {"cached_tokens": 4},
        "completion_tokens_details": {"reasoning_tokens": 3},
    }
    resp = {"input_tokens": 7, "output_tokens": 2, "input_tokens_details": {"cached_tokens": 1}}
    assert meta_client.usage_summary(chat) == {
        "input": 10,
        "cached": 4,
        "output": 5,
        "reasoning": 3,
    }
    assert meta_client.usage_summary(resp) == {"input": 7, "cached": 1, "output": 2, "reasoning": 0}
