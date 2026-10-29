"""Offline fakes that replay SAM responses recorded from the live API in M1.

``FakeSam`` patches ``framelab.clients.sam`` (upload/segment) and ``framelab.clients.meta``
(delete_file) so the real pipeline runs end-to-end without network access.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from framelab.clients import meta as meta_client
from framelab.clients import sam as sam_client
from framelab.clients import voice as voice_client

FIXTURES = Path(__file__).parent / "fixtures"
SAM_FIXTURES = FIXTURES / "sam"
MEDIA = FIXTURES / "media"


class FakeResponses:
    """Scripted Responses API (Spark): each call returns the next output text and is recorded."""

    def __init__(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, *, model: str, purpose: str, emit=None, **kw: Any):
        self.calls.append({"model": model, "purpose": purpose, **kw})
        if not self.outputs:
            raise AssertionError("FakeResponses: no scripted output left")
        text = self.outputs.pop(0)
        return SimpleNamespace(id=f"resp_fake_{len(self.calls)}", output_text=text, usage=None)

    def install(self, monkeypatch: pytest.MonkeyPatch) -> FakeResponses:
        monkeypatch.setattr(meta_client, "responses", self)
        return self


class FakeVoice:
    """Voice Transcribe returning a stored response body (or raising, if given an exception)."""

    def __init__(self, body: dict[str, Any] | None = None, error: Exception | None = None):
        self.body = body or {}
        self.error = error
        self.calls: list[str] = []

    async def transcribe(self, path: Path, **kw: Any) -> dict[str, Any]:
        self.calls.append(path.name)
        if self.error is not None:
            raise self.error
        return self.body

    def install(self, monkeypatch: pytest.MonkeyPatch) -> FakeVoice:
        monkeypatch.setattr(voice_client, "transcribe", self.transcribe)
        return self


class FakeSam:
    """Map (media, concept) → recorded fixture; record every call."""

    def __init__(self, video: dict[str, str], image: dict[str, str]) -> None:
        self.video = video
        self.image = image
        self.calls: list[tuple[str, str]] = []
        self.deleted: list[str] = []
        self.uploads = 0

    async def upload_video(self, path: Path, emit=None) -> str:
        self.calls.append(("upload", path.name))
        self.uploads += 1
        return f"file-fake-{self.uploads}"  # unique, so tests can tell uploads apart

    async def _replay(
        self, fixture: str | None, concept: str, media: str, record_path: Path, frame_index=None
    ):
        if fixture is None:  # unknown concept → valid empty result
            out = sam_client.SamOutput(
                concept=concept,
                media=media,  # type: ignore[arg-type]
                prompt_kind="text",
                status="completed",
                result=await sam_client.parse_text("", media),  # type: ignore[arg-type]
                text="",
                frame_index=frame_index,
            )
        else:
            out = await sam_client.load_record(SAM_FIXTURES / fixture)
            out.concept, out.frame_index = concept, frame_index if media == "image" else None
        out.record_path = str(record_path)
        sam_client.save_record(out, record_path)
        return out

    async def segment_video(
        self, file_id, concept, record_path, *, emit=None, frames_hint=0, prompt=None
    ):
        self.calls.append(("video", concept))
        return await self._replay(self.video.get(concept), concept, "video", record_path)

    async def segment_image(self, image_path, concept, record_path, *, emit=None, frame_index=None):
        self.calls.append(("image", concept))
        return await self._replay(
            self.image.get(concept), concept, "image", record_path, frame_index
        )

    async def delete_file(self, file_id, emit=None) -> bool:
        self.deleted.append(file_id)
        return True

    def install(self, monkeypatch: pytest.MonkeyPatch) -> FakeSam:
        monkeypatch.setattr(sam_client, "upload_video", self.upload_video)
        monkeypatch.setattr(sam_client, "segment_video", self.segment_video)
        monkeypatch.setattr(sam_client, "segment_image", self.segment_image)
        monkeypatch.setattr(meta_client, "delete_file", self.delete_file)
        return self


def freefall_fake() -> FakeSam:
    return FakeSam(
        video={"orange ball": "freefall_60fps_5s_orange_ball.sam.json.gz"},
        image={
            "ruler": "keyframe_000_ruler.sam.json.gz",
            "orange ball": "keyframe_000_orange_ball.sam.json.gz",
        },
    )


def pendulum_fake() -> FakeSam:
    """Synthetic pendulum (tests/synth_sam.py); the ruler is the live M1 recording (same box)."""
    return FakeSam(
        video={"red ball": "pendulum_60fps_12s_red_ball.sam.json.gz"},
        image={"ruler": "keyframe_000_ruler.sam.json.gz", "string": "pendulum_string.sam.json.gz"},
    )


def _reply(text: str | None, calls: list[Any]) -> Any:
    message = SimpleNamespace(content=text, tool_calls=calls or None, model_extra={})
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeChat:
    """Scripted Chat Completions for the Analyst (06 §3.1).

    A step is a list of ``(tool_name, args)`` calls (several = parallel calls in one turn), a text
    reply, an exception to raise, or a callable taking the messages and returning one of these.
    Every request is recorded, so tests can check what the model saw.
    """

    def __init__(self, steps: list[Any]) -> None:
        self.steps = list(steps)
        self.calls: list[dict[str, Any]] = []

    async def __call__(
        self, *, model: str, messages: list[dict[str, Any]], purpose: str, emit=None, **kw: Any
    ) -> Any:
        self.calls.append({"model": model, "purpose": purpose,
                           "messages": [dict(m) for m in messages], **kw})  # fmt: skip
        if not self.steps:
            raise AssertionError("FakeChat: no scripted step left")
        step = self.steps.pop(0)
        if callable(step):
            step = step(messages)
        if isinstance(step, Exception):
            raise step
        if isinstance(step, str):
            return _reply(step, [])
        calls = [
            SimpleNamespace(id=f"call_{len(self.calls)}_{i}",
                            function=SimpleNamespace(name=name, arguments=json.dumps(args)))
            for i, (name, args) in enumerate(step)
        ]  # fmt: skip
        return _reply(None, calls)

    def install(self, monkeypatch: pytest.MonkeyPatch) -> FakeChat:
        monkeypatch.setattr(meta_client, "chat", self)
        return self
