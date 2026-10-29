"""Offline fakes that replay SAM responses recorded from the live API in M1.

``FakeSam`` patches ``framelab.clients.sam`` (upload/segment) and ``framelab.clients.meta``
(delete_file) so the real pipeline runs end-to-end without network access.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from framelab.clients import meta as meta_client
from framelab.clients import sam as sam_client

FIXTURES = Path(__file__).parent / "fixtures"
SAM_FIXTURES = FIXTURES / "sam"
MEDIA = FIXTURES / "media"


class FakeSam:
    """Map (media, concept) → recorded fixture; record every call."""

    def __init__(self, video: dict[str, str], image: dict[str, str]) -> None:
        self.video = video
        self.image = image
        self.calls: list[tuple[str, str]] = []
        self.deleted: list[str] = []

    async def upload_video(self, path: Path, emit=None) -> str:
        self.calls.append(("upload", path.name))
        return "file-fake-1"

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
