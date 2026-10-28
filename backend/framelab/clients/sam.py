"""SAM 3.1 client: upload, video/image segmentation, box-prompt fallback, raw event tee.

Spec: docs/plan/02-meta-api-reference.md §4–§5
(pattern adapted from meta-model-cookbook/07_sam/01_api_basics)
Status: skeleton (M0). Implement in milestone M2.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


async def upload_video(path: Path) -> str:
    """Upload an MP4 via the Files API (purpose='user_data'); return the ``file-…`` id."""
    raise NotImplementedError("Milestone M2: see docs/plan/09-milestones-and-tasks.md")


async def segment_video(file_id: str, concept: str, raw_log: Path) -> Any:
    """Stream a SAM video request, tee raw events to JSONL, parse with meta_sam_parser."""
    raise NotImplementedError("Milestone M2: see docs/plan/09-milestones-and-tasks.md")


async def segment_image(image_path: Path, concept: str, raw_log: Path) -> Any:
    """Segment one key frame (image request) and return the parsed result."""
    raise NotImplementedError("Milestone M2: see docs/plan/09-milestones-and-tasks.md")


async def segment_video_box_prompt(
    file_id: str, size: tuple[int, int], frame: int, box: tuple[int, int, int, int], raw_log: Path
) -> Any:
    """Fallback: box prompt built with meta_sam_parser.build_object_prompt (02 §5.3)."""
    raise NotImplementedError("Milestone M2: see docs/plan/09-milestones-and-tasks.md")


def validate_result(result: Any) -> None:
    """Raise if outcome != completed or parser diagnostics contain errors (empty = valid)."""
    raise NotImplementedError("Milestone M2: see docs/plan/09-milestones-and-tasks.md")
