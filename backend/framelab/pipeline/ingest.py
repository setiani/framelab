"""Ingest: probe, effective fps, normalize to CFR, audio, analysis clip, key frames.

Spec: docs/plan/04-pipeline-and-measurement.md §2
Status: skeleton (M0). Implement in milestone M2.
"""

from __future__ import annotations

from pathlib import Path

from framelab.schemas import ExperimentSpec, IngestResult, ProbeInfo


def probe(video: Path) -> ProbeInfo:
    """Run ffprobe (list args, no shell) and parse the subset in ProbeInfo."""
    raise NotImplementedError("Milestone M2: see docs/plan/09-milestones-and-tasks.md")


def choose_fps(info: ProbeInfo) -> tuple[int, int]:
    """Target CFR fps as a rational (num, den) per 04 §2.3."""
    raise NotImplementedError("Milestone M2: see docs/plan/09-milestones-and-tasks.md")


def run(run_dir: Path, capture_fps: int | None, max_long_edge_px: int = 960) -> IngestResult:
    """Normalize video, extract audio and count frames."""
    raise NotImplementedError("Milestone M2: see docs/plan/09-milestones-and-tasks.md")


def make_analysis_clip(
    run_dir: Path, ingest: IngestResult, spec: ExperimentSpec, max_s: float
) -> IngestResult:
    """Frame-accurate trimmed clip for SAM; sets clip_offset_frame and clip_path (04 §2.4)."""
    raise NotImplementedError("Milestone M2: see docs/plan/09-milestones-and-tasks.md")


def extract_keyframe(video: Path, frame_index: int, out_png: Path) -> Path:
    """Extract one frame as PNG (04 §2.5)."""
    raise NotImplementedError("Milestone M2: see docs/plan/09-milestones-and-tasks.md")
