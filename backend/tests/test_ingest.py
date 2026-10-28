from __future__ import annotations

import shutil
import subprocess
from fractions import Fraction
from pathlib import Path

import pytest

from framelab.pipeline import ingest
from framelab.schemas import ProbeInfo
from tests.fakes import MEDIA

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")


def make_run(tmp_path: Path, src: Path) -> Path:
    run_dir = tmp_path / "run"
    (run_dir / "input").mkdir(parents=True)
    shutil.copyfile(src, run_dir / "input" / "original.mp4")
    return run_dir


def lavfi_video(path: Path, size: str, rate: str, seconds: float, audio: bool) -> Path:
    args = ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", f"testsrc=size={size}:rate={rate}"]
    if audio:
        args += ["-f", "lavfi", "-i", "sine=f=440", "-c:a", "aac", "-shortest"]
    args += ["-t", str(seconds), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)]
    subprocess.run(args, check=True)
    return path


def test_parse_probe_rotation_and_vfr() -> None:
    data = {
        "streams": [
            {"codec_type": "video", "codec_name": "h264", "width": 1920, "height": 1080,
             "avg_frame_rate": "2997/100", "r_frame_rate": "120/1", "duration": "4.0",
             "side_data_list": [{"rotation": -90}]},
            {"codec_type": "audio"},
        ]
    }  # fmt: skip
    info = ingest.parse_probe(data)
    assert (info.width, info.height) == (1080, 1920)
    assert info.is_vfr and info.has_audio and info.rotation_deg == -90


def test_choose_fps_exact_cfr_and_nearest_standard_for_vfr() -> None:
    base = dict(width=10, height=10, duration_s=1, has_audio=False, codec="h264")
    cfr = ProbeInfo(avg_fps=29.97003, r_fps=29.97003, is_vfr=False, **base)
    assert ingest.choose_fps(cfr, "30000/1001") == Fraction(30000, 1001)
    vfr = ProbeInfo(avg_fps=58.7, r_fps=120.0, is_vfr=True, **base)
    assert ingest.choose_fps(vfr) == Fraction(60)


def test_run_normalizes_fixture_video(tmp_path: Path) -> None:
    run_dir = make_run(tmp_path, MEDIA / "freefall_60fps_5s.mp4")
    res = ingest.run(run_dir, capture_fps=240)
    assert (res.norm_width, res.norm_height, res.n_frames) == (960, 540, 300)
    assert res.file_fps == 60 and res.effective_fps == 240 and res.capture_fps_source == "user"
    assert res.audio_path is None
    assert (run_dir / "ingest" / "ingest.json").is_file()


def test_downscale_audio_and_frame_accurate_clip(tmp_path: Path) -> None:
    src = lavfi_video(tmp_path / "src.mp4", "1920x1080", "60", 3, audio=True)
    run_dir = make_run(tmp_path, src)
    res = ingest.run(run_dir)
    assert (res.norm_width, res.norm_height) == (960, 540)
    assert res.n_frames == 180 and res.audio_path and res.video_with_audio_path
    clip = ingest.make_analysis_clip(run_dir, res, (2.0, 2.5), pad_before_s=0.5, pad_after_s=0.2)
    assert clip.clip_offset_frame == 90
    # t0 = 2.0 - 0.5 = 1.5 s → frame 90; t1 = 2.5 + 0.2 = 2.7 s → last frame 161
    assert ingest.count_frames(run_dir / clip.clip_path) == 161 - 90 + 1


def test_extract_keyframe(tmp_path: Path) -> None:
    png = ingest.extract_keyframe(MEDIA / "freefall_60fps_5s.mp4", 42, tmp_path / "f.png")
    assert png.stat().st_size > 1000


def test_rejects_non_video(tmp_path: Path) -> None:
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video")
    with pytest.raises(ingest.IngestError):
        ingest.probe(bad)
