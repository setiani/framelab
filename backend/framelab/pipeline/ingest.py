"""Ingest: probe, effective fps, normalize to CFR, audio, analysis clip, key frames.

Spec: docs/plan/04-pipeline-and-measurement.md §2. All ffmpeg/ffprobe calls use list arguments
(no shell) so they work on Windows and Linux alike.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path

from framelab.schemas import IngestResult, ProbeInfo
from framelab.store import write_json_atomic

STANDARD_FPS = (24, 25, 30, 50, 60, 120, 240)
MIN_DURATION_S = 1.0
MAX_DURATION_S = 120.0


class IngestError(RuntimeError):
    """Input video is unusable (not a video, too short/long, ffmpeg failure)."""


# --------------------------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------------------------


def _bin(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise IngestError(f"{name} not found on PATH (Windows: winget install Gyan.FFmpeg)")
    return path


def run_tool(args: list[str], *, timeout: float = 600) -> subprocess.CompletedProcess[str]:
    """Run ffmpeg/ffprobe; raise IngestError with stderr on failure."""
    try:
        return subprocess.run(args, capture_output=True, text=True, check=True, timeout=timeout)
    except subprocess.CalledProcessError as exc:
        raise IngestError(f"{Path(args[0]).name} failed: {exc.stderr.strip()[-800:]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise IngestError(f"{Path(args[0]).name} timed out after {timeout}s") from exc


def parse_rate(value: str | None) -> Fraction | None:
    """'30000/1001' → Fraction; '0/0' or None → None."""
    if not value or value in ("0/0", "N/A"):
        return None
    try:
        frac = Fraction(value)
    except (ValueError, ZeroDivisionError):
        return None
    return frac if frac > 0 else None


# --------------------------------------------------------------------------------------------
# Probe
# --------------------------------------------------------------------------------------------


def probe_raw(video: Path) -> dict:
    out = run_tool(
        [
            _bin("ffprobe"),
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_streams",
            "-show_format",
            str(video),
        ]
    )
    return json.loads(out.stdout)


def _rotation(stream: dict) -> int:
    for side in stream.get("side_data_list") or []:
        if "rotation" in side:
            return int(round(float(side["rotation"])))
    tag = (stream.get("tags") or {}).get("rotate")
    return int(tag) if tag else 0


def parse_probe(data: dict) -> ProbeInfo:
    """Extract ProbeInfo from ffprobe JSON (pure function, unit-tested)."""
    streams = data.get("streams") or []
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise IngestError("No video stream found")
    rotation = _rotation(video)
    width, height = int(video["width"]), int(video["height"])
    if abs(rotation) % 180 == 90:
        width, height = height, width
    avg = parse_rate(video.get("avg_frame_rate"))
    r = parse_rate(video.get("r_frame_rate"))
    avg_f = float(avg or r or 0)
    r_f = float(r or avg or 0)
    if avg_f <= 0:
        raise IngestError("Could not determine the frame rate")
    duration = float(video.get("duration") or (data.get("format") or {}).get("duration") or 0)
    nb = video.get("nb_frames")
    is_vfr = abs(avg_f - r_f) / r_f > 0.01 if r_f else False
    return ProbeInfo(
        width=width,
        height=height,
        rotation_deg=rotation,
        duration_s=duration,
        avg_fps=avg_f,
        r_fps=r_f,
        nb_frames=int(nb) if nb and str(nb).isdigit() else None,
        is_vfr=is_vfr,
        has_audio=any(s.get("codec_type") == "audio" for s in streams),
        codec=str(video.get("codec_name", "unknown")),
    )


def probe(video: Path) -> ProbeInfo:
    return parse_probe(probe_raw(video))


def choose_fps(info: ProbeInfo, raw_rate: str | None = None) -> Fraction:
    """Target CFR fps (04 §2.3): the exact file rate if CFR, else the nearest standard rate."""
    if not info.is_vfr:
        exact = parse_rate(raw_rate) if raw_rate else None
        if exact and abs(float(exact) - info.avg_fps) / info.avg_fps < 0.01:
            return exact
        return Fraction(info.avg_fps).limit_denominator(1001)
    return Fraction(min(STANDARD_FPS, key=lambda f: abs(f - info.avg_fps)))


def scale_filter(max_long_edge: int) -> str:
    """Long edge ≤ max_long_edge, even dimensions, keep aspect (single filter argument)."""
    m = int(max_long_edge)
    return (
        f"scale='if(gt(iw,ih),min({m},iw),-2)':'if(gt(iw,ih),-2,min({m},ih))':flags=lanczos,"
        "scale=trunc(iw/2)*2:trunc(ih/2)*2"
    )


def count_frames(video: Path) -> int:
    out = run_tool(
        [_bin("ffprobe"), "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(video)]
    )  # fmt: skip
    return int(out.stdout.strip().split(",")[0])


def video_size(video: Path) -> tuple[int, int]:
    out = run_tool(
        [_bin("ffprobe"), "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", str(video)]
    )  # fmt: skip
    w, h = out.stdout.strip().split(",")[:2]
    return int(w), int(h)


# --------------------------------------------------------------------------------------------
# Normalization
# --------------------------------------------------------------------------------------------


def normalize_video(src: Path, dst: Path, fps: Fraction, max_long_edge: int) -> None:
    run_tool(
        [_bin("ffmpeg"), "-y", "-v", "error", "-i", str(src), "-an",
         "-vf", f"fps={fps.numerator}/{fps.denominator},{scale_filter(max_long_edge)}",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
         "-g", str(max(1, round(float(fps)))), str(dst)]
    )  # fmt: skip


def mux_audio(video: Path, original: Path, dst: Path) -> None:
    run_tool(
        [_bin("ffmpeg"), "-y", "-v", "error", "-i", str(video), "-i", str(original),
         "-map", "0:v", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "96k",
         "-shortest", str(dst)]
    )  # fmt: skip


def extract_audio(original: Path, dst: Path) -> None:
    run_tool(
        [_bin("ffmpeg"), "-y", "-v", "error", "-i", str(original), "-vn", "-ac", "1",
         "-ar", "16000", "-sample_fmt", "s16", str(dst)]
    )  # fmt: skip


def extract_keyframe(video: Path, frame_index: int, out_png: Path) -> Path:
    """Extract one frame (0-based index of ``video``) as PNG (04 §2.5)."""
    out_png.parent.mkdir(parents=True, exist_ok=True)
    run_tool(
        [_bin("ffmpeg"), "-y", "-v", "error", "-i", str(video),
         "-vf", f"select=eq(n\\,{int(frame_index)})", "-vframes", "1", str(out_png)]
    )  # fmt: skip
    if not out_png.is_file():
        raise IngestError(f"Frame {frame_index} could not be extracted")
    return out_png


def run(
    run_dir: Path,
    *,
    capture_fps: int | None = None,
    max_long_edge_px: int = 960,
) -> IngestResult:
    """Probe and normalize ``input/original.mp4``; write ``ingest/ingest.json``."""
    original = run_dir / "input" / "original.mp4"
    out_dir = run_dir / "ingest"
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = probe_raw(original)
    info = parse_probe(raw)
    write_json_atomic(out_dir / "probe.json", info)
    if info.duration_s and not (MIN_DURATION_S <= info.duration_s <= MAX_DURATION_S):
        raise IngestError(
            f"Video duration {info.duration_s:.1f}s outside {MIN_DURATION_S}-{MAX_DURATION_S}s"
        )
    vstream = next(s for s in raw["streams"] if s.get("codec_type") == "video")
    fps = choose_fps(info, vstream.get("r_frame_rate"))

    video = out_dir / "video.mp4"
    normalize_video(original, video, fps, max_long_edge_px)
    width, height = video_size(video)
    n_frames = count_frames(video)

    audio_path: Path | None = None
    video_with_audio: Path | None = None
    if info.has_audio:
        audio_path = out_dir / "audio.wav"
        extract_audio(original, audio_path)
        video_with_audio = out_dir / "video_with_audio.mp4"
        mux_audio(video, original, video_with_audio)

    file_fps = float(fps)
    result = IngestResult(
        probe=info,
        norm_width=width,
        norm_height=height,
        fps=file_fps,
        fps_num=fps.numerator,
        fps_den=fps.denominator,
        file_fps=file_fps,
        effective_fps=float(capture_fps) if capture_fps else file_fps,
        capture_fps_source="user" if capture_fps else "file",
        n_frames=n_frames,
        video_path=str(video.relative_to(run_dir)),
        video_with_audio_path=str(video_with_audio.relative_to(run_dir))
        if video_with_audio
        else None,
        audio_path=str(audio_path.relative_to(run_dir)) if audio_path else None,
    )
    write_json_atomic(out_dir / "ingest.json", result)
    return result


def make_analysis_clip(
    run_dir: Path,
    ingest: IngestResult,
    window_s: tuple[float, float],
    *,
    pad_before_s: float = 1.0,
    pad_after_s: float = 0.5,
    max_s: float = 20.0,
) -> IngestResult:
    """Frame-accurate trimmed clip for SAM (04 §2.4). Window times are in FILE time.

    Sets ``clip_offset_frame`` and ``clip_path``; verifies the frame count.
    """
    fps = ingest.file_fps
    t0 = max(0.0, window_s[0] - pad_before_s)
    t1 = min(ingest.n_frames / fps, window_s[1] + pad_after_s)
    if t1 - t0 > max_s:
        t1 = t0 + max_s
    f0 = max(0, int(round(t0 * fps)))
    f1 = min(ingest.n_frames - 1, max(f0, int(round(t1 * fps)) - 1))
    src = run_dir / ingest.video_path
    clip = run_dir / "ingest" / "analysis.mp4"
    if f0 == 0 and f1 == ingest.n_frames - 1:
        shutil.copyfile(src, clip)
    else:
        rate = f"{ingest.fps_num}/{ingest.fps_den}"
        select = f"select='between(n\\,{f0}\\,{f1})',setpts=N/({rate}*TB)"
        run_tool(
            [_bin("ffmpeg"), "-y", "-v", "error", "-i", str(src), "-vf", select,
             "-an", "-r", f"{ingest.fps_num}/{ingest.fps_den}",
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
             str(clip)]
        )  # fmt: skip
    n = count_frames(clip)
    expected = f1 - f0 + 1
    if n != expected:
        raise IngestError(f"analysis clip has {n} frames, expected {expected}")
    updated = ingest.model_copy(
        update={"clip_offset_frame": f0, "clip_path": str(clip.relative_to(run_dir))}
    )
    write_json_atomic(run_dir / "ingest" / "ingest.json", updated)
    return updated
