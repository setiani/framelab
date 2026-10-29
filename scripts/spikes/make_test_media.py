"""Generate synthetic test media for the M1 spikes (no camera needed).

Creates in scripts/spikes/out/media/:
  freefall_60fps_5s.mp4        960x540 CFR, one drop, burnt-in frame counter
  freefall_240fps_3s.mp4       same scene at 240 fps (V3)
  freefall_60fps_20s.mp4       repeated drops, 20 s (V3 duration limit)
  freefall_60fps_5s_audio.mp4  60 fps clip + audio track (V4, V5): speech if available
  pendulum_60fps_12s.mp4       pendulum, L = 0.60 m (pivot → bob centre), θ0 = 15°, γ = 0.02 s⁻¹,
                               held 0.5 s then released; exact nonlinear ODE (M4)
  keyframe_000.png             first frame (ruler + ball) for image segmentation
  speech.wav                   mono 16 kHz s16 speech (Windows SAPI / espeak-ng / --speech file)
  *.gt.json                    ground truth: fps, scale, ruler box, ball centre/radius per frame

Physics: ball dropped from rest after 0.5 s, g = 9.81 m/s², restitution 0.6, scale 460 px = 1 m,
ball diameter 0.10 m, motion blur from 8 sub-frames over a 180° shutter.

Usage:
  uv run python scripts/spikes/make_test_media.py [--speech my_voice.m4a] [--force]
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import MEDIA, ffmpeg  # noqa: E402

W, H = 960, 540
PX_PER_M = 460.0
G = 9.81
BALL_D_M = 0.10
R_PX = BALL_D_M * PX_PER_M / 2  # 23 px
FLOOR_Y = 500.0
START_CY = 60.0 + R_PX
BALL_X = 470.0
RULER = (600, 40, 630, 500)  # left, top, right, bottom (1.00 m = 460 px)
RESTITUTION = 0.6
HOLD_S = 0.5

SPEECH_TEXT = (
    "Tinggi jatuh satu meter, diameter bola sepuluh sentimeter. "
    "My hypothesis is that the ball falls in about zero point four five seconds."
)


def ball_y(t_in_cycle: float) -> float:
    """Ball centre y (px) at time t since the cycle start (analytic bounces)."""
    if t_in_cycle <= HOLD_S:
        return START_CY
    t = t_in_cycle - HOLD_S
    g_px = G * PX_PER_M
    drop = FLOOR_Y - R_PX - START_CY
    t_fall = (2 * drop / g_px) ** 0.5
    if t < t_fall:
        return START_CY + 0.5 * g_px * t * t
    t -= t_fall
    v = g_px * t_fall * RESTITUTION
    rest_y = FLOOR_Y - R_PX
    for _ in range(12):
        t_air = 2 * v / g_px
        if t < t_air:
            return rest_y - (v * t - 0.5 * g_px * t * t)
        t -= t_air
        v *= RESTITUTION
        if v < 5:
            break
    return rest_y


def background() -> np.ndarray:
    img = np.full((H, W, 3), (240, 243, 245), np.uint8)
    cv2.rectangle(img, (0, int(FLOOR_Y)), (W, H), (170, 175, 180), -1)
    cv2.line(img, (0, int(FLOOR_Y)), (W, int(FLOOR_Y)), (90, 90, 90), 2)
    left, top, right, bottom = RULER
    cv2.rectangle(img, (left, top), (right, bottom), (60, 200, 245), -1)  # yellow ruler
    cv2.rectangle(img, (left, top), (right, bottom), (40, 120, 150), 1)
    for cm in range(0, 101, 5):
        y = int(round(top + cm / 100 * (bottom - top)))
        length = 14 if cm % 10 == 0 else 7
        cv2.line(img, (left, y), (left + length, y), (20, 20, 20), 1)
        if cm % 10 == 0:
            cv2.putText(
                img, str(cm), (right + 4, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (30, 30, 30), 1
            )
    cv2.putText(
        img,
        "FrameLab synthetic test",
        (W - 250, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (120, 120, 120),
        1,
        cv2.LINE_AA,
    )
    return img


def render(path: Path, fps: int, seconds: float, cycle_s: float, subframes: int = 8) -> dict:
    n = int(round(fps * seconds))
    bg = background()
    cmd = [
        ffmpeg(), "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
        "-g", str(fps), str(path),
    ]  # fmt: skip
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None
    shutter = 0.5 / fps
    frames_gt = []
    for k in range(n):
        t = k / fps
        cycle_start = (t // cycle_s) * cycle_s
        acc = np.zeros((H, W, 3), np.float32)
        ys = []
        for s in range(subframes):
            ts = t + shutter * (s / max(subframes - 1, 1) - 0.5)
            local = min(max(ts - cycle_start, 0.0), cycle_s - 1e-9)  # no ghosting across cycles
            y = ball_y(local)
            ys.append(y)
            img = bg.copy()
            center = (int(round(BALL_X * 16)), int(round(y * 16)))
            cv2.circle(img, center, int(round(R_PX * 16)), (0, 128, 255), -1, cv2.LINE_AA, 4)
            hl = (int(round((BALL_X - 7) * 16)), int(round((y - 7) * 16)))
            cv2.circle(img, hl, 6 * 16, (90, 180, 255), -1, cv2.LINE_AA, 4)
            acc += img
        frame = (acc / subframes).astype(np.uint8)
        cv2.rectangle(frame, (12, 10), (190, 52), (255, 255, 255), -1)
        cv2.putText(
            frame, f"F {k:04d}", (20, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 2, cv2.LINE_AA
        )
        proc.stdin.write(frame.tobytes())
        frames_gt.append(
            {
                "frame": k,
                "cx": BALL_X,
                "cy": round(float(np.mean(ys)), 3),
                "cy_at_t": ball_y(t % cycle_s),
            }
        )
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed for {path}")
    gt = {
        "video": path.name,
        "fps": fps,
        "n_frames": n,
        "width": W,
        "height": H,
        "m_per_px": 1 / PX_PER_M,
        "g": G,
        "ball_radius_px": R_PX,
        "ball_diameter_m": BALL_D_M,
        "ruler_box": RULER,
        "ruler_length_m": 1.0,
        "hold_s": HOLD_S,
        "cycle_s": cycle_s,
        "frames": frames_gt,
    }
    path.with_suffix(".gt.json").write_text(json.dumps(gt), encoding="utf-8")
    print(f"  ✓ {path.name}: {n} frames @ {fps} fps")
    return gt


# --- pendulum (M4) ---------------------------------------------------------------------------
PEND_PIVOT = (330.0, 40.0)
PEND_L_M = 0.60
PEND_THETA0_DEG = 15.0
PEND_GAMMA = 0.02
PEND_BOB_D_M = 0.06
PEND_HOLD_S = 0.5


def pendulum_theta(t: np.ndarray) -> np.ndarray:
    """θ(t): held at θ0 for PEND_HOLD_S, then the exact damped nonlinear pendulum."""
    from scipy.integrate import solve_ivp

    def rhs(_t, s):
        th, om = s
        return [om, -(G / PEND_L_M) * np.sin(th) - 2 * PEND_GAMMA * om]

    th = np.full(t.shape, np.radians(PEND_THETA0_DEG))
    moving = t > PEND_HOLD_S
    if moving.any():
        tm = t[moving] - PEND_HOLD_S
        sol = solve_ivp(rhs, (0.0, float(tm.max()) + 1e-9), [th[0], 0.0], t_eval=tm,
                        rtol=1e-11, atol=1e-12, method="DOP853")  # fmt: skip
        th[moving] = sol.y[0]
    return th


def pendulum_background() -> np.ndarray:
    img = background()
    px, py = PEND_PIVOT
    cv2.rectangle(img, (int(px) - 60, int(py) - 14), (int(px) + 60, int(py)), (70, 70, 70), -1)
    return img


def bob_position(theta: float) -> tuple[float, float]:
    r = PEND_L_M * PX_PER_M
    return PEND_PIVOT[0] + r * np.sin(theta), PEND_PIVOT[1] + r * np.cos(theta)


def render_pendulum(path: Path, fps: int, seconds: float, subframes: int = 8) -> dict:
    n = int(round(fps * seconds))
    bg = pendulum_background()
    cmd = [
        ffmpeg(), "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
        "-g", str(fps), str(path),
    ]  # fmt: skip
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None
    shutter = 0.5 / fps
    offsets = shutter * (np.arange(subframes) / max(subframes - 1, 1) - 0.5)
    t_all = np.clip((np.arange(n)[:, None] / fps + offsets[None, :]).ravel(), 0, None)
    th_all = pendulum_theta(np.sort(np.unique(t_all)))
    lookup = dict(zip(np.sort(np.unique(t_all)).tolist(), th_all.tolist(), strict=True))
    r_bob = PEND_BOB_D_M * PX_PER_M / 2
    frames_gt = []
    for k in range(n):
        acc = np.zeros((H, W, 3), np.float32)
        xs, ys = [], []
        for off in offsets:
            th = lookup[max(k / fps + off, 0.0)]
            bx, by = bob_position(th)
            xs.append(bx)
            ys.append(by)
            img = bg.copy()
            p0 = (int(round(PEND_PIVOT[0] * 16)), int(round(PEND_PIVOT[1] * 16)))
            p1 = (int(round(bx * 16)), int(round(by * 16)))
            cv2.line(img, p0, p1, (40, 40, 40), 2, cv2.LINE_AA, 4)
            cv2.circle(img, p1, int(round(r_bob * 16)), (40, 40, 220), -1, cv2.LINE_AA, 4)
            acc += img
        frame = (acc / subframes).astype(np.uint8)
        cv2.rectangle(frame, (12, 10), (190, 52), (255, 255, 255), -1)
        cv2.putText(
            frame, f"F {k:04d}", (20, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 2, cv2.LINE_AA
        )
        proc.stdin.write(frame.tobytes())
        frames_gt.append({"frame": k, "cx": round(float(np.mean(xs)), 3),
                          "cy": round(float(np.mean(ys)), 3)})  # fmt: skip
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed for {path}")
    gt = {
        "video": path.name,
        "fps": fps,
        "n_frames": n,
        "width": W,
        "height": H,
        "m_per_px": 1 / PX_PER_M,
        "g": G,
        "pivot_px": list(PEND_PIVOT),
        "length_m": PEND_L_M,
        "theta0_deg": PEND_THETA0_DEG,
        "gamma": PEND_GAMMA,
        "bob_radius_px": r_bob,
        "bob_diameter_m": PEND_BOB_D_M,
        "hold_s": PEND_HOLD_S,
        "ruler_box": RULER,
        "ruler_length_m": 1.0,
        "small_angle_period_s": float(2 * np.pi * np.sqrt(PEND_L_M / G)),
        "frames": frames_gt,
    }
    path.with_suffix(".gt.json").write_text(json.dumps(gt), encoding="utf-8")
    print(f"  ✓ {path.name}: {n} frames @ {fps} fps")
    return gt


def make_speech(out_wav: Path, user_file: Path | None) -> bool:
    raw = MEDIA / "speech_raw.wav"
    if user_file:
        src = user_file
    elif platform.system() == "Windows" and shutil.which("powershell"):
        ps = (
            "Add-Type -AssemblyName System.Speech; "
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            "$v = $s.GetInstalledVoices() | Where-Object { $_.VoiceInfo.Culture.Name -eq 'id-ID' } "
            "| Select-Object -First 1; "
            "if ($v) { $s.SelectVoice($v.VoiceInfo.Name) }; "
            f"$s.SetOutputToWaveFile('{raw}'); $s.Speak('{SPEECH_TEXT}'); $s.Dispose()"
        )
        r = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True)
        if r.returncode != 0 or not raw.is_file():
            print("  ! Windows speech synthesis failed:", r.stderr.decode(errors="ignore")[:200])
            return False
        src = raw
    elif shutil.which("espeak-ng"):
        subprocess.run(["espeak-ng", "-v", "id", "-w", str(raw), SPEECH_TEXT], check=True)
        src = raw
    else:
        print("  ! No speech source. Re-run with --speech <your recording> for the voice spike.")
        return False
    subprocess.run(
        [ffmpeg(), "-y", "-v", "error", "-i", str(src), "-ac", "1", "-ar", "16000",
         "-sample_fmt", "s16", str(out_wav)],
        check=True,
    )  # fmt: skip
    print(f"  ✓ {out_wav.name} (from {src.name})")
    return True


def mux_audio(video: Path, speech: Path | None, out: Path) -> None:
    if speech and speech.is_file():
        audio_in = ["-i", str(speech)]
        afilter = ["-af", "apad"]
    else:  # still test "video with an audio track" using a quiet tone
        audio_in = ["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=16000"]
        afilter = ["-af", "volume=0.05"]
    subprocess.run(
        [ffmpeg(), "-y", "-v", "error", "-i", str(video), *audio_in, "-map", "0:v", "-map", "1:a",
         *afilter, "-c:v", "copy", "-c:a", "aac", "-b:a", "96k", "-shortest", str(out)],
        check=True,
    )  # fmt: skip
    print(f"  ✓ {out.name} (audio: {'speech' if speech and speech.is_file() else 'tone'})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--speech", type=Path, help="your own voice recording (any format)")
    parser.add_argument("--force", action="store_true", help="regenerate existing files")
    args = parser.parse_args()

    print("Generating synthetic test media in", MEDIA)
    jobs = [
        ("freefall_60fps_5s.mp4", 60, 5.0, 5.0),
        ("freefall_240fps_3s.mp4", 240, 3.0, 3.0),
        ("freefall_60fps_20s.mp4", 60, 20.0, 2.5),
    ]
    for name, fps, seconds, cycle in jobs:
        path = MEDIA / name
        if path.is_file() and not args.force:
            print(f"  = {name} exists")
            continue
        render(path, fps, seconds, cycle, subframes=8 if fps <= 60 else 4)

    pend = MEDIA / "pendulum_60fps_12s.mp4"
    if pend.is_file() and not args.force:
        print(f"  = {pend.name} exists")
    else:
        render_pendulum(pend, 60, 12.0)

    speech = MEDIA / "speech.wav"
    needs_speech = args.force or args.speech or not speech.is_file()
    if needs_speech and not make_speech(speech, args.speech):
        speech.unlink(missing_ok=True)
    mux_audio(MEDIA / "freefall_60fps_5s.mp4", speech, MEDIA / "freefall_60fps_5s_audio.mp4")

    key = MEDIA / "keyframe_000.png"
    subprocess.run(
        [ffmpeg(), "-y", "-v", "error", "-i", str(MEDIA / "freefall_60fps_5s.mp4"),
         "-vframes", "1", str(key)],
        check=True,
    )  # fmt: skip
    print(f"  ✓ {key.name}")


if __name__ == "__main__":
    main()
