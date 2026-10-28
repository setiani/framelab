"""Run every M1 spike in order and write scripts/spikes/out/SUMMARY.md.

Usage (from the repo root):
  uv run python scripts/spikes/run_all.py                 # everything (~$0.75 total)
  uv run python scripts/spikes/run_all.py --speech my.m4a # use your own voice recording
  uv run python scripts/spikes/run_all.py --only sam_video --only voice

Each spike keeps going on errors and records them: an error IS a finding.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

# Windows consoles / redirected output may not be UTF-8; never crash on ✓/✗ symbols.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
SPIKES = ["sam_image", "sam_video", "spark_video", "structured_output", "voice"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", action="append", choices=SPIKES)
    parser.add_argument("--speech", type=Path, help="your own voice recording for the voice spike")
    parser.add_argument("--skip-long", action="store_true", help="skip the 20 s SAM clip")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    summary = OUT / "SUMMARY.md"
    summary.write_text(
        f"# FrameLab M1 spike results\n\nRun at {time.strftime('%F %T')} "
        f"on {sys.platform}, Python {sys.version.split()[0]}\n",
        encoding="utf-8",
    )
    media_cmd = [sys.executable, str(HERE / "make_test_media.py")]
    if args.speech:
        media_cmd += ["--speech", str(args.speech)]
    print("== make_test_media")
    subprocess.run(media_cmd, check=True)

    status = {}
    for spike in args.only or SPIKES:
        cmd = [sys.executable, str(HERE / f"spike_{spike}.py")]
        if spike == "sam_video" and args.skip_long:
            cmd.append("--skip-long")
        print(f"\n== spike_{spike}")
        status[spike] = subprocess.run(cmd).returncode
    print("\nExit codes:", status)
    print(f"Summary: {summary}")
    print("Tell Claude the spikes are done; it reads scripts/spikes/out/ to update the plan.")


if __name__ == "__main__":
    main()
