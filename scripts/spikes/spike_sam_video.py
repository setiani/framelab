"""Verify SAM 3.1 video behaviour (11 §1: V1, V2, V3, V4).

Milestone M1 (docs/plan/09-milestones-and-tasks.md). Use ONLY the request patterns in
docs/plan/02-meta-api-reference.md. Save raw responses under scripts/spikes/out/ (git-ignored)
and write the answers into docs/plan/11-risks-and-open-questions.md §1.

Checks:
  - Is frame_index 0-based and 1:1 with decoded frames of the uploaded file (V1)?
  - Are box/mask coordinates in the uploaded video's pixel space (V2)?
  - Max fps/frames/duration per request: test 3 s @ 240 fps and 20 s @ 60 fps (V3).
  - Is an MP4 with an audio track accepted (V4)?
  - Latency and cost (frames billed).
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("media", help="input video/image/audio file")
    parser.parse_args()
    raise NotImplementedError("Milestone M1: implement this spike")


if __name__ == "__main__":
    main()
