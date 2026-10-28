"""Verify Muse Spark 1.3 video input on the Responses API (V5, V9).

Milestone M1 (docs/plan/09-milestones-and-tasks.md). Use ONLY the request patterns in
docs/plan/02-meta-api-reference.md. Save raw responses under scripts/spikes/out/ (git-ignored)
and write the answers into docs/plan/11-risks-and-open-questions.md §1.

Checks:
  - input_file vs input_video with file_id: which works?
  - Does Spark read the embedded audio (ask it to quote the speech)?
  - usage tokens and estimated cost for a 15 s 960 px clip (V9).
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
