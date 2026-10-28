"""Verify SAM 3.1 image segmentation on a key frame (V2, V11).

Milestone M1 (docs/plan/09-milestones-and-tasks.md). Use ONLY the request patterns in
docs/plan/02-meta-api-reference.md. Save raw responses under scripts/spikes/out/ (git-ignored)
and write the answers into docs/plan/11-risks-and-open-questions.md §1.

Checks:
  - Segment 'ruler' on a key frame; print boxes and mask sizes.
  - Does the mask raster size equal the bounds size (V11)?
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
