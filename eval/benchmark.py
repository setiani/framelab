"""Accuracy benchmark on self-recorded videos vs samples/ground_truth.json.

Milestone M12 (docs/plan/08-testing-and-evaluation.md §5). Writes eval/results.md and
eval/results.json with only computed numbers.

Usage: uv run python eval/benchmark.py [--only freefall|pendulum|traps] [--use-cache]
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=["freefall", "pendulum", "traps"])
    parser.add_argument("--use-cache", action="store_true")
    parser.parse_args()
    raise NotImplementedError("Milestone M12: implement the benchmark")


if __name__ == "__main__":
    main()
