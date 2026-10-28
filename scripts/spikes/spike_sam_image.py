"""Spike: SAM 3.1 image segmentation on a key frame (VERIFY V2, V11).

Checks
  V2   coordinates are in the image's pixel space   → ruler box / ball centre vs ground truth
  V11  mask raster size equals bounds size           → mask_size_equals_bounds
  also: ruler long-axis length (PCA) vs true 460 px → calibration precision preview

Usage
  uv run python scripts/spikes/spike_sam_image.py [--image keyframe.png] [--concept ruler ...]
Approx. cost: 2 images × $2.50/1000 = $0.005
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import numpy as np
from _common import (
    MEDIA,
    RAW,
    Report,
    describe_error,
    fail_safe_main,
    sam_cost,
    sam_segment_image,
    summarize_sam,
)


def ruler_length_px(result) -> float | None:
    """PCA major-axis extent of the largest mask (image coords)."""
    from meta_sam_parser import decode_mask_to_raster

    best = None
    for rec in result.records:
        if rec.kind != "mask":
            continue
        m = np.frombuffer(decode_mask_to_raster(rec.mask), dtype=np.uint8).reshape(
            rec.mask.height, rec.mask.width
        )
        rows, cols = np.nonzero(m)
        if rows.size < 20:
            continue
        sx = (rec.bounds.right - rec.bounds.left) / rec.mask.width
        sy = (rec.bounds.bottom - rec.bounds.top) / rec.mask.height
        xs = rec.bounds.left + (cols + 0.5) * sx
        ys = rec.bounds.top + (rows + 0.5) * sy
        pts = np.vstack([xs, ys])
        evals, evecs = np.linalg.eigh(np.cov(pts))
        proj = evecs[:, 1] @ (pts - pts.mean(axis=1, keepdims=True))
        length = float(proj.max() - proj.min() + max(sx, sy))
        if best is None or length > best:
            best = length
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=MEDIA / "keyframe_000.png")
    parser.add_argument("--concept", action="append", help="default: ruler, orange ball")
    args = parser.parse_args()
    concepts = args.concept or ["ruler", "orange ball"]

    report = Report("spike_sam_image")
    gt_path = MEDIA / "freefall_60fps_5s.gt.json"
    gt = json.loads(gt_path.read_text(encoding="utf-8")) if gt_path.is_file() else None
    if not args.image.is_file():
        raise SystemExit(f"{args.image} missing (run make_test_media.py first)")
    results = {}
    for concept in concepts:
        t0 = time.perf_counter()
        raw = RAW / f"sam_image_{concept.replace(' ', '_')}.jsonl"
        try:
            result = asyncio.run(sam_segment_image(args.image, concept, raw))
            summary = summarize_sam(result)
            entry = {"ok": True, "latency_s": round(time.perf_counter() - t0, 2), **summary}
            report.cost(sam_cost(images=1))
            if gt and concept == "ruler":
                left, top, right, bottom = gt["ruler_box"]
                entry["true_ruler_box"] = [left, top, right, bottom]
                entry["pca_length_px"] = ruler_length_px(result)
                entry["true_length_px"] = bottom - top
            if gt and "ball" in concept and summary["geometry"]:
                g = summary["geometry"][0]
                entry["centroid_error_px"] = [
                    round(g["cx"] - gt["frames"][0]["cx"], 3),
                    round(g["cy"] - gt["frames"][0]["cy"], 3),
                ]
            first = summary["geometry"][0] if summary["geometry"] else {}
            report.note(
                f"{concept}: {summary['outcome']} masks={summary['n_masks']} "
                f"mask==bounds {summary['mask_size_equals_bounds']} bounds={first.get('bounds')} "
                f"pca_len={entry.get('pca_length_px')}/{entry.get('true_length_px')} "
                f"centroid_err={entry.get('centroid_error_px')} latency={entry['latency_s']}s"
            )
        except Exception as exc:  # noqa: BLE001
            entry = {"ok": False, "error": describe_error(exc)}
            report.note(f"{concept}: ✗ {describe_error(exc)}")
        results[concept] = entry
    report.add("results", results)
    report.save()


if __name__ == "__main__":
    fail_safe_main(main)
