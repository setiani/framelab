"""Masks → observations (M2) → cleaned tracks (M3).

Spec: docs/plan/04-pipeline-and-measurement.md §5. All coordinates are image pixels of the
normalized video (origin top-left, +y down). Times use ``effective_fps`` (real time).
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from meta_sam_parser import decode_mask_to_raster

from framelab.schemas import ExperimentSpec, Observation, Track, TrackPoint

MIN_MASK_PIXELS = 12
OBS_FIELDS = list(Observation.model_fields)


def mask_pixels(record: Any) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Image-space pixel centres of a mask record, plus the bounds→raster scale (sx, sy).

    The raster covers the mask bounds, not the frame (02 §5.2):
    x = left + (col + 0.5)·(right − left)/width, y = top + (row + 0.5)·(bottom − top)/height.
    """
    width, height = record.mask.width, record.mask.height
    raster = np.frombuffer(decode_mask_to_raster(record.mask), dtype=np.uint8).reshape(
        height, width
    )
    rows, cols = np.nonzero(raster)
    sx = (record.bounds.right - record.bounds.left) / width
    sy = (record.bounds.bottom - record.bounds.top) / height
    xs = record.bounds.left + (cols + 0.5) * sx
    ys = record.bounds.top + (rows + 0.5) * sy
    return xs, ys, sx, sy


def pca_axes(xs: np.ndarray, ys: np.ndarray) -> tuple[float, float, float]:
    """(major, minor, orientation_deg) from the pixel covariance (4·√λ ≈ full axis length)."""
    if xs.size < 3:
        return 0.0, 0.0, 0.0
    cov = np.cov(np.vstack([xs, ys]))
    evals, evecs = np.linalg.eigh(cov)
    evals = np.clip(evals, 0, None)
    major = 4.0 * math.sqrt(evals[1])
    minor = 4.0 * math.sqrt(evals[0])
    orientation = math.degrees(math.atan2(evecs[1, 1], evecs[0, 1]))
    return major, minor, orientation


def extent_along_major(xs: np.ndarray, ys: np.ndarray, pixel: float = 1.0) -> float:
    """Length of the pixel cloud along its principal axis (+1 pixel for the pixel width)."""
    pts = np.vstack([xs, ys])
    _, evecs = np.linalg.eigh(np.cov(pts))
    proj = evecs[:, 1] @ (pts - pts.mean(axis=1, keepdims=True))
    return float(proj.max() - proj.min() + pixel)


def mask_to_observation(
    record: Any, concept: str, frame_index: int, t: float
) -> Observation | None:
    """Sub-pixel centroid (first image moments), area, PCA axes. None if the mask is tiny."""
    xs, ys, sx, sy = mask_pixels(record)
    if xs.size < MIN_MASK_PIXELS:
        return None
    area = xs.size * sx * sy
    major, minor, orientation = pca_axes(xs, ys)
    return Observation(
        concept=concept,
        object_id=str(record.object_id),
        frame_index=frame_index,
        t=t,
        cx_px=float(xs.mean()),
        cy_px=float(ys.mean()),
        area_px=float(area),
        bbox_left=float(record.bounds.left),
        bbox_top=float(record.bounds.top),
        bbox_right=float(record.bounds.right),
        bbox_bottom=float(record.bounds.bottom),
        eq_diameter_px=float(2.0 * math.sqrt(area / math.pi)),
        major_axis_px=major,
        minor_axis_px=minor,
        orientation_deg=orientation,
        confidence=record.confidence,
        revision=int(getattr(record, "revision", 0)),
    )


def latest_masks(result: Any) -> list[Any]:
    """Keep the highest revision per (object_id, frame_index) (02 §5.2)."""
    best: dict[tuple[str, int | None], Any] = {}
    for rec in result.records:
        if rec.kind != "mask":
            continue
        frame = rec.frame.frame_index if rec.frame is not None else None
        key = (str(rec.object_id), frame)
        if key not in best or rec.revision > best[key].revision:
            best[key] = rec
    return sorted(
        best.values(), key=lambda r: (str(r.object_id), r.frame.frame_index if r.frame else -1)
    )


def observations_from_result(
    result: Any,
    concept: str,
    *,
    offset_frame: int,
    effective_fps: float,
    image_frame_index: int | None = None,
) -> list[Observation]:
    """Video results: global frame = offset + frame_index. Image results: ``image_frame_index``."""
    out: list[Observation] = []
    for rec in latest_masks(result):
        if rec.frame is not None:
            frame = offset_frame + rec.frame.frame_index
        else:
            frame = image_frame_index if image_frame_index is not None else 0
        obs = mask_to_observation(rec, concept, frame, frame / effective_fps)
        if obs is not None:
            out.append(obs)
    return out


def write_observations(path: Path, observations: list[Observation]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=OBS_FIELDS)
        writer.writeheader()
        for obs in observations:
            writer.writerow(obs.model_dump())


def read_observations(path: Path) -> list[Observation]:
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    out = []
    for row in rows:
        clean = {k: (None if v == "" else v) for k, v in row.items()}
        out.append(Observation.model_validate(clean))
    return out


def object_stats(
    observations: list[Observation], n_frames: int, motion_axis: str
) -> dict[str, dict[str, float]]:
    """Per object: coverage (fraction of clip frames), motion range (p95 − p5), mean area."""
    by_obj: dict[str, list[Observation]] = defaultdict(list)
    for o in observations:
        by_obj[o.object_id].append(o)
    stats = {}
    for obj, obs in by_obj.items():
        frames = {o.frame_index for o in obs}
        if motion_axis == "x":
            coord = np.array([o.cx_px for o in obs])
        elif motion_axis == "y":
            coord = np.array([o.cy_px for o in obs])
        else:
            coord = np.hypot([o.cx_px for o in obs], [o.cy_px for o in obs])
        rng = float(np.percentile(coord, 95) - np.percentile(coord, 5)) if coord.size else 0.0
        stats[obj] = {
            "coverage": len(frames) / max(n_frames, 1),
            "range_px": rng,
            "mean_area_px": float(np.mean([o.area_px for o in obs])),
            "first_frame": min(frames),
            "last_frame": max(frames),
            "n": len(frames),
        }
    return stats


def choose_target_track(
    observations: list[Observation], spec: ExperimentSpec, n_frames: int
) -> tuple[str | None, list[str]]:
    """Best object by coverage × motion range (04 §5.1); merge ids that look split.

    Returns (object_id, merged_ids). ``merged_ids`` lists extra ids folded into the winner
    (never visible in the same frame, each with coverage ≥ 0.4 of the remaining frames).
    """
    stats = object_stats(observations, n_frames, spec.motion_axis)
    if not stats:
        return None, []
    ranked = sorted(
        stats,
        key=lambda o: (stats[o]["coverage"] * stats[o]["range_px"], stats[o]["mean_area_px"]),
        reverse=True,
    )
    best = ranked[0]
    frames_by_obj: dict[str, set[int]] = defaultdict(set)
    for o in observations:
        frames_by_obj[o.object_id].add(o.frame_index)
    merged = [
        other
        for other in ranked[1:]
        if stats[other]["coverage"] >= 0.4 and not (frames_by_obj[other] & frames_by_obj[best])
    ]
    return best, merged


def debug_centroids_png(
    frame_png: Path, observations: list[Observation], out_png: Path, highlight: str | None = None
) -> Path:
    """Draw all centroids (and mask boxes of one object) on a frame for a visual sanity check."""
    import cv2

    img = cv2.imread(str(frame_png))
    if img is None:
        raise FileNotFoundError(frame_png)
    for o in observations:
        color = (255, 200, 0) if o.concept == highlight or highlight is None else (0, 200, 255)
        if "ruler" in o.concept or "tape" in o.concept or "paper" in o.concept:
            color = (0, 196, 255)
            cv2.rectangle(
                img,
                (int(o.bbox_left), int(o.bbox_top)),
                (int(o.bbox_right), int(o.bbox_bottom)),
                color,
                1,
            )
        cv2.circle(img, (int(round(o.cx_px)), int(round(o.cy_px))), 2, color, -1, cv2.LINE_AA)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png), img)
    return out_png


def hampel(
    series: np.ndarray, window: int = 7, k: float = 3.5, *, peaks_only: bool = False
) -> np.ndarray:
    """Boolean outlier mask: |v − rolling median| > k · 1.4826 · MAD (NaN-safe).

    The MAD is the smaller of the local and the global one (a spike next to a gap leaves few
    clean values in its window, which would otherwise inflate the local MAD), with a small
    floor so sub-pixel jitter (an object at rest) is never an outlier. ``peaks_only`` keeps a
    flag only where the deviation is a local maximum: one position spike shows up in three
    second differences (+s, −2s, +s) and only the centre one belongs to the bad point.
    """
    v = np.asarray(series, dtype=float)
    n = v.size
    half = window // 2
    out = np.zeros(n, dtype=bool)
    score = np.zeros(n)
    if n == 0 or np.all(np.isnan(v)):
        return out
    global_mad = float(np.nanmedian(np.abs(v - np.nanmedian(v))))
    floor = 0.15
    for i in range(n):
        if np.isnan(v[i]):
            continue
        seg = v[max(0, i - half) : i + half + 1]
        seg = seg[~np.isnan(seg)]
        if seg.size < 3:
            continue
        med = float(np.median(seg))
        local_mad = float(np.median(np.abs(seg - med)))
        mad = max(min(local_mad, global_mad), floor)
        score[i] = abs(v[i] - med) / (1.4826 * mad)
        out[i] = score[i] > k
    if peaks_only:
        left = np.concatenate([[0.0], score[:-1]])
        right = np.concatenate([score[1:], [0.0]])
        out &= (score >= left) & (score >= right)
    return out


def rolling_median(values: np.ndarray, window: int = 9) -> np.ndarray:
    half = window // 2
    return np.array(
        [np.median(values[max(0, i - half) : i + half + 1]) for i in range(values.size)]
    )


def clean_track(
    observations: list[Observation],
    *,
    effective_fps: float,
    concept: str,
    object_id: str,
    max_interp_gap: int = 2,
    id_switches: int = 0,
) -> Track:
    """Grid, area/position outliers, gap interpolation, metrics (04 §5.2).

    Interpolated points are flagged ``valid=False``: they are drawn in the overlay but never
    used in fits. The grid spans the first to the last observed frame of the object (SAM does
    not report static frames before the motion starts, M1 V12).
    """
    obs = sorted((o for o in observations if o.object_id == object_id), key=lambda o: o.frame_index)
    if not obs:
        raise ValueError(f"no observations for object {object_id}")
    first, last = obs[0].frame_index, obs[-1].frame_index
    by_frame = {o.frame_index: o for o in obs}
    frames = np.array(sorted(by_frame))
    x = np.array([by_frame[f].cx_px for f in frames])
    y = np.array([by_frame[f].cy_px for f in frames])
    area = np.array([by_frame[f].area_px for f in frames])

    outlier = np.zeros(frames.size, dtype=bool)
    if frames.size >= 5:
        med = rolling_median(area, 9)
        outlier |= np.abs(area / np.maximum(med, 1e-9) - 1.0) > 0.5
    if frames.size >= 7:
        consecutive = (frames[2:] - frames[:-2]) == 2
        for coord in (x, y):
            d2 = np.where(consecutive, coord[2:] - 2 * coord[1:-1] + coord[:-2], np.nan)
            flags = hampel(d2, window=7, k=3.5, peaks_only=True)
            outlier[1:-1] |= flags

    points: list[TrackPoint] = []
    n_interp = 0
    max_gap = 0
    for idx, f in enumerate(frames):
        if idx > 0:
            prev = frames[idx - 1]
            gap = int(f - prev - 1)
            if gap > 0:
                max_gap = max(max_gap, gap)
                for g in range(1, gap + 1):
                    missing_frame = prev + g
                    if gap <= max_interp_gap:
                        w = g / (gap + 1)
                        points.append(
                            TrackPoint(
                                t=missing_frame / effective_fps,
                                x_px=float(x[idx - 1] + w * (x[idx] - x[idx - 1])),
                                y_px=float(y[idx - 1] + w * (y[idx] - y[idx - 1])),
                                area_px=float(area[idx - 1] + w * (area[idx] - area[idx - 1])),
                                valid=False,
                                flag="interpolated",
                            )
                        )
                        n_interp += 1
                    else:
                        points.append(
                            TrackPoint(
                                t=missing_frame / effective_fps,
                                x_px=float("nan"),
                                y_px=float("nan"),
                                area_px=float("nan"),
                                valid=False,
                                flag="missing",
                            )
                        )
        points.append(
            TrackPoint(
                t=f / effective_fps,
                x_px=float(x[idx]),
                y_px=float(y[idx]),
                area_px=float(area[idx]),
                valid=not bool(outlier[idx]),
                flag="outlier" if outlier[idx] else "ok",
            )
        )
    grid_len = last - first + 1
    n_valid = sum(1 for p in points if p.valid)
    return Track(
        track_id=f"{_slug(concept)}_{object_id}",
        concept=concept,
        object_id=object_id,
        fps=effective_fps,
        points=points,
        coverage=n_valid / grid_len,
        n_outliers=int(outlier.sum()),
        n_interpolated=n_interp,
        max_gap_frames=max_gap,
        id_switches_suspected=id_switches,
    )


def _slug(text: str) -> str:
    return "_".join("".join(c if c.isalnum() else " " for c in text.lower()).split())


def write_track_csv(path: Path, track: Track) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["frame", "t_s", "x_px", "y_px", "area_px", "valid", "flag"])
        for p in track.points:
            writer.writerow(
                [
                    round(p.t * track.fps),
                    f"{p.t:.6f}",
                    p.x_px,
                    p.y_px,
                    p.area_px,
                    int(p.valid),
                    p.flag,
                ]
            )


def valid_arrays(track: Track) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(t, x_px, y_px) of the valid points only — the only data fits may use."""
    pts = [p for p in track.points if p.valid]
    return (
        np.array([p.t for p in pts]),
        np.array([p.x_px for p in pts]),
        np.array([p.y_px for p in pts]),
    )
