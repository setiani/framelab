from __future__ import annotations

import json
import math

import numpy as np
import pytest
from meta_sam_parser import (
    FrameReference,
    SegmentationMask,
    SegmentationMaskBounds,
    SegmentationMaskIdentity,
    SegmentationMaskRecord,
)
from meta_sam_parser._mask_codec import _encode_raster

from framelab.clients import sam as sam_client
from framelab.pipeline import measure
from framelab.schemas import ExperimentSpec
from tests.fakes import MEDIA, SAM_FIXTURES


def disk_record(cx: float, cy: float, r: float, frame: int | None = 0, scale: float = 1.0, obj="0"):
    """Mask record of a disk; raster at 1/scale resolution of its bounds (tests bounds mapping)."""
    left, top = math.floor(cx - r - 1), math.floor(cy - r - 1)
    right, bottom = math.ceil(cx + r + 1), math.ceil(cy + r + 1)
    w, h = int((right - left) / scale), int((bottom - top) / scale)
    sx, sy = (right - left) / w, (bottom - top) / h
    yy, xx = np.mgrid[0:h, 0:w]
    raster = (left + (xx + 0.5) * sx - cx) ** 2 + (top + (yy + 0.5) * sy - cy) ** 2 <= r * r
    payload = _encode_raster(bytes(raster.astype(np.uint8).ravel()), w, h)
    return SegmentationMaskRecord(
        order=0,
        object_id=obj,
        identity=SegmentationMaskIdentity(media="video", frame_index=frame, object_id=obj),
        revision=1,
        mask=SegmentationMask(encoding="one_bit", payload=payload, width=w, height=h),
        bounds=SegmentationMaskBounds(left, top, right, bottom),
        frame=FrameReference(frame_index=frame) if frame is not None else None,
    )


@pytest.mark.parametrize("scale", [1.0, 2.0])
def test_centroid_and_area_subpixel(scale: float) -> None:
    obs = measure.mask_to_observation(disk_record(100.3, 50.7, 20, scale=scale), "ball", 0, 0.0)
    assert obs is not None
    tol = 0.1 if scale == 1.0 else 0.2  # digitisation error of a pixel disk
    assert abs(obs.cx_px - 100.3) < tol and abs(obs.cy_px - 50.7) < tol
    assert obs.area_px == pytest.approx(math.pi * 400, rel=0.03 * scale)
    assert obs.eq_diameter_px == pytest.approx(40, rel=0.02 * scale)
    assert obs.major_axis_px == pytest.approx(40, rel=0.05) and obs.minor_axis_px == pytest.approx(
        40, rel=0.05
    )


def test_tiny_mask_is_ignored() -> None:
    assert measure.mask_to_observation(disk_record(10, 10, 1.2), "x", 0, 0.0) is None


def test_extent_along_major_rotated_bar() -> None:
    angle = math.radians(30)
    t = np.linspace(-100, 100, 2001)
    s = np.linspace(-5, 5, 21)
    tt, ss = np.meshgrid(t, s)
    xs = (tt * math.cos(angle) - ss * math.sin(angle)).ravel()
    ys = (tt * math.sin(angle) + ss * math.cos(angle)).ravel()
    assert measure.extent_along_major(xs, ys, pixel=0.1) == pytest.approx(200.1, abs=0.2)


class FakeResult:
    def __init__(self, records):
        self.records = records


def test_latest_revision_wins_and_offsets() -> None:
    old = disk_record(10, 10, 5, frame=3)
    new = disk_record(20, 10, 5, frame=3)
    new = SegmentationMaskRecord(
        **{
            **{
                f: getattr(new, f)
                for f in ("order", "object_id", "identity", "mask", "bounds", "frame", "confidence")
            },
            "revision": 2,
        }
    )
    obs = measure.observations_from_result(
        FakeResult([old, new]), "b", offset_frame=100, effective_fps=50
    )
    assert len(obs) == 1 and obs[0].frame_index == 103 and obs[0].t == pytest.approx(103 / 50)
    assert obs[0].cx_px == pytest.approx(20, abs=0.1)


def spec(axis="y") -> ExperimentSpec:
    return ExperimentSpec(experiment="free_fall", target_concept="ball", analysis_window_s=(0, 5),
                          motion_axis=axis, confidence=1)  # fmt: skip


def test_choose_target_prefers_moving_and_merges_split_ids() -> None:
    recs = [disk_record(100, 50 + 5 * f, 8, frame=f, obj="0") for f in range(0, 20)]
    recs += [disk_record(100, 150 + 5 * f, 8, frame=f, obj="2") for f in range(20, 40)]
    recs += [disk_record(400, 300, 8, frame=f, obj="1") for f in range(0, 40)]  # static object
    obs = measure.observations_from_result(FakeResult(recs), "b", offset_frame=0, effective_fps=60)
    best, merged = measure.choose_target_track(obs, spec(), 40)
    assert best in {"0", "2"} and merged == [{"0": "2", "2": "0"}[best]]


async def test_recorded_sam_matches_ground_truth() -> None:
    """Real SAM output (M1) vs synthetic ground truth: sub-pixel accuracy, 1:1 frames."""
    out = await sam_client.load_record(SAM_FIXTURES / "freefall_60fps_5s_orange_ball.sam.json.gz")
    obs = measure.observations_from_result(
        out.result, "orange ball", offset_frame=0, effective_fps=60
    )
    gt = json.loads((MEDIA / "freefall_60fps_5s.gt.json").read_text())
    truth = {f["frame"]: f["cy"] for f in gt["frames"]}
    errors = [abs(o.cy_px - truth[o.frame_index]) for o in obs]
    assert len(obs) == 268 and np.mean(errors) < 1.0
    assert all(abs(o.cx_px - 470) < 1.5 for o in obs)


def test_hampel_flags_spike_keeps_parabola() -> None:
    t = np.arange(40) / 60
    y = 80 + 0.5 * 4500 * t**2
    d2 = np.diff(y, 2)
    assert not measure.hampel(d2).any()
    y_spike = y.copy()
    y_spike[20] += 6
    flags = measure.hampel(np.diff(y_spike, 2))
    assert flags[19]  # d2 index 19 is centred on point 20


def _obs_track(frames, y_of, area=1700.0):
    out = []
    for f in frames:
        rec = disk_record(470, y_of(f), 23, frame=f)
        o = measure.mask_to_observation(rec, "ball", f, f / 60)
        out.append(o.model_copy(update={"area_px": area}))
    return out


def test_clean_track_flags_outliers_and_interpolates_short_gaps() -> None:
    frames = [f for f in range(32, 60) if f not in (40, 41, 50, 51, 52)]
    obs = _obs_track(frames, lambda f: 85 + 0.5 * 4512 * ((f - 31) / 60) ** 2)
    bad = obs[10].model_copy(update={"cy_px": obs[10].cy_px + 8})
    obs[10] = bad
    tr = measure.clean_track(obs, effective_fps=60, concept="orange ball", object_id="0")
    flags = {round(p.t * 60): p.flag for p in tr.points}
    assert flags[40] == flags[41] == "interpolated"
    assert flags[50] == flags[51] == flags[52] == "missing"  # gap of 3 > 2
    assert flags[bad.frame_index] == "outlier"
    assert tr.max_gap_frames == 3 and tr.n_interpolated == 2
    t, _x, y = measure.valid_arrays(tr)
    assert len(t) == len(frames) - 1


def test_clean_track_area_outlier() -> None:
    obs = _obs_track(range(32, 52), lambda f: 100 + 3 * (f - 32))
    obs[8] = obs[8].model_copy(update={"area_px": 4000.0})  # merged with a hand
    tr = measure.clean_track(obs, effective_fps=60, concept="ball", object_id="0")
    assert [p.flag for p in tr.points].count("outlier") == 1
