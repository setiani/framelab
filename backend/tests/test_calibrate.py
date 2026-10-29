from __future__ import annotations

import math

import numpy as np
import pytest
from meta_sam_parser import (
    SegmentationMask,
    SegmentationMaskBounds,
    SegmentationMaskIdentity,
    SegmentationMaskRecord,
)
from meta_sam_parser._mask_codec import _encode_raster

from framelab.pipeline import calibrate
from framelab.schemas import Observation, ScaleEstimate


def raster_record(raster: np.ndarray, left: int, top: int) -> SegmentationMaskRecord:
    h, w = raster.shape
    payload = _encode_raster(bytes(raster.astype(np.uint8).ravel()), w, h)
    return SegmentationMaskRecord(
        order=0,
        object_id="0",
        identity=SegmentationMaskIdentity(media="image", frame_index=None, object_id="0"),
        revision=1,
        mask=SegmentationMask(encoding="one_bit", payload=payload, width=w, height=h),
        bounds=SegmentationMaskBounds(left, top, left + w, top + h),
    )


@pytest.mark.parametrize("angle_deg", [0, 25, 90])
def test_ruler_length_any_orientation(angle_deg: float) -> None:
    length, width = 400.0, 24.0
    a = math.radians(angle_deg)
    size = 520
    yy, xx = np.mgrid[0:size, 0:size] + 0.5
    cx = cy = size / 2
    u = (xx - cx) * math.cos(a) + (yy - cy) * math.sin(a)
    v = -(xx - cx) * math.sin(a) + (yy - cy) * math.cos(a)
    raster = (np.abs(u) <= length / 2) & (np.abs(v) <= width / 2)
    est = calibrate.ruler_scale(raster_record(raster, 0, 0), 0.40, frame_long_edge_px=960)
    assert est.details["length_px"] == pytest.approx(length, abs=1.5)
    assert est.m_per_px == pytest.approx(0.001, rel=0.004)
    assert est.method == "ruler" and "warning" not in est.details


def obs(minor: float, major: float, eq: float) -> Observation:
    return Observation(
        concept="ball", object_id="0", frame_index=0, t=0, cx_px=0, cy_px=0, area_px=1,
        bbox_left=0, bbox_top=0, bbox_right=1, bbox_bottom=1, eq_diameter_px=eq,
        major_axis_px=major, minor_axis_px=minor, orientation_deg=0,
    )  # fmt: skip


def test_diameter_uses_minor_axis_for_blurred_frames_and_subtracts_dilation() -> None:
    sharp = [obs(47.5, 47.5, 47.5)] * 20
    blurred = [obs(47.5, 70.0, 57.0)] * 10  # stretched along the motion
    d, sigma, n = calibrate.diameter_px(sharp + blurred, dilation_px=1.5)
    assert d == pytest.approx(46.0) and n == 30 and sigma >= 1.0
    est = calibrate.diameter_scale(sharp, 0.046, dilation_px=1.5)
    assert est.m_per_px == pytest.approx(0.001)


def test_fit_circle() -> None:
    rng = np.random.default_rng(3)
    th = np.radians(np.linspace(-15, 15, 80))
    x = 480 + 300 * np.sin(th) + rng.normal(0, 0.3, th.size)
    y = 40 + 300 * np.cos(th) + rng.normal(0, 0.3, th.size)
    cx, cy, r = calibrate.fit_circle(x, y)
    assert (
        cx == pytest.approx(480, abs=3)
        and cy == pytest.approx(40, abs=6)
        and r == pytest.approx(300, abs=6)
    )


def test_combine_priority_and_agreement() -> None:
    d = ScaleEstimate(method="object_diameter", m_per_px=0.00101, sigma_m_per_px=1e-5)
    r = ScaleEstimate(method="ruler", m_per_px=0.001, sigma_m_per_px=1e-6)
    cal = calibrate.combine([d, r])
    assert cal.primary.method == "ruler" and cal.secondary.method == "object_diameter"
    assert cal.agreement_pct == pytest.approx(1.0)
