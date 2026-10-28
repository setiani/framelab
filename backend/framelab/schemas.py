"""All FrameLab data contracts (Pydantic v2).

Spec: docs/plan/03-data-contracts.md. Field names are final: add fields only with a default,
never rename. Units are SI unless the field name says otherwise (``_px``, ``_cm``, ``_ms``).
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# --------------------------------------------------------------------------------------------
# 1. Common types
# --------------------------------------------------------------------------------------------

ExperimentType = Literal["free_fall", "pendulum", "melting", "unknown"]
Lang = Literal["id", "en"]

_UNIT_TO_SI: dict[str, tuple[float, str]] = {
    "m": (1.0, "m"),
    "cm": (0.01, "m"),
    "mm": (0.001, "m"),
    "km": (1000.0, "m"),
    "s": (1.0, "s"),
    "ms": (0.001, "s"),
    "min": (60.0, "s"),
    "kg": (1.0, "kg"),
    "g": (0.001, "kg"),
    "rad": (1.0, "rad"),
    "deg": (math.pi / 180.0, "rad"),
    "m/s^2": (1.0, "m/s^2"),
    "hz": (1.0, "1/s"),
}


def utcnow() -> datetime:
    """Timezone-aware UTC now (all timestamps in FrameLab are UTC)."""
    return datetime.now(UTC)


class Quantity(BaseModel):
    """A value with a unit, optionally traced to a transcript turn."""

    value: float
    unit: str  # "cm", "m", "s", "deg", "kg", ...
    source_turn: int | None = None  # index into Transcript.turns, if spoken

    def to_si(self) -> float:
        """Return the value converted to SI (m, s, kg, rad). Raises ValueError for unknown units."""
        key = self.unit.strip().lower()
        if key not in _UNIT_TO_SI:
            raise ValueError(f"Unknown unit {self.unit!r}")
        factor, _ = _UNIT_TO_SI[key]
        return self.value * factor

    def si_unit(self) -> str:
        """The SI unit symbol ``to_si`` converts to."""
        return _UNIT_TO_SI[self.unit.strip().lower()][1]


class BoxPx(BaseModel):
    left: float
    top: float
    right: float
    bottom: float


# --------------------------------------------------------------------------------------------
# 2. Run state
# --------------------------------------------------------------------------------------------


class Stage(StrEnum):
    CREATED = "created"
    INGESTING = "ingesting"
    TRANSCRIBING = "transcribing"
    PLANNING = "planning"
    SEGMENTING = "segmenting"
    MEASURING = "measuring"
    ANALYZING = "analyzing"
    AUDITING = "auditing"
    RENDERING = "rendering"
    DESIGNING = "designing"
    DONE = "done"
    FAILED = "failed"


VerdictStatus = Literal["PASS", "PASS_WITH_WARNINGS", "NEEDS_RERECORD", "REJECT"]


class RunState(BaseModel):
    run_id: str
    created_at: datetime
    updated_at: datetime
    stage: Stage = Stage.CREATED
    stage_history: list[tuple[Stage, datetime]] = Field(default_factory=list)
    original_filename: str
    lang: Lang = "id"
    capture_fps: int | None = None  # upload form: real capture fps for slow-mo clips (04 §2.2)
    note: str | None = None  # optional user note from the upload form (≤300 chars)
    series_id: str | None = None
    experiment: ExperimentType | None = None
    verdict: VerdictStatus | None = None  # final; never "REJECT" at DONE (06 §8)
    headline: dict[str, Any] | None = None
    cost_usd: float = 0.0
    revise_rounds: int = 0
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None
    artifacts: dict[str, str] = Field(default_factory=dict)  # logical name → path rel. to run dir
    demo: bool = False


# --------------------------------------------------------------------------------------------
# 3. Ingest, transcript, spec
# --------------------------------------------------------------------------------------------


class ProbeInfo(BaseModel):
    width: int
    height: int  # of ORIGINAL video (after rotation applied)
    rotation_deg: int = 0
    duration_s: float
    avg_fps: float
    r_fps: float
    nb_frames: int | None = None
    is_vfr: bool
    has_audio: bool
    codec: str


class IngestResult(BaseModel):
    probe: ProbeInfo
    norm_width: int
    norm_height: int
    fps: float  # CONSTANT fps of the normalized video file (= file_fps)
    fps_num: int
    fps_den: int
    file_fps: float
    effective_fps: float  # REAL-TIME frames per second (capture fps for retimed slow-mo)
    capture_fps_source: Literal["file", "user"] = "file"
    n_frames: int
    video_path: str  # ingest/video.mp4
    video_with_audio_path: str | None = None
    audio_path: str | None = None  # ingest/audio.wav (None if no audio track)
    clip_offset_frame: int = 0  # global frame index of analysis.mp4 frame 0 (04 §2.4)
    clip_path: str | None = None  # ingest/analysis.mp4
    keyframes: dict[int, str] = Field(default_factory=dict)  # frame_index → png path


class TranscriptTurn(BaseModel):
    index: int
    text: str
    start_ms: int | None = None
    end_ms: int | None = None
    speaker: str | None = None


class Transcript(BaseModel):
    text: str = ""
    turns: list[TranscriptTurn] = Field(default_factory=list)
    audio_duration_ms: int | None = None
    language_hint: str | None = None
    session_id: str | None = None


class ReferenceObject(BaseModel):
    concept: str  # SAM noun phrase, e.g. "ruler", "measuring tape", "a4 paper"
    length_m: float | None = None
    length_source: Literal["spoken", "visible_marking", "standard_object", "unknown"] = "unknown"
    frame_hint_s: float | None = None


SpokenParamKey = Literal["drop_height", "string_length", "ball_diameter", "amplitude_deg", "mass"]


class ExperimentSpec(BaseModel):
    experiment: ExperimentType
    target_concept: str
    target_alt_concepts: list[str] = Field(default_factory=list)
    target_hint_box: BoxPx | None = None
    reference: ReferenceObject | None = None
    object_diameter_m: float | None = None
    spoken_params: dict[str, Quantity] = Field(default_factory=dict)
    analysis_window_s: tuple[float, float]
    motion_axis: Literal["x", "y", "xy"]
    hypothesis: str | None = None
    hypothesis_prediction: Quantity | None = None
    language: Lang = "id"
    setup_description: str = ""
    confidence: float = Field(ge=0, le=1)
    warnings: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------------------------
# 4. Segmentation and measurement
# --------------------------------------------------------------------------------------------


class SamRequestSummary(BaseModel):
    concept: str
    media: Literal["video", "image"]
    frame_index: int | None = None
    status: Literal["completed", "incomplete", "empty"]
    n_objects: int
    frames_with_mask: dict[str, int] = Field(default_factory=dict)  # object_id → count
    raw_log_path: str
    cost_usd: float = 0.0


class Observation(BaseModel):
    """One row per (object, frame) in observations.csv."""

    concept: str
    object_id: str
    frame_index: int
    t: float  # (clip_offset_frame + frame_index) / effective_fps
    cx_px: float
    cy_px: float
    area_px: float
    bbox_left: float
    bbox_top: float
    bbox_right: float
    bbox_bottom: float
    eq_diameter_px: float
    major_axis_px: float
    minor_axis_px: float
    orientation_deg: float
    confidence: float | None = None
    revision: int = 0


class TrackPoint(BaseModel):
    t: float
    x_px: float
    y_px: float
    area_px: float
    valid: bool
    flag: Literal["ok", "outlier", "interpolated", "missing"] = "ok"


class Track(BaseModel):
    track_id: str  # "<concept_slug>_<object_id>"
    concept: str
    object_id: str
    fps: float
    points: list[TrackPoint]
    coverage: float
    n_outliers: int = 0
    n_interpolated: int = 0
    max_gap_frames: int = 0
    id_switches_suspected: int = 0


class ScaleEstimate(BaseModel):
    method: Literal["ruler", "object_diameter", "spoken_length", "manual"]
    m_per_px: float
    sigma_m_per_px: float
    details: dict[str, Any] = Field(default_factory=dict)


class Calibration(BaseModel):
    primary: ScaleEstimate
    secondary: ScaleEstimate | None = None
    agreement_pct: float | None = None
    pivot_px: tuple[float, float] | None = None
    length_px_measured: float | None = None


# --------------------------------------------------------------------------------------------
# 5. Events (events.jsonl): see events.py for the writer. Types listed for validation.
# --------------------------------------------------------------------------------------------

EventType = Literal[
    "stage_started",
    "stage_completed",
    "progress",
    "model_call",
    "model_result",
    "reasoning_summary",
    "tool_call",
    "tool_result",
    "claim_added",
    "gate_result",
    "verdict",
    "artifact",
    "warning",
    "error",
    "done",
]


class Event(BaseModel):
    """One line of events.jsonl. Type-specific keys are flat extra fields (03 §5)."""

    model_config = ConfigDict(extra="allow")

    ts: datetime
    run_id: str
    seq: int
    type: EventType
    stage: str


# --------------------------------------------------------------------------------------------
# 6. Claim Ledger (analysis/ledger.jsonl)
# --------------------------------------------------------------------------------------------


class Evidence(BaseModel):
    kind: Literal["evidence"] = "evidence"
    id: str  # prefix by producer: seg_, trk_, cal_, win_, fit_, ci_, der_, gate_, fig_, cmp_, spk_
    tool: str
    created_at: datetime = Field(default_factory=utcnow)
    inputs: list[str] = Field(default_factory=list)
    data: dict[str, Any] = Field(default_factory=dict)
    summary: str  # short human-readable, generated by CODE (not LLM)


ClaimType = Literal[
    "measurement", "derived", "comparison", "hypothesis_test", "quality", "interpretation"
]


class Claim(BaseModel):
    kind: Literal["claim"] = "claim"
    id: str  # "c01", "c02", ...
    claim_type: ClaimType
    quantity: str | None = None
    value: float | None = None  # COPIED from evidence by code
    ci95: tuple[float, float] | None = None  # COPIED from evidence by code
    unit: str | None = None
    evidence: list[str] = Field(min_length=1)
    statement_en: str
    statement_id: str
    created_at: datetime = Field(default_factory=utcnow)


# --------------------------------------------------------------------------------------------
# 7. Analysis outputs
# --------------------------------------------------------------------------------------------


class GateResult(BaseModel):
    gate: Literal["G1", "G2", "G3", "G4", "G5"]
    name: str
    status: Literal["pass", "fail", "warn", "skipped"]
    hard: bool
    metrics: dict[str, float | int | str] = Field(default_factory=dict)
    thresholds: dict[str, float | int | str] = Field(default_factory=dict)
    message_en: str = ""
    message_id: str = ""
    remedy: Literal["reanalyze", "rerecord", "none"] = "none"
    diagnosis: str | None = None  # e.g. "ruler_offplane", "slowmo_retimed(8)"
    evidence_id: str | None = None


class ReportSection(BaseModel):
    heading: str
    body: str  # markdown-lite, claim placeholders only


class ReportDraft(BaseModel):
    lang: Lang
    title: str
    aim: str
    setup: str
    method: str
    results: list[ReportSection]
    hypothesis_evaluation: str | None = None
    error_analysis: str
    conclusion: str
    claims_used: list[str] = Field(default_factory=list)


class Verdict(BaseModel):
    status: Literal["PASS", "PASS_WITH_WARNINGS", "REJECT", "NEEDS_RERECORD"]
    unsupported_claims: list[str] = Field(default_factory=list)
    overreaching_statements: list[str] = Field(default_factory=list)
    gate_failures: list[str] = Field(default_factory=list)
    fix_instructions: list[str] = Field(default_factory=list)
    rerecord_tips_en: list[str] = Field(default_factory=list)
    rerecord_tips_id: list[str] = Field(default_factory=list)
    auditor_model: str = ""
    notes: str = ""


class DesignerProposal(BaseModel):
    title: str
    changed_variable: str
    controlled_variables: list[str] = Field(default_factory=list)
    new_values: list[Quantity] = Field(default_factory=list)
    prediction_en: str = ""
    prediction_id: str = ""
    protocol_steps_en: list[str] = Field(default_factory=list)
    protocol_steps_id: list[str] = Field(default_factory=list)
    safety_notes_en: list[str] = Field(default_factory=list)
    safety_notes_id: list[str] = Field(default_factory=list)
    predicted_values: list[dict[str, Any]] = Field(default_factory=list)  # filled by code


# --------------------------------------------------------------------------------------------
# 8. Series
# --------------------------------------------------------------------------------------------


class SeriesState(BaseModel):
    series_id: str
    created_at: datetime
    experiment: Literal["pendulum"] = "pendulum"
    run_ids: list[str] = Field(default_factory=list)
    independent_variable: str = "string_length"
    status: Literal["collecting", "analyzing", "done", "failed"] = "collecting"
    headline: dict[str, Any] | None = None
    artifacts: dict[str, str] = Field(default_factory=dict)


# --------------------------------------------------------------------------------------------
# 9. Ground truth (samples/ground_truth.json values)
# --------------------------------------------------------------------------------------------


class GroundTruth(BaseModel):
    experiment: ExperimentType
    expected_verdict: VerdictStatus = "PASS"
    g_ref: float | None = None
    drop_height_m: float | None = None
    ball_diameter_m: float | None = None
    ruler_length_m: float | None = None
    fps_recorded: int | None = None
    string_length_m: float | None = None
    period_stopwatch_s: float | None = None
    period_stopwatch_n: int | None = None
    amplitude_deg_approx: float | None = None
    expected_failed_gates: list[str] = Field(default_factory=list)
    notes: str = ""
