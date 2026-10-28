# 03: Data contracts

All models live in `backend/framelab/schemas.py` (Pydantic v2). Field names are final.
Add fields only with a default, and never rename. Units are SI unless the field name says
otherwise (`_px`, `_cm`, `_ms`). Times are seconds (`float`) unless suffixed `_ms`.

## 1. Common types

```python
from __future__ import annotations
from datetime import datetime
from enum import Enum
from typing import Literal, Any
from pydantic import BaseModel, Field

ExperimentType = Literal["free_fall", "pendulum", "melting", "unknown"]
Lang = Literal["id", "en"]


class Quantity(BaseModel):
    value: float
    unit: str  # "cm", "m", "s", "deg", "kg", ...
    source_turn: int | None = None  # index into Transcript.turns, if spoken

    def to_si(self) -> float: ...  # cm→m, deg→rad, ms→s, etc. (implement with a small table)


class BoxPx(BaseModel):
    left: float
    top: float
    right: float
    bottom: float
```

## 2. Run state

```python
class Stage(str, Enum):
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
    stage: Stage
    stage_history: list[tuple[Stage, datetime]] = []
    original_filename: str
    lang: Lang = "id"  # report language preference from upload form
    capture_fps: int | None = None  # upload form: real capture fps for slow-mo clips (04 §2.2)
    series_id: str | None = None
    experiment: ExperimentType | None = None
    verdict: VerdictStatus | None = None  # final (never "REJECT" at DONE; see 06 §8)
    headline: dict[str, Any] | None = (
        None  # {"quantity":"g","value":9.74,"ci95":[9.62,9.86],"unit":"m/s^2","claim_id":"c03"}
    )
    cost_usd: float = 0.0
    revise_rounds: int = 0
    warnings: list[str] = []
    error: str | None = None
    artifacts: dict[str, str] = {}  # logical name → path relative to run dir
    demo: bool = False
```

## 3. Ingest, transcript, spec

```python
class ProbeInfo(BaseModel):
    width: int
    height: int  # of ORIGINAL video (after rotation applied)
    rotation_deg: int = 0
    duration_s: float
    avg_fps: float  # avg_frame_rate
    r_fps: float  # r_frame_rate
    nb_frames: int | None
    is_vfr: bool  # |avg_fps - r_fps| / r_fps > 0.01 or ffprobe says VFR
    has_audio: bool
    codec: str


class IngestResult(BaseModel):
    probe: ProbeInfo
    norm_width: int
    norm_height: int  # after resize (long edge ≤ 960, even numbers)
    fps: float  # CONSTANT fps of normalized video file (= file_fps)
    fps_num: int
    fps_den: int
    file_fps: float  # same as fps; explicit name used in physics code
    effective_fps: float  # REAL-TIME frames per second (capture fps for retimed slow-mo)
    capture_fps_source: Literal["file", "user"] = "file"
    clip_offset_frame: int = 0  # global frame index of analysis.mp4 frame 0 (04 §2.4)
    clip_path: str | None = None  # ingest/analysis.mp4
    n_frames: int  # counted on normalized video (ffprobe -count_frames)
    video_path: str  # ingest/video.mp4
    video_with_audio_path: str | None
    audio_path: str | None  # ingest/audio.wav (None if no audio track)
    keyframes: dict[int, str] = {}  # frame_index → png path


class TranscriptTurn(BaseModel):
    index: int
    text: str
    start_ms: int | None = None
    end_ms: int | None = None
    speaker: str | None = None


class Transcript(BaseModel):
    text: str
    turns: list[TranscriptTurn]
    audio_duration_ms: int | None = None
    language_hint: str | None = None
    session_id: str | None = None


class ReferenceObject(BaseModel):
    concept: str  # SAM noun phrase, e.g. "ruler", "measuring tape", "A4 paper"
    length_m: float | None  # real length of the reference's long axis
    length_source: Literal["spoken", "visible_marking", "standard_object", "unknown"]
    frame_hint_s: float | None = None  # time where it is fully visible and unoccluded


class ExperimentSpec(BaseModel):
    experiment: ExperimentType
    target_concept: str  # SAM noun phrase for the moving object, e.g. "orange ball"
    target_alt_concepts: list[str] = []  # fallbacks, e.g. ["ball", "tennis ball"]
    target_hint_box: BoxPx | None = None  # optional box at analysis_window_s[0] (normalized px)
    reference: ReferenceObject | None
    object_diameter_m: float | None = None  # spoken/known diameter of the moving object
    spoken_params: dict[
        str, Quantity
    ] = {}  # keys: "drop_height", "string_length", "ball_diameter", "amplitude_deg", "mass"
    analysis_window_s: tuple[float, float]  # relevant segment of the video
    motion_axis: Literal["x", "y", "xy"]  # free_fall → "y", pendulum → "x"
    hypothesis: str | None = None  # verbatim quote from transcript
    hypothesis_prediction: Quantity | None = None  # e.g. period 1.5 s
    language: Lang = "id"
    setup_description: str  # one sentence, for the report
    confidence: float = Field(ge=0, le=1)
    warnings: list[str] = []  # e.g. "camera appears to move at 3 s"
```

## 4. Segmentation and measurement

```python
class SamRequestSummary(BaseModel):
    concept: str
    media: Literal["video", "image"]
    frame_index: int | None = None  # for image requests: which keyframe
    status: Literal["completed", "incomplete", "empty"]
    n_objects: int
    frames_with_mask: dict[str, int]  # object_id → count
    raw_log_path: str
    cost_usd: float


class Observation(BaseModel):  # one row per (object, frame) in observations.csv
    concept: str
    object_id: str
    frame_index: int
    t: float  # (clip_offset_frame + frame_index) / effective_fps
    cx_px: float
    cy_px: float  # sub-pixel centroid (image coords)
    area_px: float
    bbox_left: float
    bbox_top: float
    bbox_right: float
    bbox_bottom: float
    eq_diameter_px: float  # 2*sqrt(area/pi)
    major_axis_px: float
    minor_axis_px: float  # from PCA of mask pixels (4*sqrt(eigenvalue))
    orientation_deg: float  # major-axis angle
    confidence: float | None
    revision: int


class TrackPoint(BaseModel):
    t: float
    x_px: float
    y_px: float
    area_px: float
    valid: bool  # False = outlier/gap-filled
    flag: Literal["ok", "outlier", "interpolated", "missing"] = "ok"


class Track(BaseModel):
    track_id: str  # "<concept_slug>_<object_id>"
    concept: str
    object_id: str
    fps: float
    points: list[TrackPoint]
    coverage: float  # fraction of frames in window with a valid mask
    n_outliers: int
    n_interpolated: int
    max_gap_frames: int
    id_switches_suspected: int


class ScaleEstimate(BaseModel):
    method: Literal["ruler", "object_diameter", "spoken_length", "manual"]
    m_per_px: float
    sigma_m_per_px: float
    details: dict[str, Any] = {}  # e.g. {"length_px": 812.3, "length_m": 0.30}


class Calibration(BaseModel):
    primary: ScaleEstimate
    secondary: ScaleEstimate | None
    agreement_pct: float | None  # |p - s| / p * 100
    pivot_px: tuple[float, float] | None = None  # pendulum
    length_px_measured: float | None = None  # pendulum pivot→bob
```

## 5. Events (`events.jsonl`)

One JSON object per line. Required keys: `ts` (ISO-8601 UTC), `run_id`, `seq` (monotonic
int), `type`, `stage`. Other keys depend on `type`:

| type | extra keys |
|---|---|
| `stage_started` / `stage_completed` | `duration_s` (completed only) |
| `progress` | `message`, `pct` (0–100, optional) |
| `model_call` | `call_id`, `model`, `api` ("chat"\|"responses"\|"files"\|"asr"\|"images"), `purpose` ("planner"…), `input_summary` (sizes, no media) |
| `model_result` | `call_id`, `latency_s`, `usage` {input, cached, output, reasoning}, `units` {frames, images, audio_s}, `cost_usd`, `request_id`, `ok` |
| `reasoning_summary` | `call_id`, `text` (≤2,000 chars) |
| `tool_call` | `call_id`, `tool`, `args` |
| `tool_result` | `call_id`, `tool`, `evidence_ids`, `summary` (short human text), `ok`, `error` |
| `claim_added` | `claim` (Claim) |
| `gate_result` | `gate` (GateResult) |
| `verdict` | `verdict` (Verdict), `round` |
| `artifact` | `name`, `path`, `mime` |
| `warning` | `message` |
| `error` | `message`, `exception_type`, `retryable` |
| `done` | `verdict`, `headline` |

## 6. Claim Ledger (`analysis/ledger.jsonl`)

The ledger holds two record kinds in one file: **evidence** (produced only by tools) and
**claims** (created by the `add_claim` tool, whose value is copied from evidence by code).

```python
class Evidence(BaseModel):
    kind: Literal["evidence"] = "evidence"
    id: str  # prefix by producer: seg_, trk_, cal_, fit_, ci_, der_, gate_, fig_, cmp_
    tool: str
    created_at: datetime
    inputs: list[str] = []  # evidence ids this depends on
    data: dict[str, Any]  # tool-specific numbers (see 06 §4 per tool)
    summary: str  # short human-readable, generated by CODE (not LLM)


class Claim(BaseModel):
    kind: Literal["claim"] = "claim"
    id: str  # "c01", "c02", ...
    claim_type: Literal[
        "measurement", "derived", "comparison", "hypothesis_test", "quality", "interpretation"
    ]
    quantity: str | None  # "g", "period", "damping", "slope", ...
    value: float | None  # COPIED from evidence by code
    ci95: tuple[float, float] | None  # COPIED from evidence by code
    unit: str | None
    evidence: list[str]  # must be non-empty and exist
    statement_en: str  # template, may contain {value}, {ci}, {unit}
    statement_id: str  # Indonesian template
    created_at: datetime
```

Rendering: `render_claim(claim, lang)` produces e.g. `g = 9.74 ± 0.12 m/s²` using the
rounding rules in `05 §7`. Narrative text written by the model references claims as
`{{c03}}` only.

## 7. Analysis outputs

```python
class GateResult(BaseModel):
    gate: Literal["G1", "G2", "G3", "G4", "G5"]
    name: str
    status: Literal["pass", "fail", "warn", "skipped"]
    hard: bool  # hard gates block PASS
    metrics: dict[str, float | int | str]
    thresholds: dict[str, float | int | str]
    message_en: str
    message_id: str  # generated by code from templates
    remedy: Literal["reanalyze", "rerecord", "none"]
    evidence_id: str  # gate_xx evidence in the ledger


class ReportSection(BaseModel):
    heading: str
    body: str  # markdown-lite, claim placeholders only


class ReportDraft(BaseModel):  # written by the Analyst via submit_report tool
    lang: Lang
    title: str
    aim: str
    setup: str
    method: str
    results: list[ReportSection]
    hypothesis_evaluation: str | None
    error_analysis: str
    conclusion: str
    claims_used: list[str]  # ids referenced anywhere above


class Verdict(BaseModel):  # Auditor structured output
    status: Literal["PASS", "PASS_WITH_WARNINGS", "REJECT", "NEEDS_RERECORD"]
    unsupported_claims: list[str] = []  # claim ids or quoted spans
    overreaching_statements: list[str] = []
    gate_failures: list[str] = []  # gate ids
    fix_instructions: list[str] = []  # for the Analyst (REJECT)
    rerecord_tips_en: list[str] = []  # for the user (NEEDS_RERECORD)
    rerecord_tips_id: list[str] = []
    auditor_model: str
    notes: str = ""


class DesignerProposal(BaseModel):
    title: str
    changed_variable: str  # exactly one
    controlled_variables: list[str]
    new_values: list[Quantity]  # e.g. string lengths 0.25 m, 1.00 m
    prediction_en: str
    prediction_id: str  # may reference claims, numbers computed by tool `predict`
    protocol_steps_en: list[str]
    protocol_steps_id: list[str]
    safety_notes_en: list[str] = []
    safety_notes_id: list[str] = []
    predicted_values: list[dict[str, Any]] = []  # filled by code (e.g. T for each L), not LLM
```

## 8. Series

```python
class SeriesState(BaseModel):
    series_id: str  # "s_" + timestamp + hex
    created_at: datetime
    experiment: Literal["pendulum"]  # MVP: pendulum only
    run_ids: list[str]
    independent_variable: str  # "string_length"
    status: Literal["collecting", "analyzing", "done", "failed"]
    headline: dict[str, Any] | None = None
    artifacts: dict[str, str] = {}
```

## 9. Ground truth (`samples/ground_truth.json`)

```json
{
  "freefall_01.mp4": {"experiment": "free_fall", "g_ref": 9.78, "drop_height_m": 1.00,
                      "ball_diameter_m": 0.067, "ruler_length_m": 1.00, "fps_recorded": 240,
                      "expected_verdict": "PASS", "notes": "tennis ball, white wall"},
  "pendulum_050.mp4": {"experiment": "pendulum", "string_length_m": 0.50,
                       "period_stopwatch_s": 1.42, "period_stopwatch_n": 10,
                       "amplitude_deg_approx": 10, "expected_verdict": "PASS"},
  "trap_ruler_offplane.mp4": {"experiment": "free_fall", "expected_verdict": "NEEDS_RERECORD",
                              "expected_failed_gates": ["G2"]}
}
```

## 10. Settings (`config.py`)

```python
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FRAMELAB_", env_file=".env", extra="ignore")
    runs_dir: Path = Path("runs")
    demo: bool = False
    max_upload_mb: int = 200
    max_analysis_s: float = 20.0
    max_long_edge_px: int = 960
    concurrency: int = 2
    use_wonder: bool = False                # M1: Wonder unreliable on structured tasks (02 §7)
    mask_dilation_px: float = 1.5           # M1: SAM masks ~1 px larger per side (04 §6.2)
    designer_backend: Literal["meta", "baseten"] = "meta"
    analyst_max_steps: int = 12
    max_revise_rounds: int = 2
    bootstrap_n: int = 1000
    # gate thresholds: see 06 §7 (GateThresholds model, overridable via env JSON)
```
`MODEL_API_KEY` and `BASETEN_API_KEY` are read without prefix.
