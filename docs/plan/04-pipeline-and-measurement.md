# 04: Pipeline and measurement (deterministic stages)

This file specifies every stage that is **not** an LLM agent: ingest, transcription
call, segmentation calls, mask→measurement, calibration, charts, overlay, report rendering,
and the orchestrator that ties them together. Agents are in `06`, physics in `05`.

## 1. Orchestrator (`pipeline/run.py`)

```python
async def execute_run(run_id: str, settings: Settings) -> RunState:
    ctx = RunContext.load(run_id, settings)  # paths, state, emit(), clients, ledger
    try:
        ing = await stage(ctx, Stage.INGESTING, ingest.run)  # §2
        up_task = asyncio.create_task(sam.upload(ctx, ing))  # Files API (video_with_audio)
        tr = await stage(ctx, Stage.TRANSCRIBING, transcribe.run)  # §3 (skips if no audio)
        file_id = await up_task
        spec = await stage(ctx, Stage.PLANNING, planner.run, file_id, tr)  # 06 §2
        clip = await ingest.make_analysis_clip(ctx, spec)  # §2.4
        seg = await stage(ctx, Stage.SEGMENTING, segment.run, spec, clip)  # §4
        meas = await stage(ctx, Stage.MEASURING, measure.run, spec, seg)  # §5, §6
        for round_ in range(settings.max_revise_rounds + 1):
            draft = await stage(ctx, Stage.ANALYZING, analyst.run, spec, meas, round_)  # 06 §3
            verdict = await stage(ctx, Stage.AUDITING, auditor.run, draft, round_)  # 06 §5
            if verdict.status != "REJECT":
                break
            ctx.state.revise_rounds = round_ + 1
        final = finalize_verdict(verdict, ctx.ledger.gates())  # 06 §8
        await stage(ctx, Stage.RENDERING, render_outputs, spec, meas, draft, final)  # §7–§9
        await stage(ctx, Stage.DESIGNING, designer.run, spec, final)  # 06 §6
        ctx.finish(final)
    except Exception as exc:  # noqa: BLE001
        ctx.fail(exc)  # writes error.json, emits error
    return ctx.state
```

Rules:
- `stage()` sets `state.stage`, emits `stage_started` and `stage_completed`, writes
  `state.json` atomically (write to tmp + `os.replace`), and catches nothing (errors bubble
  up to `ctx.fail`).
- If `NEEDS_RERECORD` is decided before analysis (e.g. SAM finds no target even after
  fallbacks), skip to RENDERING with a minimal "rerecord" report. Do not crash.
- Designer runs only when the final verdict is PASS or PASS_WITH_WARNINGS.
- `--resume` (CLI) skips stages whose output files already exist and are valid.

## 2. Ingest (`pipeline/ingest.py`)

### 2.1 Probe
```bash
ffprobe -v error -print_format json -show_streams -show_format input/original.mp4
```
Extract: width, height, rotation (side_data `rotation` or tag `rotate`), duration,
`avg_frame_rate`, `r_frame_rate`, `nb_frames`, codec, audio presence. Set `is_vfr` if
`avg` and `r` differ by more than 1%. Reject non-video or duration > 120 s (HTTP 422 at
upload) and duration < 1 s.

### 2.2 Effective (real-time) fps: the slow-motion trap

Many phones save slow-motion clips **retimed** to 30 fps playback. The file says 30 fps,
but each frame is 1/240 s of real time. Using the file fps would make g wrong by a factor
of (240/30)² = 64.

- The upload form has an optional field `capture_fps` (`auto` | 30 | 60 | 120 | 240).
- `effective_fps = capture_fps if given else file_fps`.
- **All physics time uses `t = frame_index / effective_fps`.**
- Gate G4 also checks the slow-mo signature (06 §7): if `g_measured * k²` is within 5 % of
  9.78 for k ∈ {2, 4, 8}, the diagnosis says "this looks like a slow-motion clip saved at
  playback speed; re-upload with capture fps = {30·k}".

Add to `IngestResult`: `file_fps: float`, `effective_fps: float`, `capture_fps_source:
Literal["file","user"]`. These fields extend the 03 §3 model.

### 2.3 Normalize
Target fps = `round(file_fps)` if the file is CFR, else the nearest of {24, 25, 30, 50, 60,
120, 240} to `avg_fps`. Long edge ≤ `max_long_edge_px` (960), even dimensions. Rotation is
applied automatically by ffmpeg.

```bash
# video only (for SAM and overlay)
ffmpeg -y -i input/original.mp4 -an \
  -vf "fps=${FPS},scale='if(gt(iw,ih),min(960,iw),-2)':'if(gt(iw,ih),-2,min(960,ih))':flags=lanczos" \
  -c:v libx264 -preset veryfast -crf 18 -pix_fmt yuv420p -g ${FPS} ingest/video.mp4

# with audio (for the Spark Planner, which also reads audio)
ffmpeg -y -i ingest/video.mp4 -i input/original.mp4 -map 0:v -map 1:a? -c:v copy -c:a aac -b:a 96k \
  -shortest ingest/video_with_audio.mp4

# audio for Voice Transcribe
ffmpeg -y -i input/original.mp4 -vn -ac 1 -ar 16000 -sample_fmt s16 ingest/audio.wav

# exact frame count of the normalized video
ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of csv=p=0 ingest/video.mp4
```
Run via `subprocess.run([...], check=True, capture_output=True)` in `asyncio.to_thread`.
On Windows, the `scale` expression is passed as one list element (no shell quoting needed).

### 2.4 Analysis clip (for SAM)
SAM is billed per frame and may have frame limits. Send only the analysis window:
`[t0 − 0.3 s, t1 + 0.3 s]` in **file** time, clamped to the video and to
`max_analysis_s`. Make it frame-accurate:

```bash
ffmpeg -y -i ingest/video.mp4 \
  -vf "select='between(n\,${F0}\,${F1})',setpts=N/(${FPS}*TB)" -an \
  -c:v libx264 -preset veryfast -crf 18 -pix_fmt yuv420p ingest/analysis.mp4
```
Store `clip_offset_frame = F0`. Every SAM `frame_index` k maps to global frame `F0 + k`.
Verify with `nb_read_frames == F1 − F0 + 1` and log a warning if not.

### 2.5 Key frames
Extract PNGs for image segmentation and the report:
```bash
ffmpeg -y -i ingest/video.mp4 -vf "select='eq(n\,${K})'" -vframes 1 ingest/keyframes/frame_${K:06d}.png
```
Key frames: the reference frame (`spec.reference.frame_hint_s`, else the analysis start),
the analysis start, the middle and the end. For the Planner fallback (02 §6.3), take 8
evenly spaced frames.

## 3. Transcription (`pipeline/transcribe.py`)

- Skip if no audio. Write an empty `Transcript` with a `warning` event.
- Call the Voice client with mode `ENDPOINTING`, keyword list from 02 §8, no language bias.
- Map the response to `Transcript`: `turns[i] = {index:i, text, start_ms, end_ms}`.
- If the call fails, the run continues without a transcript (Planner relies on video
  audio), and a warning is added.

## 4. Segmentation (`pipeline/segment.py`)

Inputs: `ExperimentSpec`, analysis clip `file_id` (upload `ingest/analysis.mp4`), key frames.

Requests:
1. **Target (video):** `segment_video(clip_file_id, spec.target_concept)`.
2. If the result is empty or the best track coverage < 0.5, try each of
   `spec.target_alt_concepts` (max 2 more requests). Then try the box-prompt fallback
   (02 §5.3) if `spec.target_hint_box` exists or a motion blob is found (§4.1).
3. **Reference (image):** `segment_image(ref_keyframe, spec.reference.concept)` when a
   reference exists. Pick the mask with the largest `major_axis_px`.
4. **Pendulum string (image, optional):** `segment_image(mid_keyframe, "string")`. Its
   topmost mask point is a pivot estimate used to cross-check the circle-fit pivot.

Store the raw events (`sam/<slug>.raw.jsonl`) and the parsed summary. Return the
`SamRequestSummary[]` + parsed results in memory for measurement.

### 4.1 Motion-blob fallback (no LLM)
Compute the absolute difference between frames k and k+3 over the first 30 frames of the
clip (grayscale, Gaussian blur 5, threshold 25). Take the largest connected component that
appears in ≥ 5 diffs. Its bounding box on the first frame becomes the SAM box prompt.

## 5. Mask → observations (`pipeline/measure.py`)

For every mask record (keep the max `revision` per `(object_id, frame_index)`):

```python
raw = decode_mask_to_raster(rec.mask)
m = np.frombuffer(raw, dtype=np.uint8).reshape(rec.mask.height, rec.mask.width)
rows, cols = np.nonzero(m)
if rows.size < 12:                      # too small to trust
    skip (record as missing)
sx = (rec.bounds.right - rec.bounds.left) / rec.mask.width
sy = (rec.bounds.bottom - rec.bounds.top) / rec.mask.height
xs = rec.bounds.left + (cols + 0.5) * sx
ys = rec.bounds.top  + (rows + 0.5) * sy
cx, cy = xs.mean(), ys.mean()           # sub-pixel centroid (first image moments)
area = rows.size * sx * sy
cov = np.cov(np.vstack([xs, ys]))       # 2x2
evals, evecs = np.linalg.eigh(cov)      # ascending
major = 4 * np.sqrt(evals[1]); minor = 4 * np.sqrt(evals[0])   # ≈ full axis lengths for an ellipse
orientation = degrees(atan2(evecs[1,1], evecs[0,1]))
```

Write `observations.csv` with one row per observation (03 §4), `t = (F0 + k) / effective_fps`.

### 5.1 Choose the target track
For each `object_id` of the target concept: `coverage` = fraction of clip frames with a
mask, `range` = (p95 − p5) of the coordinate on `spec.motion_axis`. Score =
`coverage × range`, ties broken by mean area. The best object becomes the primary track.
If two objects have coverage ≥ 0.4 and are never visible in the same frame, the tracker
probably split one object into two IDs. Merge them (ordered by frame), increment
`id_switches_suspected`, and emit a warning.

### 5.2 Clean the track
1. Order by frame. Build a full frame grid over the window. Missing frames get `flag="missing"`.
2. **Area outliers:** `|area / rolling_median(area, 9) − 1| > 0.5` → `outlier` (blur,
   occlusion, merge with hand).
3. **Position outliers (Hampel):** on x and y separately, rolling median window 7,
   MAD-scaled σ; `|v − med| > 3.5·σ` → `outlier`. Use a window over **second differences**
   for free fall (the trajectory is curved, so raw Hampel would flag legit points). Implement
   `hampel(series, window, k)` and apply it to `np.diff(y, 2)`. Map flags back to the centre
   index.
4. **Gaps:** gaps ≤ 2 frames are linearly interpolated with `flag="interpolated"`,
   `valid=False` (shown in the overlay, **excluded from fits**). Longer gaps stay `missing`.
5. Metrics → `Track.coverage`, `n_outliers`, `n_interpolated`, `max_gap_frames`.
6. Write `tracks/<track_id>.csv` and a `trk_xx` evidence record (06 §4).

## 6. Calibration (`pipeline/calibrate.py`)

All methods return `ScaleEstimate(m_per_px, sigma_m_per_px, details)`.

### 6.1 Ruler / reference object
- Pixels of the reference mask (image coords as in §5) → PCA → project onto the major axis
  → `length_px = max(proj) − min(proj) + 1·s` (s = pixel scale of the mask, usually 1).
- `m_per_px = length_m / length_px`.
- σ: edge uncertainty ±1 px at each end → `σ_len_px = √2`; length uncertainty `σ_len_m =
  0.001 m` (spoken/marked) or `0.0005 m` (standard object). Then
  `σ_s / s = √((σ_len_px/length_px)² + (σ_len_m/length_m)²)`.
- Sanity: reference long axis must be ≥ 15 % of the frame's long edge. Otherwise warn
  (low precision).

### 6.2 Object diameter
- Use frames inside the window with a valid mask. For a sphere, motion blur stretches the
  mask along the motion direction, so use the axis **perpendicular** to the motion:
  `width_px = minor_axis_px` when `minor/major < 0.85`, else `eq_diameter_px`.
- `d_px = median(width_px)`, `σ_d_px = 1.4826·MAD / √n + 1.0` (edge term).
- `m_per_px = object_diameter_m / d_px`.

### 6.3 Pendulum pivot and length
- Algebraic circle fit (Kåsa) on valid `(cx, cy)` points: solve
  `x² + y² + D x + E y + F = 0` by least squares, centre = (−D/2, −E/2), `r_px = √(cx²+cy²−F)`.
- Angular span of the points about the centre must be ≥ 8°. Otherwise the pivot is
  unreliable and `length_px_measured = None` with a warning.
- Cross-check against the "string" mask's topmost point (if segmented). The distance
  between the two pivot estimates should be < 0.1·r_px. Otherwise warn.
- If `spec.spoken_params["string_length"]` exists: `ScaleEstimate(method="spoken_length",
  m_per_px = L_m / r_px)`. Note: spoken L is pivot→bob centre. Tell users this in the
  recording guide.

### 6.4 Choosing primary/secondary
Priority: `ruler` > `object_diameter` > `spoken_length`. Primary = first available,
secondary = next available. `agreement_pct = |p − s| / p · 100`. That value feeds gate G2.
If no method is available: verdict NEEDS_RERECORD ("no scale reference: put a ruler in
the plane of motion or say the ball diameter").

## 7. Charts (`pipeline/charts.py`)

For each analysis produce:
- **Plotly JSON** (`outputs/chart_<name>.json`) for the web: data points (valid = filled,
  invalid = hollow grey), fitted curve, residuals subplot (shared x), hover with frame index.
- **PNG** (`outputs/chart_<name>.png`, 1600×900, 200 dpi) via **matplotlib** for the PDF.
  Do not depend on kaleido.

Charts by experiment:
- free fall: `y(t)` in metres (downward positive) + fit; residuals in mm.
- pendulum: `x(t)` in metres + damped sine fit; residuals in mm.
- series: `T₀²` vs `L` with error bars + regression line + 95 % band.

## 8. Overlay video (`pipeline/overlay.py`)

- Read `ingest/video.mp4` with OpenCV frame by frame. For frames in the window:
  - Target mask contour (cv2.findContours on the local raster, offset by bounds, scaled by
    sx/sy) in **#00E0FF**, 2 px. Reference mask contour in **#FFC400** on the reference frame
    and faintly (alpha 0.3) throughout.
  - Trail: last 30 valid centroids as fading dots. Outliers are drawn as red ×.
  - HUD (top-left, semi-transparent black box): `t = 0.412 s`, `y = 0.83 m`, and after
    analysis the headline `g = 9.74 ± 0.12 m/s²` (rendered from the ledger claim).
  - Pendulum: pivot marker and the line pivot→bob.
- Last 1.5 s: freeze on the final frame with a result card (headline, verdict badge,
  "Built with Meta SAM 3.1 · Muse Spark · Muse Wonder").
- Encode by piping raw BGR frames to ffmpeg:
```bash
ffmpeg -y -f rawvideo -pix_fmt bgr24 -s ${W}x${H} -r ${FILE_FPS} -i - \
  -c:v libx264 -preset veryfast -crf 23 -pix_fmt yuv420p -movflags +faststart outputs/overlay.mp4
```
(`FILE_FPS` keeps slow-mo playing slow, which looks good in the demo.)

## 9. Report (`pipeline/report.py`)

- Template `templates/report.html.j2`, A4, self-contained CSS. Sections: header (title,
  date, run id, verdict badge), aim, setup (+ key-frame image, optional S-2 diagram),
  method (models used and their roles, gates), results (tables + chart PNGs),
  hypothesis evaluation, error analysis, conclusion, next experiment, **appendix A: Claim
  Ledger** (claims with evidence ids), **appendix B: gate table**, **appendix C: data
  sample** (every k-th row so it fits on one page).
- Narrative fields come from `ReportDraft` (placeholders). `narrative.render(text, ledger,
  lang)` replaces `{{cNN}}` with formatted claim strings. An unknown id raises an error
  (gate G5 should have caught it).
- Two PDFs: `report_id.pdf` and `report_en.pdf`. The Analyst writes the draft in the run's
  `lang`. The other language comes from a **second Spark call that translates the draft
  JSON** while preserving placeholders. Verify that the placeholder sets are identical
  before rendering; otherwise fall back to a single-language report.
- Rendering: Playwright Chromium, `page.set_content(html, wait_until="load")` with images
  as `file://` URIs or data URIs, `page.pdf(format="A4", print_background=True,
  margin=…)`. Windows setup: `uv run playwright install chromium`.
- NEEDS_RERECORD runs get a one-page "Recording feedback" PDF: what failed (gate
  messages), annotated key frame (e.g. ruler and ball scales drawn), and the tips list.

## 10. Data export
`outputs/data.csv`: `frame,t_s,x_m,y_m,area_m2,valid,flag` (positions relative to the
first valid point, SI units, y downward positive). Header comment lines start with `#` and
contain the run id, the scale and the fps.
