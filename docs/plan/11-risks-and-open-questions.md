# 11: Risks and open questions

## 1. VERIFY list (answered in M1 on 2026-10-28; evidence: `scripts/spikes/out/*.json`, run on Windows, Python 3.11)

Test media: synthetic 960×540 free-fall clips with burnt-in frame counter and per-frame ground truth
(`scripts/spikes/make_test_media.py`). Total spike cost ≈ $0.61.

| # | Question | Answer | Consequence for the build |
|---|---|---|---|
| V1 | SAM video `frame_index` 0-based and 1:1 with decoded frames? | **Yes.** Best offset vs ground truth = 0; frame steps all 1; `max(frame_index)+1 == ffprobe nb_read_frames` (300/300, 720/720, 1200/1200). `response.completed.usage.video_frames_processed` = all frames. | `t = (clip_offset_frame + frame_index) / effective_fps` is correct as specified. |
| V2 | Coordinates in uploaded pixel space? | **Yes.** Mean absolute cy error on moving frames 0.31–0.47 px, cx 0.36–0.41 px. Raw grammar carries the view size (`<|box;x1=..;y1=..;x2=..;y2=..;w=960;h=540|>`); box x2/y2 are inclusive, parser bounds are half-open (+1). | No rescaling needed. Use parser bounds only. |
| V3 | Max fps / frames / duration? | **No limit hit.** 240 fps × 3 s (720 frames) OK in 34 s; 60 fps × 20 s (1200 frames) OK in 61 s (~20 frames/s throughput). | Keep `max_analysis_s = 20`; budget ~1 s latency per 20 frames. |
| V4 | MP4 with audio track accepted by SAM? | **Yes.** Identical result to the silent clip. | One upload of `video_with_audio.mp4` can serve Planner + SAM; still upload the trimmed silent `analysis.mp4` for SAM to save frames. |
| V5 | Spark video input: which part type? audio read? | **All three work:** Responses `input_file`+file_id, Responses `input_video`+file_id, Chat `video_url` data URL. **Spark did NOT hear the embedded speech** (all three answered "silent" for a clip with clear speech). Frame sampling is sparse: it misread the first/last frame counters (F0000/F0033/F0003 and ~F0270 vs real F0299). | Planner MUST get the Voice transcript explicitly (already planned). Treat Spark's time window as coarse; code refines the window (04 §2.4, 05 §2.1). Use Responses `input_file`. |
| V6 | Responses `text.format` json_schema with Spark? | **Works with `strict: false`.** `strict: true` → HTTP 400 "'additionalProperties' is required to be supplied and to be false". | For strict mode generate strict-compatible schemas (closed objects, all keys required, optional = nullable) via a `strict_schema()` helper. Always validate with Pydantic `extra="forbid"`. |
| V7 | Wonder + structured output? | **Accepted but not enforced in non-strict mode.** Chat: valid keys plus extra keys and a wrong `target_concept`; Responses: extra keys and a wrong answer ("melting", "causal reasoning"). ~1.4–1.7k reasoning tokens (~$0.03/call). Strict mode: see V7b. | Auditor must use strict schemas + Pydantic `extra="forbid"` + one retry, and fall back to Spark if invalid (`FRAMELAB_USE_WONDER`). |
| V8 | `languageBias` for Indonesian? | **Both `["indonesian"]` and `["id"]` accepted (HTTP 200).** No clear accuracy difference on a Windows-TTS voice ("jatuh" heard as "jeda" in all variants, likely TTS pronunciation). Without bias numbers come back as digits ("0.45"); with `["indonesian"]` as words ("zero point four five"). ENDPOINTING returns turns with `startMs/endMs`; PUSH_TO_TALK returns no turns. | Use `ENDPOINTING`, **no** languageBias (digits are easier to parse). Re-test with a real human voice in M5. |
| V9 | Spark video token cost? | **Tiny:** 1,471 input tokens for a 5 s clip + prompt (~$0.007 per call incl. reasoning). | Video Planner cost is negligible; no need to downscale for cost. |
| V10 | Files retention / delete? | `files.retrieve` → `status: uploaded`, `expires_at: null` (no automatic expiry). `files.delete` → `deleted: true`. | Delete uploaded files at the end of every run (privacy); note in README. |
| V11 | Mask raster size vs bounds? | **Equal in 100% of masks** (268/268, 592/592, 1168/1168, images 2/2). Masks are ~1 px larger per side than the true edge: ball width 47–48 px vs 46 px true (+2–4%), ruler width 32 vs 30 px; ruler PCA long-axis length 460.0 vs 460 px (exact). | Long-axis (ruler) scale is reliable. For object-diameter scale subtract a dilation term δ≈1–2 px (calibrate δ on synthetic + benchmark), else G2 would flag ~3% disagreement. |
| V12 | **New:** SAM returned no masks for the first ~0.533 s of every clip although the ball is visible and static there. Time warm-up or motion onset? | **Motion onset.** Ball held still for 2 s: first mask at frame 122 (motion starts at 121). Clip starting mid-fall: first mask at frame 1. Once tracked, the object stays tracked while static (20 s clip: 97% coverage). | Static references (ruler, pivot, string) → image mode only. Free fall: first mask ≈ release + 1 frame (05 §2.1). Clip start padding 1.0 s is safe. S-1 melting (never moves) needs image-mode or box-prompt segmentation; verify before building S-1. |
| V7b | **New:** Wonder with strict-compatible schema + `strict: true`? | **Schema obeyed, content still wrong.** Chat strict: no extra keys, `target_concept: "Measurement"`. Responses strict: `experiment: "melting"`, essay as target. Spark strict on the same prompt: perfect (`pendulum`, `metal nut`, 50 cm). | Auditor defaults to Spark (separate context). Wonder re-evaluated once in M7 on realistic auditor input; enabled only if correct. |

## 2. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| SAM loses a fast/blurred ball | Medium | High | Slow-mo, bright light, colourful ball, plain background; alt concepts; box prompt; G1 catches the rest |
| SAM ignores static objects in video mode (V12) | Certain | Medium | Image-mode segmentation for references; S-1 needs a different approach (verify first) |
| Muse Wonder unreliable on structured output (V7b) | Observed | Medium | Spark auditor by default; M7 re-test before enabling Wonder; story = "independent auditor context" |
| Slow-mo files retimed to 30 fps | High | High | `capture_fps` upload field; G4 slow-mo signature + one-click re-run |
| Perspective (ruler off-plane) | Medium | Medium | Recording guide; dual scale + G2; it is also the demo's trap |
| Variable frame rate phone video | High | Medium | Normalize to CFR; warn; +0.5 % uncertainty |
| Planner extracts a wrong spoken number | Low | Medium | Must cite a transcript turn; spec shown in UI; user can re-run with a note |
| LLM writes numbers in the narrative | Medium | High | `add_claim` copies values; G5 regex; Auditor |
| Agent loops / burns budget | Low | Medium | Step budget, cost guard, deterministic fallback |
| API latency during judging | Medium | High | Demo replay mode; caching; clear progress UI |
| Credits run out | Low | High | Spend limit; per-IP limits; daily budget switch; fixtures for tests |
| Windows dev issues (ffmpeg, Playwright, paths) | Medium | Medium | `framelab doctor`; pathlib; list-arg subprocess; `scripts/dev.ps1` |
| Scope creep | High | High | Milestone order; stretch only after M11; daily check against 09 |
| Rule violation (pre-existing code, rights) | Low | Fatal | HACKATHON.md, daily commits, self-recorded samples, license list |
| Benchmark misses targets | Medium | Medium | Report honestly; explain error sources; that also shows scientific integrity |

## 3. Decisions log (append as you go)

| Date | Decision | Reason |
|---|---|---|
| 2026-10-28 | Python/FastAPI + vanilla TS; files instead of DB | Solo, 7 days, scientific stack |
| 2026-10-28 | Chat Completions for agent loops; Responses API for video/SAM/Image | Proven tool-calling pattern in the cookbook |
| 2026-10-28 | ~~Wonder as Auditor~~ → Spark Auditor with separate context; Wonder only if it passes an M7 re-test | M1 V7/V7b: Wonder gave wrong structured answers where Spark was correct |
| 2026-10-28 | Static objects segmented in image mode; clip start padding 1.0 s | M1 V12: SAM video reports objects only after they first move |
| 2026-10-28 | Subtract δ = 1.5 px mask dilation from object diameters | M1 V11: masks ~1 px larger per side |
| 2026-10-28 | Voice: ENDPOINTING, no languageBias | M1 V8: digits instead of words; turns with timings |
| 2026-10-28 | No GPU locally → open-weight bonus only via Baseten (S-3) | Hardware constraint |
| 2026-10-28 | matplotlib for PDF charts, Plotly for web | Avoid kaleido issues on Windows |
