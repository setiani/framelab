# 11: Risks and open questions

## 1. VERIFY list (answer in milestone M1, then keep this table updated)

| # | Question | Why it matters | How to test | Fallback if "no" | Answer (date, evidence) |
|---|---|---|---|---|---|
| V1 | Is SAM video `frame_index` 0-based and 1:1 with the decoded frames of the uploaded file (no sampling)? | Time axis correctness | Upload a CFR clip with a known frame count. Compare `max(frame_index)+1` and check a frame with a burnt-in frame number | Map with `frame_index × (n_frames / n_returned)` only if sampling is uniform and documented; otherwise upload a pre-sampled clip | |
| V2 | Are SAM box/mask coordinates in the pixel space of the uploaded video? | Calibration and overlay | Compare a mask on an image with a known object position | Scale by `uploaded_size / reported_size` | |
| V3 | Max fps / frames / duration per SAM video request? | Slow-mo 240 fps | 3 s @ 240 fps, 20 s @ 60 fps | Downsample to 120 fps; shorten the window | |
| V4 | Does SAM accept an MP4 with an audio track? | Reuse one file_id | Upload `video_with_audio.mp4` | Upload the silent `analysis.mp4` (already planned) | |
| V5 | Spark video input: `input_file` vs `input_video` with `file_id` on the Responses API? Is audio read? | Planner | `spike_spark_video.py` | 8 key frames as images + transcript | |
| V6 | Responses API `text.format` json_schema honoured by Spark? | Planner structured output | `spike_structured_output.py` | Prompt-only JSON + Pydantic validation + 1 retry | |
| V7 | Wonder on Chat Completions with `response_format`? | Auditor | spike | Responses API, or Spark fallback (`FRAMELAB_USE_WONDER=0`) | |
| V8 | `languageBias` value for Indonesian | Voice accuracy | spike with `["indonesian"]` vs none | Omit (auto-detect) | |
| V9 | Token usage / cost of a 15 s 960 px video for Spark | Budget | read `usage` | Lower resolution (640 px) for the Planner upload only | |
| V10 | Files API retention and delete endpoint | Privacy hygiene | `files.delete` | Document retention in README | |
| V11 | `meta_sam_parser` mask raster size vs bounds | Centroid mapping | spike image | General mapping formula already handles it | |

## 2. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| SAM loses a fast/blurred ball | Medium | High | Slow-mo, bright light, colourful ball, plain background; alt concepts; box prompt; G1 catches the rest |
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
| 2026-10-28 | Wonder as Auditor (fallback Spark) | Independent reviewer; strong reasoning |
| 2026-10-28 | No GPU locally → open-weight bonus only via Baseten (S-3) | Hardware constraint |
| 2026-10-28 | matplotlib for PDF charts, Plotly for web | Avoid kaleido issues on Windows |
