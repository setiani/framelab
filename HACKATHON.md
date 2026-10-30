# FrameLab: hackathon build log

Meta Global AI Developer Hackathon 2026 · Indie Developer track · Problem statement: *Push the Frontier*.
Building period: 27 Oct – 4 Nov 2026. **Everything in this repository was created during the
building period.** No pre-existing code was used.

## Log

| Date (WIB) | What was built |
|---|---|
| 2026-10-28 | Idea and technical proposal; implementation plan (`docs/plan/`); repository skeleton (M0): project config, Pydantic data contracts, run store, append-only event log, CLI `doctor`, `/api/health`, frontend skeleton, CI. Skeleton and plan drafted with AI assistance (Claude) and reviewed by the author. |
| 2026-10-28 (evening) | M1 API verification: spike scripts + synthetic test-media generator with ground truth; ran against the live Meta API (≈ $0.78). Findings: SAM frame indices 1:1 and sub-pixel accurate; SAM video reports objects only after they move; Spark does not hear video audio; Wonder unreliable on structured output (Spark auditor by default). Plan updated (02/04/05/06/11). |
| 2026-10-28 (night) | M2: ingest (probe, CFR normalisation, effective fps, audio, frame-accurate analysis clip, key frames), Meta client (retries, cost events, recording, strict schemas), SAM client (streaming, compact records + replay, box-prompt fallback), segmentation stage (alt concepts, motion-blob fallback, image-mode references, file deletion), observations (sub-pixel centroids, PCA axes, split-id merge), `framelab run --until measuring`, golden tests on recorded SAM responses. |
| 2026-10-29 | M3: track cleaning (area + Hampel outliers, gap interpolation), scale calibration (ruler PCA extent, blur-robust ball diameter with mask-dilation correction, method agreement), free-fall physics (automatic release/impact window, contact trimming, quadratic + cubic control fit, block bootstrap + analytic + window-variation σ with ≥90 % CI coverage on synthetic data), Claim Ledger evidence (`trk/cal/win/fit/ci/der/cmp`), `framelab run` reports g. On recorded SAM responses of synthetic drops (true g = 9.81): 9.83, 9.80, 9.83 m/s², all inside their 95 % CI. |
| 2026-10-29 (afternoon) | M4: pendulum (FFT + Hilbert initial guess, damped-sine least-squares fit with drift term, end trimming, block-bootstrap + analytic σ of T, finite-amplitude correction averaged over the decaying envelope, g = 4π²L/T0²), pivot by geometric circle fit with string cross-check, length spoken or measured (pivot radius × ruler scale), hypothesis test, series regression T0² vs L + power law + pairs bootstrap with `framelab series create/add/analyze`. Synthetic pendulum test video (exact nonlinear ODE, θ0 = 15°) with synthetic SAM records: g = 9.81 ± 0.09 m/s² (true 9.81), T0 within 0.1 %. |
| 2026-10-29 (evening) | M5 (code and offline tests; real-video acceptance pending): Muse Voice client (async, retries, cost events) and transcript stage; Planner on Muse Spark 1.3 (video `input_file` + transcript, strict structured output, post-validation, one retry, untraceable values dropped, `spk_` evidence); CLI flags as overrides and `--no-planner` for offline runs. Live check on the M1 spike clips (about $0.07): Voice wrote spoken numbers as words, so the traceability check reads words too; Spark's first answer failed validation and the retry fixed it. |
| 2026-10-29 (night) | M6 (free fall): claim ledger (values copied from evidence, G5 bare-number check with allow-list, EN/ID rendering), gates G1–G5 with EN/ID messages, 15 Analyst tools with schemas and 2 KB results, Analyst loop (Muse Spark chat with tools, budget signal, one retry, deterministic fallback). Live run on the M1 spike clip (≈ $0.09): the Analyst submitted a G5-clean report with 6 claims; g = 9.83 ± 0.03 m/s². Pendulum, Auditor and verdict follow in M6.1/M7. |
| 2026-10-29 (late) | M7: Auditor (Muse Spark, separate context, structured verdict with retry, deterministic pre- and post-checks, Wonder behind a flag), finalisation, revise loop (REJECT → analyst round 2), Designer (predictions from code, safety limit 2 m), second-language draft with placeholder check, slow-motion re-run helper. Live: PASS run (≈ $0.14), synthetic ruler off-plane trap gives NEEDS_RERECORD in EN and ID (≈ $0.07). |
| 2026-10-29 (night) | M8: charts (Plotly JSON, 1600×900 matplotlib PNG, series T₀² vs L), overlay video (mask contours, trail, HUD from fmt-formatted numbers, end card with the verdict), report HTML and A4 PDF in EN and ID (claims as phrases, appendices, data sample), recording-feedback page, `data.csv` in SI, and `framelab replay` offline. Live: replay on the PASS and trap runs, no model calls. |
| 2026-10-29 (night) | M9: HTTP API (uploads with validation and a size limit, run detail, ledger, track, artifact allow-list, rerun, series), SSE with heartbeats and Last-Event-ID replay, runs as background tasks limited by concurrency, error format per 07 §1. Live: curl upload → PASS run (≈ $0.14), 123 events streamed and closed on done. |
| 2026-10-30 | Audit of M5–M9 against the plan, and fixes: the pendulum now runs through the Analyst (damped-sine fit, period, amplitude-corrected period, g with a chosen length source), gates, Auditor, Designer, charts and overlay (pivot marker); the final verdict is stored before rendering and the Designer runs before the report, so live reports carry the verdict badge and the next experiment and re-recordings get the feedback page; no target / no scale / Planner stop end as NEEDS_RERECORD instead of FAILED; a student's fall-time prediction is tested (`fall_time`); a hypothesis claim must cite `compare_hypothesis` evidence; the Auditor reviews re-recording requests; run cost guard ($1.50); fallback capped at PASS_WITH_WARNINGS; `object_diameter_m` only from a spoken value; G1 coverage over the fit window plus `object_left_frame`, `camera_motion`, `air_drag_likely`; credits list Muse Wonder only when it ran. |

<!-- Add one row per day: milestone(s) finished, notable decisions, what changed. -->

## AI assistance

- Planning documents, the initial skeleton, M1–M4 and the M5–M9 audit fixes were written with an AI
  assistant (Claude), steered and reviewed by the author.
- M5–M9 were implemented with Claude Code (Claude Haiku), steered by the author, following
  `docs/plan/`.

## Third-party code, models and assets

| Item | Use | License |
|---|---|---|
| Meta Model API: SAM 3.1, Muse Spark 1.3, Muse Wonder, Muse Voice Transcribe (+ Muse Image, Muse Spark Open if stretch goals are done) | Core models | Meta terms of service |
| `meta_sam_parser` | Official SAM output parser | see package |
| Patterns adapted from `github.com/meta-models/meta-model-cookbook` (SAM streaming helper, voice file transcription, tool-calling loop, error handling) | Reference patterns, cited in code comments | see repository |
| FastAPI, Pydantic, OpenAI Python SDK, NumPy, SciPy, OpenCV, Pillow, Matplotlib, Jinja2, Playwright, Typer, Vite, TypeScript, Plotly.js | Libraries | MIT / BSD / Apache-2.0 |
| Sample videos in `samples/` | Recorded by the author | CC BY 4.0 |
