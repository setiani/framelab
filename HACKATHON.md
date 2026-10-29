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

<!-- Add one row per day: milestone(s) finished, notable decisions, what changed. -->

## AI assistance

- Planning documents and the initial skeleton were drafted with an AI assistant (Claude).
- Implementation is done with coding agents (e.g. Muse Code) steered by the author, following `docs/plan/`.

## Third-party code, models and assets

| Item | Use | License |
|---|---|---|
| Meta Model API: SAM 3.1, Muse Spark 1.3, Muse Wonder, Muse Voice Transcribe (+ Muse Image, Muse Spark Open if stretch goals are done) | Core models | Meta terms of service |
| `meta_sam_parser` | Official SAM output parser | see package |
| Patterns adapted from `github.com/meta-models/meta-model-cookbook` (SAM streaming helper, voice file transcription, tool-calling loop, error handling) | Reference patterns, cited in code comments | see repository |
| FastAPI, Pydantic, OpenAI Python SDK, NumPy, SciPy, OpenCV, Pillow, Matplotlib, Jinja2, Playwright, Typer, Vite, TypeScript, Plotly.js | Libraries | MIT / BSD / Apache-2.0 |
| Sample videos in `samples/` | Recorded by the author | CC BY 4.0 |
