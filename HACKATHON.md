# FrameLab: hackathon build log

Meta Global AI Developer Hackathon 2026 · Indie Developer track · Problem statement: *Push the Frontier*.
Building period: 27 Oct – 4 Nov 2026. **Everything in this repository was created during the
building period.** No pre-existing code was used.

## Log

| Date (WIB) | What was built |
|---|---|
| 2026-10-28 | Idea and technical proposal; implementation plan (`docs/plan/`); repository skeleton (M0): project config, Pydantic data contracts, run store, append-only event log, CLI `doctor`, `/api/health`, frontend skeleton, CI. Skeleton and plan drafted with AI assistance (Claude) and reviewed by the author. |

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
