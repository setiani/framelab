# AGENTS.md: instructions for coding agents working on FrameLab

> Copy this file to the repository root. The full plan lives in `docs/plan/`.
> Read `docs/plan/README.md` first, then the plan files referenced by the task you are on.

## What FrameLab is (one paragraph)

FrameLab turns a phone video of a simple physics experiment into verified measurements.
SAM 3.1 tracks the object in every frame. Deterministic Python converts masks into a
calibrated time series and fits physics models with uncertainty. A Muse Spark agent drives
the analysis through tools. A Muse Wonder auditor rejects any report whose claims are not
backed by evidence. The user gets an overlay video, charts, a bilingual (ID/EN) lab report
PDF, a Claim Ledger, and a proposal for the next experiment.

## Non-negotiable rules

1. **The LLM decides, code computes.** Language models choose actions, models and words.
   They never produce measured or derived numbers. All numbers come from Python tools and
   are stored in the ledger (`docs/plan/06-agents-and-verification.md` §1 and §4,
   `docs/plan/03-data-contracts.md` §6).
2. **Claims reference evidence.** Model-written text uses claim placeholders like `{{c03}}`.
   The report renderer substitutes values from the ledger. A bare digit sequence in
   model-written narrative fails gate G5 (except allow-listed tokens: see 06 §7).
3. **Never adjust data toward expected values.** Plausibility gates diagnose; they never
   modify, drop or hide data.
4. **Follow the data contracts exactly** (`docs/plan/03-data-contracts.md`). If you need to
   change a contract, update the plan file in the same commit.
5. **Use the Meta API patterns in `docs/plan/02-meta-api-reference.md`.** Do not invent
   endpoints or parameters. If something is marked `VERIFY`, check it with a spike script
   in `scripts/spikes/` and record the finding in `docs/plan/11-risks-and-open-questions.md`.
6. **No secrets in the repo.** Keys come from environment variables (`MODEL_API_KEY`,
   `BASETEN_API_KEY`). Never log full keys. `.env` is git-ignored.
7. **Tests must not call paid APIs by default.** Live tests are marked `@pytest.mark.live`
   and skipped unless `FRAMELAB_LIVE_TESTS=1`.
8. **Cross-platform.** The developer uses Windows. Use `pathlib`, pass `subprocess` args as
   lists (never `shell=True`), and do not rely on bash-only tooling in runtime code.
   Provide both `Makefile` targets and equivalent `scripts/*.ps1` or `python -m` commands.
9. **Hackathon rule: new work only.** All code is written during 27 Oct – 4 Nov 2026.
   Do not copy code from other projects. Patterns from the official Meta cookbooks are
   allowed. Cite them in a comment (`# Pattern adapted from meta-model-cookbook/07_sam/...`)
   and in `HACKATHON.md`.
10. **Scope discipline.** Build in the milestone order of `docs/plan/09-milestones-and-tasks.md`.
    Do not start stretch goals (melting ice, Muse Image, Baseten, canvas player) until the
    MVP is deployed.

## Tech stack (fixed, do not swap without updating the plan)

- Python 3.11+, `uv` for env/deps (pip fallback), FastAPI + Uvicorn, Pydantic v2
- `openai` SDK pointed at `https://api.meta.ai/v1`; `meta_sam_parser`; `httpx`
- numpy, scipy, opencv-python-headless, Pillow, Jinja2, Playwright (Chromium) for PDF
- ffmpeg/ffprobe binaries on PATH
- Frontend: Vite + TypeScript (no framework) + `plotly.js-dist-min`; built into `web/dist`
  and served by FastAPI
- pytest, ruff (lint + format), mypy (optional, non-blocking)

## Commands

```bash
uv sync                          # install backend deps
uv run framelab run samples/freefall_01.mp4    # run full pipeline from CLI (dev)
uv run uvicorn framelab.api.main:app --reload  # backend on :8000
cd web && npm install && npm run dev           # frontend on :5173 (proxy /api → :8000)
uv run pytest                    # offline tests
FRAMELAB_LIVE_TESTS=1 uv run pytest -m live    # live API tests (costs money)
uv run python eval/benchmark.py  # accuracy benchmark → eval/results.md
uv run ruff check . && uv run ruff format .
```

## Code conventions

- Package root: `backend/framelab/`. Import as `framelab.*`.
- Every pipeline stage is a pure-ish function `run_<stage>(ctx: RunContext) -> StageResult`
  that reads/writes files in the run folder and emits events through `ctx.emit(...)`.
- Every external call goes through `framelab.clients.*`. These modules are the only place
  that knows URLs, model IDs and request shapes. Each client logs a `model_call` event
  (model, latency, tokens, estimated cost, request id).
- Units: SI internally (metres, seconds, m/s²). Convert to cm only for display.
- Time: `t = (clip_offset_frame + frame_index) / effective_fps`. `effective_fps` is the
  real capture rate. It differs from the file fps for retimed slow-motion clips
  (`04-pipeline-and-measurement.md` §2.2).
- Coordinates: image pixels, origin top-left, +x right, +y **down**.
- Numbers in the UI and report: format with the uncertainty's significant figures
  (see `05-physics-and-uncertainty.md` §7).
- Logging: `structlog`-style JSON lines through the standard `logging` module. No print().
- Type hints everywhere; docstrings on public functions.

## Definition of done for any task

- Code + tests for the new behaviour, `uv run pytest` green, ruff clean.
- Acceptance criteria from `09-milestones-and-tasks.md` demonstrably met (paste the command
  output or the artifact path in the commit message or PR description).
- Plan files updated if a contract or decision changed.
- Small, descriptive commits (the judges will read the history).
