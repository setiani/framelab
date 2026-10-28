# 01: Architecture

## 1. System diagram

```mermaid
flowchart LR
    U[User: phone MP4 + spoken hypothesis] --> ING[Stage 1 · Ingest<br/>ffprobe/ffmpeg<br/>CFR video, WAV, keyframes]
    ING --> ASR[Stage 2 · Transcribe<br/>Muse Voice Transcribe]
    ING --> UP[Files API upload<br/>file_id]
    UP --> PLAN[Stage 3 · Planner<br/>Muse Spark 1.3<br/>video + transcript → ExperimentSpec]
    ASR --> PLAN
    PLAN --> SEG[Stage 4 · Segment & Track<br/>SAM 3.1 video + image]
    UP --> SEG
    SEG --> MEAS[Stage 5 · Measure<br/>Python: centroids, areas,<br/>calibration, cleaning]
    MEAS --> ANA[Stage 6 · Analyst agent<br/>Muse Spark 1.3 tool loop]
    ANA <--> TOOLS[(Deterministic tools<br/>fit · bootstrap · derive ·<br/>gates · plot · claims)]
    TOOLS --> LEDGER[(Claim Ledger<br/>ledger.jsonl)]
    ANA --> AUD[Stage 7 · Auditor<br/>Muse Wonder]
    LEDGER --> AUD
    AUD -- REJECT, max 2x --> ANA
    AUD -- PASS / NEEDS_RERECORD --> OUT[Stage 8 · Outputs<br/>overlay MP4 · charts · CSV ·<br/>report PDF ID/EN]
    OUT --> DES[Stage 9 · Designer<br/>Muse Spark 1.3<br/>next experiment]
    ANA -. every call .-> LOG[(events.jsonl audit log)]
    AUD -. .-> LOG
    SEG -. .-> LOG
```

Legend: Meta models = Muse Voice Transcribe, Muse Spark 1.3, SAM 3.1, Muse Wonder (and,
stretch, Muse Image + Muse Spark Open). Everything else is deterministic code.

## 2. Components

| Component | Module | Responsibility | Talks to |
|---|---|---|---|
| CLI | `framelab/cli.py` | `framelab run <video>`, `framelab series <run_ids…>`, `framelab replay <run_dir>` | Orchestrator |
| HTTP API | `framelab/api/main.py` | Upload, run state, SSE events, artifacts, series, demo mode | Orchestrator, RunStore |
| Orchestrator | `framelab/pipeline/run.py` | Executes stages in order, handles retries/revise loop, writes `state.json` | All stages |
| RunStore | `framelab/store.py` | Create/load run folders, atomic JSON writes, artifact paths | Filesystem |
| Event bus | `framelab/events.py` | `emit()` → append `events.jsonl` + push to SSE subscribers | API, all stages |
| Meta client | `framelab/clients/meta.py` | OpenAI SDK wrapper: chat, responses, files; retries; cost accounting | Meta Model API |
| SAM client | `framelab/clients/sam.py` | Upload video, segment video/image, parse with `meta_sam_parser` | Meta Model API |
| Voice client | `framelab/clients/voice.py` | `POST /v1/asr/transcribe` | Meta Model API |
| Image client (S-2) | `framelab/clients/image.py` | Muse Image edit from key frame | Meta Model API |
| Baseten client (S-3) | `framelab/clients/baseten.py` | Muse Spark Open chat completions | Baseten |
| Ingest | `framelab/pipeline/ingest.py` | Probe, normalize video, extract audio and key frames | ffmpeg |
| Measure | `framelab/pipeline/measure.py` | Masks → per-frame observations → cleaned tracks | numpy |
| Calibrate | `framelab/pipeline/calibrate.py` | Scale (m/px) from ruler, object diameter, spoken length; pendulum pivot | numpy |
| Physics | `framelab/physics/*` | Models, fitting, event detection, bootstrap, derivations, experiment registry | scipy |
| Agents | `framelab/agent/*` | Planner, Analyst loop, Auditor, Designer, tools, gates, ledger, prompts | Meta client |
| Overlay | `framelab/pipeline/overlay.py` | Render annotated MP4 | OpenCV, ffmpeg |
| Report | `framelab/pipeline/report.py` | Jinja2 HTML → PDF (Playwright), ID/EN | Playwright |
| Benchmark | `eval/benchmark.py` | Run all samples, compare with ground truth, write `eval/results.md` | Orchestrator |
| Web UI | `web/` | Upload, progress timeline, player, charts, agent trace, ledger, report download | HTTP API |

## 3. Run lifecycle (state machine)

```
CREATED → INGESTING → TRANSCRIBING → PLANNING → SEGMENTING → MEASURING
        → ANALYZING ⇄ AUDITING (≤2 revise rounds) → RENDERING → DESIGNING
        → DONE (verdict = PASS | PASS_WITH_WARNINGS | NEEDS_RERECORD)
any stage → FAILED (error.json written, event `error` emitted)
```

- Stages 2 (transcribe) and the Files upload run **concurrently** after ingest.
- `NEEDS_RERECORD` is a *successful* run outcome (the system worked; the data is unusable).
  The UI shows the diagnosis and recording tips instead of results.
- `PASS_WITH_WARNINGS` is when all hard gates pass but soft warnings exist (e.g. a VFR
  source, or a single scale method available).
- Each stage is idempotent given its inputs. If its output file exists and
  `--resume` is set, it is skipped (useful for dev and for replay).

## 4. Concurrency model

- One process, asyncio. Runs execute as background tasks. `asyncio.Semaphore(2)` limits
  concurrent runs (API rate limits, CPU for overlay rendering).
- CPU-heavy steps (mask decoding, bootstrap, overlay) run in `asyncio.to_thread`.
- SSE subscribers get events from an in-memory `asyncio.Queue` per run, fed by
  `ctx.emit`. Late subscribers first receive a replay of `events.jsonl`.

## 5. Run folder layout

```
runs/<run_id>/                       # run_id = "r_" + YYYYMMDD_HHMMSS + "_" + 6 hex chars
├── input/original.mp4               # uploaded file, untouched
├── state.json                       # RunState (03 §2), rewritten atomically after each stage
├── events.jsonl                     # append-only audit log (03 §5)
├── ingest/
│   ├── probe.json                   # ffprobe output (subset)
│   ├── video.mp4                    # normalized: CFR, ≤960 px long edge, H.264, no audio
│   ├── video_with_audio.mp4         # normalized + AAC audio (sent to Spark Planner)
│   ├── audio.wav                    # mono 16 kHz s16 (sent to Voice)
│   └── keyframes/frame_000000.png … # frames used for image segmentation
├── transcript.json                  # Transcript (03)
├── spec.json                        # ExperimentSpec (03)
├── sam/
│   ├── <concept_slug>.raw.jsonl     # raw streamed events (for golden replay)
│   └── <concept_slug>.records.json  # parsed records summary (boxes, mask bounds, revisions)
├── observations.parquet|csv         # per-frame per-object observations (03 §3)
├── tracks/<track_id>.csv            # cleaned time series used for fitting
├── calibration.json                 # Calibration results
├── analysis/
│   ├── ledger.jsonl                 # Claim Ledger: evidence + claims (03 §6)
│   ├── draft_report.json            # ReportDraft from Analyst (placeholders only)
│   ├── gates.json                   # GateResult[]
│   └── verdict.json                 # Verdict from Auditor
├── outputs/
│   ├── overlay.mp4
│   ├── chart_<name>.json            # Plotly figure JSON
│   ├── chart_<name>.png             # static image for the PDF
│   ├── data.csv                     # t, x, y, … in SI units
│   ├── report_en.pdf / report_id.pdf
│   └── next_experiment.json         # DesignerProposal
└── error.json                       # only if FAILED
```

Series (multi-run) analysis lives in `runs/series/<series_id>/` with `series.json`,
`ledger.jsonl`, `chart_t2_vs_l.json`, `report_*.pdf`.

## 6. Repository tree

```
framelab/
├── AGENTS.md                        # copied from docs/plan/AGENTS.md
├── README.md                        # public pitch + how to run + benchmark + model map
├── HACKATHON.md                     # what was built when; third-party code/assets & licenses
├── LICENSE                          # MIT
├── .env.example
├── .gitignore                       # .env, runs/, web/node_modules, web/dist, __pycache__, .venv
├── pyproject.toml                   # package "framelab" in backend/
├── uv.lock
├── Makefile                         # dev, test, live-test, eval, build-web, docker, demo-cache
├── scripts/
│   ├── dev.ps1                      # Windows equivalents of Makefile targets
│   ├── spikes/                      # M1 API verification scripts (kept in repo as evidence)
│   │   ├── spike_sam_video.py
│   │   ├── spike_sam_image.py
│   │   ├── spike_spark_video.py
│   │   ├── spike_structured_output.py
│   │   └── spike_voice.py
│   └── build_demo_cache.py
├── Dockerfile
├── backend/
│   ├── framelab/
│   │   ├── __init__.py
│   │   ├── config.py                # Settings (pydantic-settings), model IDs, thresholds
│   │   ├── schemas.py               # all Pydantic models (03)
│   │   ├── store.py                 # RunStore
│   │   ├── events.py                # EventBus + emit
│   │   ├── cli.py                   # typer CLI
│   │   ├── costs.py                 # price table + cost estimation per call
│   │   ├── api/
│   │   │   ├── main.py              # FastAPI app factory, routes, static web
│   │   │   ├── sse.py
│   │   │   └── demo.py              # demo-mode replay of cached runs
│   │   ├── clients/
│   │   │   ├── meta.py
│   │   │   ├── sam.py
│   │   │   ├── voice.py
│   │   │   ├── image.py             # S-2
│   │   │   └── baseten.py           # S-3
│   │   ├── pipeline/
│   │   │   ├── run.py               # orchestrator
│   │   │   ├── context.py           # RunContext
│   │   │   ├── ingest.py
│   │   │   ├── transcribe.py
│   │   │   ├── segment.py
│   │   │   ├── measure.py
│   │   │   ├── calibrate.py
│   │   │   ├── overlay.py
│   │   │   ├── charts.py
│   │   │   └── report.py
│   │   ├── physics/
│   │   │   ├── models.py            # model functions + fitters
│   │   │   ├── events.py            # release/impact detection, window selection
│   │   │   ├── uncertainty.py       # bootstrap, propagation, rounding
│   │   │   ├── derive.py            # g from fits, amplitude correction, series regression
│   │   │   └── experiments.py       # registry: free_fall, pendulum, (melting)
│   │   ├── agent/
│   │   │   ├── planner.py
│   │   │   ├── analyst.py
│   │   │   ├── auditor.py
│   │   │   ├── designer.py
│   │   │   ├── tools.py             # tool JSON schemas + implementations
│   │   │   ├── gates.py
│   │   │   ├── ledger.py
│   │   │   ├── narrative.py         # placeholder rendering + bare-number check
│   │   │   └── prompts/
│   │   │       ├── planner.md
│   │   │       ├── analyst.md
│   │   │       ├── auditor.md
│   │   │       └── designer.md
│   │   └── templates/
│   │       └── report.html.j2
│   └── tests/
│       ├── conftest.py
│       ├── fakes.py                 # FakeMetaClient, FakeSam, FakeVoice (replay fixtures)
│       ├── test_ingest.py
│       ├── test_measure.py
│       ├── test_calibrate.py
│       ├── test_physics_freefall.py
│       ├── test_physics_pendulum.py
│       ├── test_uncertainty.py
│       ├── test_gates.py
│       ├── test_ledger_narrative.py
│       ├── test_agent_loop.py
│       ├── test_api.py
│       ├── test_golden_runs.py
│       ├── test_live.py             # @pytest.mark.live
│       └── fixtures/
├── web/
│   ├── package.json
│   ├── vite.config.ts               # dev proxy /api → http://localhost:8000
│   ├── index.html
│   └── src/
│       ├── main.ts
│       ├── api.ts                   # typed fetch + EventSource helpers
│       ├── types.ts                 # mirrors backend schemas (hand-written)
│       ├── views/upload.ts
│       ├── views/run.ts
│       ├── views/series.ts
│       ├── components/timeline.ts
│       ├── components/player.ts
│       ├── components/charts.ts
│       ├── components/trace.ts
│       ├── components/ledger.ts
│       └── styles.css
├── samples/
│   ├── README.md                    # how they were recorded, license CC BY 4.0
│   ├── ground_truth.json
│   └── *.mp4                        # self-recorded (keep each < 20 MB; Git LFS if needed)
├── eval/
│   ├── benchmark.py
│   └── results.md                   # generated
└── docs/
    ├── plan/                        # this plan
    ├── architecture.svg
    ├── recording-guide.md
    └── demo-script.md
```

## 7. Tech stack decisions (and why)

| Decision | Choice | Reason |
|---|---|---|
| Language | Python 3.11 | Scientific stack (numpy/scipy/OpenCV), official `meta_sam_parser` |
| Web framework | FastAPI | Async, SSE via `StreamingResponse`, OpenAPI for free |
| Deps | `uv` | Fast, lockfile, works on Windows |
| LLM SDK | `openai` with `base_url=https://api.meta.ai/v1` | Officially supported by the Meta Model API cookbook |
| PDF | Jinja2 HTML → Playwright Chromium `page.pdf()` | Same HTML for web preview and PDF; works on Windows and Docker |
| Video | ffmpeg CLI | Normalization, audio extraction, final H.264 encode (`-preset veryfast`) |
| Frontend | Vite + TS, no framework | Small surface, fast to build solo; Plotly for charts |
| Storage | Files in `runs/` | No DB needed for hackathon; easy to cache for demo mode |
| Deploy | Docker on Railway/Render/Fly (or Modal) | One container, persistent volume for `runs/` |
| Agent loop | Chat Completions tool calling (proven in cookbook) | Simple, debuggable; Responses API used where required (video input, SAM, Image) |

## 8. Configuration (env vars)

| Variable | Default | Meaning |
|---|---|---|
| `MODEL_API_KEY` | — (required for live) | Meta Model API key (`LLM|{id}|{secret}`) |
| `BASETEN_API_KEY` | — | Only for S-3 |
| `FRAMELAB_RUNS_DIR` | `./runs` | Run storage |
| `FRAMELAB_DEMO` | `0` | `1` = serve cached demo runs; uploads disabled or queued |
| `FRAMELAB_MAX_UPLOAD_MB` | `200` | Upload limit |
| `FRAMELAB_MAX_ANALYSIS_S` | `20` | Max analysis window sent to SAM |
| `FRAMELAB_CONCURRENCY` | `2` | Parallel runs |
| `FRAMELAB_LIVE_TESTS` | `0` | Enable live tests |
| `FRAMELAB_USE_WONDER` | `1` | `0` = Auditor falls back to Muse Spark 1.3 (cost/availability) |
| `FRAMELAB_DESIGNER_BACKEND` | `meta` | `meta` or `baseten` (S-3) |
| `FRAMELAB_LOG_LEVEL` | `INFO` | |
