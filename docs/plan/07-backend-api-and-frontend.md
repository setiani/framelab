# 07: Backend HTTP API and frontend

## 1. HTTP API (`framelab/api/main.py`)

Base path `/api`. JSON everywhere except uploads (multipart) and artifacts (files). Errors
use `{"error": {"code": str, "message": str}}` with proper HTTP status.

| Method | Path | Request | Response | Notes |
|---|---|---|---|---|
| GET | `/api/health` | — | `{ok, version, demo, models: {...}}` | Also checks ffmpeg presence |
| POST | `/api/runs` | multipart: `video` (mp4/mov), `lang` ("id"\|"en", default "id"), `capture_fps` (optional int), `series_id` (optional), `note` (optional text ≤ 300 chars) | `201 {run_id}` | Validates size/type/duration, stores `input/original.mp4`, starts background task. In demo mode → `403 demo_mode` unless `FRAMELAB_DEMO_ALLOW_UPLOAD=1` |
| GET | `/api/runs` | `?limit=20` | `[RunState]` newest first | Includes demo runs flagged `demo: true` |
| GET | `/api/runs/{run_id}` | — | `RunState` + `spec`, `calibration`, `gates`, `verdict`, `claims` (if available) | |
| GET | `/api/runs/{run_id}/events` | — | `text/event-stream` | SSE (§2). Replays past events first |
| GET | `/api/runs/{run_id}/ledger` | — | `{evidence: [...], claims: [...]}` | Evidence `data` truncated for big arrays |
| GET | `/api/runs/{run_id}/track` | — | `{fps, points: [...], fit_curve: [...]}` | For custom plotting |
| GET | `/api/runs/{run_id}/artifacts/{name}` | — | file | `name` ∈ allow-list: `overlay.mp4`, `report_en.pdf`, `report_id.pdf`, `data.csv`, `chart_*.json`, `chart_*.png`, `keyframe.png`, `next_experiment.json`. Path traversal is blocked |
| POST | `/api/runs/{run_id}/rerun` | `{capture_fps?: int, lang?: str}` | `201 {run_id}` | New run with the same input (slow-mo fix button) |
| POST | `/api/series` | `{experiment: "pendulum", run_ids?: [str]}` | `201 SeriesState` | |
| POST | `/api/series/{series_id}/runs` | `{run_id}` | `SeriesState` | Adds a finished PASS run |
| POST | `/api/series/{series_id}/analyze` | — | `202` | Runs the series regression + report |
| GET | `/api/series/{series_id}` | — | `SeriesState` + results | |
| GET | `/` and static | — | `web/dist` | SPA fallback to `index.html` |

Upload validation: extension and MIME (`video/mp4`, `video/quicktime`), size ≤
`max_upload_mb`, ffprobe duration 1–120 s. Stream the upload to disk in chunks (do not read
it fully into memory).

CORS: allow `http://localhost:5173` in dev only.

## 2. Server-Sent Events

`GET /api/runs/{id}/events` returns `StreamingResponse(media_type="text/event-stream")`.
Each event: `id: <seq>\nevent: <type>\ndata: <json>\n\n`. Heartbeat comment `: ping` every
15 s. Supports the `Last-Event-ID` header (resume from seq). The stream closes after `done`
or `error` (after flushing).

Event types and payloads: exactly as `events.jsonl` (03 §5). The frontend relies on:
`stage_started`, `stage_completed`, `progress`, `model_call`, `model_result`, `tool_call`,
`tool_result`, `claim_added`, `gate_result`, `verdict`, `artifact`, `warning`, `error`, `done`.

## 3. Demo mode (`api/demo.py`)

- `scripts/build_demo_cache.py` copies selected finished runs into `demo_runs/` (committed
  to git if < 50 MB total; otherwise download at container start).
- With `FRAMELAB_DEMO=1`, these runs are listed first. "Try a sample" buttons create a
  **replay run**: a new run id whose events are re-emitted from the cached `events.jsonl`
  with original relative timing compressed ×4 (max 25 s total), and whose artifacts point
  to the cached files. This shows the full live experience without API calls.
- The UI labels replay runs: "Replay of a recorded run (no API calls)". Honesty matters to
  judges.

## 4. Frontend (`web/`)

Vite + TypeScript, no framework. Plotly via `plotly.js-dist-min`. One page with three views
(hash routing): `#/` upload, `#/run/<id>`, `#/series/<id>`.

### 4.1 Visual direction
- Clean "lab instrument" look: off-white background, dark ink, one accent (cyan `#00B8D9`,
  matching the overlay mask colour), monospace for numbers. Large headline number.
- Verdict badges: PASS (green), PASS_WITH_WARNINGS (amber), NEEDS_RERECORD (red-orange).
- Responsive: single column below 900 px.
- Footer: "Built with Meta SAM 3.1 · Muse Spark 1.3 · Muse Wonder · Muse Voice Transcribe"
  (+ "Built with Muse Spark Open" when S-3 is enabled).

### 4.2 Upload view (`#/`)
- Hero: one sentence + 6-second looping GIF/WebM of a ball drop with the mask overlay.
- Drop zone (MP4/MOV), language toggle (ID/EN), optional "Slow-motion capture fps" select
  (Auto, 120, 240), optional note.
- "Try a sample" cards (demo runs): Free fall, Pendulum 50 cm, Bad recording (trap).
- Link "How to record" → `docs/recording-guide.md` rendered as a modal.
- Recent runs list.

### 4.3 Run view (`#/run/<id>`)
Layout (desktop, 2 columns):
```
┌──────────────────────────── Header: run id · experiment · verdict badge · cost ───────────────────────────┐
│ Timeline (stages with spinner/check, durations)                                                         │
├───────────────────────────────────────────────┬───────────────────────────────────────────────────────────┤
│ Player: overlay.mp4 (or original while         │ Headline card: g = 9.74 ± 0.12 m/s²                        │
│ processing) + frame scrubber                   │ ref 9.78 · diff −0.4 % · hypothesis: not supported         │
│                                                │ Spec card: experiment, target, reference, spoken params,   │
│                                                │ hypothesis quote, transcript (collapsible)                 │
├───────────────────────────────────────────────┴───────────────────────────────────────────────────────────┤
│ Charts: trajectory + fit (Plotly), residuals                                                             │
├───────────────────────────────────────────────┬───────────────────────────────────────────────────────────┤
│ Agent trace: chronological list of model calls │ Claim Ledger: claims table; click a claim → highlights    │
│ (model badge, latency, cost), tool calls with  │ its evidence chain (evidence cards expand) and, if        │
│ args & short results, reasoning summaries      │ relevant, the chart/frames                                │
│ (collapsible), gate results, auditor verdict   │                                                           │
├───────────────────────────────────────────────┴───────────────────────────────────────────────────────────┤
│ Gates table (G1–G5 with metrics & messages) · Next experiment card · Downloads (PDF ID/EN, CSV, MP4)     │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```
- NEEDS_RERECORD state: the headline card becomes a "Recording feedback" card. It shows
  failed gates, the annotated key frame and the tips list (in the selected language). For
  `slowmo_retimed(k)`, show the "Re-run with capture fps = …" button.
- All numbers displayed come from the API (already formatted by backend `fmt`) or are
  formatted in TS with the same rules (port `fmt` to `web/src/format.ts` and test it
  against backend fixtures).
- The agent trace updates live from SSE. Auto-scroll can be paused.

### 4.4 Series view (`#/series/<id>`)
- Table of member runs (L, T₀ ± σ, verdict). "Add run" (pick from finished pendulum runs or
  upload a new one with this series id). "Analyze" button.
- Chart: T₀² vs L with error bars and regression; headline g from slope; power-law
  exponent n with "T ∝ √L: supported / not supported".

### 4.5 Frontend code structure
- `api.ts`: `createRun(form)`, `getRun(id)`, `subscribe(id, onEvent)`, etc.
- `types.ts`: hand-written TS interfaces mirroring 03 (keep field names identical).
- Components are functions returning `HTMLElement` and exposing `update(state)`.
- No global state library. One `RunStore` object per view holding the latest state and
  events.

## 5. CLI (`framelab/cli.py`, Typer)
```
framelab run VIDEO [--lang id|en] [--capture-fps N] [--runs-dir DIR] [--resume RUN_ID] [--no-overlay] [--no-report]
framelab replay RUN_DIR            # re-render outputs from saved files without API calls
framelab series create --run R1 --run R2 --run R3
framelab series analyze SERIES_ID
framelab doctor                    # checks ffmpeg, playwright chromium, API key, connectivity
```
The CLI prints stage progress and, at the end, the headline, verdict, cost and artifact paths.
