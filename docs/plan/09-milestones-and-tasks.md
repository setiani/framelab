# 09: Milestones and tasks

Times are WIB (UTC+7). Deadline: **Thu 5 Nov 09:00 WIB**. Internal submit target: **Wed 4 Nov
22:00 WIB**. Sat 31 Oct – Sun 1 Nov are light days (another deadline). Use them to record
videos.

Every milestone ends with: tests green → commit → push (public repo) → tick boxes here.

| Milestone | Day (WIB) | Output |
|---|---|---|
| M0 Bootstrap | Wed 28 Oct (evening) | Public repo, skeleton, CI, keys working |
| M1 API spikes | Thu 29 Oct (morning) | All VERIFY items answered and recorded |
| M2 Ingest + SAM + observations | Thu 29 Oct | `observations.csv` from a real video |
| M3 Measurement + free-fall physics (no agents) | Fri 30 Oct | **First real g** via CLI |
| M4 Pendulum + series physics | Sat 31 Oct (light) | Pendulum T, g; series regression on synthetic data |
| M5 Voice + Planner | Sun 1 – Mon 2 Nov | `transcript.json`, `spec.json` from real videos |
| M6 Ledger, tools, gates, Analyst | Mon 2 Nov | Agent-driven run producing claims + draft |
| M7 Auditor, Designer, Translator, verdict | Mon 2 – Tue 3 Nov | Full verified run, NEEDS_RERECORD on a trap |
| M8 Charts, overlay, report PDF | Tue 3 Nov | `overlay.mp4`, `report_id.pdf`, `report_en.pdf` |
| M9 HTTP API + SSE | Tue 3 Nov | API complete; run from curl |
| M10 Frontend | Tue 3 – Wed 4 Nov | Usable web app |
| M11 Deploy + demo mode | Wed 4 Nov (morning) | Live URL; sample replays |
| M12 Benchmark, docs, video, submit | Wed 4 Nov | Submission sent by 22:00 WIB |
| Stretch S-1…S-5 | only after M11 | See 00 §4 |

---

## M0: Bootstrap
**Tasks**
- [X] Claim the $150 Meta Model API credits (form in the hackathon guide), add a payment
      method, set a **Spend Limit**, and create an API key. (Human task.)
      Status 2026-10-28: API key is in `.env` (not tracked by git); confirm the credits show
      on the Billing page and the Spend Limit is set, then tick this box.
- [x] Optional: request Baseten credits (S-3) and Modal credits (hosting). (Human task.)
      Requested 2026-10-28; awaiting approval emails (credits not guaranteed). If both are
      approved: Baseten = open-weight path (Muse Spark Open), Modal = hosting.
- [x] Create the repo folder `framelab` (MIT), copy `plan/` → `docs/plan/` and
      `plan/AGENTS.md` → `AGENTS.md`. (Skeleton done 2026-10-28.)
- [x] Human: `git init`, create the **public** GitHub repo, first commit and push.
      Done 2026-10-28: https://github.com/setiani/framelab (commit `773e3d6`).
- [x] `pyproject.toml` (package in `backend/framelab`, entry point `framelab = framelab.cli:app`),
      deps: fastapi, uvicorn[standard], pydantic, pydantic-settings, openai, meta_sam_parser,
      httpx, numpy, scipy, opencv-python-headless, pillow, matplotlib, jinja2, playwright,
      typer, python-multipart, structlog (optional); dev: pytest, pytest-asyncio, ruff.
- [x] Skeleton modules from 01 §6 with docstrings and `NotImplementedError`.
- [x] `config.py` Settings, `schemas.py` (all of 03), `store.py`, `events.py`.
- [x] `.env.example`, `.gitignore`, `Makefile`, `scripts/dev.ps1`, `HACKATHON.md` (start log),
      README stub.
- [x] GitHub Actions CI: `uv sync`, `ruff check`, `pytest -m "not live and not slow"`.
- [x] `framelab doctor` command (ffmpeg, ffprobe, playwright chromium, API key, ping Spark).

**Acceptance:** `uv run framelab doctor` all green on the developer's Windows machine; CI
green; `schemas.py` imports; repo public.

**Prompt for the coding agent**
```
Read AGENTS.md and docs/plan/README.md, 01-architecture.md and 03-data-contracts.md.
Implement milestone M0 from docs/plan/09-milestones-and-tasks.md: project skeleton exactly as the
repository tree in 01 §6, pyproject with the listed dependencies (uv), Settings in config.py,
ALL Pydantic models from 03 in schemas.py, RunStore (atomic JSON writes) in store.py, EventBus with
emit() writing events.jsonl in events.py, Typer CLI with a working `doctor` command, Makefile +
scripts/dev.ps1, CI workflow, .env.example, .gitignore, HACKATHON.md started with today's entry.
Add unit tests for RunStore and EventBus. Do not implement pipeline logic yet.
```

## M1: API verification spikes
**Tasks** (each one is a small script in `scripts/spikes/` that prints findings. Write the
results into `docs/plan/11-risks-and-open-questions.md` §1)
- [x] Spike scripts written (2026-10-28) + `make_test_media.py` (synthetic free-fall clips with
      burnt-in frame counter and per-frame ground truth) + `run_all.py`. Offline-tested; the
      Meta API is not reachable from the build sandbox, so the spikes run on the dev machine:
      `uv run python scripts/spikes/run_all.py` (optionally `--speech <voice note>`), est. < $1.
      Results land in `scripts/spikes/out/` (git-ignored): `SUMMARY.md` + one JSON per spike.
- [x] Run the spikes on the dev machine and record the answers below. Done 2026-10-28
      (run_all ≈ $0.61 + spike_followup ≈ $0.17). Answers: 11 §1.
- [x] `spike_sam_video.py`: upload a 5 s 60 fps and a 3 s 240 fps clip (960 px). Segment
      "ball". Record: max frame_index+1 vs ffprobe frame count (1:1?), coordinate space,
      latency, cost, whether a video with an audio track is accepted, behaviour at 20 s.
- [x] `spike_sam_image.py`: segment "ruler" on a key frame. Check that the mask raster
      size equals the bounds size.
- [x] `spike_spark_video.py`: Responses API with `input_file` vs `input_video` + `file_id`;
      `usage` tokens for a 15 s clip; can it read the embedded audio?
- [x] `spike_structured_output.py`: Responses `text.format` json_schema with Spark; Chat
      Completions `response_format` with Wonder.
- [x] `spike_voice.py`: Indonesian/English mixed sentence; `ENDPOINTING` turns; check the
      `languageBias` value for Indonesian.
- [x] Decide fallbacks where needed and update 02/04 accordingly. Done: 02 §4–§8, 04 §2.4/§4/§6.2,
      05 §2.1, 06 table, config defaults (`use_wonder=False`, `mask_dilation_px=1.5`).
- [x] Extra spike `spike_followup.py` for the two surprises (V12 SAM motion onset,
      V7b Wonder strict mode).

**Acceptance:** every VERIFY item in 11 §1 has an answer with date and evidence (output
excerpt). Total spike cost < $2.

**Prompt for the coding agent**
```
Implement milestone M1 (API spikes) from docs/plan/09-milestones-and-tasks.md. Read
docs/plan/02-meta-api-reference.md first and use exactly those request patterns. Create the five
scripts in scripts/spikes/. Each takes paths/args via argparse, prints a concise findings report,
and saves raw responses under scripts/spikes/out/ (git-ignored). Generate test clips with ffmpeg
from a user-provided sample video (argument). Do not build any product code in this milestone.
```
(Then run them yourself, paste the outputs to the agent, and ask it to update 11 §1 and
any affected sections of 02/04.)

## M2: Ingest + SAM client + observations
**Tasks**
- [x] `pipeline/ingest.py` (04 §2): probe, effective fps, normalize, audio, analysis clip,
      key frames. Return `IngestResult`.
- [x] `clients/meta.py` (02 §1, §11): sync/async clients, retry/backoff, cost accounting,
      `model_call/model_result` events, recording mode (`FRAMELAB_RECORD_DIR`).
- [x] `clients/sam.py` (02 §5): upload, segment_video, segment_image, box-prompt fallback,
      raw event tee to JSONL, validation.
- [x] `pipeline/segment.py` (04 §4) incl. motion-blob fallback.
- [x] `pipeline/measure.py` §5 (observations only; cleaning comes in M3).
- [x] CLI `framelab run --until measuring` with a temporary hard-coded spec (`--concept`,
      `--ref-concept`, `--window`), until the Planner exists.
- [x] Tests from 08 §2 for ingest and measure (observations part).
- [x] Done 2026-10-28 (code + 27 new offline tests; 56 total green on Python 3.11 and 3.13).
      Golden replay: real SAM responses from M1 stored as `backend/tests/fixtures/sam/*.sam.json.gz`
      drive the whole M2 pipeline offline (`tests/fakes.py::FakeSam`).
- [x] Human: one live run on the dev machine (acceptance below), then commit. Done 2026-10-28
      (run r_20261028_155731_86e4b0, synthetic 60 fps clip, $0.06): 268 ball observations
      (frames 32–299, mean abs cy error 0.41 px vs ground truth), ruler found on key frame 0.

**Acceptance:** on a real free-fall video, `observations.csv` has one row per frame for
the ball with sub-pixel centroids, and the ruler observation is present. Overlay debug
image `debug/centroids.png` (scatter of centroids on the middle frame) looks right.

**Prompt for the coding agent**
```
Implement milestone M2 from docs/plan/09-milestones-and-tasks.md. Follow 04-pipeline-and-
measurement.md §2, §4, §5 and 02-meta-api-reference.md §1, §4, §5, §11 exactly (incl. the
VERIFY outcomes recorded in 11 §1). Use subprocess with list args (Windows-compatible). Add the
temporary CLI flags --concept, --ref-concept, --window to bypass the Planner. Write tests listed in
08 §2 for ingest and measure (synthetic rasters). Emit events for every stage and model call.
```

## M3: Measurement + free-fall physics (no agents yet)
**Tasks**
- [ ] Track selection and cleaning (04 §5.1–5.2).
- [ ] Calibration: ruler, object diameter, agreement (04 §6.1, 6.2, 6.4).
- [ ] `physics/events.py` free-fall window (05 §2.1); `physics/models.py` quadratic/cubic
      fits; DW; `physics/uncertainty.py` block bootstrap + propagation + `fmt` (05 §6–7);
      `physics/derive.py` g, v0, compare_reference.
- [ ] `tests/synth.py` generators + the free-fall/uncertainty tests (05 §8, 08 §2).
- [ ] `agent/ledger.py` minimal: evidence records written by these functions (claims later).
- [ ] CLI prints `g = … ± … m/s²` for a real video (agents bypassed via `--no-agents`).

**Acceptance:** synthetic targets in 05 §8 pass. On ≥ 2 real free-fall videos, g is
within 5 % of 9.78 (or a clear, explainable reason why not, written in HACKATHON.md).

**Prompt for the coding agent**
```
Implement milestone M3. Read 04 §5.1–§6 and all of 05 (free-fall parts, uncertainty, formatting,
synthetic generators). Implement pure functions with full unit tests first (synthetic data), then
wire them into the pipeline behind `framelab run --no-agents`, writing evidence records (seg_,
trk_, cal_, win_, fit_, ci_, der_, cmp_) to analysis/ledger.jsonl via agent/ledger.py. Never let
any value bypass the ledger. Print the headline with fmt() at the end of the CLI run.
```

## M4: Pendulum + series physics
**Tasks**
- [ ] Damped-sine guess + fit, amplitude correction, cycles count (05 §3).
- [ ] Circle-fit pivot, spoken/measured L, optional "string" segmentation cross-check (04 §6.3).
- [ ] Series regression + power law + pairs bootstrap (05 §4); `framelab series` CLI.
- [ ] Hypothesis test helper (05 §3.4).
- [ ] Tests with the nonlinear ODE generator (05 §8).

**Acceptance:** synthetic pendulum and series targets pass. One real pendulum video gives
T within 2 % of the stopwatch.

**Prompt for the coding agent**
```
Implement milestone M4 (pendulum + series) following 05 §3–§4 and 04 §6.3. Start with the
synthetic generator using scipy solve_ivp for the nonlinear pendulum and write tests that prove the
amplitude correction matters at 20°. Then wire into `framelab run --no-agents` (experiment chosen by
a temporary --experiment flag) and add `framelab series create/analyze`.
```

## M5: Voice + Planner
**Tasks**
- [ ] `clients/voice.py` + `pipeline/transcribe.py` (02 §8, 04 §3).
- [ ] `agent/planner.py` + `prompts/planner.md` (06 §2, §9.1): Responses API with video
      `file_id` (or the fallback decided in M1), post-validation, one retry, `spk_` evidence.
- [ ] Remove the temporary CLI flags' necessity (keep them as overrides).
- [ ] Record fixtures for 2 videos (`FRAMELAB_RECORD_DIR`).

**Acceptance:** for 3 real videos (free fall, pendulum, trap), `spec.json` has the
correct experiment, a usable target concept (SAM finds it), reference concept, and the
spoken params exactly as said, each with a source turn.

**Prompt for the coding agent**
```
Implement milestone M5. Follow 02 §6.3 and §8, 04 §3, 06 §2 and the planner prompt in 06 §9.1
(store it in agent/prompts/planner.md). Implement strict post-validation as listed in 06 §2 with one
retry that appends validation errors. Spoken numbers must be traceable to transcript turns (spk_
evidence). Add tests with FakeMetaClient for validation/retry behaviour.
```

## M6: Ledger, tools, gates, Analyst
**Tasks**
- [ ] Complete `ledger.py` (claims, `next_id`, copy-from-evidence) and `narrative.py`
      (placeholder rendering, G5 bare-number check with allow-list).
- [ ] `gates.py` G1–G4 with thresholds per experiment and EN/ID message templates (06 §7).
- [ ] `tools.py`: all tools in 06 §4 with JSON schemas, dispatch, compact results, events.
- [ ] `analyst.py` loop (06 §3) + `prompts/analyst.md` + deterministic fallback.
- [ ] Tests: `test_gates.py`, `test_ledger_narrative.py`, `test_agent_loop.py` (fakes).

**Acceptance:** a real free-fall run ends with ≥ 5 claims, all with evidence, a
`draft_report.json` whose narrative contains only placeholders, and gates recorded. The
agent trace in `events.jsonl` is readable.

**Prompt for the coding agent**
```
Implement milestone M6 following 06 §1–§4, §7, §9.2, §10 and 03 §6–§7. The core invariant: the model
can never introduce a number. Enforce it in add_claim (values copied from evidence) and in G5
(narrative bare-number check). Tool results returned to the model must be compact (≤2 KB). Write
the listed tests with a scripted FakeMetaClient, including the budget-exhaustion fallback.
```

## M7: Auditor, Designer, Translator, final verdict
**Tasks**
- [ ] `auditor.py` + prompt (06 §5, §9.3) with pre/post deterministic checks; default model
      Muse Spark (separate context), Wonder behind `FRAMELAB_USE_WONDER` (02 §7).
- [ ] Re-evaluate Muse Wonder as Auditor once on the PASS and trap fixtures with the strict
      `Verdict` schema (~$0.10). Enable only if both verdicts are correct; record in 11 §3.
- [ ] Revise loop in the orchestrator (04 §1) + `finalize_verdict` (06 §8).
- [ ] `designer.py` + `predict` tool + prompt (06 §6, §9.4).
- [ ] Translator for the second report language (04 §9, 06 §9.5) with placeholder-set check.
- [ ] Slow-mo diagnosis + `rerun` support in the orchestrator.

**Acceptance:** (1) free-fall video → PASS or PASS_WITH_WARNINGS with an auditor verdict;
(2) ruler-offplane trap → NEEDS_RERECORD with the `ruler_offplane` message in ID and EN;
(3) a deliberately injected bare number (test) → REJECT → fixed in round 2.

**Prompt for the coding agent**
```
Implement milestone M7 following 06 §5, §6, §8, §9.3–§9.5 and 04 §1, §9 (translation). The Auditor
must not see the Analyst conversation, only draft + ledger + gates + spec. Implement the
deterministic post-checks that override the model when hard gates failed. Add tests for verdict
finalisation and for the REJECT→revise round using fakes.
```

## M8: Charts, overlay, report PDF
**Tasks**
- [ ] `charts.py` Plotly JSON + matplotlib PNG (04 §7).
- [ ] `overlay.py` (04 §8) with HUD from ledger-formatted strings and the end card.
- [ ] `report.py` + `templates/report.html.j2` (04 §9), both languages; rerecord feedback PDF.
- [ ] `outputs/data.csv` (04 §10).
- [ ] `framelab replay RUN_DIR` re-renders outputs offline.

**Acceptance:** for one PASS and one NEEDS_RERECORD run, all artifacts exist; the PDF opens,
looks clean on A4, and every number in it matches the ledger.

**Prompt for the coding agent**
```
Implement milestone M8 following 04 §7–§10. Numbers in overlay HUD, charts and report must come
from ledger claims formatted with physics/uncertainty.fmt. Build the report template as clean,
printable A4 HTML (no external network resources), render with Playwright Chromium. Add
`framelab replay` to regenerate outputs from saved files without API calls.
```

## M9: HTTP API + SSE
**Tasks**
- [ ] All endpoints in 07 §1, SSE in 07 §2 with replay and Last-Event-ID, upload streaming
      and validation, artifact allow-list, background run execution with semaphore.
- [ ] `test_api.py`.

**Acceptance:** `curl -F video=@samples/freefall_01.mp4 localhost:8000/api/runs` → run
completes; `curl -N .../events` streams events; artifacts downloadable.

**Prompt for the coding agent**
```
Implement milestone M9 following 07 §1–§2 and §5. Use FastAPI StreamingResponse for SSE with
heartbeats and Last-Event-ID replay from events.jsonl. Execute runs as asyncio background tasks
limited by Settings.concurrency. Write test_api.py with the pipeline mocked.
```

## M10: Frontend
**Tasks**
- [ ] Vite + TS project, `api.ts`, `types.ts`, `format.ts` (port of `fmt`, tested).
- [ ] Upload view, run view (timeline, player, headline/spec cards, charts, agent trace,
      ledger with evidence highlighting, gates table, next experiment, downloads), series view
      (07 §4).
- [ ] NEEDS_RERECORD feedback card + slow-mo re-run button.
- [ ] Build into `web/dist`, served by FastAPI.

**Acceptance:** you can upload a video, watch the live trace, click a claim and see its
evidence chain, and download the PDFs, on desktop and on a phone-width screen.

**Prompt for the coding agent**
```
Implement milestone M10 following 07 §4 (layout, behaviour, visual direction). Vanilla TypeScript
components (functions returning HTMLElement with an update() method), Plotly for charts, EventSource
for SSE. Keep field names identical to backend schemas. Make it look like a precise lab instrument:
restrained palette, one cyan accent, monospace numbers, large headline. Must work at 375 px width.
```

## M11: Deploy + demo mode
**Tasks**
- [ ] `Dockerfile` (10 §1), deploy (10 §2), persistent volume for `runs/`.
- [ ] `scripts/build_demo_cache.py` + demo replay (07 §3); 3 sample cards.
- [ ] Health check, cost guard, upload rate limit (e.g. 10 runs/hour/IP) to protect credits.

**Acceptance:** public URL works in an incognito window; sample replays work with the API key
removed; a live upload works with the key.

## M12: Benchmark, docs, demo video, submission
**Tasks**
- [ ] Record the benchmark set (recording can happen on Sat/Sun) + `ground_truth.json`.
- [ ] `eval/benchmark.py` → `eval/results.md` (08 §5).
- [ ] README (10 §5), HACKATHON.md complete, `docs/recording-guide.md`, `docs/demo-script.md`,
      `docs/architecture.svg`.
- [ ] Record + edit the 3-minute video (10 §3). Upload (YouTube unlisted).
- [ ] Submit on the Cerebral Valley platform (10 §6 checklist). Post on X; submit the post link.

**Acceptance:** submission confirmation received before 22:00 WIB Wed 4 Nov.

---

## Stretch (only after M11)
| ID | Task | Pointer |
|---|---|---|
| S-1 | Melting ice (two objects, area) | 05 §5, registry entry, gates thresholds |
| S-2 | Muse Image apparatus diagram in the report | 02 §9 |
| S-3 | Designer on Muse Spark Open via Baseten + "Built with Muse Spark Open" | 02 §10 |
| S-4 | Canvas player with per-frame masks (instead of overlay MP4) | 07 §4.3 |
| S-5 | Ping-pong drag model | 05 (add `drag` model: `y'' = g − k·y'²`) |
