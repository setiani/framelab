# 08: Testing and evaluation

## 1. Test layers

| Layer | Location | Calls APIs? | Runs in CI | Purpose |
|---|---|---|---|---|
| Unit (synthetic) | `tests/test_*.py` | No | Yes | Physics, measurement, calibration, gates, ledger, formatting |
| Component with fakes | `tests/test_agent_loop.py`, `test_api.py` | No (fakes) | Yes | Agent loop control flow, tool dispatch, API endpoints, SSE |
| Golden replay | `tests/test_golden_runs.py` | No (recorded) | Yes | Full pipeline on recorded SAM/Voice/LLM responses → identical outputs |
| Live | `tests/test_live.py` (`@pytest.mark.live`) | **Yes** | No (manual) | Smoke test against the real Meta API |
| Benchmark | `eval/benchmark.py` | Yes (or cached) | No | Accuracy on real videos vs ground truth |

`pytest.ini`/`pyproject`: `markers = live: calls paid APIs`. `conftest.py` skips `live`
unless `FRAMELAB_LIVE_TESTS=1`.

## 2. Unit tests (minimum set)

| File | Tests |
|---|---|
| `test_ingest.py` | ffprobe parsing (fixture JSON), VFR detection, fps choice, scale expression output size even, frame-accurate clip (generate a 2 s test video with `ffmpeg -f lavfi -i testsrc=size=640x360:rate=60`, count frames) |
| `test_measure.py` | Synthetic mask raster with known disk → centroid within 0.05 px; bounds scaling (mask w/h ≠ box w/h); revision selection; track selection score; Hampel on 2nd differences keeps a clean parabola intact and flags an injected spike; gap interpolation flags |
| `test_calibrate.py` | Rotated ruler rectangle mask → length within 1 px; blurred ball (ellipse) → minor-axis diameter; Kåsa circle fit on noisy arc; agreement % |
| `test_physics_freefall.py` | `make_freefall` → window detection (release/impact indices ±1), g within 0.5 %, bounce ignored, 30 fps case |
| `test_physics_pendulum.py` | Nonlinear ODE data θ₀ = 5°, 20° → T within 0.1 %, amplitude correction effect, drift term absorbs camera drift |
| `test_uncertainty.py` | Block bootstrap coverage ≥ 90 % over 50 seeds (mark `slow`), formatting rules EN/ID incl. 1.96 → "2.0", 0.0123 → "0.012" |
| `test_gates.py` | Each gate pass/fail/warn boundary, slow-mo detection k = 8, G5 bare-number detection with allow-list |
| `test_ledger_narrative.py` | `add_claim` copies values; rejects unknown evidence; placeholder rendering EN/ID; unknown placeholder raises |
| `test_agent_loop.py` | FakeMetaClient scripted tool calls: happy path terminates with submit_report; bare number in draft → violations returned → fixed; budget exhaustion → fallback report; REJECT → second round gets fix_instructions |
| `test_api.py` | Upload validation (size/type), run creation (pipeline mocked), artifacts allow-list & traversal, SSE replay + Last-Event-ID |

## 3. Fakes and golden runs

- `tests/fakes.py`:
  - `FakeMetaClient`: returns scripted chat responses (tool calls or content) from a YAML/JSON
    script, records requests.
  - `FakeSam`: replays `sam/<slug>.raw.jsonl` through the real `meta_sam_parser`, so the
    parser integration is tested too.
  - `FakeVoice`: returns a stored transcript JSON.
- Recording: the live client has `FRAMELAB_RECORD_DIR`. When set, it saves every response to
  `tests/fixtures/<run_name>/` (SAM raw events, Voice JSON, chat/responses JSON). After M6,
  record two complete runs (free fall PASS, ruler-offplane NEEDS_RERECORD) and commit the
  fixtures (strip base64 media; keep the SAM raw text).
- `test_golden_runs.py` replays them end-to-end (except PDF rendering, which is optional and
  slow, behind the `slow` marker) and asserts: verdict, headline within 1e-9 of the stored
  value, gate statuses, and claim ids.

## 4. Live tests (manual, cost a few cents)
- `test_live_spark_ping`: chat completion returns text.
- `test_live_sam_image`: segment `tests/fixtures/media/ball.png` "orange ball" → ≥ 1 mask.
- `test_live_sam_video_short`: 2 s clip → track coverage ≥ 0.9.
- `test_live_voice`: 3 s WAV with "tes satu dua tiga" → non-empty transcript.
- `test_live_planner`: sample free-fall video → `experiment == "free_fall"`.

## 5. Benchmark protocol (`eval/benchmark.py`)

### 5.1 Dataset (self-recorded, see 10 §4 and `docs/recording-guide.md`)
| Set | Videos | Ground truth to record |
|---|---|---|
| Free fall | 6: heights {0.8, 1.0, 1.5} m × balls {tennis, rubber}; slow-mo 240 (or 120) fps | height, ball diameter (caliper/ruler), ruler length, capture fps |
| Pendulum | 6: L {0.25, 0.50, 1.00} m × amplitude {≈8°, ≈20°}; 60 fps | L pivot→bob centre, stopwatch time of 10 swings (3 trials, average) |
| Traps | 4: camera hand-held/moving, ruler closer to camera, wrong spoken length, ball leaves frame | expected verdict + expected failed gates |

### 5.2 Script behaviour
```
uv run python eval/benchmark.py [--only freefall|pendulum|traps] [--use-cache] [--n-repeat 1]
```
- Runs each sample through the full pipeline (or reuses finished runs with `--use-cache`).
- Free fall: abs & rel error of g vs `g_ref`; CI coverage (is g_ref inside CI?).
- Pendulum: rel error of T vs stopwatch; rel error of g vs g_ref; correction effect at 20°.
- Series: g from slope using the three L at small amplitude; exponent n.
- Traps: verdict correct? expected gates failed? diagnosis code correct?
- Costs: total and per run; latency per stage (from events).
- Ablation (if time): rerun with `--no-auditor --no-gates` and count unsupported claims and
  wrong PASS verdicts on traps.
- Writes `eval/results.md` (tables + summary paragraph with only computed numbers) and
  `eval/results.json`.

### 5.3 Targets (put actual results in README, whatever they are)
| Metric | Target |
|---|---|
| Free fall mean |rel error g| | ≤ 3 % |
| CI coverage of g_ref (free fall) | ≥ 5/6 |
| Pendulum period rel error vs stopwatch | ≤ 2 % |
| Pendulum g rel error | ≤ 4 % |
| Traps correctly rejected with correct diagnosis | 4/4 |
| Median latency per run | ≤ 90 s |
| Mean cost per run | ≤ $0.40 |

If a target is missed, report it honestly in the README with an explanation. Do not tune
thresholds on the benchmark until it passes. Thresholds may be calibrated once on
**separate** dev videos (record 2–3 extra for that).
