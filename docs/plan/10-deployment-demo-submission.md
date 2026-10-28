# 10: Deployment, demo video and submission

## 1. Dockerfile

```dockerfile
# --- web build ---
FROM node:20-slim AS web
WORKDIR /web
COPY web/package*.json ./
RUN npm ci
COPY web/ ./
RUN npm run build                      # → /web/dist

# --- runtime ---
FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 UV_SYSTEM_PYTHON=1 FRAMELAB_RUNS_DIR=/data/runs
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY backend/ backend/
RUN uv pip install --system -e . && python -m playwright install --with-deps chromium
COPY --from=web /web/dist web/dist
COPY demo_runs/ demo_runs/
EXPOSE 8000
CMD ["uvicorn", "framelab.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 2. Hosting
- **Railway or Render** (simplest): deploy from GitHub with this Dockerfile, set env vars
  (`MODEL_API_KEY`, `FRAMELAB_DEMO=0`, `FRAMELAB_DEMO_ALLOW_UPLOAD=1`), and attach a
  persistent volume at `/data`. Instance ≥ 1 vCPU / 2 GB RAM (overlay rendering + Chromium).
- **Modal** (if Modal credits were granted): `modal.asgi_app()` wrapping the FastAPI app, a
  volume for runs, and the image built from the same steps.
- Protect credits: per-IP limit (10 runs/hour), the global cost guard (06 §10), and the
  upload size limit. Show "Live uploads may be paused to protect API credits. Try the
  samples." when a daily budget env (`FRAMELAB_DAILY_BUDGET_USD`, default 10) is reached.
- After deploying, test in incognito: health, sample replay, one live upload.

## 3. Demo video (3:00): script for recording

Narration is in English (international judges). The in-video student speech is
Indonesian on purpose (code-switching). About 145 words/min. Record real app screens only.

| Time | Visual | Narration (EN) | Notes |
|---|---|---|---|
| 0:00–0:12 | Kitchen: phone on books, hand drops an orange ball. Cut to the FrameLab overlay: cyan mask follows the ball; **g = 9.74 ± 0.12 m/s²** appears. | "This is a phone, a ball, and a kitchen. FrameLab just turned them into a gravity measurement, accurate to about one percent." | Hook. No logo intro. Use the real benchmark number. |
| 0:12–0:30 | Text cards: "No photogates. No motion sensors. No clicking 300 frames." | "Most students never get to measure physics. Lab sensors are expensive, and video-analysis tools make you click through hundreds of frames by hand. FrameLab makes every phone a lab instrument, and it checks its own work." | Problem + who. |
| 0:30–0:52 | Upload a pendulum video. Transcript panel: *"Panjang talinya lima puluh senti, saya kira periodenya satu setengah detik."* Spec card fills. | "I upload a pendulum video and state my hypothesis in Indonesian. Muse Voice Transcribe captures it. Muse Spark watches the video and plans the experiment: what to track, what to use as a ruler, which seconds matter." | Model names visible on screen. |
| 0:52–1:20 | SAM 3.1 mask on the bob, trail, ruler segmented in yellow; data table fills; damped-sine chart draws. | "SAM 3.1 isn't just cutting out objects here. It's the sensor. It tracks the bob in every frame, and the ruler gives us real centimetres. From the masks we get sub-pixel positions and a clean time series." | Leave 2–3 s without narration. |
| 1:20–1:45 | Agent trace: `fit_model(damped_sine)` → `bootstrap_ci` → `run_gates` ✓✓✓✓. Headline T and g. Hypothesis card: "1.5 s predicted, not supported". | "The analyst agent picks the physics model and calls real tools. It never writes a number itself. Every value in the report links to evidence in a claim ledger. My guess of one and a half seconds? The data disagrees." | Click a claim → evidence chain highlights. |
| 1:45–2:12 | Trap video: ruler closer to the camera. G2 red. Muse Wonder auditor: **NEEDS RE-RECORD**: "ruler not in the plane of motion". Tips shown. | "Now a bad recording: the ruler is closer to the camera than the pendulum. The numbers look fine, but FrameLab catches the mismatch. Muse Wonder, acting as an independent auditor, refuses the result and tells me exactly how to re-record." | The "verifiable agent" moment. |
| 2:12–2:35 | Designer proposal: "Try 25 cm and 100 cm" with predicted periods. Series view: T₀² vs L straight line; g from slope. | "Then it closes the scientific loop: change one variable and test again. Three lengths later, T-squared against length is a straight line, and the slope gives gravity again, independently." | Speed up processing 2×. |
| 2:35–2:47 | Scroll the bilingual PDF report; flash the README benchmark table. | "Students get a full lab report in Indonesian or English. And we benchmarked it on our own recordings: the average error in g is [N] percent." | Use the real number. |
| 2:47–3:00 | Architecture diagram with the 5 Meta models highlighted; end card "FrameLab: every phone is a physics lab" + repo + URL. | "SAM as the sensor, Spark as the scientist, Wonder as the reviewer, Voice for the hypothesis, all built during this hackathon. Next: vibrations, robotics and growth measurements. FrameLab." | Say "built during this hackathon". |

Production tips: 1080p screen capture; record narration separately; captions burned in;
background music low. For X (Most Viral bonus), cut a 45–60 s vertical version from
0:00–0:12 + 1:45–2:12 + end card. If minors appear in any footage, show hands and
apparatus only.

## 4. Recording guide (publish as `docs/recording-guide.md`)
**All experiments**
- Phone completely still (tripod or leaned on books), never hand-held.
- Camera perpendicular to the plane of motion, about 1–1.5 m away.
- Ruler/tape **in the same plane** as the motion (taped to the wall right behind the path).
- Plain, contrasting background; bright light from the front; disable video stabilisation if
  possible.
- 10–20 s per clip. At the start, say the parameters: *"Tinggi jatuh satu meter, diameter
  bola enam koma tujuh senti."*
**Free fall:** slow motion 120/240 fps. Solid heavier ball (tennis/rubber). Height
0.8–1.5 m. If the phone saves slow-mo at 30 fps playback, enter the capture fps when
uploading.
**Pendulum:** 30/60 fps is fine. Small heavy bob (large nut, rubber ball) on thin string.
Amplitude < 15°. At least 10 swings. Measure L from the pivot to the centre of the bob.
**Ground truth for the benchmark:** write down height, L, ball diameter, ruler length, and
the stopwatch time of 10 swings (3 trials).

## 5. README outline (public repo)
1. Title + one-liner + 6 s GIF + links (demo video, live app).
2. The problem and who it helps (3–4 sentences).
3. How it works: architecture diagram + **"Meta models → role" table**.
4. Verifiable agent design: rule "LLM decides, code computes", gates, Claim Ledger, Auditor;
   a screenshot of a rejected run.
5. **Benchmark results** (from `eval/results.md`, honest).
6. Run it yourself: Docker one-liner; local dev (uv + npm); env vars; demo mode.
7. Built during the hackathon: link to `HACKATHON.md`; third-party libraries & licenses;
   patterns adapted from Meta cookbooks.
8. Limitations and next steps (industry/robotics/biology measurement).
9. License (MIT); sample videos CC BY 4.0.

## 6. Submission checklist
- [ ] Repo public; no secrets in history (`git log -p | grep -i "LLM|"` returns nothing)
- [ ] README complete with model→role table and benchmark
- [ ] `HACKATHON.md`: dated log of what was built; third-party code/assets; nothing pre-existing
- [ ] Demo video ≤ 3:00, publicly viewable (unlisted OK), shows only hackathon-built features
- [ ] Live URL works in incognito (sample replays + live upload)
- [ ] All team members listed (solo: you)
- [ ] Submitted on the Cerebral Valley platform before **Thu 5 Nov 09:00 WIB** (target Wed 22:00 WIB)
- [ ] X post with the demo clip; post link submitted (engagement counted until 8 Nov EOD)
- [ ] Share the gallery link for Community Vote (until 16 Nov)
- [ ] If S-3 is done: README states the Baseten-served Muse Spark Open usage (bonus flag)
