# FrameLab

**Every phone is a physics lab.** FrameLab turns a phone video of a simple experiment, such as
a falling ball or a swinging pendulum, into measured, uncertainty-quantified, *verified*
scientific data. It then designs the next experiment.

> 🚧 Work in progress, built during the **Meta Global AI Developer Hackathon 2026**
> (27 Oct – 4 Nov 2026). See [`HACKATHON.md`](HACKATHON.md) for what was built and when.

## How it works

| Stage | Meta model | Role |
|---|---|---|
| Transcribe the spoken hypothesis and parameters (Indonesian/English) | **Muse Voice Transcribe** | Speech → text with turn timings |
| Plan the experiment from the video | **Muse Spark 1.3** | Video + transcript → ExperimentSpec |
| Track the object in every frame and segment the ruler | **SAM 3.1** | The *sensor*: masks → sub-pixel positions |
| Analyse through tools (fit, bootstrap, gates, claims) | **Muse Spark 1.3** | Agent that decides; code computes |
| Independent review of the report | **Muse Wonder** | Auditor: PASS / REJECT / NEEDS_RERECORD |
| Propose the next experiment | **Muse Spark 1.3** | Change one variable, predict with code |

Core rule: **the LLM decides, code computes.** Every number in a report comes from
deterministic Python and is linked to evidence in a *Claim Ledger*.

Full design: [`docs/plan/`](docs/plan/README.md).

## Quick start (development)

Requirements: Python 3.11+, [uv](https://docs.astral.sh/uv/), Node 20+, ffmpeg/ffprobe on PATH.

```bash
uv sync
uv run playwright install chromium
cp .env.example .env          # then put your MODEL_API_KEY in .env
uv run framelab doctor        # checks ffmpeg, Chromium, API key
uv run pytest                 # offline tests
uv run uvicorn framelab.api.main:app --reload    # http://localhost:8000/api/health
cd web && npm install && npm run dev             # http://localhost:5173
```

Windows PowerShell: `.\scripts\dev.ps1 install`, `.\scripts\dev.ps1 test`, `.\scripts\dev.ps1 api`, …

## Status

- [x] M0: repository skeleton, schemas, run store, event log, CLI `doctor`, health endpoint
- [x] M1: Meta API verification spikes (findings in `docs/plan/11-risks-and-open-questions.md`)
- [x] M2: ingest, SAM client, segmentation, observations (`framelab run --until measuring`)
- [ ] M2–M12: see [`docs/plan/09-milestones-and-tasks.md`](docs/plan/09-milestones-and-tasks.md)

## License

MIT. Sample videos (in `samples/`) are self-recorded and licensed CC BY 4.0.
