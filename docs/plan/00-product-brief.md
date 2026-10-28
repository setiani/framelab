# 00: Product brief

## 1. One-liner

**FrameLab: every phone is a physics lab.** Record an experiment with your phone, say your
hypothesis out loud, upload, and get measured data, a fitted physical law with uncertainty,
an automatic quality verdict, a lab report, and a suggested next experiment.

## 2. Problem

- Quantitative physics labs need sensors (photogates, motion sensors). Many schools,
  homeschoolers and online tutoring students have none, so "labs" become videos or copied
  textbook numbers.
- Existing video-analysis tools (e.g. Tracker from Open Source Physics) require manual
  frame-by-frame clicking or template tuning, manual scale setup and manual model choice.
- Bad data goes unnoticed. Camera shake, a ruler not in the plane of motion, or a variable
  frame rate silently produce confident wrong numbers.
- The scientific loop (hypothesis → measure → analyse → conclude → change one variable →
  repeat) is rarely closed because re-analysis is tedious.

## 3. Solution: user journey

1. Record the experiment with a static phone (on books/tripod), a ruler in the plane of
   motion, and speak the parameters and hypothesis. Example: *"Panjang tali 50 senti, saya
   kira periodenya satu setengah detik."*
2. Upload the MP4 to FrameLab.
3. FrameLab transcribes speech (Muse Voice Transcribe, Indonesian/English code-switching)
   and watches the video (Muse Spark 1.3) to produce an **ExperimentSpec**: experiment
   type, object to track, scale reference, analysis window, spoken parameters and hypothesis.
4. **SAM 3.1** tracks the object in every frame and segments the reference object.
5. Deterministic Python converts masks into a calibrated time series (sub-pixel centroids,
   px→m scale, t = frame/fps) and cleans it.
6. The **Analyst agent** (Muse Spark 1.3 with tool calling) fits the physics model, computes
   bootstrap confidence intervals, runs quality gates, and writes a draft report with
   claim placeholders only.
7. The **Auditor** (Muse Wonder) checks the draft against the Claim Ledger and gate results:
   PASS, or REJECT with fix instructions (max 2 revision rounds). Data that cannot be
   fixed by re-analysis ends as **NEEDS_RERECORD** with concrete recording advice.
8. Outputs: overlay MP4, interactive chart, CSV, bilingual lab report PDF, Claim Ledger, and a
   **Designer** proposal for the next trial (change exactly one variable, with a prediction).

## 4. Scope

### MVP (must ship, in this order)
| ID | Feature |
|----|---------|
| MVP-1 | Free-fall experiment end-to-end: upload → g ± CI with overlay video, chart, CSV |
| MVP-2 | Verification: gates G1–G5, Claim Ledger, Auditor (Muse Wonder) with revise loop |
| MVP-3 | Speech: transcription + spoken parameters + hypothesis test |
| MVP-4 | Pendulum experiment: period, damping, g with amplitude correction |
| MVP-5 | Series analysis: several pendulum runs → T₀² vs L regression → g from slope |
| MVP-6 | Lab report PDF (ID and EN) + next-experiment proposal |
| MVP-7 | Web app with live progress (SSE), agent trace panel, ledger panel; deployed; demo mode |
| MVP-8 | Accuracy benchmark on ≥12 self-recorded videos + 4 "trap" videos, published in README |

### Stretch (only after MVP is deployed)
| ID | Feature |
|----|---------|
| S-1 | Melting ice: mask area vs time for two objects (salt vs no salt) |
| S-2 | Muse Image: clean apparatus diagram from a key frame (labels drawn by code) |
| S-3 | Designer agent on Muse Spark Open via Baseten (bonus "Best Build on Meta Open Weight Models") |
| S-4 | Interactive canvas player with per-frame masks (instead of pre-rendered overlay MP4) |
| S-5 | Ping-pong ball drag model (terminal velocity) |

### Non-goals
- No user accounts, no database, no multi-tenant storage (run folders on disk).
- No live/real-time camera inference (upload only).
- No medical/sports/political content (hackathon anti-project list).
- No chat interface. FrameLab is a pipeline with an agent inside, not a chatbot.
- No mobile app. Responsive web page is enough.

## 5. Users and impact

- Students and teachers without lab equipment (schools, homeschooling, online tutoring):
  quantitative labs with a phone and household objects; "Think Like a Scientist" variable
  worksheets filled with real data.
- Science-competition preparation: practise designing, measuring and reporting experiments
  with instant feedback.
- Makers/engineers (post-hackathon path, shown in the demo outro): the same architecture
  measures vibrations, robot-arm motion, conveyor speed and plant growth (time-lapse area)
  without dedicated sensors. This is the "Push the Frontier" story.

## 6. Judging fit (keep this visible while building)

| Criterion (weight) | What must be visible in the product/demo |
|---|---|
| Impact (30%) | Clear user story; real accuracy numbers; extension path to industry/research |
| Use of Meta AI tooling (25%) | SAM 3.1 used as a *measuring instrument*; Spark reads video+audio; Voice with ID/EN code-switching; Wonder as independent auditor; model→role map in README and video |
| Technical depth (20%) | Sub-pixel centroids, dual scale calibration, curve fitting + bootstrap, 5 gates, Claim Ledger, golden tests, benchmark |
| Demo (25%) | Ball drop → g appears; agent rejects a bad recording; T² vs L line; live app with demo mode |

Bonus prizes targeted: **Best Agentic Use-Case** (primary), **Most Viral** (X post),
**Community Vote**, optional **Best Open Weight** via Baseten (S-3).

## 7. Hackathon rules that affect implementation

- Repository must be **public**. Commit daily with clear messages.
- Demo must show only work built during the hackathon. Keep `HACKATHON.md` up to date.
- No code/data/assets without rights. Sample videos are self-recorded and licensed CC BY 4.0.
  Third-party libraries are listed with licenses.
- Must use Meta models (Model API and/or open weights). The model→role table must be explicit.
- 3-minute demo video + live product link + repo link in the submission.
