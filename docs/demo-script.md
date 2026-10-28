# FrameLab: 3-minute demo video script

> Source of truth: `docs/plan/10-deployment-demo-submission.md` §3.

## Script

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
