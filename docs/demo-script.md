# Demo video script (3:00)

The submitted video follows this script. Narration in English, about 145 words per minute; only
real app screens and real recordings. Numbers are from the runs in `eval/results.md`.

| Time | Visual | Narration |
|---|---|---|
| 0:00–0:12 | A phone on a support; a hand releases a nut on a string. Cut to the run page: result **g = 9.65 ± 0.12 m/s²** with the interval gauge. | "A phone, a string, and a nut. FrameLab just turned them into a gravity measurement, within one and a half percent." |
| 0:12–0:30 | Three text cards: "No photogates." "No motion sensors." "No clicking through 300 frames." | "Most students never get to measure physics. Lab sensors are expensive, and video-analysis tools make you click through hundreds of frames by hand. FrameLab makes every phone a lab instrument, and it checks its own work." |
| 0:30–0:52 | Upload the 40 cm pendulum video; the stage timeline moves; the transcript shows the spoken length and guess; the setup card fills. | "I upload a pendulum video and say the string length and my guess for the period. Muse Voice Transcribe captures it. Muse Spark watches the video and plans the experiment: what to track and which seconds matter." |
| 0:52–1:20 | Annotated video: SAM 3.1 mask on the nut, trail, pivot marker; frame scrubbing; trajectory chart with the damped-sine fit and residuals. | "SAM 3.1 is not just cutting out objects here. It is the sensor. It finds the nut in every frame, and code turns the masks into positions over time." |
| 1:20–1:45 | Agent trace (fit_model, bootstrap_ci, derive, run_gates); a click on a claim lights up its evidence chain; the hypothesis line: 1.5 s predicted, about 1.28 s measured. | "The analyst agent picks the physics model and calls real tools. It never writes a number itself: every value links to evidence in a claim ledger. My guess of one and a half seconds? The data say about 1.28." |
| 1:45–2:12 | Sample "Bad recording", replayed (labelled: no API calls): G2 fails, ruler and ball scales disagree by 61 %; the "Record this experiment again" card with the frame and tips. | "Now a bad recording: the ruler sits on a box, closer to the camera than the ball. The ruler and the ball's own size give two scales that disagree by sixty percent. The auditor refuses the result and says how to record it again." |
| 2:12–2:35 | Next-experiment panel with "Designed by Muse Spark Open" and predicted periods; series of 30, 40 and 50 cm: exponent n = 0.467 ± 0.019, √L law supported, T₀² against L. | "Then it closes the loop. Muse Spark Open, Meta's open-weight model, designs the next experiment: change only the length. Three lengths later, the period grows with the square root of the length, as theory says." |
| 2:35–2:47 | The lab report PDF in Indonesian: result, chart, gates, models. | "Students get a full lab report in Indonesian or English. Our three pendulum runs landed within 1.6 percent of local gravity." |
| 2:47–3:00 | Architecture diagram (`docs/architecture.svg`); end card with the repository and the app URL. | "SAM as the sensor, Spark as the scientist, an independent auditor, Voice for the hypothesis, and Spark Open for the next experiment, all built during this hackathon. FrameLab." |

Not in the video on purpose: the free-fall result (12 % low; see `eval/results.md`).
