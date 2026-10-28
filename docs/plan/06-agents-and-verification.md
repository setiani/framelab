# 06: Agents and verification

## 1. Principles

1. **The LLM decides, code computes.** Agents choose *which* tool, model, window or
   wording. Tools compute every number and write it to the ledger as evidence.
2. **Claims are bound to evidence.** `add_claim` copies values from evidence in code. The
   model cannot type a number into a claim.
3. **Narrative uses placeholders only** (`{{c03}}`). Bare numbers fail G5 mechanically,
   before any LLM review.
4. **Independent review.** The Auditor (Muse Wonder) sees the draft, ledger and gate
   results, not the Analyst's conversation.
5. **Bounded autonomy.** Max 12 Analyst steps per round, max 2 revise rounds, max 2 extra
   SAM requests per run. All limits live in `Settings`.
6. **Honest failure.** If data is bad, the right output is NEEDS_RERECORD with specific
   tips. That counts as a success of the system.

| Agent | Model | API | Reasoning | Output |
|---|---|---|---|---|
| Planner | `muse-spark-1.3` | Responses (video file) | medium | `ExperimentSpec` |
| Analyst | `muse-spark-1.3` | Chat Completions + tools | high | ledger entries + `ReportDraft` |
| Auditor | `muse-spark-1.3` by default; `muse-wonder-1.0` only if it passes the M7 re-evaluation (02 §7) | Chat Completions + strict structured output | high | `Verdict` |
| Designer | `muse-spark-1.3` (S-3: Spark Open via Baseten) | Chat Completions + 1 tool | low | `DesignerProposal` |
| Translator | `muse-spark-1.3` | Chat Completions + structured output | minimal | `ReportDraft` (other language) |

## 2. Planner (`agent/planner.py`)

**Input:** video file (`file_id` of `video_with_audio.mp4`), `Transcript` JSON, `ProbeInfo`
(duration, fps, size), user language, optional user note from the upload form.
**Output:** `ExperimentSpec` (03 §3).

Post-validation in code (reject → one retry with the error list appended):
- `target_concept`: 1–4 words, lowercase, concrete noun phrase, no verbs. Same for
  `reference.concept` and `target_alt_concepts`.
- `analysis_window_s`: inside `[0, duration]`, length ≥ 0.5 s, ≤ `max_analysis_s` (clip it
  and warn if longer).
- `spoken_params` keys must be in the allowed set (03 §3). Values must come from transcript
  turns (`source_turn` must exist). Units must be convertible. If the Planner reads a
  value that is not in the transcript, drop it.
- `experiment == "unknown"` or `confidence < 0.4` → stop the run with verdict
  NEEDS_RERECORD and the Planner's `warnings` as tips.
- Numbers in `spoken_params` are *extracted*, not computed. This is the one allowed case of
  an LLM producing a number, and it is traceable to a transcript turn. The ledger records
  them as evidence `spk_xx` with the turn text.

## 3. Analyst (`agent/analyst.py`)

### 3.1 Loop
```python
def run_analyst(ctx, spec, meas, round_) -> ReportDraft | RerecordRequest:
    messages = [
        {"role": "system", "content": prompts.ANALYST_SYSTEM},  # stable → cached
        {"role": "user", "content": analyst_context(ctx, spec, meas, round_)},  # §3.2
    ]
    for step in range(settings.analyst_max_steps):
        resp = meta.chat(
            model=SPARK,
            messages=messages,
            tools=TOOLS,
            reasoning_effort="high",
            prompt_cache_key="framelab-analyst-v1",
        )
        msg = resp.choices[0].message
        ctx.emit_reasoning(msg)
        if not msg.tool_calls:
            messages.append(
                {
                    "role": "user",
                    "content": "You must act through tools. Call submit_report or request_rerecord to finish.",
                }
            )
            continue
        messages.append(msg)
        for tc in msg.tool_calls:
            result = tools.dispatch(ctx, tc.function.name, json.loads(tc.function.arguments))
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result.to_json()})
            if result.terminal:  # submit_report accepted / request_rerecord
                return result.payload
    # step budget exhausted → deterministic fallback:
    return fallback_report(ctx, spec, meas)  # §3.4
```
Tool results are compact JSON (≤ 2 KB): `{ok, evidence_id, summary, data_excerpt, warnings}`.
Large arrays never go to the model.

### 3.2 Context message (`analyst_context`)
A single user message containing:
- `ExperimentSpec` JSON.
- Experiment definition (model, headline quantity, gate thresholds, min points).
- Track summary: track ids, coverage, outliers, gaps, auto-detected fit window, n points.
- Calibration summary: methods available, m/px, agreement %.
- Pre-computed evidence list (`seg_`, `trk_`, `cal_`, `spk_` ids with summaries).
- Round > 0: the previous `Verdict.fix_instructions` and the list of existing claims.
- The required report sections and the language (`lang`).

### 3.3 Expected tool sequence (the prompt describes it; the model may deviate if justified)
Free fall: `get_run_summary → set_fit_window(auto) → fit_model(quadratic) →
bootstrap_ci → derive(g) → compare_reference(g) → compare_hypothesis (if any) → run_gates
→ make_chart → add_claim ×N → submit_report`.
Pendulum: same with `damped_sine`, `derive(period)`, `derive(g)`.
If gates fail with remedy `reanalyze` (e.g. outliers, wrong window), try **one** corrective
action (`set_fit_window`, `choose_track`, `calibrate_scale` with another method,
`resegment`) and rerun the gates. If the remedy is `rerecord`, call `request_rerecord`.

### 3.4 Deterministic fallback
If the step budget runs out or the model API fails twice, produce a minimal report from the
auto pipeline: the default window and model, all evidence computed in code, and a
template-based narrative. Verdict is at most PASS_WITH_WARNINGS with the warning
`analyst_fallback`. The demo should never hit this, but the product must not crash.

## 4. Tools (`agent/tools.py`)

Each tool has a JSON schema (OpenAI function format), an implementation, and an evidence
record. IDs are allocated by `ledger.next_id(prefix)`.

| Tool | Args (JSON schema summary) | Does | Evidence `data` |
|---|---|---|---|
| `get_run_summary` | `{}` | Returns spec, tracks, calibration, existing evidence & claims (summaries) | — |
| `choose_track` | `{track_id: str, reason: str}` | Sets the primary track | `trk_` {track_id, coverage, n_valid} |
| `resegment` | `{concept?: str, use_box_hint?: bool, reason: str}` | New SAM request (max 2/run), re-measure | `seg_` {concept, status, n_objects, frames} |
| `calibrate_scale` | `{method: "ruler"\|"object_diameter"\|"spoken_length"}` | Computes the scale | `cal_` {m_per_px, sigma, rel_sigma, details} |
| `set_fit_window` | `{mode: "auto"\|"manual", t_start?: num, t_end?: num, reason: str}` | Auto = physics event detection (05 §2.1); manual must lie within the track | `win_` {t_start, t_end, n_valid, method} |
| `fit_model` | `{track_id: str, window_id: str, model: "quadratic"\|"cubic"\|"damped_sine"\|"linear_area"}` | Least-squares fit in px | `fit_` {params, cov, r2, rms_px, dw, n, model} |
| `bootstrap_ci` | `{fit_id: str, quantity: "g"\|"period"\|"damping"\|"melt_rate"}` | Block bootstrap + propagation (05 §6) | `ci_` {value, lo, hi, sigma_*, n, seed} |
| `derive` | `{quantity: "g"\|"period"\|"damping"\|"v0"\|"theta0", fit_id: str, scale_id?: str, length_source?: "spoken"\|"measured"}` | Physical quantity from fit | `der_` {quantity, value, unit, inputs, formula} |
| `compare_reference` | `{derived_id: str}` | vs g_ref (05 §1) | `cmp_` {ref, value, rel_diff_pct, within_ci} |
| `compare_hypothesis` | `{derived_id: str}` | vs `spec.hypothesis_prediction` (05 §3.4) | `cmp_` {p, m, lo, hi, outcome} |
| `run_gates` | `{}` | Runs G1–G4 on the current state (G5 runs at submit) | `gate_` per gate (GateResult) |
| `make_chart` | `{kind: "trajectory"\|"residuals"\|"series", fit_id: str}` | Plotly JSON + PNG | `fig_` {path_json, path_png} |
| `add_claim` | `{claim_type, quantity?: str, evidence_id: str, statement_en: str, statement_id: str}` | Creates a claim. **Value/CI/unit copied from evidence** (`der_`, `ci_`, `cmp_`, `gate_`). Templates may use `{value}`, `{ci}`, `{unit}`, `{outcome}`, `{rel_diff}` | claim `cNN` |
| `submit_report` | `ReportDraft` (03 §7) as JSON | Runs G5 (§7) on the draft; on failure returns the violations (not terminal); on success stores the draft (terminal) | — |
| `request_rerecord` | `{reasons: [gate ids], tips_en: [str], tips_id: [str]}` | Terminal; the Auditor still reviews | — |

Rules enforced in code:
- `add_claim` rejects evidence ids that do not exist. It also rejects `statement_*`
  templates containing digits outside the allow-list (§7 G5) or placeholders other than the
  allowed ones.
- `bootstrap_ci` requires a `fit_` produced in the current round.
- `derive(g)` for free fall requires `scale_id`. For the pendulum it requires `length_source`.
- Every tool emits `tool_call` / `tool_result` events and returns `warnings`.

## 5. Auditor (`agent/auditor.py`)

**Before calling the model (deterministic pre-checks):**
1. G5 mechanical check on the stored draft (should already pass).
2. Gate summary: list of hard failures.
3. If a hard gate with remedy `rerecord` failed and the Analyst did not request a
   re-record, mark it. The Auditor will be told.

**Model input:** one user message with:
- The rendered draft (placeholders replaced by formatted values **and** shown as
  `[c03: g = 9.74 ± 0.12 m/s²]` so the model sees what each claim is).
- The ledger: claims (type, value, CI, evidence ids) and the evidence summaries they
  reference (code-generated summaries).
- Gate results.
- Spec (hypothesis, spoken params) and verdict-status definitions.

**Output:** `Verdict` via structured output.

**After the model (deterministic post-checks, override with logged reason):**
- Any hard gate failed → status cannot be PASS/PASS_WITH_WARNINGS. Force REJECT (if the
  remedy is `reanalyze` and rounds remain) or NEEDS_RERECORD.
- Status PASS while soft warnings exist → downgrade to PASS_WITH_WARNINGS.
- REJECT without `fix_instructions` → add a generic instruction listing the failing items.

## 6. Designer (`agent/designer.py`)

Input: spec, headline claims, gate warnings, experiment def.
One tool: `predict(changed_variable, values[])`, computed in code:
- pendulum: `T = 2π·√(L/g_measured)` for each L (and with amplitude correction if θ given);
- free fall: `t_fall = √(2h/g_measured)` for each h.
Output `DesignerProposal` (03 §7). `predicted_values` are filled by code from the `predict`
result. Prediction texts reference them as `{{p1}}`, `{{p2}}`.
Safety constraints in the prompt: household objects only, heights ≤ 2 m, no fire, no
mains electricity, no glass, adult supervision note for children.

## 7. Gates (`agent/gates.py`)

```python
class GateThresholds(BaseModel):
    g1_min_coverage: float = 0.90
    g1_max_jump_factor: float = 5.0  # centroid step vs median step
    g1_min_points: int = 8  # pendulum: 60
    g1_min_cycles: float = 3.0  # pendulum only
    g2_max_scale_disagreement_pct: float = 5.0  # pendulum L check: 8.0
    g3_min_r2: float = 0.995  # pendulum: 0.98
    g3_max_rms_px: float = 2.0
    g3_min_durbin_watson: float = 1.0  # below → warn; below 0.5 → fail
    g4_range: tuple[float, float] = (9.0, 10.6)
    g4_slowmo_k: tuple[int, ...] = (2, 4, 8)
```

| Gate | Name | Hard? | Fail condition | Remedy | Diagnosis code → message |
|---|---|---|---|---|---|
| G1 | Track integrity | hard | coverage < 0.90, or valid points < min, or max jump > 5× median step, or id switches > 0 unresolved, or (pendulum) cycles < 3 | `rerecord` (if few points/blur) or `reanalyze` (if a wrong track/window) | `too_few_points`, `low_coverage_blur`, `object_left_frame`, `id_switch`, `too_few_cycles` |
| G2 | Scale consistency | hard if 2 methods exist; `warn` if only 1; `skipped` if pendulum without L | disagreement > 5 % (8 % for pendulum L) | `rerecord` | `ruler_offplane` (ruler scale ≠ ball scale), `spoken_length_mismatch` |
| G3 | Fit quality | hard | R² < min or RMS > 2 px; DW < 0.5 fail, DW < 1.0 warn; pendulum drift `|d|·T > 0.1·A` warn | `reanalyze` (window) then `rerecord` | `camera_motion`, `systematic_residuals`, `wrong_window` |
| G4 | Physical plausibility | soft (warn), **never alters data** | headline outside range | `none` (diagnose) | `slowmo_retimed(k)`, `air_drag_likely` (light object, g low, DW low), `length_reference_point` (pendulum L offset), `unexplained` |
| G5 | Ledger completeness | hard | any bare number in narrative outside the allow-list; unknown claim id; claim without evidence; `claims_used` ≠ ids found in the text; hypothesis present but no `hypothesis_test` claim | `reanalyze` (Analyst fixes the text) | `bare_number`, `unknown_claim`, `missing_hypothesis_claim` |

**G5 bare-number rule (`narrative.py`):** remove every `{{cNN}}`/`{{pN}}` placeholder,
then search `\d`. Allow-list (case-insensitive, matched before the digit search and
stripped): `G1`–`G5`, `SAM 3.1`, `Muse Spark 1.3`, `muse-spark-1.3`, `Muse Wonder 1.0`,
`95%`/`95 %` (only in the phrase "95% confidence"/"95% selang kepercayaan"), ordinals in
headings (`1.`, `2.` at line start), and `T₀²`-style symbol exponents. Everything else
fails.

Gate messages are **templates in code** (EN + ID) filled with formatted metrics, e.g.
`ruler_offplane`: EN "Ruler scale ({s1}) and ball-diameter scale ({s2}) disagree by
{pct}. The ruler is probably closer to or farther from the camera than the ball's path.
Tape the ruler to the wall directly behind the falling ball." / ID "Skala penggaris ({s1})
dan skala diameter bola ({s2}) berbeda {pct}. Penggaris kemungkinan tidak sejajar dengan
lintasan bola. Tempel penggaris di dinding tepat di belakang lintasan bola."

**Slow-mo check (G4):** if `|g·k² − g_ref|/g_ref < 0.05` for some k in (2, 4, 8), the
diagnosis is `slowmo_retimed(k)`: "This looks like a slow-motion video saved at normal
playback speed. Re-upload and set capture fps = {30·k}." Because this is fixable without
re-recording, the UI offers a **"Re-run with capture fps = {30·k}"** button (creates a new
run on the same input).

## 8. Final verdict (`finalize_verdict`)
```python
hard_failed = [g for g in gates if g.hard and g.status == "fail"]
if hard_failed:
    status = "NEEDS_RERECORD"  # data cannot support the measurement (re-analysis already tried)
elif auditor.status == "REJECT":  # rounds exhausted; only wording/claim issues remain
    status = "PASS_WITH_WARNINGS"
    warnings.append(
        "auditor_unresolved"
    )  # auditor notes are printed in the report's error analysis
else:
    status = auditor.status  # PASS | PASS_WITH_WARNINGS | NEEDS_RERECORD
```
`RunState.verdict` is never REJECT at DONE. The headline is shown only for PASS and
PASS_WITH_WARNINGS.

## 9. Prompts (store in `agent/prompts/*.md`, version header `<!-- v1 -->`)

### 9.1 `planner.md` (system)
```
You are the Planner of FrameLab, a system that measures physics experiments from phone videos.
You watch the uploaded video (frames and audio) and read a speech transcript, then return ONE JSON
object that matches the ExperimentSpec schema. You do not measure anything; later stages do.

Decide:
1. experiment: "free_fall" (an object dropped and falling vertically), "pendulum" (a mass swinging on
   a string), "melting" (ice/objects shrinking over time), or "unknown".
2. target_concept: a SHORT concrete noun phrase (1–4 words, lowercase) that a segmentation model can
   use to find the moving object in every frame, e.g. "orange ball", "metal nut", "ice cube".
   Prefer colour + object. Add 1–2 target_alt_concepts.
3. reference: the best visible scale reference in the plane of motion (e.g. "ruler", "measuring
   tape", "a4 paper"). Give its real length in metres only if it is spoken in the transcript, clearly
   readable in the video, or a standard object (A4 long side = 0.297 m). Otherwise null length.
   Give frame_hint_s: a time when it is fully visible and not covered by hands.
4. spoken_params: ONLY values that are explicitly spoken in the transcript. Include the transcript
   turn index in source_turn. Never guess or compute values. Keys: drop_height, string_length,
   ball_diameter, amplitude_deg, mass.
5. object_diameter_m: only if spoken (copy from spoken_params.ball_diameter).
6. analysis_window_s: the time range where the experiment happens (from just before release to
   just after the first impact for free fall; several full swings for a pendulum). At most 20 s.
7. motion_axis: "y" for free fall, "x" for pendulum, "xy" for melting.
8. hypothesis: verbatim quote of the student's hypothesis if they stated one; and
   hypothesis_prediction if it contains a number (with unit and source_turn).
9. setup_description: one neutral sentence describing the setup.
10. warnings: anything that will hurt measurement: camera moving, object leaving the frame, ruler
    not in the plane of motion, motion blur, poor lighting, reference occluded, slow-motion.
11. confidence in [0,1] for the whole spec.

Language: transcripts may mix Indonesian and English. Keep quotes verbatim. Write setup_description
in the requested language.
Return only the JSON object.
```
User message: `{"requested_language": "id", "probe": {...}, "transcript": {...}, "user_note": "..."}` + the video part.

### 9.2 `analyst.md` (system)
```
You are the Analyst agent of FrameLab. You turn tracked object positions into a verified
measurement and a lab report. You act ONLY through the provided tools.

Hard rules:
- You never write numbers yourself. All numbers come from tools and live in the evidence ledger.
- To state a result, create a claim with add_claim, which copies values from evidence. In report
  text, refer to claims ONLY with placeholders like {{c03}}. Text with any other digits is rejected.
- Never adjust, drop or cherry-pick data to approach a textbook value. Plausibility checks are for
  diagnosis only.
- If a gate fails with remedy "reanalyze", try at most one targeted fix (window, track, scale
  method, re-segmentation) and run the gates again. If the remedy is "rerecord", or the fix did not
  help, call request_rerecord with specific, practical tips in English and Indonesian.
- Be efficient: you have at most 12 steps. Call independent tools in parallel when possible.

Standard procedure:
1. get_run_summary.
2. set_fit_window(mode="auto").
3. fit_model with the experiment's primary model.
4. bootstrap_ci, then derive the headline quantity (and period for pendulum).
5. compare_reference; compare_hypothesis if the student stated a numeric prediction.
6. run_gates. Handle failures as above.
7. make_chart (trajectory, residuals).
8. add_claim for: the headline measurement, supporting quantities, comparison with reference,
   hypothesis test (if any), and one quality claim per gate result worth reporting.
9. submit_report with all sections in the requested language. Explain the method in plain words
   for a secondary-school student, describe error sources honestly, and keep conclusions within
   what the claims support. If submit_report returns violations, fix them and submit again.
```

### 9.3 `auditor.md` (system)
```
You are the independent Auditor of FrameLab lab reports. You did not write the report. Your job is
to protect scientific honesty. You receive: the report draft (each claim shown inline as
[cNN: value]), the claim ledger with evidence summaries, the quality-gate results, and the
experiment spec.

Check:
1. Every factual or numeric statement is backed by a claim whose evidence supports it.
2. Conclusions do not overreach (e.g. "proves", generalisations beyond the data, ignoring a
   failed or warning gate, calling a hypothesis supported when the comparison says otherwise).
3. Failed hard gates are acknowledged and the status reflects them.
4. Error analysis mentions the relevant gate warnings.
5. The hypothesis evaluation matches the hypothesis_test claim outcome.

Decide status:
- PASS: all checks fine, no warnings.
- PASS_WITH_WARNINGS: fine, but soft warnings exist and are disclosed.
- REJECT: fixable by re-analysis or rewriting; give precise fix_instructions (which section, what
  to change, which claim/gate).
- NEEDS_RERECORD: the data itself cannot support a measurement; give practical recording tips in
  English and Indonesian.
Return only the Verdict JSON.
```

### 9.4 `designer.md` (system)
```
You are the Designer of FrameLab. Propose the single most informative NEXT experiment that changes
exactly ONE variable, keeps the others controlled, and tests a physical relationship (e.g. period vs
string length, fall time vs height). Use the predict tool to get predicted values; never compute
them yourself, refer to them as {{p1}}, {{p2}}... Write protocol steps a secondary-school student
can follow at home with household objects. Safety: heights at most 2 m, no fire, no mains
electricity, no glass; mention adult supervision for children. Write every text field in both
English (_en) and Indonesian (_id). Return only the DesignerProposal JSON after using the tool.
```

### 9.5 `translator.md` (system)
```
Translate this FrameLab ReportDraft JSON from {src} to {dst}. Keep JSON structure and keys
identical. Keep every placeholder like {{c03}} or {{p1}} exactly as is, in the same sentence
position where natural. Do not add numbers. Use clear, school-level scientific language.
```

## 10. Budgets and limits
| Item | Limit |
|---|---|
| Analyst steps per round | 12 |
| Revise rounds | 2 |
| Extra SAM requests per run | 2 (+1 box-prompt fallback) |
| Tool result size to model | 2 KB |
| Planner retries | 1 (validation) |
| Max run cost guard | $1.50 (abort with error `cost_guard` if exceeded) |
