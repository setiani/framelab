"""Orchestrator: executes the stages in order (docs/plan/04-pipeline-and-measurement.md §1).

INGESTING → TRANSCRIBING → PLANNING → SEGMENTING → MEASURING → ANALYZING ⇄ AUDITING (revise
rounds) → DESIGNING → RENDERING → DONE. Free fall and pendulum both go through the Analyst agent
(deterministic fallback); ``--no-planner`` runs stay deterministic and stop after ANALYZING.

The final verdict is stored in ``state.json`` before DESIGNING and RENDERING, so the report, the
overlay's end card and the recording-feedback page see it, and the report carries the Designer's
next experiment (audit fix, 2026-10-30). Data that cannot support a measurement (no target, no
scale, Planner stop) ends as NEEDS_RERECORD with a feedback page, not as FAILED (04 §1).

Spec source: the Planner (default; Voice and Spark calls), or ``spec_override`` with
``--no-planner`` (flags only, no API calls for transcript or spec). ``overrides`` are typed flags
applied on top of the Planner's spec.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from framelab.agent import analyst, auditor, designer, planner, translator
from framelab.agent import gates as gates_mod
from framelab.agent import session as session_mod
from framelab.agent.ledger import Ledger
from framelab.agent.session import AnalysisSession
from framelab.clients import meta as meta_client
from framelab.clients import sam as sam_client
from framelab.config import Settings
from framelab.physics.experiments import REGISTRY
from framelab.pipeline import analysis_auto, measure, outputs, segment
from framelab.pipeline import ingest as ingest_mod
from framelab.pipeline import transcribe as transcribe_stage
from framelab.pipeline.context import RunContext
from framelab.schemas import (
    ExperimentSpec,
    GateResult,
    IngestResult,
    Lang,
    Observation,
    ReportDraft,
    RunState,
    Stage,
    Transcript,
    Verdict,
)
from framelab.store import write_json_atomic

ORDER = [
    Stage.INGESTING,
    Stage.TRANSCRIBING,
    Stage.PLANNING,
    Stage.SEGMENTING,
    Stage.MEASURING,
    Stage.ANALYZING,
    Stage.AUDITING,
    Stage.DESIGNING,
    Stage.RENDERING,
]


def _reached(stage: Stage, until: Stage | None) -> bool:
    return until is not None and ORDER.index(stage) >= ORDER.index(until)


async def execute_run(
    run_id: str,
    settings: Settings,
    *,
    spec_override: ExperimentSpec | None = None,
    overrides: dict[str, Any] | None = None,
    until: Stage | None = None,
) -> RunState:
    """Run the pipeline. Never raises: failures end in FAILED with error.json written."""
    ctx = RunContext.load(run_id, settings)
    upload: asyncio.Task[str] | None = None
    try:
        with ctx.stage(Stage.INGESTING):
            ing = ingest_mod.run(
                ctx.run_dir,
                capture_fps=ctx.state.capture_fps,
                max_long_edge_px=settings.max_long_edge_px,
            )
            ctx.add_artifact("video", ctx.run_dir / ing.video_path, "video/mp4")
        if _reached(Stage.INGESTING, until):
            ctx.finish(None, partial=True)
            return ctx.state

        transcript = Transcript()
        with ctx.stage(Stage.TRANSCRIBING):
            if spec_override is None:
                # The Planner needs the video's own audio (with sound) and the transcript.
                # The upload runs while Voice transcribes (04 §1).
                planner_video = ctx.run_dir / (ing.video_with_audio_path or ing.video_path)
                upload = asyncio.create_task(sam_client.upload_video(planner_video, ctx.emit))
                audio = ctx.run_dir / ing.audio_path if ing.audio_path else None
                transcript = await transcribe_stage.run(audio, emit=ctx.emit)
                write_json_atomic(ctx.run_dir / "transcript.json", transcript)
                ctx.add_artifact("transcript", ctx.run_dir / "transcript.json", "application/json")
            else:
                ctx.emit("progress", message="Transcript skipped: manual spec (--no-planner)")

        with ctx.stage(Stage.PLANNING):
            if spec_override is not None:
                spec = spec_override
            else:
                ledger = Ledger(ctx.run_dir / "analysis" / "ledger.jsonl")
                file_id = await upload
                try:
                    spec = await planner.run(
                        file_id,
                        transcript,
                        ing.probe,
                        ctx.state.lang,
                        ctx.state.note,
                        emit=ctx.emit,
                        ledger=ledger,
                    )
                finally:
                    await meta_client.delete_file(file_id, emit=ctx.emit)  # V10: no expiry
                spec = planner.apply_overrides(spec, overrides or {})
            write_json_atomic(ctx.run_dir / "spec.json", spec)
            ctx.add_artifact("spec", ctx.run_dir / "spec.json", "application/json")
            ctx.state.experiment = spec.experiment
            stopped = planner.needs_rerecord(spec)
            if not stopped:
                if spec.experiment not in ("free_fall", "pendulum"):
                    raise NotImplementedError(
                        f"Experiment {spec.experiment!r} is not supported yet (stretch S-1)"
                    )
                duration = ing.n_frames / ing.file_fps
                t0, t1 = spec.analysis_window_s
                if t1 - t0 > settings.max_analysis_s:
                    ctx.emit(
                        "warning",
                        message=f"Analysis window {t1 - t0:.1f} s clipped to "
                        f"{settings.max_analysis_s:.0f} s",
                    )
                spec = spec.model_copy(
                    update={"analysis_window_s": (max(0.0, t0), min(duration, t1))}
                )
                ing = ingest_mod.make_analysis_clip(
                    ctx.run_dir, ing, spec.analysis_window_s, max_s=settings.max_analysis_s
                )
                # the clip offset is needed to read masks back in the overlay and in replays
                write_json_atomic(ctx.run_dir / "ingest" / "ingest.json", ing)
        if stopped:
            # 06 §2: unknown experiment or low confidence ends the run before measurement;
            # 04 §1: skip to RENDERING with the recording-feedback page.
            ctx.emit("warning", message="Planner could not identify the experiment; re-record")
            write_json_atomic(ctx.run_dir / "ingest" / "ingest.json", ing)
            _store_verdict(ctx, _planner_stop_verdict(spec), None)
            with ctx.stage(Stage.RENDERING):
                _register_outputs(ctx, await asyncio.to_thread(outputs.render_outputs, ctx.run_dir))
            ctx.finish("NEEDS_RERECORD")
            return ctx.state
        if _reached(Stage.PLANNING, until):
            ctx.finish(None, partial=True)
            return ctx.state

        with ctx.stage(Stage.SEGMENTING):
            seg = await segment.run(
                ctx.run_dir,
                spec,
                ing,
                emit=ctx.emit,
                max_extra_requests=settings.max_extra_sam_requests,
            )
            write_json_atomic(
                ctx.run_dir / "sam" / "summary.json",
                {
                    "target_concept_used": seg.target_concept_used,
                    "attempts": [a.model_dump() for a in seg.attempts],
                    "warnings": seg.warnings,
                },
            )
        if _reached(Stage.SEGMENTING, until):
            ctx.finish(None, partial=True)
            return ctx.state

        with ctx.stage(Stage.MEASURING):
            observations, summary = measure_observations(ctx, spec, ing, seg)
        if _reached(Stage.MEASURING, until):
            ctx.finish(None, partial=True)
            return ctx.state

        if spec_override is not None:
            with ctx.stage(Stage.ANALYZING):
                # --no-planner: deterministic analysis without an LLM (M3/M4), no report
                headline = analysis_auto.run(ctx, spec, ing, seg, observations, summary)
            if _reached(Stage.AUDITING, until):  # asked for a review or report: say why not
                ctx.emit("warning", message="No Auditor or report for --no-planner runs")
            ctx.finish(None, headline, partial=True)
            return ctx.state

        try:
            review = await _review(ctx, spec, ing, seg, observations, summary, until)
        except analysis_auto.AnalysisError as exc:
            review = _early_rerecord(ctx, spec, exc)
        if review.stopped_after_analyzing:
            ctx.finish(None, review.headline, partial=True)
            return ctx.state
        # The verdict is final here: store it before anything is rendered (audit fix).
        _store_verdict(ctx, None, review)
        headline = ctx.state.headline  # None unless PASS / PASS_WITH_WARNINGS (06 §8)
        if _reached(Stage.AUDITING, until):
            ctx.finish(review.status, headline, partial=True)
            return ctx.state
        with ctx.stage(Stage.DESIGNING):
            # 04 §1: only for PASS / PASS_WITH_WARNINGS; before RENDERING so the report has it
            if review.status in PASSING and review.session is not None:
                gate_warnings = [g.message_en for g in review.gates if g.status in ("warn", "fail")]
                await designer.run(
                    review.session, headline=review.headline, gate_warnings=gate_warnings
                )
        if _reached(Stage.DESIGNING, until):
            ctx.finish(review.status, headline, partial=True)
            return ctx.state
        with ctx.stage(Stage.RENDERING):
            await _translate(ctx, review)
            rendered = await asyncio.to_thread(outputs.render_outputs, ctx.run_dir)
            _register_outputs(ctx, rendered)
        ctx.finish(review.status, headline)
        return ctx.state
    except Exception as exc:  # noqa: BLE001
        ctx.fail(exc)
    return ctx.state


PASSING = ("PASS", "PASS_WITH_WARNINGS")


@dataclass
class Review:
    """Outcome of the Analyst, Auditor and revise rounds (04 §1, 06 §8)."""

    session: AnalysisSession | None
    status: str | None
    headline: dict[str, Any] | None
    draft: ReportDraft | None
    gates: list[GateResult]
    stopped_after_analyzing: bool = False


async def _review(
    ctx: RunContext,
    spec: ExperimentSpec,
    ing: IngestResult,
    seg: segment.SegmentationOutput,
    observations: list[Observation],
    summary: dict[str, Any],
    until: Stage | None,
) -> Review:
    """Analyst → gates → Auditor, repeated while the Auditor says REJECT (04 §1)."""
    session: AnalysisSession | None = None
    fix: list[str] = []
    thresholds = REGISTRY[spec.experiment].gates
    headline: dict[str, Any] | None = None
    draft: ReportDraft | None = None
    gates: list[GateResult] = []
    verdict: Verdict | None = None
    used_fallback = False
    outcome = None
    for round_ in range(ctx.settings.max_revise_rounds + 1):
        with ctx.stage(Stage.ANALYZING):
            if session is None:
                session = session_mod.prepare(ctx, spec, ing, seg, observations, summary)
            session.round = round_
            outcome = await analyst.run(session, fix_instructions=fix)
            used_fallback = used_fallback or outcome.fallback
            if outcome.fallback:
                headline = ctx.state.headline
            else:
                headline = session.headline()
                ctx.state.headline = headline or ctx.state.headline
            draft = outcome.draft if outcome.kind == "draft" else None
        if round_ == 0 and _reached(Stage.ANALYZING, until):
            return Review(session, None, headline, draft, [], stopped_after_analyzing=True)
        with ctx.stage(Stage.AUDITING):
            gates = gates_mod.run_gates(session, thresholds)
            if outcome.kind == "rerecord":
                verdict = await auditor.review_rerecord(
                    session, outcome.rerecord, gates, round_=round_
                )
            else:
                verdict = await auditor.run(session, outcome.draft, gates, round_=round_)
        if verdict.status != "REJECT":
            break
        ctx.state.revise_rounds = round_ + 1
        fix = verdict.fix_instructions
    assert verdict is not None and session is not None  # the loop runs at least once
    if verdict.status == "REJECT" and not any(g.hard and g.status == "fail" for g in gates):
        # Rounds are used up and only the text (G5) or a refused re-record request is left:
        # the deterministic report replaces the text, so no unverified wording is printed.
        last_is_text = outcome is not None and (outcome.kind == "rerecord" or any(
            g == "G5" for g in verdict.gate_failures))  # fmt: skip
        if last_is_text:
            with ctx.stage(Stage.ANALYZING):
                outcome = analyst.fallback(session, steps=0)
                used_fallback = True
                headline = ctx.state.headline
                draft = outcome.draft
    status, extra = auditor.finalize_verdict(verdict, gates, fallback=used_fallback)
    for message in extra:
        ctx.emit("warning", message=message)
    path = ctx.run_dir / "analysis" / "verdict.json"
    write_json_atomic(path, verdict.model_copy(update={"status": status}))
    ctx.add_artifact("verdict", path, "application/json")
    ctx.emit("verdict", status=status, final=True, revise_rounds=ctx.state.revise_rounds)
    _slowmo_hint(ctx, gates)
    if status not in PASSING:
        draft = None  # NEEDS_RERECORD: the feedback page, not a lab report
    return Review(session, status, headline, draft, gates)


def _store_verdict(ctx: RunContext, verdict: Verdict | None, review: Review | None) -> None:
    """Write the final verdict to verdict.json (if given) and state.json before rendering."""
    if verdict is not None:
        path = ctx.run_dir / "analysis" / "verdict.json"
        write_json_atomic(path, verdict)
        ctx.add_artifact("verdict", path, "application/json")
        ctx.emit("verdict", status=verdict.status, final=True, revise_rounds=0)
        ctx.state.verdict = verdict.status  # type: ignore[assignment]
    if review is not None:
        ctx.state.verdict = review.status  # type: ignore[assignment]
        ctx.state.headline = review.headline if review.status in PASSING else None
    ctx.save()


def _planner_stop_verdict(spec: ExperimentSpec) -> Verdict:
    tips_en = [
        "FrameLab could not recognise the experiment in this video. Film one free fall or "
        "one swinging pendulum, with the whole motion in view, and say what you do."
    ]
    tips_id = ["FrameLab tidak dapat mengenali eksperimen di video ini. Rekam satu benda jatuh "
               "atau satu bandul yang berayun, seluruh gerakan terlihat, dan ucapkan apa yang "
               "Anda lakukan."]  # fmt: skip
    return Verdict(status="NEEDS_RERECORD", rerecord_tips_en=tips_en + list(spec.warnings),
                   rerecord_tips_id=tips_id, auditor_model="code",
                   notes=f"Planner stop: experiment {spec.experiment}, "
                   f"confidence {spec.confidence:.2f}")  # fmt: skip


def _early_rerecord(ctx: RunContext, spec: ExperimentSpec, exc: Exception) -> Review:
    """04 §1: no target or no scale → NEEDS_RERECORD with gate messages, not a crash."""
    ledger = Ledger(ctx.run_dir / "analysis" / "ledger.jsonl")
    diagnosis = getattr(exc, "diagnosis", "unusable_data")
    if diagnosis == "no_scale":
        gate = gates_mod.gate_scale(None, REGISTRY[spec.experiment].gates)
    else:
        gate = gates_mod.gate_no_target(spec.target_concept)
    gate = gates_mod.record(ledger, gate, ctx.emit)
    verdict = Verdict(status="NEEDS_RERECORD", gate_failures=[gate.gate],
                      rerecord_tips_en=[gate.message_en], rerecord_tips_id=[gate.message_id],
                      auditor_model="code", notes=f"stopped before analysis: {exc}")  # fmt: skip
    path = ctx.run_dir / "analysis" / "verdict.json"
    write_json_atomic(path, verdict)
    ctx.add_artifact("verdict", path, "application/json")
    ctx.emit("verdict", status="NEEDS_RERECORD", final=True, revise_rounds=0)
    return Review(None, "NEEDS_RERECORD", None, None, [gate])


def _slowmo_hint(ctx: RunContext, gates: list[GateResult]) -> None:
    """G4 slow-motion diagnosis: a re-run with the right capture fps fixes it (06 §7)."""
    for gate in gates:
        if gate.gate == "G4" and gate.diagnosis == "slowmo_retimed":
            fps = 30 * int(gate.metrics.get("slowmo_k", 0))
            path = ctx.run_dir / "analysis" / "rerun.json"
            write_json_atomic(path, {"capture_fps": fps, "diagnosis": gate.diagnosis})
            ctx.emit("warning", message=gate.message_en)


def _register_outputs(ctx: RunContext, rendered: outputs.RenderResult) -> None:
    """Artifacts for the rendered files, and a warning for each output that failed."""
    mimes = {".pdf": "application/pdf", ".png": "image/png", ".mp4": "video/mp4",
             ".json": "application/json", ".csv": "text/csv"}  # fmt: skip
    for name, path in sorted(rendered.paths.items()):
        ctx.add_artifact(name, path, mimes.get(path.suffix, "application/octet-stream"))
    for message in rendered.warnings:
        ctx.emit("warning", message=message)


async def _translate(ctx: RunContext, review: Review) -> None:
    """The draft in the other language (04 §9). Kept single-language if the check fails."""
    if review.draft is None or review.session is None:
        return
    dst: Lang = "id" if review.draft.lang == "en" else "en"
    translated, problem = await translator.translate(
        review.draft, review.session.ledger, dst=dst, emit=ctx.emit
    )
    if translated is None:
        ctx.emit("warning", message=f"translation to {dst} skipped: {problem}")
        return
    path = ctx.run_dir / f"draft_{dst}.json"
    write_json_atomic(path, translated)
    ctx.add_artifact(f"draft_{dst}", path, "application/json")


def measure_observations(
    ctx: RunContext, spec: ExperimentSpec, ing: IngestResult, seg: segment.SegmentationOutput
) -> tuple[list[Observation], dict]:
    """Observations for target (+merged split ids), reference and string; csv + debug image."""
    observations: list[Observation] = []
    clip_frames = (
        ingest_mod.count_frames(ctx.run_dir / ing.clip_path) if ing.clip_path else ing.n_frames
    )
    summary: dict = {"target": None, "reference": None, "string": None}

    if seg.target is not None:
        target_obs = measure.observations_from_result(
            seg.target.result,
            seg.target.concept,
            offset_frame=ing.clip_offset_frame,
            effective_fps=ing.effective_fps,
        )
        best, merged = measure.choose_target_track(target_obs, spec, clip_frames)
        if merged:
            ctx.emit("warning", message=f"Merged split SAM object ids {merged} into {best}")
            target_obs = [
                o.model_copy(update={"object_id": best}) if o.object_id in merged else o
                for o in target_obs
            ]
        stats = measure.object_stats(target_obs, clip_frames, spec.motion_axis)
        summary["target"] = {
            "concept": seg.target_concept_used,
            "object_id": best,
            "merged_ids": merged,
            "objects": stats,
            "first_mask_frame": stats[best]["first_frame"] if best else None,
        }
        observations += target_obs
    else:
        ctx.emit("warning", message="No usable target track: the run will need a re-recording")

    for key, out in (("reference", seg.reference), ("string", seg.string)):
        if out is None:
            continue
        obs = measure.observations_from_result(
            out.result,
            out.concept,
            offset_frame=0,
            effective_fps=ing.effective_fps,
            image_frame_index=out.frame_index,
        )
        observations += obs
        if obs:
            longest = max(obs, key=lambda o: o.major_axis_px)
            summary[key] = {
                "concept": out.concept,
                "frame_index": out.frame_index,
                "major_axis_px": longest.major_axis_px,
                "minor_axis_px": longest.minor_axis_px,
                "bbox": [
                    longest.bbox_left,
                    longest.bbox_top,
                    longest.bbox_right,
                    longest.bbox_bottom,
                ],
            }

    csv_path = ctx.run_dir / "observations.csv"
    measure.write_observations(csv_path, observations)
    ctx.add_artifact("observations", csv_path, "text/csv")
    write_json_atomic(ctx.run_dir / "measurement.json", summary)

    # Debug image: all centroids on the reference key frame (or the clip's middle frame).
    frame = (
        seg.reference.frame_index
        if seg.reference is not None
        else ing.clip_offset_frame + clip_frames // 2
    )
    png = ctx.run_dir / "ingest" / "keyframes" / f"frame_{frame:06d}.png"
    if not png.is_file():
        ingest_mod.extract_keyframe(ctx.run_dir / ing.video_path, frame, png)
    debug = measure.debug_centroids_png(
        png, observations, ctx.run_dir / "debug" / "centroids.png", seg.target_concept_used
    )
    ctx.add_artifact("debug_centroids", debug, "image/png")
    target = summary["target"] or {}
    n_target = target.get("objects", {}).get(target.get("object_id"), {}).get("n", 0)
    ctx.emit("progress", message=f"{len(observations)} observations ({n_target} target frames)")
    return observations, summary


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
