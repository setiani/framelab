"""Orchestrator: executes the stages in order (docs/plan/04-pipeline-and-measurement.md §1).

Implemented so far (M2–M5): INGESTING → TRANSCRIBING → PLANNING → SEGMENTING → MEASURING →
ANALYZING: free fall by the Analyst agent (M6, deterministic fallback); pendulum and --no-planner
deterministic (M3). Later stages raise until their milestone lands.

Spec source: the Planner (default; Voice and Spark calls), or ``spec_override`` with
``--no-planner`` (flags only, no API calls for transcript or spec). ``overrides`` are typed flags
applied on top of the Planner's spec.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from framelab.agent import analyst, planner
from framelab.agent import session as session_mod
from framelab.agent.ledger import Ledger
from framelab.clients import meta as meta_client
from framelab.clients import sam as sam_client
from framelab.config import Settings
from framelab.pipeline import analysis_auto, measure, segment
from framelab.pipeline import ingest as ingest_mod
from framelab.pipeline import transcribe as transcribe_stage
from framelab.pipeline.context import RunContext
from framelab.schemas import (
    ExperimentSpec,
    IngestResult,
    Observation,
    RunState,
    Stage,
    Transcript,
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
    Stage.RENDERING,
    Stage.DESIGNING,
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
        if stopped:
            # 06 §2: unknown experiment or low confidence ends the run before measurement.
            ctx.emit("warning", message="Planner could not identify the experiment; re-record")
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

        rerecord = False
        with ctx.stage(Stage.ANALYZING):
            if spec_override is None and spec.experiment == "free_fall":
                # M6: the Analyst drives the tools; the deterministic path is its fallback (06 §3.4)
                session = session_mod.prepare(ctx, spec, ing, seg, observations, summary)
                outcome = await analyst.run(session)
                if outcome.fallback:
                    headline = ctx.state.headline
                else:
                    headline = session.headline()
                    ctx.state.headline = headline
                rerecord = outcome.kind == "rerecord"
                for name, rel in (("ledger", "analysis/ledger.jsonl"),
                                  ("draft_report", "draft_report.json")):  # fmt: skip
                    if (ctx.run_dir / rel).is_file():
                        ctx.add_artifact(name, ctx.run_dir / rel, "application/json")
            else:
                # M3: deterministic analysis: --no-planner (no LLM) and pendulum (agent in M6.1)
                headline = analysis_auto.run(ctx, spec, ing, seg, observations, summary)
        if rerecord:
            # Without the Auditor (M7) an Analyst request for re-recording is final.
            ctx.finish("NEEDS_RERECORD")
            return ctx.state
        if _reached(Stage.ANALYZING, until):
            ctx.finish(None, headline, partial=True)
            return ctx.state

        raise NotImplementedError("Stages after ANALYZING arrive in M7–M8.")
    except Exception as exc:  # noqa: BLE001
        ctx.fail(exc)
    return ctx.state


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
