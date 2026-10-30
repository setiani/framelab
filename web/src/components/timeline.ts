// Stage timeline (07 §4.3): the pipeline is a real sequence, so it is drawn as one.
import { h } from "../dom";
import { t, type Key } from "../i18n";
import type { FrameLabEvent, RunState, Stage } from "../types";

export const STAGES: Stage[] = [
  "ingesting", "transcribing", "planning", "segmenting", "measuring", "analyzing", "auditing",
  "designing", "rendering",
];

export function timeline() {
  const el = h("ol.timeline", { "aria-label": "Pipeline" });
  el.style.listStyle = "none";
  el.style.margin = "0 0 20px";
  el.style.padding = "0";
  function update(state: RunState, events: FrameLabEvent[]) {
    const durations = new Map<string, number>();
    const started = new Set<string>();
    let failedAt: string | null = null;
    for (const e of events) {
      if (e.type === "stage_started") started.add(e.stage);
      if (e.type === "stage_completed") durations.set(e.stage, (durations.get(e.stage) ?? 0) + Number(e.duration_s ?? 0));
      if (e.type === "error") failedAt = e.stage;
    }
    for (const [stage] of state.stage_history ?? []) started.add(stage);
    const finished = state.stage === "done" || state.stage === "failed";
    el.replaceChildren(...STAGES.map((stage) => {
      let cls = "";
      if (failedAt === stage || (state.stage === "failed" && !failedAt && state.stage_history?.at(-2)?.[0] === stage)) cls = "failed";
      else if (!finished && state.stage === stage) cls = "active";
      else if (durations.has(stage) || (started.has(stage) && finished)) cls = "done";
      const dur = durations.get(stage);
      return h(`li.step${cls ? "." + cls : ""}`, { "aria-current": cls === "active" ? "step" : null },
        h("span.bar"),
        h("span.label", null, cls === "active" ? h("span.spin", { "aria-hidden": "true" }) : null,
          t(`stage_${stage}` as Key)),
        h("span.dur", null, dur !== undefined ? `${dur < 10 ? dur.toFixed(1) : Math.round(dur)} s` : " "));
    }));
  }
  return { el, update };
}
