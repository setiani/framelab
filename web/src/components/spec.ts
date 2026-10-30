// Spec card (07 §4.3): what the Planner understood, with the transcript it read.
import { clear, h } from "../dom";
import { fmt } from "../format";
import { getLang, t } from "../i18n";
import type { RunDetail } from "../types";
import { experimentLabel } from "./common";

const KEY_LABEL: Record<string, { en: string; id: string }> = {
  drop_height: { en: "Drop height", id: "Tinggi jatuh" },
  string_length: { en: "String length", id: "Panjang tali" },
  ball_diameter: { en: "Ball diameter", id: "Diameter bola" },
  amplitude_deg: { en: "Amplitude", id: "Amplitudo" },
  mass: { en: "Mass", id: "Massa" },
};

export function specCard() {
  const el = h("section.panel", { "aria-labelledby": "spec-title" });
  function update(run: RunDetail) {
    const lang = getLang();
    const spec = run.spec;
    const head = h("div.panel-head", null, h("h2#spec-title", null, t("setup")));
    el.hidden = !spec && run.stage === "failed";
    if (!spec) {
      clear(el, head, h("p.empty", null, t("plan_pending")));
      return;
    }
    const rows: [string, Node | string][] = [
      [lang === "id" ? "Eksperimen" : "Experiment", experimentLabel(spec.experiment)],
      [t("target"), h("span.num.small", null, spec.target_concept)],
    ];
    if (spec.reference) {
      const len = spec.reference.length_m ? ` (${fmt(spec.reference.length_m, null, "m", lang)})` : "";
      rows.push([t("ruler"), `${spec.reference.concept}${len}`]);
    }
    const spoken = Object.entries(spec.spoken_params ?? {});
    if (spoken.length) {
      rows.push([t("spoken"), h("span", null, ...spoken.map(([k, q], i) => h("span", null,
        i ? "; " : "", `${KEY_LABEL[k]?.[lang] ?? k} `,
        h("span.num.small", null, `${q.value} ${q.unit}`),
        q.source_turn !== null ? h("span.muted.tiny", null, ` (${t("turn")} ${q.source_turn})`) : null)))]);
    }
    if (spec.hypothesis) rows.push([t("hypothesis"), `“${spec.hypothesis}”`]);
    const kv = h("dl.kv", null, ...rows.flatMap(([k, v]) => [h("dt", null, k), h("dd", null, v)]));
    const turns = run.transcript?.turns ?? [];
    const transcript = h("details", { style: "margin-top:14px" },
      h("summary", null, `${t("transcript")} (${turns.length})`),
      turns.length
        ? h("ol", { start: 0, style: "margin:8px 0 0;padding-left:28px;display:grid;gap:4px" },
            ...turns.map((turn) => h("li.small", null, turn.text)))
        : h("p.empty", null, t("no_transcript")));
    clear(el, head, spec.setup_description ? h("p.small.muted", { style: "margin-bottom:12px" }, spec.setup_description) : "",
      kv, transcript);
  }
  return { el, update };
}
