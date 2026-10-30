// Gates table, next experiment and downloads (07 §4.3, bottom row).
import { artifactUrl } from "../api";
import { clear, h } from "../dom";
import { getLang, t } from "../i18n";
import type { RunDetail } from "../types";

const STATUS_CLASS: Record<string, string> = { pass: "pass", warn: "warn", fail: "redo", skipped: "" };

export function gatesPanel() {
  const el = h("section.panel", { "aria-labelledby": "gates-title" });
  function update(run: RunDetail) {
    const lang = getLang();
    const head = h("div.panel-head", null, h("h2#gates-title", null, t("gates")));
    if (!run.gates.length) {
      clear(el, head, h("p.empty", null, t("waiting_result")));
      return;
    }
    const rows = run.gates.map((g) => h("tr", null,
      h("td.gid", null, g.gate),
      h("td", null, h(`span.chip${STATUS_CLASS[g.status] ? "." + STATUS_CLASS[g.status] : ""}`, null, g.status)),
      h("td", null, h("div", null, g.name, g.diagnosis ? h("span.tiny.muted", null, `  ${g.diagnosis}`) : null),
        h("div.small", null, lang === "id" ? g.message_id : g.message_en),
        Object.keys(g.metrics).length
          ? h("div.metrics", null, ...Object.entries(g.metrics).map(([k, v]) => h("span", null, `${k} ${v}`)))
          : null)));
    clear(el, head, h("table.gates", null,
      h("thead", null, h("tr", null, h("th", null, t("gate")), h("th", null, t("status")), h("th", null, t("message")))),
      h("tbody", null, ...rows)));
  }
  return { el, update };
}

export function nextPanel() {
  const el = h("section.panel.next", { "aria-labelledby": "next-title" });
  function update(run: RunDetail) {
    const p = run.next_experiment;
    el.hidden = !p;
    if (!p) return;
    const lang = getLang();
    const steps = lang === "id" ? p.protocol_steps_id : p.protocol_steps_en;
    const safety = lang === "id" ? p.safety_notes_id : p.safety_notes_en;
    const prediction = (lang === "id" ? p.prediction_id_rendered : p.prediction_en_rendered) ?? "";
    clear(el, 
      h("div.panel-head", null, h("h2#next-title", null, t("next"))),
      h("h3", { style: "margin-bottom:10px" }, p.title),
      h("dl.kv", null, h("dt", null, t("changed")), h("dd", null, p.changed_variable.replace("_", " ")),
        h("dt", null, t("controlled")), h("dd", null, p.controlled_variables.join(", "))),
      prediction ? h("p.pred", { style: "margin-top:12px" }, h("strong", null, `${t("predictions")}: `), prediction) : null,
      steps.length ? h("div", { style: "margin-top:12px" }, h("h3", null, t("protocol")), h("ol", null, ...steps.map((s) => h("li", null, s)))) : null,
      safety.length ? h("div", { style: "margin-top:12px" }, h("h3", null, t("safety")), h("ul", null, ...safety.map((s) => h("li", null, s)))) : null,
    );
  }
  return { el, update };
}

const DOWNLOADS: [string, string, "report_id" | "report_en" | "feedback_pdf" | "data_csv" | "overlay_mp4"][] = [
  ["report_id", "report_id.pdf", "report_id"],
  ["report_en", "report_en.pdf", "report_en"],
  ["feedback", "recording_feedback.pdf", "feedback_pdf"],
  ["data_csv", "data.csv", "data_csv"],
  ["overlay", "overlay.mp4", "overlay_mp4"],
];

export function downloadsPanel(runId: string) {
  const el = h("section.panel", { "aria-labelledby": "dl-title" });
  function update(run: RunDetail) {
    const items = DOWNLOADS.filter(([name]) => run.artifacts[name]);
    el.hidden = !items.length;
    clear(el, h("div.panel-head", null, h("h2#dl-title", null, t("downloads"))),
      h("div.downloads", null, ...items.map(([, file, label]) => h("a", { href: artifactUrl(runId, file), download: file },
        h("span", null, t(label)), h("span.ext", null, file.split(".").pop()!.toUpperCase())))));
  }
  return { el, update };
}
