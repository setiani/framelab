// NEEDS_RERECORD card (07 §4.3): failed gates, annotated frame, tips, and the slow-motion fix.
import { artifactUrl, rerun } from "../api";
import { clear, h } from "../dom";
import { getLang, t } from "../i18n";
import type { RunDetail } from "../types";

export function feedbackCard(runId: string) {
  const el = h("section.panel.feedback", { "aria-labelledby": "fb-title" });
  function update(run: RunDetail) {
    const lang = getLang();
    const failed = run.gates.filter((g) => g.status === "fail");
    const messages = failed.map((g) => (lang === "id" ? g.message_id : g.message_en));
    const tips = (lang === "id" ? run.verdict_detail?.rerecord_tips_id : run.verdict_detail?.rerecord_tips_en) ?? [];
    const otherTips = tips.filter((tip) => !messages.includes(tip));
    const slow = run.gates.find((g) => g.diagnosis === "slowmo_retimed");
    const k = slow ? Number(slow.metrics.slowmo_k) : 0;
    const button = k
      ? h("button.btn.primary", { type: "button", onclick: async (e: Event) => {
          (e.currentTarget as HTMLButtonElement).disabled = true;
          const res = await rerun(runId, { capture_fps: 30 * k });
          location.hash = `#/run/${res.run_id}`;
        } }, `${t("rerun_fps")} ${30 * k} fps`)
      : null;
    const frame = run.artifacts.feedback_frame
      ? h("img", { src: artifactUrl(runId, "feedback_frame.png"), alt: t("what_failed"), loading: "lazy" })
      : null;
    clear(el, 
      h("h2#fb-title", null, t("feedback")),
      messages.length ? h("div", { style: "margin-top:12px" }, h("h3", null, t("what_failed")),
        h("ul", null, ...failed.map((g, i) => h("li", null, h("span.tag", null, g.gate), " ", messages[i])))) : null,
      frame,
      otherTips.length ? h("div", null, h("h3", null, t("tips")), h("ul", null, ...otherTips.map((tip) => h("li", null, tip)))) : null,
      button ? h("div", { style: "margin-top:14px" }, button) : null,
    );
  }
  return { el, update };
}
