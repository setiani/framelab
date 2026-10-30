// Small shared pieces: verdict chip, stage chip, experiment label, recording-guide dialog.
import { RECORDING_GUIDE } from "../content/recordingGuide";
import { h } from "../dom";
import { getLang, t, type Key } from "../i18n";
import type { RunState } from "../types";

export function verdictChip(state: Pick<RunState, "verdict" | "stage">): HTMLElement {
  if (state.stage === "failed") return h("span.chip.redo", null, t("failed"));
  if (state.verdict) {
    const cls = state.verdict === "PASS" ? "pass" : state.verdict === "PASS_WITH_WARNINGS" ? "warn" : "redo";
    return h(`span.chip.${cls}`, null, t(`v_${state.verdict}` as Key));
  }
  if (state.stage === "done") return h("span.chip", null, t("stage_done"));
  return h("span.chip.run", null, t("running"));
}

export function experimentLabel(exp: string | null | undefined): string {
  if (exp === "free_fall" || exp === "pendulum" || exp === "unknown") return t(`exp_${exp}` as Key);
  return t("exp_none");
}

export function guideDialog(): HTMLDialogElement {
  const guide = RECORDING_GUIDE[getLang()];
  const dlg = h("dialog", { "aria-labelledby": "guide-title" });
  const body = h("div.dlg", null, h("h2#guide-title", null, guide.title));
  for (const section of guide.sections) {
    body.append(h("section", null, h("h3", null, section.title),
      h("ul", null, ...section.items.map((item) => h("li", null, item)))));
  }
  body.append(h("div", null, h("button.btn", { type: "button", onclick: () => dlg.close() }, t("close"))));
  dlg.append(body);
  dlg.addEventListener("click", (e) => e.target === dlg && dlg.close());
  return dlg;
}

/** The headline as shown in lists: the backend's fmt string in the UI language. */
export function headlineText(state: RunState): string {
  const hl = state.headline;
  if (!hl) return "";
  const symbol = hl.quantity === "period" ? "T" : hl.quantity;
  const formatted = getLang() === "id" ? hl.formatted_id : hl.formatted_en;
  return formatted ? `${symbol} = ${formatted}` : "";
}
