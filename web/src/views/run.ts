// Run view (#/run/<id>), docs/plan/07 §4.3. One RunStore holds the latest state and the events.
import { ApiError, getRun, getTrack, subscribe } from "../api";
import { chartsPanel } from "../components/charts";
import { experimentLabel, verdictChip } from "../components/common";
import { feedbackCard } from "../components/feedback";
import { ledgerPanel } from "../components/ledger";
import { player } from "../components/player";
import { readout } from "../components/readout";
import { downloadsPanel, gatesPanel, nextPanel } from "../components/sections";
import { specCard } from "../components/spec";
import { timeline } from "../components/timeline";
import { trace } from "../components/trace";
import { clear, h } from "../dom";
import { plain } from "../format";
import { getLang, t } from "../i18n";
import type { View } from "../main";
import type { FrameLabEvent, RunDetail } from "../types";

const REFRESH_ON = new Set(["stage_completed", "artifact", "verdict", "claim_added", "gate_result", "done", "error"]);

export function runView(runId: string): View {
  const lang = getLang();
  const events: FrameLabEvent[] = [];
  let detail: RunDetail | null = null;
  let trackFps: number | undefined;

  const title = h("h1", null, "…");
  const meta = h("div.run-meta");
  const chip = h("span");
  const errorBox = h("div");
  const tl = timeline();
  const play = player(runId);
  const result = readout();
  const feedback = feedbackCard(runId);
  const spec = specCard();
  const charts = chartsPanel(runId);
  const tr = trace();
  const ledger = ledgerPanel(runId, (ids) => tr.highlight(ids));
  const gates = gatesPanel();
  const next = nextPanel();
  const downloads = downloadsPanel(runId);

  const el = h("div", null,
    h("div.run-head", null, title, chip),
    meta, h("div", { style: "height:14px" }), errorBox, tl.el,
    h("div.grid-2", null, play.el, h("div.stack", null, result.el, feedback.el, spec.el)),
    h("div", { style: "height:20px" }), charts.el,
    h("div", { style: "height:20px" }),
    h("div.grid-even", null, tr.el, ledger.el),
    h("div", { style: "height:20px" }),
    h("div.grid-even", null, gates.el, h("div.stack", null, next.el, downloads.el)));

  function render(run: RunDetail) {
    title.textContent = run.original_filename;
    clear(chip, verdictChip(run));
    clear(meta,
      h("span", null, h("b", null, experimentLabel(run.experiment))),
      h("span", null, `${t("cost")} `, h("b.num", null, `$${plain(run.cost_usd, 3, lang)}`)),
      h("span.num.tiny", null, run.run_id),
      run.demo ? h("span", null, t("sample_note")) : null);
    clear(errorBox, run.stage === "failed"
      ? h("div.error-box", { style: "margin-bottom:16px" }, h("strong", null, t("error_title")), h("div.small", null, run.error ?? ""))
      : null);
    tl.update(run, events);
    play.update(run, trackFps);
    const rerecord = run.verdict === "NEEDS_RERECORD";
    result.el.hidden = rerecord;
    feedback.el.hidden = !rerecord;
    if (rerecord) feedback.update(run);
    else result.update(run);
    spec.update(run);
    void charts.update(run);
    ledger.update(run.claims);
    gates.update(run);
    next.update(run);
    downloads.update(run);
    if (!trackFps && run.artifacts.track) {
      getTrack(runId).then((tv) => {
        trackFps = tv.fps;
        play.update(run, trackFps);
      }).catch(() => undefined);
    }
  }

  let timer: number | undefined;
  const refresh = () => {
    window.clearTimeout(timer);
    timer = window.setTimeout(async () => {
      try {
        detail = await getRun(runId);
        render(detail);
      } catch {
        /* keep the last state; the next event retries */
      }
    }, 350);
  };

  getRun(runId)
    .then((run) => {
      detail = run;
      render(run);
    })
    .catch((err) => {
      const notFound = err instanceof ApiError && err.status === 404;
      el.replaceChildren(h("section.panel.stack", null,
        h("h1", null, notFound ? t("not_found") : t("backend_down")),
        h("p", null, h("a", { href: "#/" }, t("back")))));
    });

  const unsubscribe = subscribe(runId, (e) => {
    events.push(e);
    tr.push(e);
    if (detail && (e.type === "stage_started" || e.type === "stage_completed")) {
      if (e.type === "stage_started") detail.stage = e.stage as RunDetail["stage"];
      tl.update(detail, events);
    }
    if (REFRESH_ON.has(e.type)) refresh();
  });

  return {
    el,
    destroy: () => {
      unsubscribe();
      window.clearTimeout(timer);
      charts.destroy();
    },
  };
}
