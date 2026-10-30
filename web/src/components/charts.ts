// Charts (07 §4.3): the backend's Plotly figure (outputs/chart_trajectory.json), restyled to
// the interface. Plotly loads only when a chart is shown.
import { artifactUrl } from "../api";
import { h } from "../dom";
import { t } from "../i18n";
import type { RunState } from "../types";

export const FONT = { family: "Instrument Sans, system-ui, sans-serif", size: 13, color: "#33434c" };

export async function plotly() {
  return (await import("plotly.js-dist-min")).default;
}

export function restyle(layout: Record<string, unknown>): Record<string, unknown> {
  const axis = (a: unknown) => ({ ...(a as object), gridcolor: "#e9eeef", zerolinecolor: "#d9e0e2",
                                   linecolor: "#d9e0e2", tickfont: { family: "Martian Mono, monospace", size: 11 } });
  const out: Record<string, unknown> = { ...layout, font: FONT, paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "#ffffff", margin: { l: 60, r: 12, t: 64, b: 48 },
    legend: { orientation: "h", y: 1.0, yanchor: "bottom", x: 0, font: { size: 12 } }, template: undefined };
  for (const key of Object.keys(layout)) if (/^[xy]axis\d*$/.test(key)) out[key] = axis(layout[key]);
  const title = layout.title as { text?: string } | undefined;
  out.title = { text: title?.text ?? "", x: 0, xanchor: "left", font: { size: 14, color: "#142027" } };
  return out;
}

export function chartsPanel(runId: string) {
  const plot = h("div.chart");
  const empty = h("p.empty", null, t("charts_pending"));
  const el = h("section.panel", { "aria-labelledby": "chart-title" },
    h("div.panel-head", null, h("h2#chart-title", null, t("charts"))), empty, plot);
  plot.hidden = true;
  let loaded = "";
  const onResize = () => !plot.hidden && plotly().then((P) => P.Plots.resize(plot));
  window.addEventListener("resize", onResize);
  async function update(state: RunState) {
    const rel = state.artifacts.chart_trajectory_json;
    // A finished run without a fit (re-record before analysis) has no chart: hide the panel.
    el.hidden = !rel && (state.stage === "done" || state.stage === "failed");
    if (!rel || rel === loaded) return;
    loaded = rel;
    try {
      const fig = await (await fetch(artifactUrl(runId, "chart_trajectory.json"))).json();
      const P = await plotly();
      const traces = (fig.data as Record<string, unknown>[]).map((tr) => {
        const name = String(tr.name ?? "");
        if (tr.mode === "lines") return { ...tr, line: { color: "#142027", width: 2 } };
        if (String(tr.yaxis) === "y2") return { ...tr, marker: { color: "#00b8d9", size: 4 } };
        if ((tr.marker as { symbol?: string })?.symbol === "circle-open") return tr;
        return { ...tr, name, marker: { color: "#00a3c2", size: 5 } };
      });
      plot.hidden = false;
      empty.hidden = true;
      await P.react(plot, traces, restyle(fig.layout), { responsive: true, displaylogo: false,
        modeBarButtonsToRemove: ["lasso2d", "select2d"] });
    } catch {
      loaded = "";
    }
  }
  return { el, update, destroy: () => window.removeEventListener("resize", onResize) };
}
