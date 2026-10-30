// Series view (#/series and #/series/<id>), docs/plan/07 §4.4: pendulum runs at several lengths,
// T₀² against L with the regression, g from the slope and the exponent of T ∝ L^n.
import { addSeriesRun, analyzeSeries, createSeries, getSeries, listRuns, listSeries } from "../api";
import { plotly, restyle } from "../components/charts";
import { headlineText, verdictChip } from "../components/common";
import { gauge } from "../components/readout";
import { clear, h, relativeTime } from "../dom";
import { fmt, fmtInterval, sigmaFromCi } from "../format";
import { getLang, t } from "../i18n";
import type { View } from "../main";
import type { RunState, SeriesResult, SeriesState } from "../types";

const G_REF = 9.781; // config.G_REF_DEFAULT (Jakarta), comparison only

function usable(r: RunState): boolean {
  const hl = r.headline;
  return r.experiment === "pendulum" && r.stage === "done" && !!hl && typeof hl.length_m === "number"
    && (r.verdict === "PASS" || r.verdict === "PASS_WITH_WARNINGS");
}

function seriesIndex(): View {
  const lang = getLang();
  const list = h("div", null, h("p.empty", null, "…"));
  const picker = h("div.stack", null, h("p.empty", null, "…"));
  const el = h("div.stack", null,
    h("div.hero-copy", null, h("h1", null, t("series_title")), h("p", null, t("series_intro"))),
    h("div.grid-even", null,
      h("section.panel", { "aria-labelledby": "sl" }, h("div.panel-head", null, h("h2#sl", null, t("series_list"))), list),
      h("section.panel", { "aria-labelledby": "sn" }, h("div.panel-head", null, h("h2#sn", null, t("new_series"))), picker)));
  listSeries().then((items) => {
    if (!items.length) return clear(list, h("p.empty", null, t("no_series")));
    clear(list, h("table.runs", null, h("tbody", null, ...items.map((s) => h("tr", null,
      h("td", null, h("a.num.small", { href: `#/series/${s.series_id}` }, s.series_id)),
      h("td.small", null, `${s.run_ids.length} runs`),
      h("td.small.num", null, s.headline && typeof s.headline.formatted_en === "string"
        ? `g = ${lang === "id" ? s.headline.formatted_id : s.headline.formatted_en}` : s.status),
      h("td.small.muted.hide-sm", null, relativeTime(s.created_at, lang)))))));
  }).catch(() => clear(list, h("p.empty", null, t("backend_down"))));
  listRuns(200).then((runs) => {
    const candidates = runs.filter(usable);
    if (!candidates.length) return clear(picker, h("p.empty", null, t("no_pendulum_runs")));
    const boxes = candidates.map((r) => h("label", { style: "display:flex;gap:10px;align-items:center" },
      h("input", { type: "checkbox", value: r.run_id }),
      h("span", null, r.original_filename, " ",
        h("span.num.tiny.muted", null, `L ${fmt(Number(r.headline!.length_m), null, "m", lang)} · ${headlineText(r)}`))));
    const msg = h("div", { role: "alert" });
    const button = h("button.btn.primary", { type: "button", onclick: async () => {
      const ids = boxes.map((b) => b.querySelector("input")!).filter((i) => i.checked).map((i) => i.value);
      try {
        const s = await createSeries(ids);
        location.hash = `#/series/${s.series_id}`;
      } catch (err) {
        clear(msg, h("div.error-box", null, (err as Error).message));
      }
    } }, t("new_series"));
    clear(picker, h("p.small.muted", null, t("select_runs")), ...boxes, msg, h("div", null, button));
  }).catch(() => clear(picker, h("p.empty", null, t("backend_down"))));
  return { el };
}

async function seriesChart(target: HTMLElement, res: SeriesResult): Promise<void> {
  const P = await plotly();
  const pts = [...res.points].sort((a, b) => a.L - b.L);
  const lo = Math.min(...pts.map((p) => p.L)) * 0.9;
  const hi = Math.max(...pts.map((p) => p.L)) * 1.05;
  const xs = Array.from({ length: 50 }, (_, i) => lo + ((hi - lo) * i) / 49);
  const line = xs.map((x) => res.linear.k * x + res.linear.b);
  const band = xs.map((x) => Math.hypot(x * res.linear.sigma_k, res.linear.sigma_b) * 1.96);
  await P.react(target, [
    { x: [...xs, ...xs.slice().reverse()], y: [...line.map((y, i) => y + band[i]), ...line.map((y, i) => y - band[i]).reverse()],
      fill: "toself", fillcolor: "rgba(0,184,217,0.15)", line: { width: 0 }, hoverinfo: "skip", name: "95%" },
    { x: xs, y: line, mode: "lines", line: { color: "#142027", width: 2 }, name: "T₀² = kL + b" },
    { x: pts.map((p) => p.L), y: pts.map((p) => p.T0 * p.T0), mode: "markers", name: "runs",
      marker: { color: "#00a3c2", size: 9 },
      error_y: { type: "data", array: pts.map((p) => 2 * p.T0 * p.sigma_T * 1.96), color: "#00718c" },
      error_x: { type: "data", array: pts.map((p) => p.sigma_L * 1.96), color: "#00718c" } },
  ], restyle({ xaxis: { title: { text: "L (m)" } }, yaxis: { title: { text: "T₀² (s²)" } }, title: { text: "" } }),
  { responsive: true, displaylogo: false });
}

function seriesDetail(id: string): View {
  const lang = getLang();
  const members = h("div", null, h("p.empty", null, "…"));
  const add = h("div");
  const resultBox = h("div.stack");
  const chart = h("div.chart");
  const analyzeBtn = h("button.btn.primary", { type: "button" }, t("analyze"));
  const msg = h("div", { role: "alert" });
  const el = h("div.stack", null,
    h("div.hero-copy", null, h("h1", null, t("series_title")), h("p.num.small.muted", null, id)),
    h("div.grid-2", null,
      h("section.panel", { "aria-labelledby": "sm" }, h("div.panel-head", null, h("h2#sm", null, t("members")), analyzeBtn),
        members, add, msg),
      h("section.panel.readout-panel", { "aria-labelledby": "sr" }, h("h2#sr.readout-label", null, t("slope_g")), resultBox)),
    h("section.panel", { "aria-label": "T₀² against L" }, chart));
  chart.hidden = true;
  let poll: number | undefined;

  async function load(): Promise<void> {
    const [series, runs] = await Promise.all([getSeries(id), listRuns(200)]);
    const byId = new Map(runs.map((r) => [r.run_id, r]));
    const rows = series.run_ids.map((rid) => {
      const r = byId.get(rid);
      const hl = r?.headline;
      return h("tr", null,
        h("td", null, h("a.file", { href: `#/run/${rid}` }, r?.original_filename ?? rid)),
        h("td.num.small", null, hl && typeof hl.length_m === "number" ? fmt(hl.length_m, (hl.length_sigma_m as number) ?? null, "m", lang) : "—"),
        h("td.num.small", null, hl && typeof hl.period0 === "number" ? fmt(hl.period0, (hl.period0_sigma as number) ?? null, "s", lang) : "—"),
        h("td", null, r ? verdictChip(r) : ""));
    });
    clear(members, rows.length
      ? h("table.runs", null, h("thead", null, h("tr", null, h("th", null, t("video")), h("th", null, t("length")),
          h("th", null, t("period0")), h("th", null, t("verdict")))), h("tbody", null, ...rows))
      : h("p.empty", null, t("series_empty")));
    const options = runs.filter((r) => usable(r) && !series.run_ids.includes(r.run_id));
    if (options.length) {
      const select = h("select.input", { "aria-label": t("series_pick") },
        h("option", { value: "" }, t("series_pick")),
        ...options.map((r) => h("option", { value: r.run_id }, `${r.original_filename} (L ${fmt(Number(r.headline!.length_m), null, "m", lang)})`)));
      const btn = h("button.btn", { type: "button", onclick: async () => {
        if (!select.value) return;
        try {
          await addSeriesRun(id, select.value);
          await load();
        } catch (err) {
          clear(msg, h("div.error-box", null, (err as Error).message));
        }
      } }, t("add_run"));
      clear(add, h("div", { style: "display:flex;gap:8px;margin-top:14px;flex-wrap:wrap" }, select, btn));
    } else clear(add);
    analyzeBtn.disabled = series.run_ids.length < 3 || series.status === "analyzing";
    analyzeBtn.textContent = series.status === "analyzing" ? t("analyzing") : t("analyze");
    renderResult(series);
    if (series.status === "analyzing") poll = window.setTimeout(() => void load(), 1500);
  }

  function renderResult(series: SeriesState): void {
    const res = series.results;
    if (series.status === "failed") {
      clear(resultBox, h("div.error-box", null, String(series.headline?.error ?? "failed")));
      return;
    }
    if (!res) {
      clear(resultBox, h("div.readout-value", null, h("span.sig", null, "—")), h("p.muted", null, t("series_empty")));
      chart.hidden = true;
      return;
    }
    const [lo, hi] = res.ci_g;
    const formatted = fmt(res.g, res.sigma_g, "m/s^2", lang);
    const [value, rest = ""] = formatted.split(" ± ");
    const [sig, ...unit] = rest.split(" ");
    clear(resultBox,
      h("div.readout-value", null,
        h("span.grp", null, h("span.sym", null, "g"), h("span", null, "="), h("span", null, value)),
        h("span.grp", null, h("span.pm", null, "±"), h("span.sig", null, sig), h("span.unit", null, unit.join(" ")))),
      gauge(res.g, lo, hi, G_REF, "m/s^2"),
      h("div.readout-facts", null,
        h("div.fact", null, h("span.k", null, t("interval")), h("span.v", null, fmtInterval(lo, hi, "m/s^2", sigmaFromCi(lo, hi), lang))),
        h("div.fact", null, h("span.k", null, t("exponent")), h("span.v", null, fmt(res.n, res.sigma_n, "", lang))),
        h("div.fact", null, h("span.k", null, t("sqrt_law")), h("span.v", null, res.sqrt_law_supported ? t("supported") : t("not_supported")))),
      ...res.flags.map((f) => h("p.small", null, h("span.chip.warn", null, f))));
    chart.hidden = false;
    void seriesChart(chart, res);
  }

  analyzeBtn.addEventListener("click", async () => {
    analyzeBtn.disabled = true;
    clear(msg);
    try {
      await analyzeSeries(id);
      await load();
    } catch (err) {
      clear(msg, h("div.error-box", null, (err as Error).message));
    }
  });
  load().catch(() => clear(members, h("p.empty", null, t("not_found"))));
  return { el, destroy: () => window.clearTimeout(poll) };
}

export function seriesView(id: string | null): View {
  return id ? seriesDetail(id) : seriesIndex();
}
