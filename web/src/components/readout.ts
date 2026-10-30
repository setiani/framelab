// The result readout: the headline value in large mono digits, and a gauge that draws the 95 %
// interval, the measured value and the reference on one scale (05 §7 numbers, 06 §8 headline).
import { clear, h, svg } from "../dom";
import { decimalsForSigma, fmt, fmtInterval, fmtPct, plain, sigmaFromCi, unitDisplay } from "../format";
import { getLang, t } from "../i18n";
import type { RunDetail } from "../types";

function splitFormatted(formatted: string): { value: string; sigma: string; unit: string } {
  const [left, rest = ""] = formatted.split(" ± ");
  const idx = rest.indexOf(" ");
  if (!rest) {
    const sp = left.indexOf(" ");
    return { value: sp > 0 ? left.slice(0, sp) : left, sigma: "", unit: sp > 0 ? left.slice(sp + 1) : "" };
  }
  return { value: left, sigma: idx > 0 ? rest.slice(0, idx) : rest, unit: idx > 0 ? rest.slice(idx + 1) : "" };
}

export function gauge(value: number, lo: number, hi: number, ref: number | null, unit: string): SVGElement {
  const lang = getLang();
  const W = 420;
  const H = 84;
  const pts = [lo, hi, value, ...(ref !== null ? [ref] : [])];
  let min = Math.min(...pts);
  let max = Math.max(...pts);
  const span = Math.max(max - min, Math.abs(value) * 1e-4);
  min -= span * 0.45;
  max += span * 0.45;
  const x = (v: number) => 14 + ((v - min) / (max - min)) * (W - 28);
  const root = svg("svg", { viewBox: `0 0 ${W} ${H}`, class: "gauge", role: "img" }) as SVGSVGElement;
  const d = decimalsForSigma(sigmaFromCi(lo, hi));
  root.setAttribute("aria-label", `${t("interval", lang)} ${fmtInterval(lo, hi, unit, sigmaFromCi(lo, hi), lang)}`);
  // 95 % interval band, then the scale under it, the measured value and the reference
  root.append(svg("rect", { x: x(lo), y: 24, width: Math.max(2, x(hi) - x(lo)), height: 22, rx: 3,
                            fill: "#00b8d9", "fill-opacity": 0.2, stroke: "#00b8d9" }));
  root.append(svg("line", { x1: 14, x2: W - 14, y1: 56, y2: 56, stroke: "#c3cdd0", "stroke-width": 1 }));
  for (let i = 0; i <= 4; i++) {
    const v = min + ((max - min) * i) / 4;
    root.append(svg("line", { x1: x(v), x2: x(v), y1: 56, y2: 61, stroke: "#c3cdd0" }));
    const label = svg("text", { x: x(v), y: 76, "text-anchor": i === 0 ? "start" : i === 4 ? "end" : "middle" });
    label.textContent = plain(v, Math.min(d + 1, 6), lang);
    root.append(label);
  }
  root.append(svg("line", { x1: x(value), x2: x(value), y1: 18, y2: 56, stroke: "#142027", "stroke-width": 2.5 }));
  if (ref !== null) {
    root.append(svg("line", { x1: x(ref), x2: x(ref), y1: 14, y2: 56, stroke: "#c4431c",
                              "stroke-width": 1.5, "stroke-dasharray": "3 3" }));
    const lbl = svg("text", { x: x(ref), y: 9, "text-anchor": x(ref) < 80 ? "start" : x(ref) > W - 80 ? "end" : "middle", class: "lbl" });
    lbl.textContent = `${t("reference", lang)} ${fmt(ref, null, unit, lang)}`;
    root.append(lbl);
  }
  return root;
}

export function readout() {
  const el = h("section.panel.readout-panel", { "aria-labelledby": "readout-title" });
  function update(run: RunDetail) {
    const lang = getLang();
    const hl = run.headline;
    if (!hl) {
      clear(el, h("h2#readout-title.readout-label", null, t("result")),
        h("div.readout-value", { "aria-hidden": "true" }, h("span.sig", null, "—")),
        h("p.muted", null, run.stage === "failed" ? t("error_title") : t("waiting_result")));
      return;
    }
    const symbol = hl.quantity === "period" ? "T" : hl.quantity;
    const parts = splitFormatted(lang === "id" ? hl.formatted_id : hl.formatted_en);
    const [lo, hi] = hl.ci95;
    const ref = typeof hl.ref === "number" ? hl.ref : null;
    const sigma = sigmaFromCi(lo, hi);
    const facts: [string, string][] = [[t("interval"), fmtInterval(lo, hi, hl.unit, sigma, lang)]];
    if (ref !== null) facts.push([t("reference"), fmt(ref, null, hl.unit, lang)]);
    if (typeof hl.rel_diff_pct === "number") facts.push([t("difference"), fmtPct(hl.rel_diff_pct, lang, 2)]);
    if (hl.experiment === "pendulum") {
      if (typeof hl.period0 === "number") facts.push(["T₀", fmt(hl.period0, (hl.period0_sigma as number) ?? null, "s", lang)]);
      if (typeof hl.length_m === "number") {
        facts.push([`L (${hl.length_source})`, fmt(hl.length_m, (hl.length_sigma_m as number) ?? null, "m", lang)]);
      }
      if (typeof hl.n_cycles === "number") facts.push(["n", plain(hl.n_cycles, 1, lang)]);
    }
    const hyp = run.claims.find((c) => c.claim_type === "hypothesis_test");
    const hypText = hyp ? (lang === "id" ? hyp.text_id : hyp.text_en) : null;
    clear(el, 
      h("h2#readout-title.readout-label", null, t("result")),
      h("div.readout-value", { role: "text", "aria-label": `${symbol} = ${parts.value} ± ${parts.sigma} ${parts.unit}` },
        h("span.grp", null, h("span.sym", null, symbol), h("span", null, "="), h("span", null, parts.value)),
        h("span.grp", null, parts.sigma ? h("span.pm", null, "±") : null,
          parts.sigma ? h("span.sig", null, parts.sigma) : null,
          h("span.unit", null, parts.unit || unitDisplay(hl.unit)))),
      gauge(hl.value, lo, hi, ref, hl.unit),
      ref !== null ? h("p.small", null, lo <= ref && ref <= hi ? t("inside") : t("outside")) : null,
      hl.experiment === "pendulum" && hl.quantity === "period" ? h("p.small.muted", null, t("period_only")) : null,
      h("div.readout-facts", null, ...facts.map(([k, v]) => h("div.fact", null, h("span.k", null, k), h("span.v", null, v)))),
      hypText ? h("p.hyp", null, h("strong", null, `${t("hypothesis")}: `), hypText) : null,
    );
  }
  return { el, update };
}
