// Port of backend physics/uncertainty.py fmt / fmt_interval / fmt_pct (docs/plan/05 §7).
// Numbers shown in the UI come pre-formatted from the API where possible; this is for the rest.
import type { Lang } from "./types";

const UNIT_DISPLAY: Record<string, string> = {
  "m/s^2": "m/s²", "m/s2": "m/s²", "s^-1": "s⁻¹", "1/s": "s⁻¹", "m^2": "m²", deg: "°",
};

export function unitDisplay(unit: string): string {
  return UNIT_DISPLAY[unit] ?? unit;
}

/** Python round(x, d) for the decimal places used here (half to even on exact ties). */
function roundTo(value: number, decimals: number): number {
  const f = 10 ** decimals;
  return Math.round(value * f) / f;
}

export function decimalsForSigma(sigma: number): number {
  if (!(sigma > 0) || !Number.isFinite(sigma)) return 3;
  const exponent = Math.floor(Math.log10(sigma));
  const leading = Math.floor(sigma / 10 ** exponent + 1e-9);
  const sigFigs = leading === 1 || leading === 2 ? 2 : 1;
  const decimals = sigFigs - 1 - exponent;
  const rounded = roundTo(sigma, decimals);
  if (rounded > 0 && Math.floor(Math.log10(rounded)) !== exponent) {
    return decimalsForSigma(rounded);
  }
  return Math.max(decimals, 0);
}

function num(value: number, decimals: number, lang: Lang): string {
  let text = value.toFixed(decimals);
  if (text.startsWith("-") && Number(text) === 0) text = text.slice(1);
  return lang === "id" ? text.replace(".", ",") : text;
}

/** "9.74 ± 0.12 m/s²" (EN) / "9,74 ± 0,12 m/s²" (ID). */
export function fmt(value: number, sigma: number | null, unit: string, lang: Lang = "en"): string {
  const u = unitDisplay(unit);
  const sep = u === "°" || u === "%" ? "" : " ";
  if (sigma === null || !Number.isFinite(sigma) || sigma <= 0) {
    const decimals = value === 0 ? 2 : Math.max(0, 2 - Math.floor(Math.log10(Math.abs(value))));
    return `${num(value, decimals, lang)}${sep}${u}`.trimEnd();
  }
  const d = decimalsForSigma(sigma);
  return `${num(value, d, lang)} ± ${num(sigma, d, lang)}${sep}${u}`.trimEnd();
}

/** "[9.62, 9.86] m/s²" / "[9,62; 9,86] m/s²". */
export function fmtInterval(lo: number, hi: number, unit: string, sigma: number, lang: Lang = "en"): string {
  const d = decimalsForSigma(sigma);
  const sep = lang === "id" ? "; " : ", ";
  return `[${num(lo, d, lang)}${sep}${num(hi, d, lang)}] ${unitDisplay(unit)}`.trimEnd();
}

export function fmtPct(value: number, lang: Lang = "en", decimals = 1): string {
  const sign = value > 0 ? "+" : "";
  return lang === "id" ? `${sign}${num(value, decimals, lang)} %` : `${sign}${value.toFixed(decimals)}%`;
}

/** σ of a normal distribution whose 95 % interval is [lo, hi]. */
export function sigmaFromCi(lo: number, hi: number): number {
  return (hi - lo) / 3.92;
}

/** Plain number with a fixed number of decimals in the UI language (cost, metrics). */
export function plain(value: number, decimals: number, lang: Lang = "en"): string {
  return num(value, decimals, lang);
}
