// Same cases as backend/tests/test_uncertainty.py (05 §7): the two fmt implementations agree.
import { describe, expect, it } from "vitest";
import { fmt, fmtInterval, fmtPct, sigmaFromCi } from "./format";

describe("fmt matches the backend", () => {
  const cases: [number, number | null, string, "en" | "id", string][] = [
    [9.7432, 0.1234, "m/s^2", "en", "9.74 ± 0.12 m/s²"],
    [9.7432, 0.1234, "m/s^2", "id", "9,74 ± 0,12 m/s²"],
    [9.8236, 0.0318, "m/s^2", "en", "9.82 ± 0.03 m/s²"],
    [1.4271, 1.96, "s", "en", "1.4 ± 2.0 s"],
    [0.56789, 0.0123, "m", "en", "0.568 ± 0.012 m"],
    [9.81, 0.0996, "m/s^2", "en", "9.81 ± 0.10 m/s²"],
    [123.4, 25.0, "s", "en", "123 ± 25 s"],
    [9.81, null, "m/s^2", "en", "9.81 m/s²"],
    [12.0, 0.5, "deg", "en", "12.0 ± 0.5°"],
    [1.55923, 0.00005, "s", "en", "1.55923 ± 0.00005 s"],
    [0.31942, null, "s", "id", "0,319 s"],
  ];
  for (const [value, sigma, unit, lang, expected] of cases) {
    it(`${value} ± ${sigma} ${unit} (${lang})`, () => {
      expect(fmt(value, sigma, unit, lang)).toBe(expected);
    });
  }

  it("formats intervals and percentages", () => {
    expect(fmtInterval(9.6234, 9.8611, "m/s^2", 0.12, "en")).toBe("[9.62, 9.86] m/s²");
    expect(fmtInterval(9.6234, 9.8611, "m/s^2", 0.12, "id")).toBe("[9,62; 9,86] m/s²");
    expect(fmtPct(0.4567, "en", 2)).toBe("+0.46%");
    expect(fmtPct(-1.26, "id", 1)).toBe("-1,3 %"); // exact binary ties (-1.25) round differently in JS
    expect(sigmaFromCi(-1.96, 1.96)).toBeCloseTo(1.0);
  });
});
