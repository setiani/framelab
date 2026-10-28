# 05: Physics, fitting and uncertainty

All code in `backend/framelab/physics/`. Pure functions, numpy/scipy only, fully unit-tested
with synthetic data (08 §2). Inputs are SI-converted series. **Conventions:** image
y axis points down, so in free fall y increases with time and g comes out positive.

## 1. Experiment registry (`experiments.py`)

```python
@dataclass(frozen=True)
class ExperimentDef:
    name: ExperimentType
    motion_axis: Literal["x", "y", "xy"]
    primary_model: str  # "quadratic" | "damped_sine" | "linear_area"
    headline_quantity: str  # "g" | "period" | "melt_rate"
    gates: GateThresholds  # per-experiment thresholds (06 §7)
    min_valid_points: int
    plausible_range: tuple[float, float]  # for G4 on the headline
    required_scale: bool  # pendulum period needs no scale; g needs L


REGISTRY = {
    "free_fall": ExperimentDef(
        "free_fall",
        "y",
        "quadratic",
        "g",
        ...,
        min_valid_points=8,
        plausible_range=(9.0, 10.6),
        required_scale=True,
    ),
    "pendulum": ExperimentDef(
        "pendulum",
        "x",
        "damped_sine",
        "g",
        ...,
        min_valid_points=60,
        plausible_range=(9.0, 10.6),
        required_scale=False,
    ),
    "melting": ExperimentDef("melting", "xy", "linear_area", "melt_rate", ...),  # S-1
}
G_REF_DEFAULT = 9.781  # m/s², Jakarta (φ≈−6.2°, h≈8 m) via the International Gravity Formula
```
`g_ref(lat_deg, h_m) = 9.780327·(1 + 0.0053024·sin²φ − 0.0000058·sin²2φ) − 3.086e-6·h`
(used for comparison only, never to change data).

## 2. Free fall

### 2.1 Window detection (`events.py::freefall_window`)
Input: valid track points `t`, `y_m` (downward positive).
1. Smooth velocity: `v = savgol_filter(y, window=min(7, odd(n/5)), poly=2, deriv=1, delta=dt)`.
2. `v_max = max(v)`. **Impact** index `i_imp` = first index after `argmax(v)` where
   `v < 0.3·v_max` **or** where `y` reaches its global max (floor), whichever is first. If
   neither exists, the object left the frame: `i_imp = last valid index` and add a warning
   `object_left_frame`.
3. **Release** index `i_rel` = last index before `argmax(v)` where `v < 0.05·v_max`
   (object still in the hand). If none, the release happened before the first mask:
   `i_rel = first valid index`. **This is the normal case with SAM**, which reports a static
   object only from the frame after it starts moving (11 §1 V12): the first mask frame ≈
   release + 1 frame. Record `release_frame_estimate = first_mask_frame − 1` in the `win_`
   evidence. The quadratic fit keeps `v0` free, so the exact release instant is not needed.
4. Fit window = `[i_rel + 1, i_imp − 1]` (exclude the contact frames).
5. Require ≥ `min_valid_points` (8) valid points in the window. Else gate G1 fails with
   remedy `rerecord`, and the message suggests slow motion (`too_few_points`).
6. Bounces: later arcs (between impacts) are ignored in the MVP. Optional extra evidence:
   fit each later arc too (`fit_xx` with `arc=2,3…`).

### 2.2 Model and fit
`y(t) = c0 + c1·t + c2·t²`, with `t` shifted so the window starts at 0 (numerical
stability). Ordinary least squares (`np.polyfit(t, y_px, 2, cov=True)`) in **pixels**, so
fit quality is independent of scale.
- `a_px = 2·c2` (px/s²), `g = a_px · m_per_px`.
- Report `r2`, `rms_px`, `n_points`, `durbin_watson` of residuals, `max_abs_resid_px`.
- Initial velocity `v0 = c1·m_per_px` (useful sanity: ≈0 if released at the window start).

### 2.3 Residual pattern check (feeds G3)
- Durbin–Watson `DW = Σ(eᵢ−eᵢ₋₁)² / Σeᵢ²`. Values < 1.0 indicate strongly correlated
  residuals (systematic misfit: camera motion, perspective, drag) → warning or fail
  depending on thresholds.
- Also fit a cubic. If the cubic term is significant (|c3| > 3σ) and rms improves > 30 %,
  record `systematic_curvature=True`.

## 3. Pendulum

### 3.1 Signal
Use `x_px(t)` of the bob centroid (horizontal displacement). The angle is not needed for
period estimation.

### 3.2 Initial guesses (`models.py::guess_damped_sine`)
1. `c0 = median(x)`. Detrend: `x − c0`.
2. FFT with zero-padding ×8 (Hann window). Peak frequency f₀ with parabolic interpolation.
   Ignore f < 0.2 Hz and f > 5 Hz.
3. `A0 = (p95 − p5)/2`. `γ0` from a linear fit of `log(|hilbert(x−c0)|)` vs t, clipped
   to [0, 1] s⁻¹. `φ0` from the phase of the analytic signal at t₀.

### 3.3 Model and fit
`x(t) = A·e^{−γt}·cos(ω_d·t + φ) + c + d·t` (linear term `d` absorbs slow camera drift;
a large |d| is itself a quality signal).
Fit with `scipy.optimize.least_squares(method="trf", loss="linear")`, bounds A>0, γ≥0,
ω_d∈[2π·0.2, 2π·5]. Then:
- Damped period `T_d = 2π/ω_d`. Undamped period `T = 2π/√(ω_d² + γ²)` (difference is tiny
  but correct).
- **Amplitude correction:** `θ = asin(A_mean_m / L)` where `A_mean_m` = mean envelope
  amplitude over the window in metres (needs scale; if no scale, use `A_px / r_px` from
  the circle fit). Finite-amplitude series: `T₀ = T / (1 + θ²/16 + 11θ⁴/3072)`.
- `g = 4π²·L / T₀²`.
- `L` priority: spoken string length (pivot→bob centre) → measured `r_px·m_per_px` (circle
  fit + ruler scale) → none (then report T only, the headline becomes `period`, and G2 is
  skipped).
- Require ≥ 5 full cycles in the window (`n_cycles = duration/T`). Fewer gives a warning.
  Fewer than 3 fails G1 (`too_few_cycles`).

### 3.4 Hypothesis test helper (`derive.py::hypothesis_test`)
Given a prediction `p` (e.g. spoken "1.5 s") and a measurement `m` with CI `[lo, hi]`:
- `supported` if `lo ≤ p ≤ hi`
- `close` if `|p − m|/m ≤ 0.05`
- `not_supported` otherwise.
Return `{p, m, lo, hi, rel_diff_pct, outcome}` as `cmp_xx` evidence. The LLM only words it.

## 4. Series: T₀² vs L (`derive.py::pendulum_series`)

Inputs: ≥ 3 PASS runs with `(L_i, T0_i, σ_T_i, σ_L_i)`.
- Weighted linear regression `T₀² = k·L + b`, `σ_{T²,i} = 2·T0_i·σ_T_i` (L errors folded
  in as `k·σ_L`, effective variance). Use `np.polyfit(L, T2, 1, w=1/σ_eff, cov="unscaled")`.
- `g = 4π²/k`, `σ_g = g·σ_k/k`. Intercept `b` should be ≈ 0. If `|b| > 2σ_b`, flag
  `length_offset` (L probably measured to the wrong point).
- Power law: regress `ln T` on `ln L` → exponent `n ± σ_n`. Hypothesis "T ∝ √L" is
  supported if `|n − 0.5| ≤ 2σ_n`.
- Bootstrap over runs (pairs bootstrap, 2,000 resamples) for CI of g and n.

## 5. Melting ice (S-1, stretch)
Per object: `A(t)` in m² (area_px·s²). Fit linear `A = A0 − r·t` and exponential
`A = A0·e^{−λt}`. Choose by AIC. Headline: ratio of rates (salt vs no salt) with bootstrap
CI. Both objects must be tracked with distinct `object_id`s in one video.

## 6. Uncertainty (`uncertainty.py`)

### 6.1 Residual moving-block bootstrap (statistical part)
Residuals of a time series are autocorrelated, so resample them in blocks:
```python
def block_bootstrap(t, y, fit_fn, predict_fn, n=1000, block=None, seed=1234):
    rng = np.random.default_rng(seed)
    params = fit_fn(t, y)
    yhat = predict_fn(t, params)
    e = y - yhat
    L = block or max(3, int(round(len(e) ** (1 / 3))))
    starts = np.arange(len(e) - L + 1)
    out = []
    for _ in range(n):
        idx = np.concatenate(
            [np.arange(s, s + L) for s in rng.choice(starts, size=len(e) // L + 1)]
        )[: len(e)]
        out.append(fit_fn(t, yhat + e[idx]))  # warm-start curve fits from `params`
    return np.array(out)
```
- `n = settings.bootstrap_n` (1000) for free fall (polyfit is fast), `n = 300` for the
  pendulum (least_squares, warm start). The seed is fixed for reproducibility.
- CI95 = percentiles 2.5 / 97.5 of the derived quantity. `σ_stat = (hi − lo)/3.92`.

### 6.2 Systematic parts (propagated)
- Scale: `σ_s/s` from calibration (04 §6). For free fall `σ_g,scale = g·σ_s/s`.
- Length (pendulum): `σ_L` = 0.005 m if spoken, or measured via scale + circle-fit σ.
- Frame timing: CFR → 0. If the source was VFR, add 0.5 % relative and a warning.
- Combined: `σ_total = √(σ_stat² + Σσ_sys²)`, `CI95_total = value ± 1.96·σ_total`.
- Store both: `ci_xx.data = {stat_lo, stat_hi, sigma_stat, sigma_scale, sigma_len,
  sigma_total, lo, hi, method:"block_bootstrap+propagation", n, block, seed}`.

### 6.3 Pendulum g
`σ_g/g = √((σ_L/L)² + (2σ_T/T)² + (σ_corr)²)`. `σ_corr` = half of the amplitude
correction's size (conservative).

## 7. Number formatting (`uncertainty.py::fmt`)
- Round the uncertainty to 2 significant figures if its first digit is 1 or 2, else 1.
  Round the value to the same decimal place.
- English: `9.74 ± 0.12 m/s²`. Indonesian: decimal comma `9,74 ± 0,12 m/s²`.
- Intervals: `[9.62, 9.86]` / `[9,62; 9,86]`.
- Units displayed: m/s², s, m, cm (lengths < 1 m shown in cm), s⁻¹, %.
- This function is the **only** way numbers enter the report, overlay HUD and UI.

## 8. Synthetic test generators (`tests/synth.py`, used heavily)
- `make_freefall(g=9.78, m_per_px=0.0011, fps=240, n=110, noise_px=0.4, release_hold=20,
  bounce=True)` → t, y_px with a stationary segment, the fall, and a bounce.
- `make_pendulum(L=0.5, theta0_deg=10, gamma=0.03, fps=60, seconds=15, noise_px=0.5,
  drift_px_s=0)` → x_px, y_px on the true arc (exact nonlinear ODE via `solve_ivp`, not
  the small-angle formula, so the amplitude correction is actually tested).
- Synthetic SAM-like masks: draw filled circles/ellipses with motion blur into rasters
  with bounds, for measure/calibrate tests.

Acceptance targets on synthetic data (CI tests):
| Test | Target |
|---|---|
| free fall, noise 0.4 px, 240 fps | |g − 9.78| < 0.5 %; CI covers truth in ≥ 90 % of 50 seeds |
| free fall, 30 fps (≈13 pts) | completes, CI wider, G1 warns about few points |
| pendulum θ₀ = 20°, nonlinear ODE | with correction |g − true| < 0.3 %; without > 0.5 % |
| series 3 lengths | |g − true| < 0.5 %, n ≈ 0.5 ± 0.02 |
| slow-mo trap (fps wrong ×8) | G4 diagnosis `slowmo_retimed` with k = 8 |
