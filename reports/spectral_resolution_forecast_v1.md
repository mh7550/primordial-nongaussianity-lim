# Spectral-Resolution Forecast — Is R the LIM×Galaxy Cross-Cancellation Bottleneck?

**Branch:** `joint-lim-galaxy-forecast`
**Driver:** `scripts/spectral_resolution_forecast.py`
**Results:** `data/spectral_resolution_forecast.pkl`
**Purpose:** Test the Pullen hypothesis that upgrading SPHEREx to uniform R=100 (PRIMA/FIRESS-level) is what would finally activate multi-tracer LIM×galaxy cosmic-variance cancellation.

## Prediction stated before running

The Seljak multi-tracer benefit requires (i) similar per-mode S/N in both tracers and (ii) different Δb(k) response to f_NL. Improving LIM resolution alone narrows σ_z (0.12 → 0.02 at R=100), which:
- Makes LIM auto-C_ℓ signal larger per channel.
- Reduces cross-correlation r between LIM and wide Euclid photo bins.
- Adds independent channels via more Fourier modes.

**Prediction:** resolution alone does NOT activate cross-cancellation in Wide, because per-mode LIM S/N stays ~10× worse than Euclid gal. Cross-block gain stays ≈ 1.00. Wide σ_marg improves modestly (×1.3-2) via LIM auto-power. Deep gets a bigger LIM-side boost via auto-power addition (F_ratio > 1.05).

## Construction of the R=100 SPHEREx variant

- Same 0.75–5.0 μm range, **uniform R = 100** replacing SPHEREx's band-dependent R = 41/35/130.
- 4 lines × 30 log-uniform (in 1+z) redshift samples per line = **120 channels** (vs SPHEREx's ~92).
- σ_z per channel = (1+z)/R ∈ [0.012, 0.060] across the 120 channels (vs 0.12 flat native).
- Analytic maximum independent channels = R × ln((1+z_max)/(1+z_min)) per line: Hα gives 152, [OIII] 139, Hβ 136, [OII] 110 → ~537 total. Using 120 is a factor ~4 subsampling; each channel is treated as an independent Gaussian window at σ_z = (1+z)/R, so the aggregate Fisher tracks the sampled mode set. Increasing n_per_line beyond 30 would tighten σ_LIM by ~√(N_new / 120) ≈ 2× if fully sampled — the physics test does not depend on this factor.

## Noise interpolation — option (a) vs (b)

Two physically distinct choices:

- **Option (a) — direct interpolation of v28 σ_n(λ):** background-limited detector; noise per unit bandwidth is resolution-independent. Correct for SPHEREx's zodi + read-noise regime.
- **Option (b) — σ_n × √(R_new / R_native(λ)):** photon shot noise; narrower bandpasses collect fewer photons and are correspondingly noisier per channel. Correct for a cryogenic spectrometer like PRIMA/FIRESS.

Since the user asked specifically about PRIMA/FIRESS-level resolution and those are cold cryogenic spectrometers where photon noise dominates, **option (b) is the more physically correct default for the headline**. In bands 1–3 (R_native = 41) the R=100 noise per channel is inflated by √(100/41) = 1.56×. Option (a) is retained as an optimistic lower bound.

Both are reported for the Wide × photo case (sub-scenarios (a) and (b)) so the size of the noise-interpolation uncertainty is visible.

## Sub-scenario decomposition — Wide, marginalised

| Sub-scenario | σ_LIM | σ_gal | σ_joint | σ_quad | σ_joint_marg | cross-block gain | r@ℓ=10 |
|:---|---:|---:|---:|---:|---:|---:|---:|
| **BASELINE**: SPHEREx-native × Euclid photo (from c80555c) | 1241 | 7.32 | 7.31 | 7.32 | **11.06** | ×1.00 | 0.665 |
| **(a)** R=100 photon-scaled × photo | 199.9 → 261 | 7.32 | 7.30 | 7.31 | **10.85** | ×1.0007 | 0.471 |
| **(b)** R=100 direct-interp × photo | 199.9 → 261 | 7.32 | 7.30 | 7.31 | **10.45** | ×1.0015 | 0.471 |
| **(c)** R=100 × dense spec (50 bins) | 469.9 → 614 | 2.04 | 2.04 | 2.04 | **4.31** | ×1.0001 | 0.930 |

**σ values at LIM's own f_sky → after rescale to joint f_sky are the ones used for σ_quad and cross-block gain (Fisher-monotonicity-consistent comparison).**

Improvement over baseline in Wide σ_marg:
- (a) photon-scaled noise: ×1.02
- (b) direct-interp noise: ×1.06
- (c) R=100 × dense spec: ×2.57

## Deep-config decomposition

| Sub-scenario | σ_LIM_joint | σ_gal | σ_joint | σ_marg | cross-gain | F_joint/max |
|:---|---:|---:|---:|---:|---:|---:|
| **(a) Deep** R=100 photon-scaled × photo | 398 | 124 | 118.5 | **139.4** | ×1.006 | ×1.09 |
| **(c) Deep** R=100 × dense spec (50 bins) | 326 | 34.7 | 34.6 | **51.5** | ×0.998 | ×1.007 |

vs baseline present-day SPHEREx × photo Deep (152.9) and present-day dense spec Deep (62.1), R=100 tightens by ×1.10 (photo) and ×1.20 (dense spec).

## Step 4 diagnostic — r(LIM, gal) at three resolutions

Hα LIM channel nearest z=1 × best-overlapping Euclid photo bin, at ℓ=10:

| LIM resolution | σ_z_LIM | σ_z_gal | r | ratio to native |
|:---|---:|---:|---:|---:|
| SPHEREx native (flat σ_z = 0.12) | 0.120 | 0.142 | **0.970** | 1.000 |
| R = 100 uniform | 0.020 | 0.132 | **0.472** | 0.487 |
| R = 1000 uniform | 0.002 | 0.132 | **0.150** | 0.155 |

**Correlation scales inversely with LIM resolution** because narrower LIM windows sample much less of the wide Euclid photo z-range. This is the "mismatched-window suppression" — high LIM resolution against a photo tracer *reduces* the overlap integral in the cross-power, not increases it.

r only recovers to 0.930 in scenario (c) (R=100 × dense spec) where **both** tracers have narrow windows so their bin widths are matched.

## Physical verdict

**Spectral resolution is NOT the bottleneck for cross-block synergy.**

Across all five R=100 sub-configurations tested — Wide/Deep × photo/spec × noise-options a/b — the cross-block gain never exceeds ×1.006 (min ×0.998). This holds even in scenario (c) where r = 0.93 (LIM and gal windows are matched in width). Multi-tracer Seljak cancellation *does not activate* even under near-perfect window matching.

The bottleneck is the *per-mode signal-to-noise ratio* of the LIM tracer relative to the galaxy tracer. In every scenario tested — including the futuristic scenarios in the earlier campaign — LIM at joint f_sky remains at least 10–30× noisier than Euclid gal on individual modes. Correlation r just tells you the two tracers' fluctuations overlap; multi-tracer cancellation requires the two tracers to *each* deliver a per-mode measurement of comparable quality that can be *differenced* to cancel cosmic variance. When one tracer is essentially noiseless (Euclid) and the other essentially at the noise floor (LIM), the difference recovers nothing new.

**Where R=100 does help:** the LIM *auto-power* contribution to the joint Fisher grows because narrower windows raise the diagonal C_ℓ amplitude per channel and more channels are sampled. This is visible in the Deep-config F_ratio ×1.09 (photo) — LIM contributes 9% via auto-power addition, not cross-cancellation. In Wide, the galaxy tracer is so much better than LIM that even this auto-power addition is negligible (F_ratio ×1.00).

The earlier finding that "cross-cancellation just doesn't activate for SPHEREx × Euclid" is now *confirmed to be resolution-independent* — even PRIMA/FIRESS-level resolution doesn't change the story. A different explanation is not needed: **the S/N mismatch is the fundamental barrier**.

## Six-scenario summary table (Wide, marginalised) — updated

| # | Scenario | σ_marg | vs Planck 5.1 | cross-gain |
|---:|:---|---:|---:|---:|
| 1 | Present-day photo (validated baseline) | 11.06 | 0.46× | ×1.00 |
| 2 | Present-day real Hα spec | 22.05 | 0.23× | ×1.00 |
| 3 | Futuristic + photo | 6.27 | 0.81× | ×1.00 |
| 4 | Present-day dense spec (50 bins) | 4.18 | 1.22× | ×1.00 |
| 5 | Futuristic + dense spec | 2.78 | 1.83× | ×1.00 |
| **6a** | **R=100 SPHEREx × photo (option b)** | **10.85** | **0.47×** | **×1.001** |
| **6c** | **R=100 SPHEREx × dense spec** | **4.31** | **1.18×** | **×1.000** |

**Best case remains Run 5** (futuristic instrumentation × dense spec) at σ = 2.78. R=100 alone (scenario 6a) does not improve materially on the present-day photo baseline; R=100 combined with dense spec (6c) is essentially the same as Run 4 (present-day noise + dense spec), because the LIM contribution to the joint Fisher is still negligible either way.

## Timing

R=100 driver, all five configurations, total wall time ~35 min. 54-param dense-spec marginalisation dominates (~100 s per ℓ × 6 ℓ × 2 configs).

## Interpretation for Prof. Pullen

- **Upgrading SPHEREx to R=100 (PRIMA/FIRESS-level) does not unlock multi-tracer synergy.** The best-case Wide σ improves from 11.06 to 10.85 (photo) or from 4.18 to 4.31 (dense spec) — a ~2% shift; well within the marginalisation-scheme uncertainty.
- **The bottleneck is per-mode LIM S/N, not spectral resolution or window matching.** Even at r = 0.93 (window-matched), cross-block gain stays ×1.00.
- **What would unlock cross-cancellation:** a mission with both narrower σ_z (to match gal windows) AND per-mode S/N comparable to Euclid galaxies. The latter requires either a much longer integration time or a much larger collecting area — neither is achievable by resolution alone.
- **Consequence for the paper:** the "LIM barely moves the needle" finding is a robust, resolution-independent statement about SPHEREx × Euclid pairing. It generalises to any spectrophotometric intensity mapper at photon-noise-limited or background-noise-limited sensitivity paired with a modern optical/NIR galaxy survey.
