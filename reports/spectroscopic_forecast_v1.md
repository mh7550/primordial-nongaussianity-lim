# Spectroscopic Cross-Correlation Forecast — Runs 2a and 2b

**Branch:** `joint-lim-galaxy-forecast`
**Driver:** `scripts/spectroscopic_forecast.py`
**Results:** `data/spectroscopic_forecast.pkl`

## Scenarios

Same corrected pipeline as before (Steps A/B/D active; Step C outlier-tail broadening dropped since spec z has negligible scatter), with two new galaxy configurations:

- **2a — Real Euclid Hα spec (present-day):** n = 1900 gal/deg² over z ∈ [0.9, 1.8], 5 tomographic bins, σ_z/(1+z) = 10⁻³. Bias b_g(z) = 0.79(1+z) (Blanchard Hα fiducial).
- **2b — Hypothetical dense spec:** photo-density (30 gal/arcmin²) but spec-quality z (σ_z/(1+z) = 10⁻³), split into 50 narrow z-bins across [0, 2]. Bias b_g(z) = √(1+z).

## Three-way comparison — Wide, marginalised σ(f_NL)

| Configuration | N_bins | σ_z / (1+z) | n_gal | σ(f_NL, marg.) | r(LIM, gal) at ℓ = 10 | cross-block gain |
|:---|---:|---:|---:|---:|---:|---:|
| **Euclid photo (validated baseline)** | 10 | 0.05 | 30 gal/arcmin² | **11.06** | 0.665 | ×1.00 |
| **2a Real Euclid Hα spec** | 5 | 0.001 | 0.53 gal/arcmin² | **22.05** | **0.832** | ×1.00 |
| **2b Hypothetical dense spec** | 50 | 0.001 | 30 gal/arcmin² | **4.18** | 0.443 | ×1.00 |

## Full decomposition (all four scenarios × two configs)

### Real Euclid Hα spec (Run 2a)

| Config | σ_LIM | σ_gal | σ_joint | σ_quad | σ_joint_marg | r@ℓ=10 |
|:---|---:|---:|---:|---:|---:|---:|
| Deep | 341.85 (at LIM f_sky) → 680 (at joint) | 134.77 | 130.28 | 132.20 | 186.62 | 0.832 |
| Wide | 1241.43 → 1622 (at joint) | 7.95 | 7.95 | 7.95 | 22.05 | 0.832 |

**Deep marginalisation penalty ×1.43. Wide penalty ×2.77** — much larger than photo (×1.05 for Deep, ×1.29 for Wide) because we have only 5 galaxy biases to marginalise, and each carries less information density individually (lower total gal count).

### Hypothetical dense spec (Run 2b)

| Config | σ_LIM | σ_gal | σ_joint | σ_quad | σ_joint_marg | r@ℓ=10 |
|:---|---:|---:|---:|---:|---:|---:|
| Deep | 343.50 → 684 (at joint) | 34.68 | 34.64 | 34.64 | 62.08 | 0.443 |
| Wide | 1241.43 → 1622 (at joint) | **2.04** | **2.04** | 2.04 | **4.18** | 0.443 |

## Key physical findings

### 1. Narrower z-bins with same density unlocks Fourier modes

Compare **photo (10 bins, 30 gal/arcmin²)** at σ_marg = **11.06** vs **hypothetical dense (50 bins, 30 gal/arcmin²)** at σ_marg = **4.18**. Same total galaxy count, but spec-quality redshifts spread over 50 narrow bins gives a **×2.6 tighter constraint on f_NL**. The mechanism is textbook: the k^-2 scale-dependent bias signal has power along the line of sight as well as transverse to it; a survey that resolves the LOS direction with σ_z ~ 0.01 (spec) rather than σ_z ~ 0.1 (photo) can access ~10× more Fourier modes at any fixed k. Fisher information roughly scales with the number of accessible modes.

### 2. Density matters, but not more than z resolution here

Compare **real Hα spec (5 bins, 0.53 gal/arcmin²)** at σ_marg = **22.05** vs the photo baseline at 11.06. Even with much sharper z resolution, the real Hα spec loses to photo because it has ~57× fewer galaxies. The 5-bin restriction and low density combine to make it worse than photo overall.

### 3. Correlation coefficient tracks bin width, but total σ tracks number of modes

- Photo (broad bins, wide σ_z = 0.05): r = 0.665 — LIM and gal windows overlap moderately.
- **Real Hα spec (narrow bins, σ_z = 0.052 from bin-width):** r = **0.832** — bin width matches LIM σ_z = 0.12 well, giving highest r.
- Hypothetical dense (very narrow bins, σ_z = 0.012): r = 0.443 — each narrow bin overlaps only a fraction of the LIM Gaussian, so individual r is low. But there are 50 bins, so the aggregate cross information is large.

Higher r *per bin* does not automatically mean tighter σ — the total joint constraint depends on the *number* of correlated modes as well.

### 4. Cross-block gain is ×1.00 in every scenario

All three galaxy configurations show cross-block gain ≈ 1.00. **The LIM × galaxy multi-tracer Seljak-style benefit is not activating in any configuration tested,** including the tight-correlation real Hα spec case (r = 0.83). LIM's noise level relative to signal is the limiting factor; even when correlations with gal are strong, the LIM data doesn't have enough per-mode S/N to add non-trivial cross-cancellation information.

## Physical interpretation

**Does finer z-binning help?** Yes, dramatically — but not through LIM cross-correlation. The mechanism is standard: spec-quality z resolution unlocks LOS Fourier modes that photometric surveys average out. The result is a tighter *galaxy-only* Fisher, and the joint constraint tracks it because LIM contributes negligibly.

**Does the tightest r translate to the tightest σ(f_NL)?** No. The real Hα spec has the highest r (0.832) but the loosest σ (22.05) among the three, because it has 57× fewer galaxies. **σ(f_NL) is set by mode count and shot noise, not by cross-correlation coefficient.**

**Is there any scenario where LIM helps?** In every configuration tested, cross-block gain ≈ 1.00. LIM either doesn't help (photo, real spec, dense hypothetical) or adds independent auto-power (deep futuristic). Even the tight-correlation real Hα spec case, where LIM and gal see the same LSS through nearly the same windows, doesn't yield the cosmic-variance cancellation.

## Comparison to Planck (σ = 5.1)

| Config | σ_marg (Wide) | vs Planck |
|:---|---:|---:|
| Real Hα spec | 22.05 | 0.23× (much worse) |
| Photo baseline | 11.06 | 0.46× (worse) |
| Hypothetical dense spec | **4.18** | **1.22× (better)** |

Only the hypothetical dense spec configuration reaches sub-Planck sensitivity. **σ = 4.18 crosses σ = 5.1** and lands just below Planck. This is not a multi-field-discriminating measurement (σ = 1 threshold not reached), but it is a genuine improvement over the CMB constraint.

## For Prof. Pullen — the summary

- **The single most impactful survey design change is going spec-quality with the total galaxy count preserved.** A hypothetical Euclid Wide-scale survey with photo-density but spec-quality redshifts would give σ(f_NL) ≈ 4 — a real ~×1.2 improvement over Planck's σ = 5.1.
- **The real Euclid Hα spec survey (present-day) does NOT deliver this.** Its galaxy count is ~57× too low; σ_wide is 22, worse than photo.
- **The LIM cross-correlation contributes negligibly in every scenario tested.** Whether photo, real spec, or dense hypothetical spec — cross-block gain ≈ 1.00.
- **The correlation coefficient r is not a proxy for constraining power.** r = 0.83 (real Hα spec) but σ_marg = 22; r = 0.44 (dense hypothetical) but σ_marg = 4.18. Mode count dominates.

## Explicit assumptions

Same as the corrected joint forecast, plus:

- **Run 2a spec bins:** 5 uniform-in-z bins over z ∈ [0.9, 1.8], b_g(z) = 0.79(1+z), σ_z_eff dominated by top-hat bin width ≈ 0.052.
- **Run 2b spec bins:** 50 uniform-in-z bins over z ∈ [0, 2], b_g(z) = √(1+z) (photo fiducial), σ_z_eff ≈ 0.012.
- **Both scenarios**: no photo-z outlier broadening (Step C dropped); everything else (A, B, D) unchanged.
- **Marginalisation:** 4 cosmological + N_gal biases (5 for real spec, 50 for hypothetical). LIM A_i = B_i = 1 fixed as in the physical marginalisation baseline.

## Timing

- Real Hα spec (5 bins): ~4 s per ℓ for 9×9 marginalisation.
- Hypothetical dense spec (50 bins): ~40 s per ℓ for 54×54 marginalisation.
- Both configs each: ~5 min total wall time.
