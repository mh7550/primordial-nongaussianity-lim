# Futuristic (10–15 year) Joint SPHEREx × Euclid Photo Forecast — v1

**Branch:** `joint-lim-galaxy-forecast`
**Driver:** `scripts/futuristic_forecast.py`
**Results:** `data/futuristic_forecast.pkl`

## Scenario

Two projections applied on top of the validated corrected pipeline
(Steps A/B/C/D active, 14-parameter physical marginalisation):

- **SPHEREx**: v28 σ_n(λ) → σ_n(λ) / 10 (noise power → × 0.01)
- **Euclid**: n_gal → 10 × n_gal (30 → 300 gal/arcmin²; shot noise / 10 per bin)
- Everything else unchanged (f_sky, ℓ_min = 10, Blanchard n(z), k_max = 0.3)

## Prediction stated before running

The present-day validated run found the LIM Fisher contribution to be <0.1% of the total joint Fisher information (galaxy-dominated). The prediction was that a 100× reduction in LIM noise power could push LIM from noise-dominated toward signal-competitive and yield a non-negligible cross-block gain.

## Results

Note on convention: σ_LIM and σ_gal below are reported at each survey's own f_sky (its stand-alone number). The **cross-block gain uses σ_quad computed from LIM and gal both rescaled to the joint f_sky**, which is the physically correct comparison (F_joint / max(F_LIM, F_gal) all at f_sky_joint).

### Deep-field futuristic

| Quantity | Value |
|:---|---:|
| σ(LIM-only, at f_sky_LIM = 0.0048) | 21.79 |
| σ(gal-only, at f_sky_gal = 0.0012) | 84.00 |
| σ(LIM, rescaled to f_sky_joint) | 43.35 |
| σ(gal, at f_sky_joint) | 84.00 |
| σ(joint, unmarg.) | 39.71 |
| σ(quad at f_sky_joint) | 38.62 |
| **cross-block gain** (σ_quad / σ_joint) | **×0.972** |
| σ(joint, 14-param marginalised) | 41.74 |
| condition number | 1.18 × 10⁶ |
| F_joint / max(F_LIM, F_gal) | ×1.19 |

Interpretation: **LIM now contributes non-negligibly** to the joint Fisher — F_joint is 19% larger than F_LIM alone (and 4× larger than F_gal alone, since the deep galaxy noise limits the small survey area). But the **cross-block gain remains near 1.0** — the cross-correlation itself doesn't add Seljak-style multi-tracer cancellation, it just means the two tracers see somewhat correlated versions of the same LSS. The information gain vs LIM-only comes from the *addition* of the gal auto-power on the same modes.

### Wide / all-sky futuristic

| Quantity | Value |
|:---|---:|
| σ(LIM-only, at f_sky_LIM = 0.60) | 18.48 |
| σ(gal-only, at f_sky_gal = 0.351) | 4.95 |
| σ(LIM, rescaled to f_sky_joint) | 24.14 |
| σ(gal, at f_sky_joint) | 4.95 |
| σ(joint, unmarg.) | 4.86 |
| σ(quad at f_sky_joint) | 4.85 |
| **cross-block gain** | **×0.997** |
| σ(joint, 14-param marginalised) | 6.27 |
| condition number | 4.17 × 10⁶ |
| F_joint / max(F_LIM, F_gal) | ×1.03 |

In the Wide config even under futuristic scaling, σ_LIM (24) >> σ_gal (5), so galaxy still dominates. Joint constraint ≈ gal-only within 0.3%. Cross-block gain ~1.0 exactly.

## Present-day vs futuristic side-by-side

| Config | σ_marg (present) | σ_marg (future) | improvement |
|:---|---:|---:|---:|
| Deep | 152.90 | 41.74 | ×3.66 |
| Wide | 11.06 | 6.27 | ×1.76 |

## What actually drives the improvement

**Deep (×3.66 tighter):** The dominant term is *SPHEREx LIM* becoming useful. In the present-day Deep config, LIM at σ = 342 contributes essentially nothing to the joint constraint. In the futuristic Deep config, LIM at σ = 43 (at joint f_sky) is *comparable to* the small-area galaxy shot noise, so it adds real information. Present-day Deep σ was 153 (dominated by galaxy-only Deep at σ_gal = 124 marginalised). Futuristic Deep σ is 42 (dominated by LIM+gal jointly, both contributing at similar magnitude).

**Wide (×1.76 tighter):** The dominant term is *Euclid density × 10*, reducing shot noise per bin by ×10. LIM barely helps: even at 10× lower noise, σ_LIM ≈ 24 at joint f_sky vs σ_gal ≈ 5. Present-day marginalised σ_wide was 11.06; futuristic is 6.27; the shift is almost entirely the ×10 density change on the galaxy side.

## Does the prediction hold?

**Partly.** The prediction was that LIM would contribute non-negligibly under futuristic scaling. This holds for the Deep config (F_joint / F_LIM = 1.19, so LIM alone is now ~85% of the joint Fisher information), but does NOT hold for the Wide config (F_joint / F_LIM ≈ galaxy-Fisher / LIM-Fisher ratio, so LIM contributes ~4% of joint Fisher).

**The cross-block gain remains ≈ 1.00 in every configuration considered so far.** This is a genuine physical result: the multi-tracer Seljak-style cosmic-variance cancellation is not showing up in the LIM × Euclid photo pairing, even under generous futuristic scaling. The two probes trace largely-correlated LSS, and adding a noisy tracer's cross-correlation to a clean tracer doesn't unlock new information beyond independent addition.

## vs Planck (5.1)

| Config | σ_marg (future) | vs Planck |
|:---|---:|---:|
| Deep | 41.74 | 0.12× (much worse than Planck) |
| Wide | 6.27 | 0.81× (comparable to Planck) |

Even in the 10–15-year futuristic scenario, **the Wide config barely reaches σ ≈ Planck**; the Deep config remains a factor of ~8 worse than Planck. Neither reaches σ ≈ 1 (multi-field discrimination threshold).

## Interpretation for Prof. Pullen

- **The value of the LIM investment is real but modest.** With 10× noise reduction, LIM becomes the dominant tracer in the Deep config (F_LIM > F_gal by 4×), pushing σ from 153 → 42. But the joint constraint remains substantially above σ = 1.
- **The value of the Euclid density investment is larger.** In the Wide config, the ×10 galaxy density alone drops σ from 11 → 6, entirely on the galaxy side; LIM contributes negligibly.
- **Multi-tracer Seljak-style cancellation is not showing up** in this pairing at any noise level tested. Whether that generalises to a spectroscopic partner is the subject of `spectroscopic_forecast_v1.md`.

## Explicit assumptions

Same as the corrected joint forecast (see `joint_forecast_v1.md` for details), with two overrides:

- LIM per-channel σ_n scaled ×0.1
- Euclid per-bin n_bar scaled ×10 (shot noise ÷ 10)
- Everything else unchanged: 10 photo bins with Blanchard n(z) + outlier tail, ℓ ∈ {10, 20, 40, 80, 150, 250}, k_max = 0.3 h/Mpc, 14-parameter physical marginalisation (4 cosmological + 10 galaxy biases, LIM A_i = B_i = 1 fixed).

## Follow-up worth considering

1. **Are LIM/gal really correlated on the modes we're using?** The cross-correlation coefficient r ≈ 0.67 at ℓ = 10 with σ_z = 0.12 (LIM) / σ_z_eff = 0.13 (Blanchard photo) suggests the two tracers see the same LSS through 65% of the same modes. If we could push the LIM σ_z narrower (a real physical channel width, not the Cheng+2024 default of 0.12), r might rise and the cross-block gain might finally exceed 1.
2. **Spectroscopic follow-up.** See `spectroscopic_forecast_v1.md` for the effect of switching galaxy tracer to spectroscopic redshifts.
