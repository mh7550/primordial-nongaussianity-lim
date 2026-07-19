# Combined Scenario Summary — Five Configurations (v1)

**Branch:** `joint-lim-galaxy-forecast`
**Drivers:** `scripts/{joint_forecast_corrected,futuristic_forecast,spectroscopic_forecast,combined_futuristic_dense}.py`
**Result files:** `data/{joint_forecast_CORRECTED,futuristic_forecast,spectroscopic_forecast,combined_futuristic_dense}.pkl`
**Purpose:** Internal reference while awaiting Prof. Pullen. Not for paper prose.

## The five scenarios in one table

All rows use the same validated corrected pipeline: Steps A + B + D active on every block (Step C's photo-z outlier tail applied only where photo bins are used). ℓ ∈ {10, 20, 40, 80, 150, 250}. k_max = 0.3 h/Mpc. **14-parameter physical marginalisation** (4 cosmological + N_gal galaxy biases; LIM A_i = B_i = 1 fixed).

σ values reported at each block's *own* f_sky for LIM-only and gal-only; σ_quad computed with LIM rescaled to joint f_sky for a valid Fisher-monotonicity comparison.

| # | Scenario | Deep σ_marg | Wide σ_marg | Wide r@ℓ=10 | Wide cross-block gain | Wide F_joint / max(F_L, F_g) |
|---:|:---|---:|---:|---:|---:|---:|
| 1 | **Present-day photo** (validated baseline; 10 bins, σ_z=0.05, 30 gal/arcmin²) | 152.90 | 11.06 | 0.665 | ×1.00 | ×1.00 |
| 2 | **Present-day real Hα spec** (5 bins, σ_z=10⁻³, 0.53 gal/arcmin²) | 186.62 | 22.05 | 0.832 | ×1.00 | ×1.00 |
| 3 | **Futuristic + photo** (LIM σ_n×0.1, gal×10 density, 10 photo bins) | 41.74 | 6.27 | 0.665 | ×1.00 | ×1.03 |
| 4 | **Present-day dense spec** (50 spec bins, photo total density) | 62.08 | 4.18 | 0.443 | ×1.00 | ×1.00 |
| 5 | **Futuristic + dense spec** (LIM σ_n×0.1, 300 gal/arcmin², 50 spec bins) | **34.25** | **2.78** | 0.443 | ×1.00 | ×1.00 |

vs Planck (σ = 5.1):

| # | Scenario | Wide vs Planck |
|---:|:---|---:|
| 1 | Present-day photo | 0.46× (worse than Planck) |
| 2 | Real Hα spec | 0.23× (much worse) |
| 3 | Futuristic photo | 0.81× (close to Planck) |
| 4 | Present-day dense spec | 1.22× (better than Planck) |
| 5 | **Combined futuristic + dense spec** | **1.83× (best)** |

**Best-case ceiling: σ(f_NL) = 2.78 in Run 5 (Wide, marginalised).** No scenario crosses σ = 1.

## Did Run 5 show multiplicative stacking or new synergy?

Neither, precisely. See the stacking check for the multiplicative prediction:

| Config | Baseline (Run 1) | Run 3 (fut+photo) | Run 4 (dense spec) | Predicted mult | Actual Run 5 | Actual / mult |
|:---|---:|---:|---:|---:|---:|---:|
| Deep | 152.90 | 41.74 (×3.66) | 62.08 (×2.46) | 16.95 (×9.02) | 34.25 (×4.46) | **×2.02 looser** |
| Wide | 11.06 | 6.27 (×1.76) | 4.18 (×2.65) | 2.37 (×4.66) | 2.78 (×3.98) | **×1.17 looser** |

- **Multiplicative stacking (prediction a)** overpredicted the improvement by ×1.2 (Wide) to ×2.0 (Deep). The two individual improvements *overlap* because they both operate primarily on the galaxy side — Run 3 improves via ×10 density; Run 4 improves via 5× finer z-binning. Combining them doesn't double the gain because both are pushing the galaxy Fisher toward the same cosmic-variance floor.
- **Cross-block synergy (prediction b)** did not materialise. Cross-block gain remained ×0.96–1.00 in every scenario. The correlation coefficient r stayed at 0.44 in Run 5 (same as Run 4, because the LIM σ_z = 0.12 window still dominates the cross overlap regardless of galaxy binning). Narrower galaxy bins reduced *per-bin* r but did not activate a new multi-tracer channel.

**What Run 5 actually delivered:** the honest single most impactful change — replacing Euclid photo with a hypothetical Euclid-density spec sample — dropped σ_wide from 11.06 to 4.18 (Run 4). Adding the futuristic ×10 gal density and ×0.1 LIM noise on top of that (Run 5) delivered a further modest tightening to 2.78. Most of the improvement came from Run 4 already; Run 5's extra gain over Run 4 is only ×1.5 in Wide (not the ×2–4 the multiplicative estimate would suggest).

## Deep-config finding: LIM auto-power finally contributes meaningfully

The Wide config remains gal-dominated in every scenario (F_joint / max ≤ ×1.03). But the Deep config in Run 5 shows a real change:

- Present-day Deep (Run 1): F_joint / F_gal ≈ ×1.00 (LIM contributes nothing).
- Futuristic photo Deep (Run 3): F_joint / F_LIM = ×1.19 (LIM auto-power adds ~19%).
- **Combined futuristic + dense spec Deep (Run 5): F_joint / max(F_L, F_g) = ×1.44** — LIM's contribution to the joint Fisher is now genuinely non-trivial (44% larger than either single tracer alone).

This is Fisher-information *addition*, not multi-tracer cross-correlation cancellation. The mechanism: in Deep, the small survey area (200 deg² of overlap) limits the galaxy cosmic-variance floor to σ_gal ≈ 32; futuristic LIM at σ_LIM ≈ 43 (at joint f_sky) is now comparable, and their independent auto-powers add roughly in quadrature to give σ_joint ≈ 27. Cross-block gain still ×0.96 — the cross-correlation *itself* subtracts a small amount of the naive independent-tracer gain.

## Cross-block gain remains ≈ 1.00 across all five scenarios

This is the persistent negative result of this entire follow-up campaign. In no scenario tested — present-day, futuristic, real spec, dense spec, futuristic × dense spec — does the LIM × galaxy cross-block gain exceed ×1.00 by more than a percent. The Seljak-style multi-tracer cosmic-variance cancellation is not activating for the SPHEREx LIM × Euclid galaxy pairing at any noise level, tracer density, or z-resolution combination examined.

Two independent reasons appear to be operating together:
1. **LIM σ_z = 0.12 dominates the effective cross-window.** Even with 50 narrow galaxy bins (σ_z_gal = 0.012), the cross-overlap integral is dominated by the LIM Gaussian width. Narrower gal bins raise the *total* cross information (more bins) but do not raise the per-bin correlation coefficient meaningfully — narrow gal bins × wide LIM window still have gal-bin-width-limited overlap.
2. **LIM per-mode S/N is not the bottleneck for the cross-cancellation.** Even Run 5's LIM at 10× lower noise has σ_LIM_joint = 24 vs σ_gal_joint = 1.9 in Wide — LIM is still 12× noisier than gal on individual modes. Cross-cancellation Seljak-style requires *comparable* S/N in both tracers.

For cross-block gain to activate meaningfully, both LIM's σ_z would need to narrow (matching the gal windows) AND LIM's per-mode S/N would need to rise to be comparable to Euclid's. The σ_z change is the deeper physical constraint: it would require a SPHEREx-scale mission with proper spectroscopic-resolution channels (R ≳ 1000) rather than R ≈ 41–135.

## Marginalisation-penalty pattern

| Scenario | Deep penalty | Wide penalty |
|:---|---:|---:|
| Present-day photo | ×1.03 | ×1.02 |
| Present-day real Hα spec | ×1.43 | ×2.77 |
| Futuristic photo | ×1.05 | ×1.29 |
| Present-day dense spec | ×1.79 | ×2.05 |
| Futuristic + dense spec | ×1.27 | ×1.46 |

The dense-spec scenarios show larger marginalisation penalties (×1.4–×2.8) because we marginalise over one bias parameter per bin — 5 or 50 bias parameters for spec configs vs 10 for photo. Each free bias soaks up some cosmology-degenerate direction. **Larger N_gal_bins → larger marg penalty in the same-density regime.** In the futuristic dense case (Run 5), the penalty drops relative to the present-day dense (Run 4) because the increased density per bin makes each bias parameter better-constrained by its own auto-power.

## Cross-correlation coefficient r vs σ(f_NL)

Compiled from all scenarios (Wide, marginalised):

| Scenario | r@ℓ=10 | σ_marg | rank(σ) |
|:---|---:|---:|---:|
| Present-day real Hα spec (narrow bins, low density) | **0.832** | 22.05 | worst |
| Present-day photo (broad bins, high density) | 0.665 | 11.06 | 4 |
| Futuristic photo (broad bins, ×10 density) | 0.665 | 6.27 | 3 |
| Present-day dense spec (narrow bins, photo density) | 0.443 | 4.18 | 2 |
| **Combined futuristic + dense spec (narrow bins, ×10 density)** | **0.443** | **2.78** | **best** |

**r does not track σ(f_NL) at all.** The best-r configuration (real Hα spec) gives the worst σ; the two lowest-r configurations give the best σ. This confirms that in this regime, **σ(f_NL) is set by mode count and shot noise, not by cross-correlation strength.**

## For the paper (once Prof. Pullen returns)

- **The best-case ceiling for σ(f_NL) from an all-sky SPHEREx × Euclid joint forecast, with all four Blanchard corrections applied and a hypothetical 10–15 year survey configuration, is σ = 2.78** — approximately ×1.8 tighter than Planck's 5.1. This is real physical improvement but does not cross σ = 1.
- **The best case is dominated by the galaxy side.** LIM contributes ~0% to the Wide joint Fisher information in Run 5. The only Deep-config LIM contribution is via auto-power addition (F_ratio ×1.44), not via cross-correlation cancellation.
- **The multi-tracer Seljak-style cross-cancellation between SPHEREx LIM and Euclid galaxies does not activate under any physically defensible SPHEREx configuration examined.** This is a firm negative result across five diverse scenarios.

## Timing

Run 3 took 42 s/ℓ × 6 ℓ × 2 configs = **~8 min** for the 54×54 marginalised Fisher. Total wall time for the full 5-scenario campaign is now ~50 min of compute.
