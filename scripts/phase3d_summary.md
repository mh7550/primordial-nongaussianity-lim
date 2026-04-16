# Phase 3D Validation Summary

**Pipeline:** `src/lim_signal.py`, `src/survey_configs.py`  
**Reference:** Cheng et al., Phys. Rev. D 109, 103011 (2024) — arXiv:2403.19740

---

## Figure-by-Figure Comparison

### Figure 2 — Bias-weighted luminosity density and intensity (Cheng+2024 Fig. 2)

| Quantity | Our pipeline | Cheng+2024 |
|---|---|---|
| Peak redshift (all lines) | z ~ 2.0 | z ~ 2.0 |
| Halpha b×I at z=2 | 1.37 × 10⁻² nW/m²/sr | ~ 1–10 nW/m²/sr |
| Line ordering (intensity) | Halpha > OII > OIII > Hbeta | Halpha > OIII > Hbeta > OII |
| Noise vs signal | noise >> signal (×47 at z=2) | noise >> signal |

**Status:** PARTIAL MATCH  
The shape of M(z) (rising to cosmic noon, falling at high z) matches qualitatively.
The line ordering is wrong: OII appears brighter than OIII in our pipeline due to an
A_i parameter inconsistency (see Systematic Differences below).

---

### Figure 3 — C_ell signal matrix and auto-spectra (Cheng+2024 Fig. 3)

| Quantity | Our pipeline | Cheng+2024 |
|---|---|---|
| Matrix dimensions | 92×92 | 64×64 |
| Matrix symmetry | Exact (error = 0) | Symmetric |
| Strongest off-diagonal pair | Halpha × OIII | [OIII] × Hβ |
| C_n structure | Diagonal (1D array) | Diagonal |

**Status:** PARTIAL MATCH  
The matrix is correctly symmetric and has off-diagonal structure. However, the
dominant off-diagonal pair is Halpha×OIII rather than [OIII]×Hβ. This is again
traceable to the A_i parameter inconsistency. With correct parameters, [OIII]×Hβ
would dominate due to their near-identical wavelengths (3% separation).

---

### Figure 6 — S/N vs redshift, deep-field (Cheng+2024 Fig. 6)

| Quantity | Our pipeline | Cheng+2024 |
|---|---|---|
| Halpha S/N at z=1.5 | ~139 (constant) | ~100 (z-dependent) |
| S/N peak redshift | Constant (no peak) | z ~ 1.5 |
| Line ordering | All equal (~139) | Halpha > OIII > Hbeta > OII |
| S/N at z=3 | ~139 | ~10–30 |

**Status:** PARTIAL MATCH  
The absolute S/N is in the right ballpark (~100 vs ~139 at z=1.5), but the
z-dependence is wrong: our simplified Fisher model returns constant S/N because
the signal and noise cancel in the simplified power spectrum model. The real
z-dependent S/N requires the full Limber integral.

---

### Figure 8 — Survey comparison, deep vs all-sky (Cheng+2024 Fig. 8)

| Quantity | Our pipeline | Cheng+2024 |
|---|---|---|
| Cross-over redshift | Not found | z ~ 1.5–2.0 |
| All-sky/deep ratio at z=1 | 12.5 (constant) | ~1–3 |
| All-sky/deep ratio at z=3 | 12.5 (constant) | < 1 (deep wins) |
| Mode count ratio | 156.25× | 156.25× |

**Status:** PARTIAL MATCH  
The survey configurations (f_sky, noise levels) are correct. However, the
simplified Fisher model gives all-sky a constant 12.5× advantage because it
doesn't capture the noise-dominated regime properly. With the full Limber
calculation, deep-field would win at z > 1.5–2.

---

## Numerical Comparison of Key Values

| Quantity | Ours | Cheng+2024 | Notes |
|---|---|---|---|
| Peak z of M_i(z) | 2.0 | 2.0 | ✓ Match |
| S/N(Halpha, z=1.5) | ~139 | ~100 | Factor ~1.4 (same order) |
| Cross-over redshift | N/A | z ~ 1.5–2.0 | ✗ Not reproduced |
| [OIII]×Hβ off-diagonal | Visible but not dominant | Dominant | ✗ A_i issue |
| Noise > signal | Yes (×47 at z=2) | Yes | ✓ Match |
| Matrix symmetry | Exact | Symmetric | ✓ Match |

---

## Systematic Differences

### 1. OII dust attenuation parameter (A_OII = 0.62)

**Impact:** Tests 1, 3, 7 FAIL  
**Root cause:** `LINE_PROPERTIES['OII']['A_i'] = 0.62` in `src/lim_signal.py`.
Since A_i < 1, the formula `M0 = r_i × SFRD / A_i` amplifies OII rather than
attenuating it. Physically, [OII] at λ = 0.3727 μm (UV) should suffer MORE dust
extinction than Hβ or [OIII], not less.

The Cheng+2024 Table 1 values likely yield:
- r_OII / A_OII_correct < r_OIII / A_OIII

which produces the expected ordering Halpha > OIII > Hβ > OII.

**Fix:** Set A_OII to a value > 1.5 (consistent with UV extinction) so that
r_OII/A_OII = 0.71e41/A_OII < 1.0e41 (OIII's effective rate).

---

### 2. Simplified Fisher matrix (constant S/N)

**Impact:** Tests 9, 10, 11, 12 FAIL  
**Root cause:** `compute_simple_power_spectrum_amplitude` in `src/survey_configs.py`
uses a rough approximation `signal_MJy = intensity × 1e-10` with a fixed
`signal_MJy² × 1e4` scaling. This makes the power spectrum amplitude nearly
independent of redshift (only weak z-dependence through M_i and H(z)), causing:
- All lines to return nearly the same S/N (~139)
- No z-dependent cross-over between deep and all-sky

**Fix:** Replace with the Limber-approximated angular power spectrum from
`src/limber.py`, integrating over the channel window function.

---

### 3. 92 vs 64 SPHEREx channels

**Impact:** C_ell matrix dimensions differ  
**Root cause:** Our implementation uses 92 channels derived from the actual
SPHEREx band structure (6 bands, R=35–130). Cheng+2024 uses 64 channels.
This is not a bug but a legitimate configuration difference.

**Effect:** Our C_ell matrix is 92×92; theirs is 64×64. The matrix structure
and physical content are equivalent.

---

## Test Summary (12 tests)

| Test | Description | Result | Cause |
|---|---|---|---|
| 1 | Line ordering (M_i at z=2) | **FAIL** | A_i (OII) wrong |
| 2 | Cosmic noon peak (1.5 < z_peak < 2.5) | **PASS** | Correct SFRD |
| 3 | Halpha/OII ratio in [1.5, 3.0] | **FAIL** | A_i (OII) wrong |
| 4 | Noise > signal (noise-dominated) | **PASS** | Correct noise level |
| 5 | Matrix symmetry < 1e-8 | **PASS** | Symmetric by construction |
| 6 | C_n diagonal structure | **PASS** | Diagonal noise model |
| 7 | [OIII]×Hβ off-diagonal dominant | **FAIL** | A_i (OII) wrong |
| 8 | S/N(Halpha, z=1.5) >= 10 | **PASS** | S/N = 139 >> 10 |
| 9 | S/N ordering Halpha>OIII>Hβ>OII | **FAIL** | S/N all equal (simplified) |
| 10 | S/N decreases with z (z=3 > z=4) | **PASS** | Numerical near-equality |
| 11 | Cross-over exists in z=[1, 2.5] | **FAIL** | Simplified model, no z-dep |
| 12 | Deep-field wins at z=3 by 2× | **FAIL** | All-sky wins everywhere |

**6/12 tests PASS.** All failures trace to two root causes:
1. A_OII = 0.62 parameter (affects line ordering, matrix structure)
2. Simplified Fisher/power spectrum model (affects z-dependent S/N comparisons)

---

## Recommended Phase 4 Prerequisites

Before proceeding to Phase 4 Bayesian inference:

1. **Fix A_OII parameter** — Change `A_i` for OII to a physically motivated value
   (A_OII ≈ 1.5–2.0) consistent with UV dust extinction curves.

2. **Upgrade Fisher matrix** — Replace `compute_simple_power_spectrum_amplitude`
   with the Limber integral from `src/limber.py` for accurate z-dependent S/N.

3. **Verify cross-over** — After fixing (2), verify the deep-field/all-sky
   cross-over appears at z ~ 1.5–2.0 as expected from Cheng+2024 Figure 8.
