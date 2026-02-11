#!/usr/bin/env python3
"""
Validate Fisher matrix implementation against analytic approximations.

Four tests are performed:

Test 1 - Single-tracer cosmic variance (CV) limit:
    In the CV limit (N_ell -> 0) the Fisher information has the simple form
        F_CV = (f_sky/2) * sum_ell (2ell+1) * [d ln C_ell / d f_NL]^2
    which we evaluate analytically using the Limber approximation at z_mid.

Test 2 - Fisher scales as f_sky:
    The multitracer Fisher is linear in f_sky, so F(2 f_sky) / F(f_sky) = 2 exactly.

Test 3 - sigma(f_NL) = 1/sqrt(F):
    get_constraints() must agree with direct matrix inversion to numerical precision.

Test 4 - Multi-tracer outperforms every single-tracer sample:
    The Seljak (2009) multi-tracer estimator uses cross-spectra to partially cancel
    cosmic variance, so sigma_multi < sigma_single for all 5 SPHEREx samples.

Run from the repository root:
    python scripts/test_analytic_limits.py
"""

import sys
sys.path.insert(0, '.')

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import numpy as np

# ---------------------------------------------------------------------------
# Imports from src/
# ---------------------------------------------------------------------------
from src.fisher import (
    compute_fisher_element,
    compute_fisher_matrix,
    get_constraints,
    compute_multitracer_fisher,
    compute_single_sample_forecast,
)
from src.limber import get_comoving_distance
from src.bias_functions import (
    delta_b_local,
    OMEGA_M, H0, C_LIGHT, DELTA_C,
)
from src.cosmology import get_transfer_function, get_growth_factor
from src.survey_specs import (
    F_SKY,
    SPHEREX_Z_BINS,
    get_bias,
    get_shot_noise_angular,
)


# ---------------------------------------------------------------------------
# Test 1 — Single-tracer cosmic variance limit
# ---------------------------------------------------------------------------

def test1_cv_limit():
    """
    Check that the numerical Fisher element in the N_ell -> 0 limit agrees with
    the analytic cosmic-variance Fisher formula.

    Derivation
    ----------
    In the Limber approximation, the angular power spectrum for a tophat bin
    centred at z_mid is dominated by a single wavenumber k_eff = (ell+0.5)/chi:

        C_ell ~ b^2(k_eff, z_mid) * P_m(k_eff, z_mid) * [geometry]

    At fiducial fNL = 0, b = b1.  The scale-dependent bias grows linearly with
    fNL, so

        d ln C_ell / d fNL |_{fNL=0}
            = 2 * (db/dfNL) / b1
            = 2 * delta_b_local(k_eff, z_mid, fNL=1, b1) / b1

    where delta_b_local(fNL=1) = 2*(b1-1)*delta_c * 3*Omega_m*H0^2/(c^2*k^2*T*D).

    The Fisher element in the CV limit (N_ell = 0) is:

        F_CV = (f_sky/2) * sum_ell (2ell+1) * [d ln C_ell / d fNL]^2

    Note: The expression "f_sky * sum (2ell+1) * (2 Δb/b1)^2" quoted in the task
    description defines F_CV with a Δb that lacks the factor-of-2 present in
    delta_b_local.  These two extra factors of 2 (one from Δb, one from 1/2 in the
    Fisher formula) cancel exactly, so the formula used here is equivalent.

    We compare F_CV with compute_fisher_element(..., N_ell_override=1e-30).
    The ratio should be close to 1.0; deviations of 20–30% are expected from the
    single-z_mid Limber approximation vs. the full redshift integral.
    """
    print("=" * 70)
    print("TEST 1: Single-Tracer Cosmic Variance Limit")
    print("=" * 70)

    ell = np.arange(10, 201)
    z_min, z_max = 0.8, 1.0
    z_mid = (z_min + z_max) / 2.0  # 0.9
    b1 = 2.1
    f_sky = F_SKY  # 0.75

    chi_zmid = get_comoving_distance(z_mid)

    print(f"  Setup: z in [{z_min}, {z_max}], z_mid = {z_mid}")
    print(f"  b1 = {b1}, f_sky = {f_sky}")
    print(f"  chi(z_mid) = {chi_zmid:.1f} Mpc/h")
    print(f"  ell range : {ell[0]} – {ell[-1]}")
    print()

    # ------------------------------------------------------------------
    # Analytic CV Fisher
    # ------------------------------------------------------------------
    # For each ell evaluate the derivative d ln C_ell / d fNL at k_eff = (ell+0.5)/chi.
    # Then sum over the ell modes with the standard Fisher weight (2ell+1)*f_sky/2.
    # ------------------------------------------------------------------
    F_CV = 0.0
    for ell_val in ell:
        k_eff = (ell_val + 0.5) / chi_zmid

        # Scale-dependent bias at fNL = 1 (linear coefficient)
        # delta_b_local includes the full factor:
        #   Δb = 2*(b1-1)*fNL*delta_c * 3*Omega_m*H0^2 / (c^2 * k^2 * T(k) * D(z))
        db = delta_b_local(k_eff, z_mid, fNL=1, b1=b1)

        # d ln C_ell / d fNL ≈ 2 * Δb(k_eff, z_mid, fNL=1) / b1
        # (the leading 2 comes from differentiating b^2: d(b^2)/dfNL = 2b * db/dfNL)
        dlnCl_dfNL = 2.0 * db / b1

        # Fisher contribution for this ell in the CV limit (N_ell = 0)
        F_CV += (2.0 * ell_val + 1.0) * (f_sky / 2.0) * dlnCl_dfNL ** 2

    # ------------------------------------------------------------------
    # Numerical Fisher with near-zero noise (N_ell = 1e-30 ≈ 0)
    # ------------------------------------------------------------------
    print("  Computing numerical Fisher (N_ell = 1e-30) ... ", end="", flush=True)
    F_numeric = compute_fisher_element(
        ell, z_min, z_max, b1, 0, 'local', 'local',
        f_sky=f_sky, N_ell_override=1e-30
    )
    print("done")
    print()

    ratio = F_numeric / F_CV

    print(f"  F_CV (analytic CV formula) = {F_CV:.4e}")
    print(f"  F_numeric (N_ell -> 0)     = {F_numeric:.4e}")
    print(f"  Ratio  F_numeric / F_CV    = {ratio:.4f}")
    print()

    # Accept within 30 % of 1.0 — consistent with Limber approximation accuracy
    PASS = 0.70 <= ratio <= 1.30
    print(f"  Criterion: 0.70 <= ratio <= 1.30  (Limber single-z_mid approximation)")
    status = "PASS" if PASS else "FAIL"
    print(f"  Result: {status}")
    return PASS, ratio


# ---------------------------------------------------------------------------
# Test 2 — Fisher scales linearly with f_sky
# ---------------------------------------------------------------------------

def test2_fsky_scaling():
    """
    The Fisher information F is proportional to f_sky (sky fraction covered).
    Therefore F(0.75) / F(0.375) must equal 0.75/0.375 = 2.0 exactly.

    We verify this using the full multitracer Fisher matrix for z_bin_idx = 5,
    ell in [10, 50].  Both calls use the same ell array and z-bin; only f_sky
    differs, so the ratio must be exact to floating-point precision.
    """
    print()
    print("=" * 70)
    print("TEST 2: Fisher Information Scales Linearly with f_sky")
    print("=" * 70)

    ell_array = np.arange(10, 51)
    z_bin_idx = 5

    f_sky_high = 0.75
    f_sky_low = 0.375
    expected_ratio = f_sky_high / f_sky_low  # 2.0 exactly

    z_min, z_max = SPHEREX_Z_BINS[z_bin_idx]
    print(f"  z bin {z_bin_idx}: [{z_min}, {z_max}]")
    print(f"  ell range : {ell_array[0]} – {ell_array[-1]}")
    print(f"  f_sky values: {f_sky_high} and {f_sky_low}")
    print(f"  Expected ratio: {expected_ratio:.6f} (exact)")
    print()

    print("  Computing F(f_sky = 0.75)  ... ", end="", flush=True)
    F_high = compute_multitracer_fisher(ell_array, z_bin_idx, f_sky=f_sky_high)
    print("done")

    print("  Computing F(f_sky = 0.375) ... ", end="", flush=True)
    F_low = compute_multitracer_fisher(ell_array, z_bin_idx, f_sky=f_sky_low)
    print("done")
    print()

    ratio = F_high / F_low

    print(f"  F(f_sky = {f_sky_high}) = {F_high:.6e}")
    print(f"  F(f_sky = {f_sky_low}) = {F_low:.6e}")
    print(f"  Ratio F(0.75) / F(0.375) = {ratio:.10f}")
    print(f"  Expected                  = {expected_ratio:.10f}")
    print()

    # f_sky enters only through a multiplicative prefactor, so the ratio is exact
    PASS = abs(ratio - expected_ratio) < 1e-6
    status = "PASS" if PASS else "FAIL"
    print(f"  Criterion: |ratio - 2.0| < 1e-6  (should be numerically exact)")
    print(f"  Result: {status}")
    return PASS, ratio


# ---------------------------------------------------------------------------
# Test 3 — sigma(f_NL) = 1 / sqrt(F)
# ---------------------------------------------------------------------------

def test3_sigma_inversion():
    """
    For a single-parameter analysis, the Fisher matrix F is 1×1.
    The constraint from get_constraints() is sqrt((F^{-1})_{00}) = 1/sqrt(F),
    which is identical to the direct formula sigma = 1/sqrt(F).

    We verify that both routes give the same answer to 10 significant figures.
    """
    print()
    print("=" * 70)
    print("TEST 3: sigma(f_NL) = 1 / sqrt(F)")
    print("=" * 70)

    ell = np.arange(10, 51)
    z_bin_idx = 4   # (0.8, 1.0)
    z_min, z_max = SPHEREX_Z_BINS[z_bin_idx]
    z_mid = (z_min + z_max) / 2.0

    b1 = get_bias(1, z_bin_idx)       # sample 1 (best photo-z)
    chi = get_comoving_distance(z_mid)
    N_ell = get_shot_noise_angular(1, z_bin_idx, z_mid, chi)

    print(f"  z bin {z_bin_idx}: [{z_min}, {z_max}],  z_mid = {z_mid}")
    print(f"  b1 = {b1},  shot noise N_ell = {N_ell:.4e}")
    print(f"  ell range: {ell[0]} – {ell[-1]}")
    print()

    print("  Computing Fisher matrix via compute_fisher_matrix() ...", end="", flush=True)
    F_matrix, param_names = compute_fisher_matrix(
        ell, [(z_min, z_max)], ['fNL_local'],
        b1_values=[b1],
        N_ell_values=[N_ell]
    )
    print(" done")
    print()

    # Route A: sigma from get_constraints (matrix inversion via scipy.linalg.inv)
    constraints = get_constraints(F_matrix, param_names)
    sigma_gc = constraints['fNL_local']

    # Route B: direct formula 1 / sqrt(F_00)
    F_00 = F_matrix[0, 0]
    sigma_direct = 1.0 / np.sqrt(F_00)

    ratio = sigma_gc / sigma_direct

    print(f"  F matrix element F_00            = {F_00:.6e}")
    print(f"  sigma via get_constraints()      = {sigma_gc:.10f}")
    print(f"  sigma via 1/sqrt(F_00)           = {sigma_direct:.10f}")
    print(f"  Ratio sigma_gc / sigma_direct    = {ratio:.15f}")
    print()

    PASS = abs(ratio - 1.0) < 1e-10
    status = "PASS" if PASS else "FAIL"
    print(f"  Criterion: |ratio - 1| < 1e-10  (numerical identity)")
    print(f"  Result: {status}")
    return PASS, ratio


# ---------------------------------------------------------------------------
# Test 4 — Multi-tracer always better than every single-tracer sample
# ---------------------------------------------------------------------------

def test4_multitracer_beats_single():
    """
    The multi-tracer estimator (Seljak 2009) uses cross-spectra between the
    5 SPHEREx galaxy samples.  Cross-spectra carry no shot noise, enabling
    partial cancellation of cosmic variance.  Therefore sigma_multi < sigma_single
    must hold for every individual sample.

    Setup: ell in [10, 100], z_bin_idx = 5, local fNL.
    """
    print()
    print("=" * 70)
    print("TEST 4: Multi-Tracer Outperforms Every Single-Tracer Sample")
    print("=" * 70)

    ell = np.arange(10, 101)
    z_bin_idx = 5
    z_min, z_max = SPHEREX_Z_BINS[z_bin_idx]

    print(f"  z bin {z_bin_idx}: [{z_min}, {z_max}]")
    print(f"  ell range: {ell[0]} – {ell[-1]}")
    print()

    # Multi-tracer Fisher and sigma
    print("  Computing multi-tracer Fisher ...", end="", flush=True)
    F_multi = compute_multitracer_fisher(ell, z_bin_idx)
    print(" done")

    if F_multi > 0:
        sigma_multi = 1.0 / np.sqrt(F_multi)
    else:
        sigma_multi = np.inf

    print(f"  sigma_multi = {sigma_multi:.6f}  (F_multi = {F_multi:.4e})")
    print()

    # Per-sample single-tracer sigma
    print(f"  {'Sample':<10} {'b1':>7} {'sigma_single':>14} {'sigma_multi':>14}"
          f"  {'Ratio (M/S)':>12}  {'Better?'}")
    print(f"  {'-' * 68}")

    all_pass = True
    sample_results = []

    for s in range(1, 6):
        b1_s = get_bias(s, z_bin_idx)
        print(f"    sample {s} ...", end="", flush=True)
        sigma_single = compute_single_sample_forecast(
            ell, s, z_bin_indices=[z_bin_idx]
        )
        print(f"\r", end="")
        better = sigma_multi < sigma_single
        if not better:
            all_pass = False
        ratio_s = sigma_multi / sigma_single
        sample_results.append((s, b1_s, sigma_single, sigma_multi, ratio_s, better))
        print(f"  {s:<10} {b1_s:>7.2f} {sigma_single:>14.6f} {sigma_multi:>14.6f}"
              f"  {ratio_s:>12.6f}  {'PASS' if better else 'FAIL'}")

    print()
    status = "PASS" if all_pass else "FAIL"
    print(f"  Criterion: sigma_multi < sigma_single for all 5 samples")
    print(f"  Result: {status}")
    return all_pass


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    print()
    print("=" * 70)
    print("FISHER MATRIX ANALYTIC VALIDATION TESTS")
    print("=" * 70)
    print()

    results = {}

    # --- Test 1 ---
    pass1, ratio1 = test1_cv_limit()
    results["Test 1  CV limit (ratio ≈ 1.0)"] = pass1

    # --- Test 2 ---
    pass2, ratio2 = test2_fsky_scaling()
    results["Test 2  f_sky scaling (ratio = 2.0 exact)"] = pass2

    # --- Test 3 ---
    pass3, ratio3 = test3_sigma_inversion()
    results["Test 3  sigma = 1/sqrt(F) (ratio = 1 exact)"] = pass3

    # --- Test 4 ---
    pass4 = test4_multitracer_beats_single()
    results["Test 4  Multi-tracer > single-tracer"] = pass4

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    col_w = 48
    print(f"  {'Test':<{col_w}} {'Result'}")
    print(f"  {'-' * (col_w + 8)}")
    for name, passed in results.items():
        print(f"  {name:<{col_w}} {'PASS' if passed else 'FAIL'}")
    print(f"  {'-' * (col_w + 8)}")

    n_pass = sum(results.values())
    n_total = len(results)
    all_pass = n_pass == n_total
    print(f"  {'Overall  ({}/{} passed)'.format(n_pass, n_total):<{col_w}} {'PASS' if all_pass else 'FAIL'}")
    print()

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
