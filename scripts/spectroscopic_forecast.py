"""
spectroscopic_forecast.py — Runs 2a and 2b: SPHEREx × Euclid spec.

Configurations, running the same corrected pipeline (A/B/D + 14-param
physical marginalisation; Step-C outlier tail dropped since spec doesn't
have photo-z broadening):

  2a  Real Euclid Hα spec (present-day):
        n = 1900 gal/deg², z ∈ [0.9, 1.8], 5 bins, σ_z/(1+z) ~ 1e-3.
  2b  Hypothetical dense spec (same total density as photo, spec z):
        n = 30 gal/arcmin², z ∈ [0, 2], 50 bins, σ_z/(1+z) ~ 1e-3.

Cross-block gain reported at the joint f_sky (correct comparison).
"""
import numpy as np
import sys, os, time, pickle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

import cosmology
from lim_channels import build_92_channels
from survey_specs import (build_euclid_spec_bins,
                           build_euclid_hypothetical_spec_bins,
                           EUCLID_F_SKY_DEEP, EUCLID_F_SKY_WIDE)
from limber import (clear_intensity_cache, compute_lim_cl_limber,
                    compute_gal_cl_limber, compute_lim_x_galaxy_cross_cl)
from joint_forecast_corrected import (
    ELL_GRID, K_MAX_H_PER_MPC, FNL_FID, DELTA_FNL, DELTA_NUISANCE,
    C_lim_kmax, C_gal_kmax, C_cross_kmax, joint_C_kmax,
    dC_dsigma8_block, dC_dns_block, dC_dOm_joint, _entries_z_bar,
    fisher_fNL_scalar,
)
from futuristic_forecast import reduced_marg_fisher


def sigma_from_F(F):
    return 1.0 / np.sqrt(F) if F > 0 else np.inf


def rescale_sigma_to_fsky(sigma, fsky_from, fsky_to):
    """σ scales as 1/√f_sky."""
    return sigma * np.sqrt(fsky_from / fsky_to)


def run_scenario(name, lim_ch, gal_bins, fsky_lim, fsky_gal, fsky_joint):
    print("\n" + "═" * 72)
    print(f"SCENARIO: {name}")
    print(f"  N_lim = {len(lim_ch)}   N_gal = {len(gal_bins)}   "
          f"f_sky (LIM,gal,joint) = ({fsky_lim}, {fsky_gal:.4f}, {fsky_joint:.4f})")
    print("═" * 72)
    N_lim = np.asarray([c['noise'] for c in lim_ch])
    N_gal = np.asarray([b['noise'] for b in gal_bins])

    # Individual Fishers at joint f_sky (for a valid cross-gain comparison)
    F_lim_joint = fisher_fNL_scalar(ELL_GRID,
        lambda ell, f: C_lim_kmax(ell, lim_ch, f), N_lim, fsky_joint)
    F_gal_joint = fisher_fNL_scalar(ELL_GRID,
        lambda ell, f: C_gal_kmax(ell, gal_bins, f), N_gal, fsky_joint)

    sigma_lim_own = sigma_from_F(fisher_fNL_scalar(ELL_GRID,
        lambda ell, f: C_lim_kmax(ell, lim_ch, f), N_lim, fsky_lim))
    sigma_gal_own = sigma_from_F(fisher_fNL_scalar(ELL_GRID,
        lambda ell, f: C_gal_kmax(ell, gal_bins, f), N_gal, fsky_gal))
    print(f"  σ(LIM-only)  at f_sky_LIM  = {sigma_lim_own:.3f}")
    print(f"  σ(gal-only)  at f_sky_gal  = {sigma_gal_own:.3f}")

    # Joint (unmarg)
    N_joint = np.concatenate([N_lim, N_gal])
    F_joint = fisher_fNL_scalar(ELL_GRID,
        lambda ell, f: joint_C_kmax(ell, lim_ch, gal_bins, f), N_joint, fsky_joint)
    sigma_joint = sigma_from_F(F_joint)
    sigma_lim_at_joint = sigma_from_F(F_lim_joint)
    sigma_gal_at_joint = sigma_from_F(F_gal_joint)
    sigma_quad_joint = 1.0 / np.sqrt(1.0/sigma_lim_at_joint**2 +
                                     1.0/sigma_gal_at_joint**2)
    gain = sigma_quad_joint / sigma_joint if sigma_joint > 0 else np.nan
    print(f"  σ(LIM) at f_sky_joint     = {sigma_lim_at_joint:.3f}")
    print(f"  σ(gal) at f_sky_joint     = {sigma_gal_at_joint:.3f}")
    print(f"  σ(joint, unmarg.)          = {sigma_joint:.3f}")
    print(f"  σ(quad at f_sky_joint)     = {sigma_quad_joint:.3f}")
    print(f"  cross-block gain           = ×{gain:.3f}")
    print(f"  F_joint / max(F_LIM,F_gal) = "
          f"×{F_joint / max(F_lim_joint, F_gal_joint):.3f}")

    # Diagnostic: cross correlation at ell=10, best-overlapping bin
    ha = [c for c in lim_ch if c['line'] == 'Halpha']
    lim_z = np.asarray([c['z_peak'] for c in ha])
    # find gal bin closest in z to Hα peak at cosmic noon
    gal_zs = np.asarray([b['z_peak'] for b in gal_bins])
    lim_ch_ref = ha[int(np.argmin(np.abs(lim_z - 1.5)))]  # around cosmic noon
    j_ref = int(np.argmin(np.abs(gal_zs - lim_ch_ref['z_peak'])))
    gal_ref = gal_bins[j_ref]
    C_l = compute_lim_cl_limber(10, lim_ch_ref, lim_ch_ref, fNL=0.0)
    C_g = compute_gal_cl_limber(10, gal_ref, gal_ref, fNL=0.0)
    C_x = compute_lim_x_galaxy_cross_cl(10, lim_ch_ref, gal_ref, fNL=0.0)
    r = C_x / np.sqrt(C_l * C_g) if C_l * C_g > 0 else np.nan
    print(f"\n  cross-correlation diagnostic at ℓ=10:")
    print(f"    LIM (Hα, z={lim_ch_ref['z_peak']:.2f}) × gal "
          f"(z_c={gal_ref['z_peak']:.2f}, σ_z={gal_ref['sigma_z']:.4f})")
    print(f"    r = {r:.3f}")

    # 14-parameter physical marginalisation
    print(f"\n  14-parameter physical marginalisation "
          f"(4 cosmo + {len(gal_bins)} gal biases):")
    sigma_marg, cond = reduced_marg_fisher(
        lim_ch, gal_bins, N_lim, N_gal, fsky_joint,
        n_bg=len(gal_bins))
    print(f"    σ(joint, marg) = {sigma_marg:.3f}   cond = {cond:.2e}")
    print(f"    marginalisation penalty vs joint unmarg = "
          f"×{sigma_marg/sigma_joint:.2f}")

    return dict(sigma_lim_own=sigma_lim_own, sigma_gal_own=sigma_gal_own,
                sigma_lim_at_joint=sigma_lim_at_joint,
                sigma_gal_at_joint=sigma_gal_at_joint,
                sigma_joint=sigma_joint, sigma_quad_joint=sigma_quad_joint,
                cross_gain=gain, sigma_joint_marg=sigma_marg,
                cond=cond, r_diag=r,
                N_bins=len(gal_bins))


def main():
    print("=" * 72)
    print("RUN 2 — Spectroscopic cross-correlations")
    print("=" * 72)

    lim_deep  = build_92_channels(mode='deep')
    lim_wide  = build_92_channels(mode='all-sky')

    # --------------------------------------------------------------
    # 2a: real Euclid Hα spec
    # --------------------------------------------------------------
    # Deep-field spec (same f_sky as photo deep, i.e. limited overlap area)
    bins_2a_deep, fsky_gal_deep = build_euclid_spec_bins('deep', n_bins=5)
    bins_2a_wide, fsky_gal_wide = build_euclid_spec_bins('wide', n_bins=5)
    fsky_jd = min(0.0048, fsky_gal_deep)
    fsky_jw = min(0.60, fsky_gal_wide)
    res_2a_deep = run_scenario("2a Real Hα spec — DEEP",
                               lim_deep, bins_2a_deep,
                               0.0048, fsky_gal_deep, fsky_jd)
    res_2a_wide = run_scenario("2a Real Hα spec — WIDE",
                               lim_wide, bins_2a_wide,
                               0.60, fsky_gal_wide, fsky_jw)

    # --------------------------------------------------------------
    # 2b: hypothetical dense spec (photo density, spec z, 50 bins)
    # --------------------------------------------------------------
    bins_2b_deep, _ = build_euclid_hypothetical_spec_bins('deep', n_bins=50)
    bins_2b_wide, _ = build_euclid_hypothetical_spec_bins('wide', n_bins=50)
    res_2b_deep = run_scenario("2b Hypothetical dense spec — DEEP",
                               lim_deep, bins_2b_deep,
                               0.0048, EUCLID_F_SKY_DEEP,
                               min(0.0048, EUCLID_F_SKY_DEEP))
    res_2b_wide = run_scenario("2b Hypothetical dense spec — WIDE",
                               lim_wide, bins_2b_wide,
                               0.60, EUCLID_F_SKY_WIDE,
                               min(0.60, EUCLID_F_SKY_WIDE))

    # --------------------------------------------------------------
    # Three-way comparison table
    # --------------------------------------------------------------
    print("\n\n" + "=" * 72)
    print("THREE-WAY GALAXY CONFIGURATION COMPARISON (Wide, marginalised)")
    print("=" * 72)
    # Load the corrected-photo Wide result
    now_wide = 11.06  # 14-param physical marg from c80555c
    print(f"  {'configuration':<45}  {'σ_marg':>7}  {'r@ℓ=10':>8}  {'gain':>6}")
    print(f"  {'photo (10 bins, σ_z=0.05, validated baseline)':<45}  "
          f"{now_wide:>7.3f}  {'0.665':>8}  {'~1.00':>6}")
    print(f"  {'2a real Hα spec (5 bins, σ_z=1e-3, 1900/deg²)':<45}  "
          f"{res_2a_wide['sigma_joint_marg']:>7.3f}  "
          f"{res_2a_wide['r_diag']:>8.3f}  "
          f"×{res_2a_wide['cross_gain']:>4.2f}")
    print(f"  {'2b hypothetical spec (50 bins, photo density)':<45}  "
          f"{res_2b_wide['sigma_joint_marg']:>7.3f}  "
          f"{res_2b_wide['r_diag']:>8.3f}  "
          f"×{res_2b_wide['cross_gain']:>4.2f}")

    # Save
    out = os.path.join(os.path.dirname(__file__), '..',
                       'data', 'spectroscopic_forecast.pkl')
    with open(out, 'wb') as f:
        pickle.dump({'2a_deep': res_2a_deep, '2a_wide': res_2a_wide,
                     '2b_deep': res_2b_deep, '2b_wide': res_2b_wide}, f)
    print(f"\n  Results pickled: {out}")


if __name__ == "__main__":
    main()
