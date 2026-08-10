"""
spectral_resolution_forecast.py — Test whether SPHEREx spectral resolution
is the bottleneck for LIM × galaxy cross-cancellation.

Runs three sub-scenarios of the validated corrected pipeline:
  (a)  R=100 LIM (photon-scaled noise, option b)  × present-day Euclid photo
  (a') R=100 LIM (option-a noise, direct interp)  × present-day Euclid photo
  (c)  R=100 LIM (option-b noise)                 × hypothetical dense spec (50 bins)

Baseline for comparison: present-day SPHEREx (σ_z=0.12) × Euclid photo (Run 1
in the corrected pipeline), σ_marg_wide = 11.06.

Reports σ_LIM, σ_gal, σ_joint, σ_quad at joint f_sky, σ_joint_marg, and the
cross-block gain — the critical number here.
"""
import numpy as np
import sys, os, time, pickle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

from lim_channels import build_uniform_R_channels
from validate_euclid_step_c import build_euclid_bins_blanchard
from survey_specs import (build_euclid_hypothetical_spec_bins,
                          EUCLID_F_SKY_DEEP, EUCLID_F_SKY_WIDE)
from limber import (clear_intensity_cache, compute_lim_cl_limber,
                    compute_gal_cl_limber, compute_lim_x_galaxy_cross_cl)
from joint_forecast_corrected import (
    ELL_GRID, C_lim_kmax, C_gal_kmax, joint_C_kmax,
    FNL_FID, DELTA_FNL, fisher_fNL_scalar,
)
from futuristic_forecast import reduced_marg_fisher


def sigma_from_F(F):
    return 1.0/np.sqrt(F) if F > 0 else np.inf


def run_config(name, lim_ch, gal_bins, fsky_lim, fsky_gal, fsky_joint,
               config_short):
    print("\n" + "═" * 76)
    print(f"SCENARIO: {name}")
    print(f"  N_lim = {len(lim_ch)}   N_gal = {len(gal_bins)}   "
          f"f_sky (LIM,gal,joint) = ({fsky_lim}, {fsky_gal:.4f}, {fsky_joint:.4f})")
    print("═" * 76)
    N_lim = np.asarray([c['noise'] for c in lim_ch])
    N_gal = np.asarray([b['noise'] for b in gal_bins])
    N_joint = np.concatenate([N_lim, N_gal])

    F_lim_j = fisher_fNL_scalar(ELL_GRID,
        lambda ell, f: C_lim_kmax(ell, lim_ch, f), N_lim, fsky_joint)
    F_gal_j = fisher_fNL_scalar(ELL_GRID,
        lambda ell, f: C_gal_kmax(ell, gal_bins, f), N_gal, fsky_joint)
    F_lim_own = fisher_fNL_scalar(ELL_GRID,
        lambda ell, f: C_lim_kmax(ell, lim_ch, f), N_lim, fsky_lim)
    F_gal_own = fisher_fNL_scalar(ELL_GRID,
        lambda ell, f: C_gal_kmax(ell, gal_bins, f), N_gal, fsky_gal)
    sigma_lim_own = sigma_from_F(F_lim_own)
    sigma_gal_own = sigma_from_F(F_gal_own)
    sigma_lim_j = sigma_from_F(F_lim_j)
    sigma_gal_j = sigma_from_F(F_gal_j)

    F_joint = fisher_fNL_scalar(ELL_GRID,
        lambda ell, f: joint_C_kmax(ell, lim_ch, gal_bins, f),
        N_joint, fsky_joint)
    sigma_joint = sigma_from_F(F_joint)
    sigma_quad_j = 1.0/np.sqrt(1.0/sigma_lim_j**2 + 1.0/sigma_gal_j**2)
    gain = sigma_quad_j / sigma_joint if sigma_joint>0 else np.nan
    F_ratio = F_joint / max(F_lim_j, F_gal_j)

    print(f"\n  σ(LIM-only, own f_sky)   = {sigma_lim_own:.3f}")
    print(f"  σ(gal-only, own f_sky)   = {sigma_gal_own:.3f}")
    print(f"  σ(LIM at joint f_sky)    = {sigma_lim_j:.3f}")
    print(f"  σ(gal at joint f_sky)    = {sigma_gal_j:.3f}")
    print(f"  σ(joint, unmarg.)        = {sigma_joint:.3f}")
    print(f"  σ(quad at joint f_sky)   = {sigma_quad_j:.3f}")
    print(f"  cross-block gain          = ×{gain:.4f}")
    print(f"  F_joint / max(F_L, F_g)   = ×{F_ratio:.3f}")

    # r diagnostic at reference (Hα near z=1 × best-overlapping gal bin)
    ha = [c for c in lim_ch if c['line']=='Halpha']
    ha_z = np.asarray([c['z_peak'] for c in ha])
    lim_ref = ha[int(np.argmin(np.abs(ha_z - 1.0)))]
    gal_zs = np.asarray([b['z_peak'] for b in gal_bins])
    gal_ref = gal_bins[int(np.argmin(np.abs(gal_zs - lim_ref['z_peak'])))]
    C_l = compute_lim_cl_limber(10, lim_ref, lim_ref, fNL=0.0)
    C_g = compute_gal_cl_limber(10, gal_ref, gal_ref, fNL=0.0)
    C_x = compute_lim_x_galaxy_cross_cl(10, lim_ref, gal_ref, fNL=0.0)
    r = C_x/np.sqrt(C_l*C_g) if C_l*C_g > 0 else np.nan
    print(f"  r(LIM,gal) at ℓ=10, z≈1  = {r:.3f}")

    print(f"\n  14-parameter physical marginalisation:")
    sigma_marg, cond = reduced_marg_fisher(
        lim_ch, gal_bins, N_lim, N_gal, fsky_joint, n_bg=len(gal_bins))
    print(f"    σ(joint, marg) = {sigma_marg:.3f}   cond = {cond:.2e}")
    print(f"    marg. penalty  = ×{sigma_marg/sigma_joint:.2f}")

    return dict(config=config_short,
                sigma_lim_own=sigma_lim_own, sigma_gal_own=sigma_gal_own,
                sigma_lim_j=sigma_lim_j, sigma_gal_j=sigma_gal_j,
                sigma_joint=sigma_joint, sigma_quad_j=sigma_quad_j,
                cross_gain=gain, F_ratio=F_ratio,
                sigma_joint_marg=sigma_marg, cond=cond, r_ref=r,
                n_lim=len(lim_ch), n_gal=len(gal_bins))


def main():
    print("=" * 76)
    print("SPECTRAL RESOLUTION FORECAST — Pullen R=100 test")
    print("=" * 76)

    # LIM channel builds (R=100)
    lim_R100_deep_a = build_uniform_R_channels(R=100, mode='deep',
                                               n_per_line=30, noise_scaling='a')
    lim_R100_deep_b = build_uniform_R_channels(R=100, mode='deep',
                                               n_per_line=30, noise_scaling='b')
    lim_R100_wide_a = build_uniform_R_channels(R=100, mode='all-sky',
                                               n_per_line=30, noise_scaling='a')
    lim_R100_wide_b = build_uniform_R_channels(R=100, mode='all-sky',
                                               n_per_line=30, noise_scaling='b')
    print(f"\n  R=100 LIM channel count: {len(lim_R100_deep_a)}  (30 per line × 4)")
    print(f"  Noise option (b) inflates σ_n by √(100/R_native)")
    print(f"    Bands 1-3 (R=41):  ×{np.sqrt(100/41):.2f}")
    print(f"    Band 4 (R=35):     ×{np.sqrt(100/35):.2f}")
    print(f"    Bands 5-6 (R≥110): ×{np.sqrt(100/110):.2f}–{np.sqrt(100/130):.2f}")

    # -----------------------------------------------------------
    # (a) R=100 × photo, noise option (b), Wide + Deep
    # -----------------------------------------------------------
    photo_deep, fsky_photo_deep = build_euclid_bins_blanchard('deep')
    photo_wide, fsky_photo_wide = build_euclid_bins_blanchard('wide')

    print("\n\n" + "#" * 76)
    print("# SUB-SCENARIO (a): R=100 × Euclid photo (10 bins) — noise option b")
    print("#" * 76)
    res_a_deep = run_config("(a) R=100 photon-scaled × photo — DEEP",
                            lim_R100_deep_b, photo_deep,
                            0.0048, fsky_photo_deep,
                            min(0.0048, fsky_photo_deep), "a_deep")
    res_a_wide = run_config("(a) R=100 photon-scaled × photo — WIDE",
                            lim_R100_wide_b, photo_wide,
                            0.60, fsky_photo_wide,
                            min(0.60, fsky_photo_wide), "a_wide")

    # -----------------------------------------------------------
    # (b) same but noise option (a) — Wide only for the check
    # -----------------------------------------------------------
    print("\n\n" + "#" * 76)
    print("# SUB-SCENARIO (b): same as (a) but noise option a (direct interp)")
    print("#" * 76)
    res_b_wide = run_config("(b) R=100 direct-interp × photo — WIDE",
                            lim_R100_wide_a, photo_wide,
                            0.60, fsky_photo_wide,
                            min(0.60, fsky_photo_wide), "b_wide")

    # -----------------------------------------------------------
    # (c) R=100 × dense spec (50 bins)
    # -----------------------------------------------------------
    dspec_deep, _ = build_euclid_hypothetical_spec_bins('deep', n_bins=50)
    dspec_wide, _ = build_euclid_hypothetical_spec_bins('wide', n_bins=50)

    print("\n\n" + "#" * 76)
    print("# SUB-SCENARIO (c): R=100 × hypothetical dense spec (50 bins)")
    print("#" * 76)
    res_c_deep = run_config("(c) R=100 × dense spec — DEEP",
                            lim_R100_deep_b, dspec_deep,
                            0.0048, EUCLID_F_SKY_DEEP,
                            min(0.0048, EUCLID_F_SKY_DEEP), "c_deep")
    res_c_wide = run_config("(c) R=100 × dense spec — WIDE",
                            lim_R100_wide_b, dspec_wide,
                            0.60, EUCLID_F_SKY_WIDE,
                            min(0.60, EUCLID_F_SKY_WIDE), "c_wide")

    # Summary
    print("\n\n" + "=" * 76)
    print("SPECTRAL-RESOLUTION SUMMARY  (Wide, marginalised)")
    print("=" * 76)
    print(f"  {'scenario':<40}  {'σ_marg':>7}  {'cross gain':>10}  "
          f"{'r@ℓ=10':>7}")
    baseline = 11.06
    print(f"  {'BASELINE: SPHEREx-native × photo':<40}  {baseline:>7.2f}  "
          f"{'×1.00':>10}  {'0.665':>7}  (from c80555c)")
    for r in [res_a_wide, res_b_wide, res_c_wide]:
        print(f"  {r['config']+':':<40}  {r['sigma_joint_marg']:>7.3f}  "
              f"×{r['cross_gain']:>8.4f}  {r['r_ref']:>7.3f}")

    print("\n  Deep-config summary:")
    for r in [res_a_deep, res_c_deep]:
        print(f"  {r['config']+':':<40}  {r['sigma_joint_marg']:>7.3f}  "
              f"×{r['cross_gain']:>8.4f}  {r['r_ref']:>7.3f}")

    out = os.path.join(os.path.dirname(__file__), '..',
                       'data', 'spectral_resolution_forecast.pkl')
    with open(out, 'wb') as f:
        pickle.dump({'a_deep': res_a_deep, 'a_wide': res_a_wide,
                     'b_wide': res_b_wide,
                     'c_deep': res_c_deep, 'c_wide': res_c_wide}, f)
    print(f"\n  Results pickled: {out}")


if __name__ == "__main__":
    main()
