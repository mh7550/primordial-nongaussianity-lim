"""
combined_futuristic_dense.py — Run 3: futuristic instrumentation × dense spec.

  * SPHEREx v28 σ_n × 0.1                     (noise power × 0.01)
  * Euclid density × 10 (300 gal/arcmin²)      applied to 50 spec bins
  * σ_z/(1+z) = 1e-3 spec-quality per bin
  * Same corrected pipeline (A/B/D active; C outlier-tail dropped for spec)
  * 14-parameter physical marginalisation (4 cosmo + 50 gal biases)
"""
import numpy as np
import sys, os, time, pickle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

from lim_channels import build_92_channels
from survey_specs import (build_euclid_hypothetical_spec_bins,
                           EUCLID_F_SKY_DEEP, EUCLID_F_SKY_WIDE)
from limber import (clear_intensity_cache, compute_lim_cl_limber,
                    compute_gal_cl_limber, compute_lim_x_galaxy_cross_cl)
from joint_forecast_corrected import (
    ELL_GRID, C_lim_kmax, C_gal_kmax, joint_C_kmax,
    FNL_FID, DELTA_FNL, fisher_fNL_scalar,
)
from futuristic_forecast import (rescale_lim_noise, rescale_gal_density,
                                  reduced_marg_fisher)


def sigma_from_F(F):
    return 1.0/np.sqrt(F) if F > 0 else np.inf


def run(name, lim_mode, euclid_config, fsky_lim, fsky_gal, fsky_joint,
        n_gal_bins=50):
    print("\n" + "═" * 72)
    print(f"SCENARIO: {name}")
    print("═" * 72)
    lim_ch = rescale_lim_noise(build_92_channels(mode=lim_mode), 0.1)
    gal_bins, _ = build_euclid_hypothetical_spec_bins(euclid_config,
                                                     n_bins=n_gal_bins)
    gal_bins = rescale_gal_density(gal_bins, 10.0)   # 30 → 300 gal/arcmin²
    N_lim = np.asarray([c['noise'] for c in lim_ch])
    N_gal = np.asarray([b['noise'] for b in gal_bins])
    N_joint = np.concatenate([N_lim, N_gal])
    print(f"  LIM channels: {len(lim_ch)}  (σ_n scaled × 0.1)")
    print(f"  gal bins:     {len(gal_bins)}  (density × 10 → 300 gal/arcmin², "
          f"spec-quality σ_z ≈ {gal_bins[0]['sigma_z']:.4f})")
    print(f"  f_sky (LIM, gal, joint) = ({fsky_lim}, {fsky_gal:.4f}, "
          f"{fsky_joint:.4f})")

    # Individual Fishers at each survey's own f_sky and at joint f_sky
    F_lim_own = fisher_fNL_scalar(ELL_GRID,
        lambda ell, f: C_lim_kmax(ell, lim_ch, f), N_lim, fsky_lim)
    F_gal_own = fisher_fNL_scalar(ELL_GRID,
        lambda ell, f: C_gal_kmax(ell, gal_bins, f), N_gal, fsky_gal)
    F_lim_joint = fisher_fNL_scalar(ELL_GRID,
        lambda ell, f: C_lim_kmax(ell, lim_ch, f), N_lim, fsky_joint)
    F_gal_joint = fisher_fNL_scalar(ELL_GRID,
        lambda ell, f: C_gal_kmax(ell, gal_bins, f), N_gal, fsky_joint)

    sigma_lim_own = sigma_from_F(F_lim_own)
    sigma_gal_own = sigma_from_F(F_gal_own)
    sigma_lim_j = sigma_from_F(F_lim_joint)
    sigma_gal_j = sigma_from_F(F_gal_joint)

    # Joint Fisher (unmarg)
    F_joint = fisher_fNL_scalar(ELL_GRID,
        lambda ell, f: joint_C_kmax(ell, lim_ch, gal_bins, f), N_joint, fsky_joint)
    sigma_joint = sigma_from_F(F_joint)
    sigma_quad_j = 1.0/np.sqrt(1.0/sigma_lim_j**2 + 1.0/sigma_gal_j**2)
    gain = sigma_quad_j / sigma_joint if sigma_joint>0 else np.nan
    F_ratio = F_joint / max(F_lim_joint, F_gal_joint)
    print(f"\n  σ(LIM-only, own f_sky)   = {sigma_lim_own:.3f}")
    print(f"  σ(gal-only, own f_sky)   = {sigma_gal_own:.3f}")
    print(f"  σ(LIM at joint f_sky)    = {sigma_lim_j:.3f}")
    print(f"  σ(gal at joint f_sky)    = {sigma_gal_j:.3f}")
    print(f"  σ(joint, unmarg.)        = {sigma_joint:.3f}")
    print(f"  σ(quad at joint f_sky)   = {sigma_quad_j:.3f}")
    print(f"  cross-block gain         = ×{gain:.3f}")
    print(f"  F_joint / max(F_L, F_g)  = ×{F_ratio:.3f}")

    # Diagnostic: cross correlation r at ℓ=10, best-overlapping bin
    ha = [c for c in lim_ch if c['line']=='Halpha']
    lim_ch_ref = ha[int(np.argmin(np.abs(
        np.asarray([c['z_peak'] for c in ha]) - 1.5)))]
    gal_zs = np.asarray([b['z_peak'] for b in gal_bins])
    j_ref = int(np.argmin(np.abs(gal_zs - lim_ch_ref['z_peak'])))
    gal_ref = gal_bins[j_ref]
    C_l = compute_lim_cl_limber(10, lim_ch_ref, lim_ch_ref, fNL=0.0)
    C_g = compute_gal_cl_limber(10, gal_ref, gal_ref, fNL=0.0)
    C_x = compute_lim_x_galaxy_cross_cl(10, lim_ch_ref, gal_ref, fNL=0.0)
    r = C_x/np.sqrt(C_l*C_g) if C_l*C_g > 0 else np.nan
    print(f"  cross-corr r(ℓ=10) at z ≈ {lim_ch_ref['z_peak']:.2f}: "
          f"{r:.3f}")

    # Marginalisation (4 cosmo + 50 gal biases)
    print(f"\n  14-parameter physical marginalisation:")
    sigma_marg, cond = reduced_marg_fisher(
        lim_ch, gal_bins, N_lim, N_gal, fsky_joint, n_bg=len(gal_bins))
    print(f"    σ(joint, marg) = {sigma_marg:.3f}   cond = {cond:.2e}")
    print(f"    marg. penalty  = ×{sigma_marg/sigma_joint:.2f}")

    return dict(sigma_lim_own=sigma_lim_own, sigma_gal_own=sigma_gal_own,
                sigma_lim_j=sigma_lim_j, sigma_gal_j=sigma_gal_j,
                sigma_joint=sigma_joint, sigma_quad_j=sigma_quad_j,
                cross_gain=gain, F_ratio=F_ratio,
                sigma_joint_marg=sigma_marg, cond=cond, r_diag=r,
                n_gal_bins=len(gal_bins))


def main():
    print("=" * 72)
    print("RUN 3 — Combined futuristic + dense spectroscopic")
    print("=" * 72)

    _, fsky_gal_deep = build_euclid_hypothetical_spec_bins('deep', n_bins=50)
    _, fsky_gal_wide = build_euclid_hypothetical_spec_bins('wide', n_bins=50)
    fsky_joint_deep = min(0.0048, fsky_gal_deep)
    fsky_joint_wide = min(0.60,   fsky_gal_wide)

    res_deep = run("Run 3 DEEP  (LIM deep, futuristic × dense spec, Euclid deep)",
                   "deep", "deep", 0.0048, fsky_gal_deep, fsky_joint_deep,
                   n_gal_bins=50)
    res_wide = run("Run 3 WIDE  (LIM all-sky, futuristic × dense spec, Euclid wide)",
                   "all-sky", "wide", 0.60, fsky_gal_wide, fsky_joint_wide,
                   n_gal_bins=50)

    # Multiplicative stacking prediction
    baseline_deep = 152.9
    baseline_wide = 11.06
    fut_photo_deep = 41.74
    fut_photo_wide = 6.27
    dense_deep = 62.08
    dense_wide = 4.18
    predicted_deep = baseline_deep / (baseline_deep/fut_photo_deep *
                                      baseline_deep/dense_deep)
    predicted_wide = baseline_wide / (baseline_wide/fut_photo_wide *
                                      baseline_wide/dense_wide)

    print("\n" + "=" * 72)
    print("RUN 3 SUMMARY vs multiplicative-stacking prediction")
    print("=" * 72)
    print(f"  {'config':<8}  {'σ_marg':>8}  {'predicted (mult.)':>18}  "
          f"{'actual/pred':>13}  {'cross gain':>11}")
    print(f"  {'deep':<8}  {res_deep['sigma_joint_marg']:8.3f}  "
          f"{predicted_deep:18.3f}  "
          f"{res_deep['sigma_joint_marg']/predicted_deep:13.3f}  "
          f"×{res_deep['cross_gain']:9.3f}")
    print(f"  {'wide':<8}  {res_wide['sigma_joint_marg']:8.3f}  "
          f"{predicted_wide:18.3f}  "
          f"{res_wide['sigma_joint_marg']/predicted_wide:13.3f}  "
          f"×{res_wide['cross_gain']:9.3f}")

    print("\n  vs Planck (5.1):")
    for name, r in [("deep", res_deep), ("wide", res_wide)]:
        val = r['sigma_joint_marg']
        print(f"    {name:<5}  σ_marg = {val:.3f}  → "
              f"{5.1/val:.2f}× Planck  "
              f"({'crosses σ=1' if val < 1 else 'above σ=1'})")

    out = os.path.join(os.path.dirname(__file__), '..',
                       'data', 'combined_futuristic_dense.pkl')
    with open(out, 'wb') as f:
        pickle.dump({'deep': res_deep, 'wide': res_wide,
                     'predicted_deep': predicted_deep,
                     'predicted_wide': predicted_wide}, f)
    print(f"\n  Results pickled: {out}")


if __name__ == "__main__":
    main()
