"""
futuristic_forecast.py — Run 1: 10–15 year projection for SPHEREx × Euclid photo.

Applies TWO scenario-specific changes on top of the validated corrected pipeline:
  * SPHEREx v28 σ_n(λ) → σ_n(λ) / 10   (noise power → power / 100)
  * Euclid photo n_gal → n_gal × 10    (shot noise / 10 per bin)

All four Blanchard corrections (Steps A–D) remain active.
14-parameter physical marginalisation (LIM A_i = B_i = 1 fixed).
"""
import numpy as np
import sys, os, time, pickle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

import cosmology
import limber
import bias_functions
from lim_channels import build_92_channels, LINE_ORDER
from validate_euclid_step_c import build_euclid_bins_blanchard
from limber import clear_intensity_cache, compute_lim_cl_limber

from joint_forecast_corrected import (
    ELL_GRID, K_MAX_H_PER_MPC, FNL_FID, DELTA_FNL, DELTA_NUISANCE,
    C_lim_kmax, C_gal_kmax, C_cross_kmax, joint_C_kmax,
    dC_dsigma8_block, dC_dns_block, dC_dOm_joint, _entries_z_bar,
    fisher_fNL_scalar,
)

NOISE_SCALE_LIM = 0.1     # σ_n / 10  → N_ℓ / 100
DENSITY_SCALE_GAL = 10.0  # n_gal × 10 → shot / 10


def rescale_lim_noise(lim_channels, factor):
    for ch in lim_channels:
        ch['sigma_n'] = ch['sigma_n'] * factor
        ch['noise'] = ch['noise'] * (factor ** 2)
    return lim_channels


def rescale_gal_density(gal_bins, factor):
    for b in gal_bins:
        b['n_bar'] = b['n_bar'] * factor
        b['noise'] = b['noise'] / factor
    return gal_bins


def reduced_marg_fisher(lim_ch, gal_bins, N_lim, N_gal, f_sky, n_bg=None):
    """
    14-parameter marginalisation (fixed LIM A_i = B_i = 1; marginalise
    over f_NL, σ_8, n_s, Ω_m, and one b_g per galaxy bin).
    """
    if n_bg is None:
        n_bg = len(gal_bins)
    Nl, Ng = len(lim_ch), len(gal_bins)
    N_joint = np.concatenate([N_lim, N_gal])
    peaks_all = np.array([c['z_peak'] for c in lim_ch] +
                         [b['z_peak'] for b in gal_bins])
    zbar_map = _entries_z_bar(peaks_all, peaks_all)

    n_par = 4 + n_bg

    def _apply_gal_bg(bg):
        out = []
        for i, b in enumerate(gal_bins):
            c = dict(b); c['b_scale'] = bg[i]
            out.append(c)
        return out

    def _build_S(bg, fNL):
        C = joint_C_kmax(int(ell), lim_ch, _apply_gal_bg(bg), fNL)
        return C + np.diag(N_joint)

    bg0 = np.ones(Ng)
    F = np.zeros((n_par, n_par))
    for ell in ELL_GRID:
        clear_intensity_cache()
        t0 = time.time()
        S = _build_S(bg0, FNL_FID)
        try:
            Sinv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            continue

        C_signal = S - np.diag(N_joint)
        dS = [
            (_build_S(bg0, FNL_FID + DELTA_FNL) -
             _build_S(bg0, FNL_FID - DELTA_FNL)) / (2.0 * DELTA_FNL),
            dC_dsigma8_block(C_signal),
            dC_dns_block(ell, zbar_map, C_signal),
            dC_dOm_joint(ell, lim_ch, gal_bins, FNL_FID),
        ]
        for a in range(n_bg):
            bp = bg0.copy(); bp[a] = 1.0 + DELTA_NUISANCE
            bm = bg0.copy(); bm[a] = 1.0 - DELTA_NUISANCE
            dS.append((_build_S(bp, FNL_FID) - _build_S(bm, FNL_FID)) /
                      (2.0 * DELTA_NUISANCE))

        M = [Sinv @ d for d in dS]
        weight = (2.0 * ell + 1.0) * f_sky / 2.0
        for a in range(n_par):
            for b in range(a, n_par):
                v = weight * np.trace(M[a] @ M[b])
                F[a, b] += v
                if a != b:
                    F[b, a] += v
        print(f"    ℓ={ell:>4d}: {time.time()-t0:5.1f}s  "
              f"{n_par}×{n_par} Fisher assembled")

    cond = np.linalg.cond(F)
    cov = np.linalg.pinv(F)
    return float(np.sqrt(cov[0, 0])), cond


def run_config(name, lim_mode, euclid_mode, fsky_lim, fsky_joint):
    print("\n" + "═" * 72)
    print(f"CONFIG: {name}  (LIM {lim_mode}, Euclid Wide-photo × future scale)")
    print("═" * 72)

    lim_ch = build_92_channels(mode=lim_mode)
    for ch in lim_ch:
        ch['f_sky_config'] = fsky_lim
    lim_ch = rescale_lim_noise(lim_ch, NOISE_SCALE_LIM)
    N_lim = np.asarray([ch['noise'] for ch in lim_ch])

    gal_bins, fsky_gal = build_euclid_bins_blanchard(euclid_mode)
    gal_bins = rescale_gal_density(gal_bins, DENSITY_SCALE_GAL)
    N_gal = np.asarray([b['noise'] for b in gal_bins])

    print(f"  LIM σ_n scaled × {NOISE_SCALE_LIM} → noise power × "
          f"{NOISE_SCALE_LIM**2}")
    print(f"  Euclid density × {DENSITY_SCALE_GAL} → per-bin shot noise "
          f"× {1/DENSITY_SCALE_GAL}")
    print(f"  f_sky (LIM) = {fsky_lim}, gal = {fsky_gal:.4f}, "
          f"joint = {fsky_joint:.4f}")

    print("\n(i) LIM-only Fisher:")
    F = fisher_fNL_scalar(ELL_GRID,
        lambda ell, f: C_lim_kmax(ell, lim_ch, f), N_lim, fsky_lim)
    sigma_lim = 1.0/np.sqrt(F) if F > 0 else np.inf
    print(f"    σ(LIM-only) = {sigma_lim:.3f}")

    print("(ii) Gal-only Fisher:")
    F = fisher_fNL_scalar(ELL_GRID,
        lambda ell, f: C_gal_kmax(ell, gal_bins, f), N_gal, fsky_gal)
    sigma_gal = 1.0/np.sqrt(F) if F > 0 else np.inf
    print(f"    σ(gal-only) = {sigma_gal:.3f}")

    print("(iii) Joint Fisher (unmarg.):")
    N_joint = np.concatenate([N_lim, N_gal])
    F = fisher_fNL_scalar(ELL_GRID,
        lambda ell, f: joint_C_kmax(ell, lim_ch, gal_bins, f),
        N_joint, fsky_joint)
    sigma_joint = 1.0/np.sqrt(F) if F > 0 else np.inf
    sigma_quad = 1.0/np.sqrt(1.0/sigma_lim**2 + 1.0/sigma_gal**2)
    gain = sigma_quad / sigma_joint if sigma_joint>0 else np.nan
    print(f"    σ(joint) = {sigma_joint:.3f}   σ(quad) = {sigma_quad:.3f}"
          f"   cross-block gain = ×{gain:.3f}")

    print("(iv) 14-parameter physical marginalisation:")
    sigma_marg, cond = reduced_marg_fisher(lim_ch, gal_bins, N_lim, N_gal,
                                           fsky_joint)
    print(f"    σ(joint, marg) = {sigma_marg:.3f}   cond = {cond:.2e}")
    print(f"    marginalisation penalty vs joint unmarg = "
          f"×{sigma_marg/sigma_joint:.2f}")

    return dict(sigma_lim=sigma_lim, sigma_gal=sigma_gal,
                sigma_joint=sigma_joint, sigma_quad=sigma_quad,
                sigma_joint_marg=sigma_marg, cond=cond,
                cross_gain=gain)


def main():
    print("=" * 72)
    print("RUN 1 — 'Futuristic (10-15 year)' forecast")
    print("     LIM σ_n × 0.1  (noise power × 0.01)")
    print("     Euclid n_gal × 10 (photo density → 300 gal/arcmin²)")
    print("=" * 72)

    # DEEP
    _, fsky_gal_deep = build_euclid_bins_blanchard('deep')
    fsky_joint_deep = min(0.0048, fsky_gal_deep)
    res_deep = run_config("DEEP-FIELD (future)", "deep", "deep",
                          fsky_lim=0.0048, fsky_joint=fsky_joint_deep)

    # WIDE
    _, fsky_gal_wide = build_euclid_bins_blanchard('wide')
    fsky_joint_wide = min(0.60, fsky_gal_wide)
    res_wide = run_config("WIDE / ALL-SKY (future)", "all-sky", "wide",
                          fsky_lim=0.60, fsky_joint=fsky_joint_wide)

    print("\n" + "=" * 72)
    print("FUTURISTIC DECOMPOSITION (10× SPHEREx noise reduction, 10× Euclid density)")
    print("=" * 72)
    print(f"  {'config':<12}  {'σ_LIM':>8}  {'σ_gal':>8}  "
          f"{'σ_joint':>10}  {'σ_quad':>10}  {'σ_joint_marg':>13}  "
          f"{'cross gain':>11}")
    for name, r in [("deep", res_deep), ("wide", res_wide)]:
        print(f"  {name:<12}  {r['sigma_lim']:8.3f}  {r['sigma_gal']:8.3f}  "
              f"{r['sigma_joint']:10.3f}  {r['sigma_quad']:10.3f}  "
              f"{r['sigma_joint_marg']:13.3f}  ×{r['cross_gain']:9.3f}")

    print("\n  vs present-day (from validated run c80555c):")
    print(f"  {'config':<12}  {'now σ_marg':>11}  {'future σ_marg':>13}  "
          f"{'ratio':>7}")
    now_deep = 152.9
    now_wide = 11.06
    print(f"  {'deep':<12}  {now_deep:>11.2f}  "
          f"{res_deep['sigma_joint_marg']:>13.3f}  "
          f"×{now_deep/res_deep['sigma_joint_marg']:>6.2f}")
    print(f"  {'wide':<12}  {now_wide:>11.2f}  "
          f"{res_wide['sigma_joint_marg']:>13.3f}  "
          f"×{now_wide/res_wide['sigma_joint_marg']:>6.2f}")

    out = os.path.join(os.path.dirname(__file__), '..',
                       'data', 'futuristic_forecast.pkl')
    with open(out, 'wb') as f:
        pickle.dump({'deep': res_deep, 'wide': res_wide}, f)
    print(f"\n  Results pickled: {out}")


if __name__ == "__main__":
    main()
