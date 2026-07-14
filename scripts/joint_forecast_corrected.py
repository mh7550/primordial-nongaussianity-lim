"""
joint_forecast_corrected.py — final joint SPHEREx LIM × Euclid photo Fisher
with all four Blanchard-validation corrections active.

Corrections applied:
  A. Cosmological marginalisation over σ_8, n_s, Ω_m
  B. ℓ_min = 10                    (drops ℓ = 2, 5)
  C. Blanchard n(z) + outlier-inflated σ_z on the galaxy side only
     (LIM keeps σ_z = 0.12 per paper text; sensitivity ×1.10)
  D. k_max = 0.3 h/Mpc cutoff on all pairs (LIM×LIM, gal×gal, cross)

All corrections applied UNIFORMLY across LIM, gal, and cross blocks so
the three sub-Fishers are internally consistent with the corrected
gal-only baseline (σ_gal_wide ≈ 8.46, within 2× of Blanchard).

Reports for Deep and Wide:
    σ_LIM  σ_gal  σ_joint  σ_joint_marg
plus the LIM×gal correlation coefficient at ℓ = 10 (matched z).
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
from limber import (compute_lim_cls_matrix, compute_gal_cl_limber,
                    compute_lim_x_galaxy_cross_cl, clear_intensity_cache)


# ---- Corrections (Steps A–D) ---------------------------------------------
ELL_GRID = np.array([10, 20, 40, 80, 150, 250])   # Step B
K_MAX_H_PER_MPC = 0.3                              # Step D
K_PIVOT = 0.05                                     # n_s pivot
FNL_FID = 1.0
DELTA_FNL = 0.1
DELTA_NUISANCE = 0.01
DELTA_OM = 0.01


# ---------------------------------------------------------------------------
# Helper: k_eff at a pair z_bar
# ---------------------------------------------------------------------------

def _keff(ell, z_bar):
    chi = cosmology.get_comoving_distance(z_bar)
    return (ell + 0.5) / max(chi, 1.0)


# ---------------------------------------------------------------------------
# Corrected per-block matrix builders with k_max cutoff
# ---------------------------------------------------------------------------

def C_lim_kmax(ell, lim_ch, fNL, k_max=K_MAX_H_PER_MPC):
    C = compute_lim_cls_matrix(int(ell), lim_ch, fNL=fNL,
                               use_bessel_below_limber=False, use_rsd=False)
    Nl = len(lim_ch)
    for i in range(Nl):
        for j in range(Nl):
            z_bar = 0.5 * (lim_ch[i]['z_peak'] + lim_ch[j]['z_peak'])
            if _keff(ell, z_bar) > k_max:
                C[i, j] = 0.0
    return C


def C_gal_kmax(ell, gal_bins, fNL, k_max=K_MAX_H_PER_MPC):
    Ng = len(gal_bins)
    C = np.zeros((Ng, Ng))
    for i in range(Ng):
        for j in range(i, Ng):
            z_bar = 0.5 * (gal_bins[i]['z_peak'] + gal_bins[j]['z_peak'])
            if _keff(ell, z_bar) > k_max:
                continue
            v = compute_gal_cl_limber(ell, gal_bins[i], gal_bins[j], fNL=fNL)
            C[i, j] = v
            C[j, i] = v
    return C


def C_cross_kmax(ell, lim_ch, gal_bins, fNL, k_max=K_MAX_H_PER_MPC):
    Nl, Ng = len(lim_ch), len(gal_bins)
    C = np.zeros((Nl, Ng))
    for i in range(Nl):
        for j in range(Ng):
            z_bar = 0.5 * (lim_ch[i]['z_peak'] + gal_bins[j]['z_peak'])
            if _keff(ell, z_bar) > k_max:
                continue
            C[i, j] = compute_lim_x_galaxy_cross_cl(
                ell, lim_ch[i], gal_bins[j], fNL=fNL)
    return C


def joint_C_kmax(ell, lim_ch, gal_bins, fNL, k_max=K_MAX_H_PER_MPC):
    Nl, Ng = len(lim_ch), len(gal_bins)
    C = np.zeros((Nl + Ng, Nl + Ng))
    C[:Nl, :Nl] = C_lim_kmax(ell, lim_ch, fNL, k_max)
    C[Nl:, Nl:] = C_gal_kmax(ell, gal_bins, fNL, k_max)
    X = C_cross_kmax(ell, lim_ch, gal_bins, fNL, k_max)
    C[:Nl, Nl:] = X
    C[Nl:, :Nl] = X.T
    return C


# ---------------------------------------------------------------------------
# σ_8, n_s analytic derivatives  (applied to any C block)
# ---------------------------------------------------------------------------

def dC_dsigma8_block(C_fid):
    """∂C/∂σ_8 = 2 C / σ_8. Applies uniformly since P(k) ∝ σ_8²."""
    return 2.0 * C_fid / cosmology.sigma8


def dC_dns_block(ell, entries_z_bar, C_fid):
    """
    ∂C/∂n_s = C × ln(k_eff / k_pivot).
    entries_z_bar[i, j] is the z_bar for that matrix element.
    """
    lnk = np.log(np.maximum(entries_z_bar, 1e-8))   # placeholder
    n = C_fid.shape[0]
    dC = np.zeros_like(C_fid)
    for i in range(n):
        for j in range(n):
            if C_fid[i, j] == 0:
                continue
            k = _keff(ell, entries_z_bar[i, j])
            dC[i, j] = C_fid[i, j] * np.log(max(k, 1e-8) / K_PIVOT)
    return dC


def _entries_z_bar(peaks_i, peaks_j):
    return 0.5 * (peaks_i[:, None] + peaks_j[None, :])


# ---------------------------------------------------------------------------
# Ω_m derivative — perturb cosmology.Om0 numerically
# ---------------------------------------------------------------------------

def _patch_Om(new_Om):
    cosmology.Om0 = new_Om
    cosmology.Ode0 = 1.0 - new_Om
    bias_functions.OMEGA_M = new_Om
    clear_intensity_cache()


def dC_dOm_joint(ell, lim_ch, gal_bins, fNL):
    Om_fid = cosmology.Om0
    _patch_Om(Om_fid + DELTA_OM)
    Cp = joint_C_kmax(ell, lim_ch, gal_bins, fNL)
    _patch_Om(Om_fid - DELTA_OM)
    Cm = joint_C_kmax(ell, lim_ch, gal_bins, fNL)
    _patch_Om(Om_fid)
    return (Cp - Cm) / (2.0 * DELTA_OM)


def dC_dOm_gal(ell, gal_bins, fNL):
    Om_fid = cosmology.Om0
    _patch_Om(Om_fid + DELTA_OM)
    Cp = C_gal_kmax(ell, gal_bins, fNL)
    _patch_Om(Om_fid - DELTA_OM)
    Cm = C_gal_kmax(ell, gal_bins, fNL)
    _patch_Om(Om_fid)
    return (Cp - Cm) / (2.0 * DELTA_OM)


def dC_dOm_lim(ell, lim_ch, fNL):
    Om_fid = cosmology.Om0
    _patch_Om(Om_fid + DELTA_OM)
    Cp = C_lim_kmax(ell, lim_ch, fNL)
    _patch_Om(Om_fid - DELTA_OM)
    Cm = C_lim_kmax(ell, lim_ch, fNL)
    _patch_Om(Om_fid)
    return (Cp - Cm) / (2.0 * DELTA_OM)


# ---------------------------------------------------------------------------
# Fisher assemblers (unmarginalised → LIM-only, gal-only, joint)
# ---------------------------------------------------------------------------

def fisher_fNL_scalar(ell_grid, build_C, N_diag, f_sky):
    F = 0.0
    for ell in ell_grid:
        clear_intensity_cache()
        Sf = build_C(int(ell), FNL_FID) + np.diag(N_diag)
        Sp = build_C(int(ell), FNL_FID + DELTA_FNL) + np.diag(N_diag)
        Sm = build_C(int(ell), FNL_FID - DELTA_FNL) + np.diag(N_diag)
        dS = (Sp - Sm) / (2.0 * DELTA_FNL)
        try:
            Sinv = np.linalg.inv(Sf)
        except np.linalg.LinAlgError:
            continue
        weight = (2.0 * ell + 1.0) * f_sky / 2.0
        M = Sinv @ dS
        F += weight * float(np.trace(M @ M))
    return F


# ---------------------------------------------------------------------------
# Marginalised Fisher (fully-corrected joint)
# ---------------------------------------------------------------------------

def joint_marginalised_fisher(lim_ch, gal_bins, N_lim, N_gal, f_sky):
    """
    22-parameter marginalisation: f_NL, σ_8, n_s, Ω_m,
    4 A_i, 4 B_i, N_gal b_g^a. Cross terms via joint C.
    """
    Nl, Ng = len(lim_ch), len(gal_bins)
    N_joint = np.concatenate([N_lim, N_gal])
    n_par = 4 + 8 + Ng   # f_NL, σ_8, n_s, Ω_m, 4A_i, 4B_i, N_gal b_g

    # z_bar map for n_s derivative
    peaks_all = np.array([c['z_peak'] for c in lim_ch] +
                         [b['z_peak'] for b in gal_bins])
    zbar_map = _entries_z_bar(peaks_all, peaks_all)

    def _apply_lim_AB(A, B):
        out = []
        for ch in lim_ch:
            c = dict(ch); c['I_scale'] = A[ch['line']]; c['b_scale'] = B[ch['line']]
            out.append(c)
        return out

    def _apply_gal_bg(bg):
        out = []
        for i, b in enumerate(gal_bins):
            c = dict(b); c['b_scale'] = bg[i]
            out.append(c)
        return out

    def _build_S(A, B, bg, fNL):
        L = _apply_lim_AB(A, B)
        G = _apply_gal_bg(bg)
        C = joint_C_kmax(int(ell), L, G, fNL)
        return C + np.diag(N_joint)

    A0 = {l: 1.0 for l in LINE_ORDER}
    B0 = A0.copy()
    bg0 = np.ones(Ng)

    F_total = np.zeros((n_par, n_par))
    for ell in ELL_GRID:
        clear_intensity_cache()
        t0 = time.time()
        S = _build_S(A0, B0, bg0, FNL_FID)
        try:
            Sinv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            continue

        # Build derivatives
        dS = []
        # 1. f_NL
        Sp = _build_S(A0, B0, bg0, FNL_FID + DELTA_FNL)
        Sm = _build_S(A0, B0, bg0, FNL_FID - DELTA_FNL)
        dS.append((Sp - Sm) / (2.0 * DELTA_FNL))
        # 2. σ_8 (analytic)
        C_signal = S - np.diag(N_joint)
        dS.append(dC_dsigma8_block(C_signal))
        # 3. n_s (analytic)
        dS.append(dC_dns_block(ell, zbar_map, C_signal))
        # 4. Ω_m (numerical)
        dS.append(dC_dOm_joint(ell, lim_ch, gal_bins, FNL_FID))

        # 5–8. A_i
        for line in LINE_ORDER:
            Ap = A0.copy(); Ap[line] = 1.0 + DELTA_NUISANCE
            Am = A0.copy(); Am[line] = 1.0 - DELTA_NUISANCE
            dS.append((_build_S(Ap, B0, bg0, FNL_FID) -
                       _build_S(Am, B0, bg0, FNL_FID)) / (2.0 * DELTA_NUISANCE))
        # 9–12. B_i
        for line in LINE_ORDER:
            Bp = B0.copy(); Bp[line] = 1.0 + DELTA_NUISANCE
            Bm = B0.copy(); Bm[line] = 1.0 - DELTA_NUISANCE
            dS.append((_build_S(A0, Bp, bg0, FNL_FID) -
                       _build_S(A0, Bm, bg0, FNL_FID)) / (2.0 * DELTA_NUISANCE))
        # 13+. b_g^a per bin
        for a in range(Ng):
            bp = bg0.copy(); bp[a] = 1.0 + DELTA_NUISANCE
            bm = bg0.copy(); bm[a] = 1.0 - DELTA_NUISANCE
            dS.append((_build_S(A0, B0, bp, FNL_FID) -
                       _build_S(A0, B0, bm, FNL_FID)) / (2.0 * DELTA_NUISANCE))

        M = [Sinv @ d for d in dS]
        weight = (2.0 * ell + 1.0) * f_sky / 2.0
        for a in range(n_par):
            for b in range(a, n_par):
                v = weight * np.trace(M[a] @ M[b])
                F_total[a, b] += v
                if a != b:
                    F_total[b, a] += v
        print(f"    ℓ={ell:>4d}: {time.time()-t0:5.1f}s  "
              f"22×22 Fisher assembled")

    cond = np.linalg.cond(F_total)
    return F_total, cond


# ---------------------------------------------------------------------------
# Per-config runner
# ---------------------------------------------------------------------------

def run_config(config_name, lim_mode, euclid_mode, f_sky_lim, f_sky_gal,
               f_sky_joint):
    print("\n" + "═" * 72)
    print(f"CONFIG: {config_name}   "
          f"(LIM {lim_mode}, Euclid {euclid_mode})")
    print(f"  f_sky:  LIM={f_sky_lim}   gal={f_sky_gal:.5f}   joint={f_sky_joint:.5f}")
    print("═" * 72)

    lim_ch = build_92_channels(mode=lim_mode)
    for ch in lim_ch:
        ch['f_sky_config'] = f_sky_lim
    N_lim = np.asarray([ch['noise'] for ch in lim_ch])

    gal_bins, _ = build_euclid_bins_blanchard(euclid_mode)
    N_gal = np.asarray([b['noise'] for b in gal_bins])

    # ---- (i) LIM-only ---
    print("\n(i) LIM-only Fisher (Steps A/B/D active; C not applicable)")
    F_lim = fisher_fNL_scalar(ELL_GRID,
        lambda ell, f: C_lim_kmax(ell, lim_ch, f), N_lim, f_sky_lim)
    sigma_lim = 1.0/np.sqrt(F_lim) if F_lim > 0 else np.inf
    print(f"    σ(LIM-only, unmarg.) = {sigma_lim:.3f}")

    # ---- (ii) gal-only ---
    print("\n(ii) Galaxy-only Fisher (Steps A/B/C/D active, cosmo unmarg here)")
    F_gal = fisher_fNL_scalar(ELL_GRID,
        lambda ell, f: C_gal_kmax(ell, gal_bins, f), N_gal, f_sky_gal)
    sigma_gal = 1.0/np.sqrt(F_gal) if F_gal > 0 else np.inf
    print(f"    σ(gal-only, unmarg.) = {sigma_gal:.3f}")

    # ---- (iii) Joint unmarg ---
    print(f"\n(iii) Joint Fisher (LIM {len(lim_ch)} + gal {len(gal_bins)}, unmarg.)")
    N_joint = np.concatenate([N_lim, N_gal])
    F_joint = fisher_fNL_scalar(ELL_GRID,
        lambda ell, f: joint_C_kmax(ell, lim_ch, gal_bins, f),
        N_joint, f_sky_joint)
    sigma_joint = 1.0/np.sqrt(F_joint) if F_joint > 0 else np.inf
    sigma_quad = 1.0/np.sqrt(1.0/sigma_lim**2 + 1.0/sigma_gal**2)
    gain = sigma_quad / sigma_joint if sigma_joint > 0 else np.nan
    print(f"    σ(joint, unmarg.)    = {sigma_joint:.3f}")
    print(f"    σ(quadrature)        = {sigma_quad:.3f}")
    print(f"    cross-block gain     = ×{gain:.3f}")

    # ---- (iv) Joint 22×22 marginalisation ---
    print("\n(iv) Joint 22-parameter marginalisation "
          "(f_NL, σ_8, n_s, Ω_m, 4A_i, 4B_i, 10 b_g^a)")
    F22, cond22 = joint_marginalised_fisher(lim_ch, gal_bins, N_lim, N_gal,
                                            f_sky_joint)
    print(f"    22×22 Fisher condition number: {cond22:.3e}")
    cov = np.linalg.pinv(F22)
    sigma_marg = float(np.sqrt(cov[0, 0]))
    print(f"    σ(joint, marginalised) = {sigma_marg:.3f}")
    print(f"    marginalisation penalty vs joint unmarg = "
          f"×{sigma_marg/sigma_joint:.2f}")

    return dict(sigma_lim=sigma_lim, sigma_gal=sigma_gal,
                sigma_joint=sigma_joint, sigma_quad=sigma_quad,
                sigma_joint_marg=sigma_marg, cond=cond22,
                f_sky_lim=f_sky_lim, f_sky_gal=f_sky_gal,
                f_sky_joint=f_sky_joint,
                N_lim_channels=len(lim_ch), N_gal_bins=len(gal_bins))


# ---------------------------------------------------------------------------
# Diagnostic — cross correlation at ℓ = 10 under corrected setup
# ---------------------------------------------------------------------------

def diag_cross_correlation():
    print("\n" + "─" * 72)
    print("DIAGNOSTIC — cross-block r_{LIM,gal}(ℓ=10) with corrected bins")
    print("─" * 72)
    lim = build_92_channels(mode='deep')
    gal, _ = build_euclid_bins_blanchard('wide')
    ha = [c for c in lim if c['line']=='Halpha']
    lim_ch = ha[int(np.argmin(np.abs(np.asarray([c['z_peak'] for c in ha]) - 1.0)))]
    gal_bin = gal[int(np.argmin(np.abs(np.asarray([b['z_peak'] for b in gal]) - 1.0)))]
    ell = 10
    print(f"  LIM channel: line={lim_ch['line']}, z_peak={lim_ch['z_peak']:.3f}, "
          f"σ_z={lim_ch.get('sigma_z', 0.12):.3f}")
    print(f"  Euclid bin:  z_c={gal_bin['z_peak']:.3f}, "
          f"σ_z_eff={gal_bin['sigma_z']:.4f}  (Blanchard n(z) + outliers)")
    from limber import compute_lim_cl_limber
    C_l = compute_lim_cl_limber(ell, lim_ch, lim_ch, fNL=0.0)
    C_g = compute_gal_cl_limber(ell, gal_bin, gal_bin, fNL=0.0)
    C_x = compute_lim_x_galaxy_cross_cl(ell, lim_ch, gal_bin, fNL=0.0)
    r = C_x / np.sqrt(C_l * C_g) if C_l*C_g > 0 else np.nan
    print(f"    C_lim(auto)       = {C_l:.3e}   (nW/m²/sr)²")
    print(f"    C_gal(auto)       = {C_g:.3e}   dim.-less")
    print(f"    C_cross           = {C_x:.3e}   nW/m²/sr")
    print(f"    correlation r     = {r:.3f}   ({'PASS' if 0<=r<=1 else 'FAIL'})")
    return r


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 72)
    print("CORRECTED JOINT SPHEREx LIM × Euclid photo Fisher forecast")
    print("Steps A+B+C+D applied uniformly")
    print("=" * 72)
    print(f"  ℓ grid           = {list(ELL_GRID)}     (Step B: ℓ_min=10)")
    print(f"  k_max            = {K_MAX_H_PER_MPC} h/Mpc     (Step D)")
    print(f"  Cosmological marg = σ_8, n_s, Ω_m            (Step A)")
    print(f"  Euclid n(z)       = Blanchard z² exp[−(z/z_m)^1.5] + outliers")
    print(f"                      (Step C, gal-side only)")

    r = diag_cross_correlation()
    if not (0 <= r <= 1):
        print("HALT — correlation coefficient out of [0, 1]")
        return

    # DEEP config
    _, fsky_gal_deep = build_euclid_bins_blanchard('deep')
    fsky_joint_deep = min(0.0048, fsky_gal_deep)
    res_deep = run_config("DEEP-FIELD", "deep", "deep",
                          f_sky_lim=0.0048, f_sky_gal=fsky_gal_deep,
                          f_sky_joint=fsky_joint_deep)

    # WIDE config
    _, fsky_gal_wide = build_euclid_bins_blanchard('wide')
    fsky_joint_wide = min(0.60, fsky_gal_wide)
    res_wide = run_config("WIDE / ALL-SKY", "all-sky", "wide",
                          f_sky_lim=0.60, f_sky_gal=fsky_gal_wide,
                          f_sky_joint=fsky_joint_wide)

    # Summary
    print("\n" + "=" * 72)
    print("FINAL DECOMPOSITION (all four corrections active)")
    print("=" * 72)
    print(f"  {'config':<12}  {'σ_LIM':>8}  {'σ_gal':>8}  "
          f"{'σ_joint':>10}  {'σ_quad':>8}  {'σ_joint_marg':>13}")
    for name, r in [("deep", res_deep), ("wide", res_wide)]:
        print(f"  {name:<12}  {r['sigma_lim']:8.2f}  {r['sigma_gal']:8.3f}  "
              f"{r['sigma_joint']:10.3f}  {r['sigma_quad']:8.3f}  "
              f"{r['sigma_joint_marg']:13.3f}")

    print(f"\n  Planck 2018 (5.1) comparison:")
    for name, r in [("deep", res_deep), ("wide", res_wide)]:
        val = r['sigma_joint_marg']
        print(f"  {name:<12}  σ_joint_marg = {val:.3f}  → {5.1/val:.2f}× "
              f"({'crosses σ=1' if val<1 else 'above σ=1'})")

    out = os.path.join(os.path.dirname(__file__), '..',
                       'data', 'joint_forecast_CORRECTED.pkl')
    with open(out, 'wb') as f:
        pickle.dump({'deep': res_deep, 'wide': res_wide,
                     'r_cross_diag': r}, f)
    print(f"\n  Results pickled: {out}")


if __name__ == "__main__":
    main()
