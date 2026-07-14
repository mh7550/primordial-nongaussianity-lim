"""
joint_forecast_driver.py — SPHEREx LIM × Euclid photometric joint Fisher forecast.

Runs, for both the Deep and Wide configurations:
  (i)   LIM-only
  (ii)  Galaxy-only  (Step 2 baseline)
  (iii) Joint (LIM + gal + cross)
Then marginalises the joint case over (A_i, B_i, b_g^a) — Step 6.

No target tuning. Reports whatever the pipeline produces.
"""
import numpy as np
import sys, os, time, pickle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from lim_channels import build_92_channels, LINE_ORDER
from survey_specs import build_euclid_bins, EUCLID_N_ZBINS
from limber import (compute_lim_cls_matrix, compute_gal_cl_limber,
                    compute_lim_x_galaxy_cross_cl, compute_joint_cls_matrix,
                    clear_intensity_cache)


ELL_GRID = np.array([2, 5, 10, 20, 40, 80, 150, 250])
FNL_FID = 1.0
DELTA_FNL = 0.1
DELTA_NUISANCE = 0.01
PLANCK_SIGMA = 5.1


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def gal_cls_matrix(ell, gal_bins, fNL):
    Ng = len(gal_bins)
    C = np.zeros((Ng, Ng))
    for i in range(Ng):
        for j in range(i, Ng):
            v = compute_gal_cl_limber(ell, gal_bins[i], gal_bins[j], fNL=fNL)
            C[i, j] = v
            C[j, i] = v
    return C


def _apply_lim_AB(lim_ch, A_by_line, B_by_line):
    out = []
    for ch in lim_ch:
        c = dict(ch)
        c['I_scale'] = A_by_line[ch['line']]
        c['b_scale'] = B_by_line[ch['line']]
        out.append(c)
    return out


def _apply_gal_bias(gal_bins, b_scale_vec):
    out = []
    for i, b in enumerate(gal_bins):
        c = dict(b)
        c['b_scale'] = b_scale_vec[i]
        out.append(c)
    return out


# ---------------------------------------------------------------------------
# Fisher trace utilities
# ---------------------------------------------------------------------------

def trace_fisher_from_sigmas(Sigma, Sp, Sm, delta, ell, f_sky):
    """Return the 1×1 f_NL Fisher contribution from a single ℓ."""
    dS = (Sp - Sm) / (2.0 * delta)
    Sinv = np.linalg.inv(Sigma)
    weight = (2.0 * ell + 1.0) * f_sky / 2.0
    M = Sinv @ dS
    return weight * float(np.trace(M @ M))


# ---------------------------------------------------------------------------
# Single-config runner
# ---------------------------------------------------------------------------

def run_config(config_name, f_sky_joint, lim_channels, gal_bins,
               N_lim, N_gal, do_marginalise=False):
    """
    Run LIM-only, gal-only, and joint Fisher for one configuration.

    f_sky_joint is used for the joint trace (min of the two surveys' f_sky).
    Individual LIM-only / gal-only use their own f_sky (passed via
    lim_channels['f_sky_config'] / gal_bins[0]['f_sky_config']).
    """
    Nl = len(lim_channels)
    Ng = len(gal_bins)
    print("\n" + "═" * 70)
    print(f"CONFIG: {config_name}")
    print(f"  f_sky (LIM):   {lim_channels[0].get('f_sky_config', None)}")
    print(f"  f_sky (gal):   {gal_bins[0].get('f_sky_config', None)}")
    print(f"  f_sky (joint): {f_sky_joint}")
    print("═" * 70)

    # ---- (i) LIM-only Fisher ---------------------------------------------
    print("\n(i) LIM-only Fisher (92×92 Limber, unmarg.)")
    f_sky_lim = lim_channels[0].get('f_sky_config', 0.60)
    F_lim = 0.0
    for ell in ELL_GRID:
        clear_intensity_cache()
        Sf = compute_lim_cls_matrix(int(ell), lim_channels, fNL=FNL_FID,
                                    use_bessel_below_limber=False,
                                    use_rsd=False) + np.diag(N_lim)
        Sp = compute_lim_cls_matrix(int(ell), lim_channels, fNL=FNL_FID+DELTA_FNL,
                                    use_bessel_below_limber=False,
                                    use_rsd=False) + np.diag(N_lim)
        Sm = compute_lim_cls_matrix(int(ell), lim_channels, fNL=FNL_FID-DELTA_FNL,
                                    use_bessel_below_limber=False,
                                    use_rsd=False) + np.diag(N_lim)
        F_lim += trace_fisher_from_sigmas(Sf, Sp, Sm, DELTA_FNL, ell, f_sky_lim)
    sigma_lim = 1.0/np.sqrt(F_lim) if F_lim > 0 else np.inf
    print(f"    σ(LIM-only)  = {sigma_lim:.3f}")

    # ---- (ii) Gal-only Fisher --------------------------------------------
    print("\n(ii) Galaxy-only Fisher (Euclid photo, N_bins × N_bins)")
    f_sky_gal = gal_bins[0].get('f_sky_config', 0.35)
    F_gal = 0.0
    for ell in ELL_GRID:
        Sf = gal_cls_matrix(int(ell), gal_bins, FNL_FID) + np.diag(N_gal)
        Sp = gal_cls_matrix(int(ell), gal_bins, FNL_FID+DELTA_FNL) + np.diag(N_gal)
        Sm = gal_cls_matrix(int(ell), gal_bins, FNL_FID-DELTA_FNL) + np.diag(N_gal)
        F_gal += trace_fisher_from_sigmas(Sf, Sp, Sm, DELTA_FNL, ell, f_sky_gal)
    sigma_gal = 1.0/np.sqrt(F_gal) if F_gal > 0 else np.inf
    print(f"    σ(gal-only)  = {sigma_gal:.3f}")

    # ---- (iii) Joint Fisher ---------------------------------------------
    print("\n(iii) Joint Fisher ({0}+{1})×({0}+{1}) with cross-block".format(Nl, Ng))
    N_joint = np.concatenate([N_lim, N_gal])
    F_joint = 0.0
    for ell in ELL_GRID:
        clear_intensity_cache()
        t0 = time.time()
        Sf = compute_joint_cls_matrix(int(ell), lim_channels, gal_bins,
                                       fNL=FNL_FID) + np.diag(N_joint)
        Sp = compute_joint_cls_matrix(int(ell), lim_channels, gal_bins,
                                       fNL=FNL_FID+DELTA_FNL) + np.diag(N_joint)
        Sm = compute_joint_cls_matrix(int(ell), lim_channels, gal_bins,
                                       fNL=FNL_FID-DELTA_FNL) + np.diag(N_joint)
        F_joint += trace_fisher_from_sigmas(Sf, Sp, Sm, DELTA_FNL, ell, f_sky_joint)
        sig = 1.0/np.sqrt(F_joint) if F_joint > 0 else np.inf
        print(f"    ℓ={ell:>4d}: {time.time()-t0:5.1f}s  σ_joint_run = {sig:.3f}")
    sigma_joint = 1.0/np.sqrt(F_joint) if F_joint > 0 else np.inf

    # Quadrature reference
    sigma_quad = 1.0/np.sqrt(1.0/sigma_lim**2 + 1.0/sigma_gal**2)

    print(f"\n    σ(LIM-only)                   = {sigma_lim:.3f}")
    print(f"    σ(gal-only)                   = {sigma_gal:.3f}")
    print(f"    σ(joint, unmarg.)             = {sigma_joint:.3f}")
    print(f"    σ(quadrature, no cross-block) = {sigma_quad:.3f}")
    if sigma_joint > 0:
        gain_over_naive = sigma_quad / sigma_joint
        print(f"    cross-block gain vs quadrature = ×{gain_over_naive:.3f}")

    out = dict(sigma_lim=sigma_lim, sigma_gal=sigma_gal,
               sigma_joint=sigma_joint, sigma_quad=sigma_quad,
               f_sky_joint=f_sky_joint, f_sky_lim=f_sky_lim, f_sky_gal=f_sky_gal)

    # ---- (iv) Marginalisation --------------------------------------------
    if do_marginalise:
        print("\n(iv) Joint 9+N_gal Fisher — marginalise over "
              "(f_NL, {A,B}_i × 4, b_g × N_gal)")
        sigma_marg = _joint_marginalised_fisher(
            lim_channels, gal_bins, N_lim, N_gal, f_sky_joint)
        out['sigma_joint_marg'] = sigma_marg
        print(f"    σ(joint, marginalised)        = {sigma_marg:.3f}")
        print(f"    marginalisation penalty       = ×{sigma_marg/sigma_joint:.2f}")

    return out


# ---------------------------------------------------------------------------
# Joint 9+N_gal marginalised Fisher (Step 6)
# ---------------------------------------------------------------------------

def _joint_marginalised_fisher(lim_ch, gal_bins, N_lim, N_gal, f_sky):
    """
    Marginalise over: f_NL,  4 A_i (LIM intensity), 4 B_i (LIM bias),
    N_gal b_g^a (galaxy biases). Cosmological params (n_s, σ_8) not
    included per Prof. Pullen's argument that Planck constrains them
    at percent level.
    """
    Nl = len(lim_ch)
    Ng = len(gal_bins)
    N_joint = np.concatenate([N_lim, N_gal])

    A0 = {l: 1.0 for l in LINE_ORDER}
    B0 = A0.copy()
    bg0 = np.ones(Ng)

    def _sig(A, B, bg, fNL):
        L = _apply_lim_AB(lim_ch, A, B)
        G = _apply_gal_bias(gal_bins, bg)
        C = compute_joint_cls_matrix(ell, L, G, fNL=fNL)
        return C + np.diag(N_joint)

    n_par = 1 + 8 + Ng   # f_NL + (A_i,B_i)×4 + b_g^a per z-bin
    F_total = np.zeros((n_par, n_par))

    for ell in ELL_GRID:
        clear_intensity_cache()
        t0 = time.time()
        S = _sig(A0, B0, bg0, FNL_FID)
        try:
            Sinv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            continue

        # Derivatives
        dS_list = []
        # f_NL
        Sp = _sig(A0, B0, bg0, FNL_FID + DELTA_FNL)
        Sm = _sig(A0, B0, bg0, FNL_FID - DELTA_FNL)
        dS_list.append((Sp - Sm) / (2.0 * DELTA_FNL))
        # A_i, B_i (line-by-line)
        for line in LINE_ORDER:
            Ap = A0.copy(); Ap[line] = 1.0 + DELTA_NUISANCE
            Am = A0.copy(); Am[line] = 1.0 - DELTA_NUISANCE
            dS_list.append((_sig(Ap, B0, bg0, FNL_FID) -
                            _sig(Am, B0, bg0, FNL_FID)) / (2.0 * DELTA_NUISANCE))
            Bp = B0.copy(); Bp[line] = 1.0 + DELTA_NUISANCE
            Bm = B0.copy(); Bm[line] = 1.0 - DELTA_NUISANCE
            dS_list.append((_sig(A0, Bp, bg0, FNL_FID) -
                            _sig(A0, Bm, bg0, FNL_FID)) / (2.0 * DELTA_NUISANCE))
        # b_g^a per z-bin
        for a in range(Ng):
            bgp = bg0.copy(); bgp[a] = 1.0 + DELTA_NUISANCE
            bgm = bg0.copy(); bgm[a] = 1.0 - DELTA_NUISANCE
            dS_list.append((_sig(A0, B0, bgp, FNL_FID) -
                            _sig(A0, B0, bgm, FNL_FID)) / (2.0 * DELTA_NUISANCE))

        M_list = [Sinv @ dS for dS in dS_list]
        weight = (2.0 * ell + 1.0) * f_sky / 2.0
        for a in range(n_par):
            for b in range(a, n_par):
                v = weight * np.trace(M_list[a] @ M_list[b])
                F_total[a, b] += v
                if a != b:
                    F_total[b, a] += v
        print(f"    ℓ={ell:>4d}: {time.time()-t0:5.1f}s   9+{Ng} Fisher assembled")

    cond = np.linalg.cond(F_total)
    print(f"    Joint marginalised Fisher condition number: {cond:.3e}")
    if cond > 1e12:
        print(f"    ⚠ condition > 10^12 — regularising with 1e-30 × I")
        F_total = F_total + 1e-30 * np.eye(n_par)
    cov = np.linalg.pinv(F_total)
    return float(np.sqrt(cov[0, 0]))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("SPHEREx LIM × Euclid Photometric — Joint Fisher Forecast")
    print("=" * 70)

    # DEEP-FIELD CONFIG
    lim_deep = build_92_channels(mode='deep')
    for ch in lim_deep:
        ch['f_sky_config'] = 0.0048
    N_lim_deep = np.asarray([ch['noise'] for ch in lim_deep])
    gal_deep, fsky_gal_deep = build_euclid_bins('deep')
    N_gal_deep = np.asarray([b['noise'] for b in gal_deep])
    # joint f_sky = min(surveys), Euclid Deep 50 deg² ≈ overlaps with SPHEREx 200 deg²
    fsky_joint_deep = min(0.0048, fsky_gal_deep)
    res_deep = run_config("DEEP-FIELD", fsky_joint_deep,
                          lim_deep, gal_deep, N_lim_deep, N_gal_deep,
                          do_marginalise=True)

    # ALL-SKY CONFIG
    lim_wide = build_92_channels(mode='all-sky')
    for ch in lim_wide:
        ch['f_sky_config'] = 0.60
    N_lim_wide = np.asarray([ch['noise'] for ch in lim_wide])
    gal_wide, fsky_gal_wide = build_euclid_bins('wide')
    N_gal_wide = np.asarray([b['noise'] for b in gal_wide])
    fsky_joint_wide = min(0.60, fsky_gal_wide)
    res_wide = run_config("WIDE / ALL-SKY", fsky_joint_wide,
                          lim_wide, gal_wide, N_lim_wide, N_gal_wide,
                          do_marginalise=True)

    # -------------------------------------------------------------
    # Summary table
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("DECOMPOSITION SUMMARY")
    print("=" * 70)
    print(f"  {'config':<12}  {'σ_LIM':>8}  {'σ_gal':>8}  {'σ_joint':>10}  "
          f"{'σ_quad':>8}  {'σ_joint_marg':>13}")
    for name, r in [("deep", res_deep), ("wide", res_wide)]:
        print(f"  {name:<12}  {r['sigma_lim']:8.2f}  {r['sigma_gal']:8.3f}  "
              f"{r['sigma_joint']:10.3f}  {r['sigma_quad']:8.3f}  "
              f"{r.get('sigma_joint_marg', float('nan')):13.3f}")
    print()
    print("vs Planck (5.1):")
    for name, r in [("deep", res_deep), ("wide", res_wide)]:
        print(f"  {name:<12}  joint marg σ = {r.get('sigma_joint_marg', np.nan):.3f}  "
              f"→ {5.1/r.get('sigma_joint_marg', np.nan):.2f}× Planck")

    # Save
    out = os.path.join(os.path.dirname(__file__), '..',
                       'data', 'joint_forecast_results.pkl')
    with open(out, 'wb') as f:
        pickle.dump({'deep': res_deep, 'wide': res_wide}, f)
    print(f"\n  Results pickled: {out}")

    return res_deep, res_wide


if __name__ == "__main__":
    main()
