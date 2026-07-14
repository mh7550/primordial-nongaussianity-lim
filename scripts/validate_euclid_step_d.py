"""
validate_euclid_step_d.py — add k_max = 0.3 h/Mpc nonlinear cutoff
(cumulative on Steps A + B + C).
"""
import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

import cosmology
from validate_euclid_step_a import (dC_dsigma8, dC_dns_matrix, dC_dOm,
                                     FNL_FID, DELTA_FNL, FSKY_WIDE)
from validate_euclid_step_c import build_euclid_bins_blanchard
import limber

ELL_GRID_D = np.array([10, 20, 40, 80, 150, 250])   # same ℓ grid as Step C — isolate k_max effect
K_MAX = 0.3        # h/Mpc, Blanchard 'mean-of-two-recipes' choice


def _k_eff(ell, z_bar):
    chi = cosmology.get_comoving_distance(z_bar)
    return (ell + 0.5) / max(chi, 1.0)


def gal_cls_matrix_kmax(ell, gal_bins, fNL, k_max=K_MAX):
    """Same as gal_cls_matrix but zero any pair whose k_eff > k_max."""
    Ng = len(gal_bins)
    C = np.zeros((Ng, Ng))
    for i in range(Ng):
        for j in range(i, Ng):
            z_bar = 0.5 * (gal_bins[i]['z_peak'] + gal_bins[j]['z_peak'])
            if _k_eff(ell, z_bar) > k_max:
                continue        # nonlinear scale — drop
            v = limber.compute_gal_cl_limber(ell, gal_bins[i], gal_bins[j], fNL=fNL)
            C[i, j] = v
            C[j, i] = v
    return C


def dC_dOm_kmax(ell, gal_bins, k_max=K_MAX):
    Om_fid = cosmology.Om0
    cosmology.Om0 = Om_fid + 0.01
    cosmology.Ode0 = 1.0 - cosmology.Om0
    import bias_functions
    bias_functions.OMEGA_M = cosmology.Om0
    limber.clear_intensity_cache()
    C_plus = gal_cls_matrix_kmax(ell, gal_bins, FNL_FID, k_max)
    cosmology.Om0 = Om_fid - 0.01
    cosmology.Ode0 = 1.0 - cosmology.Om0
    bias_functions.OMEGA_M = cosmology.Om0
    limber.clear_intensity_cache()
    C_minus = gal_cls_matrix_kmax(ell, gal_bins, FNL_FID, k_max)
    cosmology.Om0 = Om_fid
    cosmology.Ode0 = 1.0 - cosmology.Om0
    bias_functions.OMEGA_M = cosmology.Om0
    limber.clear_intensity_cache()
    return (C_plus - C_minus) / 0.02


def main():
    print("=" * 72)
    print(f"STEP D — add k_max = {K_MAX} h/Mpc nonlinear cutoff "
          f"(cumulative on A+B+C)")
    print("=" * 72)

    gal_bins, fsky = build_euclid_bins_blanchard('wide')
    N_gal = np.asarray([b['noise'] for b in gal_bins])

    # Report which ℓ×bin-pairs get cut
    print(f"  Wide config, N_bins = {len(gal_bins)}, "
          f"ℓ grid = {list(ELL_GRID_D)}")
    print(f"\n  k_eff (ℓ, z_bar) [h/Mpc] — cells beyond k_max={K_MAX} will be dropped")
    header_zs = ["z={:.2f}".format(b['z_peak']) for b in gal_bins]
    print(f"  {'ℓ':>4}  " + "  ".join(header_zs))
    for ell in ELL_GRID_D:
        row = [f"{_k_eff(ell, b['z_peak']):.3f}" for b in gal_bins]
        row_marked = [f"{v:>6s}{'*' if float(v)>K_MAX else ' '}" for v in row]
        print(f"  {ell:>4d}  " + " ".join(row_marked))

    n_par = 4
    F = np.zeros((n_par, n_par))
    sigma8 = cosmology.sigma8
    for ell in ELL_GRID_D:
        C_fid = gal_cls_matrix_kmax(int(ell), gal_bins, FNL_FID)
        if not np.any(C_fid):
            continue
        S = C_fid + np.diag(N_gal)
        C_plus  = gal_cls_matrix_kmax(int(ell), gal_bins, FNL_FID + DELTA_FNL)
        C_minus = gal_cls_matrix_kmax(int(ell), gal_bins, FNL_FID - DELTA_FNL)
        dS_fNL = (C_plus - C_minus) / (2.0 * DELTA_FNL)
        dS_s8  = dC_dsigma8(C_fid, sigma8)
        dS_ns  = dC_dns_matrix(ell, gal_bins, C_fid)
        dS_Om  = dC_dOm_kmax(ell, gal_bins)
        try:
            Sinv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            continue
        weight = (2.0 * ell + 1.0) * FSKY_WIDE / 2.0
        M = [Sinv @ dS for dS in (dS_fNL, dS_s8, dS_ns, dS_Om)]
        for a in range(n_par):
            for b in range(a, n_par):
                v = weight * np.trace(M[a] @ M[b])
                F[a, b] += v
                if a != b:
                    F[b, a] += v

    cov = np.linalg.pinv(F)
    sig_marg = float(np.sqrt(cov[0, 0]))
    print(f"\n  σ_gal_wide (marg., A+B+C+D)   = {sig_marg:.3f}")
    print(f"  factor change vs Step C (8.99) = ×{sig_marg/8.99:.2f}")
    print(f"  factor change vs baseline 1.07 = ×{sig_marg/1.07:.2f}")
    print(f"  Blanchard target range          = 5–6")
    print(f"  within 2× of Blanchard (2.5..12) = {2.5 <= sig_marg <= 12.0}")


if __name__ == "__main__":
    main()
