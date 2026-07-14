"""
validate_euclid_step_a.py — Add cosmological marginalisation to Euclid photo.

Parameters marginalised:
    f_NL, σ_8, n_s, Ω_m       (4 × 4 Fisher)

σ_8, n_s derivatives are analytic:
    ∂P/∂σ_8 = 2P/σ_8       (σ_8 is P amplitude prefactor)
    ∂P/∂n_s = P × ln(k/k_p)  (k_p = 0.05 h/Mpc, Planck convention)

Ω_m derivative is numerical: perturb src.cosmology.Om0 by ±1%, rebuild
the growth factor and transfer function and re-compute the C_ℓ matrix.
"""
import numpy as np
import sys, os, time, importlib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import cosmology
from survey_specs import build_euclid_bins
import limber

ELL_GRID = np.array([2, 5, 10, 20, 40, 80, 150, 250])
K_PIVOT = 0.05          # h/Mpc (Planck convention)
FNL_FID = 1.0
DELTA_FNL = 0.1
DELTA_OM = 0.01
FSKY_WIDE = 0.35


def gal_cls_matrix(ell, gal_bins, fNL):
    Ng = len(gal_bins)
    C = np.zeros((Ng, Ng))
    for i in range(Ng):
        for j in range(i, Ng):
            v = limber.compute_gal_cl_limber(ell, gal_bins[i], gal_bins[j], fNL=fNL)
            C[i, j] = v
            C[j, i] = v
    return C


def keff_ell(ell, gal_bin):
    """Effective k for this ℓ and bin (Limber, at z_bar of the bin)."""
    z = gal_bin['z_peak']
    chi = cosmology.get_comoving_distance(z)
    return (ell + 0.5) / max(chi, 1.0)


def dC_dsigma8(C_fid, sigma8):
    """∂C/∂σ_8 = 2 C / σ_8 (analytic, C ∝ σ_8²)."""
    return 2.0 * C_fid / sigma8


def dC_dns_matrix(ell, gal_bins, C_fid):
    """
    ∂C/∂n_s = C × ln(k_eff / k_pivot), where k_eff is the Limber k at bin z_bar.
    For a cross entry (i, j) we use the geometric mean of the two k_eff values
    (since the pair overlap is dominated by the shared z_bar).
    """
    Ng = len(gal_bins)
    k_arr = np.asarray([keff_ell(ell, b) for b in gal_bins])
    lnk = np.log(k_arr / K_PIVOT)
    D = np.zeros((Ng, Ng))
    for i in range(Ng):
        for j in range(i, Ng):
            # symmetric geometric mean of the two log(k)
            factor = 0.5 * (lnk[i] + lnk[j])
            D[i, j] = C_fid[i, j] * factor
            D[j, i] = D[i, j]
    return D


def _rebuild_cosmology_for_Om(Om_new):
    """Monkey-patch cosmology.Om0 and Ode0 for spatially flat ΛCDM."""
    cosmology.Om0 = Om_new
    cosmology.Ode0 = 1.0 - Om_new
    # Bias functions cache Om0 at import — patch that too.
    import bias_functions
    bias_functions.OMEGA_M = Om_new
    # Clear intensity cache since Ī depends on H(z) → Om.
    limber.clear_intensity_cache()


def dC_dOm(ell, gal_bins, C_fid, N_diag):
    """Numerical ∂C/∂Ω_m via ±1% perturbation of cosmology.Om0."""
    Om_fid = cosmology.Om0
    _rebuild_cosmology_for_Om(Om_fid + DELTA_OM)
    C_plus = gal_cls_matrix(ell, gal_bins, FNL_FID)
    _rebuild_cosmology_for_Om(Om_fid - DELTA_OM)
    C_minus = gal_cls_matrix(ell, gal_bins, FNL_FID)
    _rebuild_cosmology_for_Om(Om_fid)  # restore
    return (C_plus - C_minus) / (2.0 * DELTA_OM)


def main():
    print("=" * 72)
    print("STEP A — cosmological marginalisation on Euclid-photo-alone Wide")
    print("=" * 72)

    gal_bins, fsky = build_euclid_bins('wide')
    N_gal = np.asarray([b['noise'] for b in gal_bins])
    Ng = len(gal_bins)
    n_par = 4     # f_NL, σ_8, n_s, Ω_m

    sigma8 = cosmology.sigma8
    Om0    = cosmology.Om0
    ns     = cosmology.ns
    print(f"  Fiducial cosmology: σ_8 = {sigma8}, n_s = {ns}, Ω_m = {Om0}")
    print(f"  Nuisance / cosmo parameters marginalised: [f_NL, σ_8, n_s, Ω_m]")
    print(f"  f_sky = {FSKY_WIDE}, N_bins = {Ng}, ℓ grid = {list(ELL_GRID)}")
    print()

    F_total = np.zeros((n_par, n_par))
    for ell in ELL_GRID:
        t0 = time.time()
        C_fid = gal_cls_matrix(int(ell), gal_bins, FNL_FID)
        S = C_fid + np.diag(N_gal)
        # f_NL derivative (finite difference)
        C_plus = gal_cls_matrix(int(ell), gal_bins, FNL_FID + DELTA_FNL)
        C_minus = gal_cls_matrix(int(ell), gal_bins, FNL_FID - DELTA_FNL)
        dS_fNL = (C_plus - C_minus) / (2.0 * DELTA_FNL)
        # σ_8 derivative (analytic)
        dS_s8  = dC_dsigma8(C_fid, sigma8)
        # n_s derivative (analytic)
        dS_ns  = dC_dns_matrix(ell, gal_bins, C_fid)
        # Ω_m derivative (numerical, module-patch)
        dS_Om  = dC_dOm(ell, gal_bins, C_fid, N_gal)

        try:
            Sinv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            continue
        weight = (2.0 * ell + 1.0) * FSKY_WIDE / 2.0
        M = [Sinv @ dS for dS in (dS_fNL, dS_s8, dS_ns, dS_Om)]
        for a in range(n_par):
            for b in range(a, n_par):
                v = weight * np.trace(M[a] @ M[b])
                F_total[a, b] += v
                if a != b:
                    F_total[b, a] += v
        cov = np.linalg.pinv(F_total)
        sig = float(np.sqrt(cov[0, 0]))
        print(f"    ℓ={ell:>4d}: {time.time()-t0:5.2f}s   σ_marg = {sig:.3f}")

    cond = np.linalg.cond(F_total)
    print(f"\n  4×4 Fisher condition number: {cond:.3e}")
    cov = np.linalg.pinv(F_total)
    sig_marg = float(np.sqrt(cov[0, 0]))
    sig_unm  = float(1.0 / np.sqrt(F_total[0, 0]))
    print(f"\n  σ_gal_wide (unmarg.)         = {sig_unm:.3f}")
    print(f"  σ_gal_wide (marg. σ_8,n_s,Ω_m) = {sig_marg:.3f}")
    print(f"  factor change vs baseline 1.07 = ×{sig_marg/1.07:.2f}")
    print(f"  Blanchard target range         = 5–6")
    print(f"  within 2× of Blanchard         = {1.5 <= sig_marg <= 12.0}")


if __name__ == "__main__":
    main()
