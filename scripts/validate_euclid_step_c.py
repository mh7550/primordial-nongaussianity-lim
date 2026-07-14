"""
validate_euclid_step_c.py — replace Euclid n(z) and photo-z scatter
with Blanchard+2020 IST fiducial (cumulative on Steps A + B).

Changes vs Step B:
  * n(z) ∝ z² exp[−(z/z_m)^{1.5}],  z_m = z_median/√2 ≈ 0.636  (Euclid photo)
    → recompute per-bin n_bar (galaxies actually falling into each bin)
  * Effective σ_z per bin = σ_core × √(1 + 4 f_out) with f_out = 0.10
    → 18% broadening on top of the Gaussian core
"""
import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

import cosmology
from validate_euclid_step_a import (gal_cls_matrix, dC_dsigma8, dC_dns_matrix,
                                     dC_dOm, FNL_FID, DELTA_FNL, FSKY_WIDE)
from survey_specs import (build_euclid_bins, EUCLID_Z_BINS,
                          EUCLID_N_STER, euclid_bias, euclid_photoz_sigma,
                          _bin_effective_sigma)

ELL_GRID_C = np.array([10, 20, 40, 80, 150, 250])   # ℓ_min = 10 (Step B)


def blanchard_nz(z, z_median=0.9):
    """Un-normalised Euclid photo n(z) — Blanchard+2020 Eq. C.5."""
    z_m = z_median / np.sqrt(2.0)
    return z ** 2 * np.exp(-(z / z_m) ** 1.5)


def build_euclid_bins_blanchard(config='wide', f_out=0.10):
    """
    Rebuild the Euclid bins with:
      (a) per-bin n_bar from ∫nz(z) dz inside the bin (n(z) shape from
          Blanchard),
      (b) effective σ_z inflated by √(1 + 4 f_out) to account for the
          10% outlier tail.
    """
    if config == 'wide':
        f_sky = 0.35148
    else:
        f_sky = 0.00121

    # Normalise n(z) so the total over z ∈ [0, 2] equals EUCLID_N_STER
    z_dense = np.linspace(0.0, 2.5, 500)
    nz_raw = blanchard_nz(z_dense)
    norm = np.trapezoid(nz_raw, z_dense)
    nz_scaled = nz_raw * (EUCLID_N_STER / norm)

    bins = []
    for z_lo, z_hi in EUCLID_Z_BINS:
        z_c = 0.5 * (z_lo + z_hi)
        mask = (z_dense >= z_lo) & (z_dense <= z_hi)
        n_bar = float(np.trapezoid(nz_scaled[mask], z_dense[mask]))
        # Core σ_z from bin width + Gaussian photo-z scatter
        sigma_core = _bin_effective_sigma(z_lo, z_hi)
        # Outlier broadening: √(1 + 4 f_out) per Blanchard+2020 Eq. C.28
        sigma_eff = sigma_core * np.sqrt(1.0 + 4.0 * f_out)
        bins.append(dict(
            kind='gal',
            z_peak=float(z_c),
            z_edges=(float(z_lo), float(z_hi)),
            sigma_z=float(sigma_eff),
            b_g_of_z=(lambda z, _zc=z_c: euclid_bias(z)),
            n_bar=n_bar,
            noise=float(1.0 / n_bar) if n_bar > 0 else np.inf,
            b_scale=1.0,
            I_scale=1.0,
            f_sky_config=f_sky,
        ))
    return bins, f_sky


def main():
    print("=" * 72)
    print("STEP C — Blanchard n(z) + outlier-inflated σ_z (cumulative on A+B)")
    print("=" * 72)
    gal_bins, fsky = build_euclid_bins_blanchard('wide')
    N_gal = np.asarray([b['noise'] for b in gal_bins])
    print(f"  Fiducial cosmology: σ_8 = {cosmology.sigma8}, "
          f"n_s = {cosmology.ns}, Ω_m = {cosmology.Om0}")
    print(f"  Blanchard n(z) shape:  z² exp[−(z/z_m)^1.5],  z_m ≈ 0.636")
    print(f"  Outlier fraction:      f_out = 0.10 → σ_z × √1.4 = ×1.18")
    print(f"  f_sky = {fsky:.5f}, N_bins = {len(gal_bins)}, "
          f"ℓ grid = {list(ELL_GRID_C)}")
    print()
    print(f"  {'bin':<3} {'z_c':>5} {'σ_eff':>7} {'n_bar (gal/sr)':>15} "
          f"{'shot noise':>12}")
    for i, b in enumerate(gal_bins):
        print(f"  {i:<3d} {b['z_peak']:>5.2f} {b['sigma_z']:>7.4f} "
              f"{b['n_bar']:>15.3e} {b['noise']:>12.3e}")

    n_par = 4
    F = np.zeros((n_par, n_par))
    sigma8 = cosmology.sigma8
    for ell in ELL_GRID_C:
        C_fid = gal_cls_matrix(int(ell), gal_bins, FNL_FID)
        S = C_fid + np.diag(N_gal)
        C_plus  = gal_cls_matrix(int(ell), gal_bins, FNL_FID + DELTA_FNL)
        C_minus = gal_cls_matrix(int(ell), gal_bins, FNL_FID - DELTA_FNL)
        dS_fNL = (C_plus - C_minus) / (2.0 * DELTA_FNL)
        dS_s8  = dC_dsigma8(C_fid, sigma8)
        dS_ns  = dC_dns_matrix(ell, gal_bins, C_fid)
        dS_Om  = dC_dOm(ell, gal_bins, C_fid, N_gal)
        Sinv = np.linalg.inv(S)
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
    print(f"\n  σ_gal_wide (marg., Blanchard n(z) + outliers) = {sig_marg:.3f}")
    print(f"  factor change vs Step B (6.80)  = ×{sig_marg/6.80:.2f}")
    print(f"  factor change vs baseline (1.07) = ×{sig_marg/1.07:.2f}")
    print(f"  within 2× of Blanchard 5–6      = {2.5 <= sig_marg <= 12.0}")


if __name__ == "__main__":
    main()
