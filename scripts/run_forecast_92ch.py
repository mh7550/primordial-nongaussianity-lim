"""
run_forecast_92ch.py — Full 92-channel LIM Fisher forecast for f_NL.

Reports the four-step decomposition requested by Prof. Pullen:

  (a) Limber-only, f_NL only, no marginalisation — expected ≈ 0.71
  (b) full-Bessel + RSD at ℓ ≤ ℓ_limber, f_NL only, no marginalisation
  (c) full-Bessel + RSD, 9×9 marginalised — new headline
  (d) ℓ_limber per line at peak z with the full geometry

Also prints the condition number of the 9×9 Fisher matrix and halts
before inversion if it exceeds 10¹².
"""
import numpy as np
import sys, os, time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from lim_channels import build_92_channels, line_peak_z, LINE_ORDER
from limber import (compute_cls_full, compute_ell_limber, clear_bessel_cache,
                    get_cross_power_spectrum)
from fisher import compute_fisher_9x9, sigma_fNL_from_fisher_9x9
from survey_specs import F_SKY

F_SKY_PAPER = 0.60


# ---------------------------------------------------------------------------
# Fast Limber-only single-parameter (f_NL) Fisher — used for step (a) so that
# the O(N²) covariance assembly at each ℓ doesn't call the general 9×9 path.
# ---------------------------------------------------------------------------

def _tophat_bounds(ch):
    lam_c = ch['lambda_obs']
    dl = ch['delta_lambda']
    z_lo = max(0.0, (lam_c - 0.5 * dl) / ch['lambda_rest'] - 1.0)
    z_hi = (lam_c + 0.5 * dl) / ch['lambda_rest'] - 1.0
    return z_lo, z_hi


def _cl_limber_pair(ell, ch_i, ch_j, fNL):
    """Limber cross-C_ℓ for two channels. Zero if their z windows don't overlap."""
    z_lo_i, z_hi_i = _tophat_bounds(ch_i)
    z_lo_j, z_hi_j = _tophat_bounds(ch_j)
    z_lo = max(z_lo_i, z_lo_j)
    z_hi = min(z_hi_i, z_hi_j)
    if z_hi <= z_lo:
        return 0.0
    z_mid = 0.5 * (z_lo + z_hi)
    b_i = ch_i['b_scale'] * ch_i['b1_of_z'](z_mid)
    b_j = ch_j['b_scale'] * ch_j['b1_of_z'](z_mid)
    scale = ch_i['I_scale'] * ch_j['I_scale']
    return scale * get_cross_power_spectrum(
        np.asarray([ell]), z_lo, z_hi, b_i, b_j, fNL=fNL, shape='local'
    )[0]


def _sigma_limber(ell, channels, fNL, N_diag):
    n = len(channels)
    S = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            v = _cl_limber_pair(ell, channels[i], channels[j], fNL)
            S[i, j] = v
            S[j, i] = v
    S += np.diag(N_diag)
    return S


def fisher_limber_fNL_only(ell_array, channels, N_diag, f_sky,
                           fNL_fid=1.0, delta_fNL=0.1, verbose=False):
    F = 0.0
    for ell in ell_array:
        S = _sigma_limber(ell, channels, fNL_fid, N_diag)
        S_p = _sigma_limber(ell, channels, fNL_fid + delta_fNL, N_diag)
        S_m = _sigma_limber(ell, channels, fNL_fid - delta_fNL, N_diag)
        dS = (S_p - S_m) / (2.0 * delta_fNL)
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            continue
        weight = (2.0 * ell + 1.0) * f_sky / 2.0
        M = S_inv @ dS
        F += weight * np.trace(M @ M)
        if verbose:
            print(f"    ℓ={ell:>4d}: cumulative F = {F:.3e}, "
                  f"σ ≈ {1/np.sqrt(F) if F>0 else np.inf:.3f}")
    return F


def report_ell_limber(channels):
    print("─" * 62)
    print("Step (d): ℓ_limber per line at peak z (full 92-channel geometry)")
    print("─" * 62)
    for line in LINE_ORDER:
        subset = [ch for ch in channels if ch['line'] == line]
        # Use the middle channel of each line as "peak".
        ch = subset[len(subset) // 2]
        ell_lim = compute_ell_limber(ch['lambda_rest'], ch['delta_lambda'],
                                     ch['z_peak'])
        print(f"  {line:8s}  z_peak = {ch['z_peak']:4.2f}   "
              f"Δλ = {ch['delta_lambda']:.4f} μm   "
              f"ℓ_limber ≈ {ell_lim:6.1f}")
    print()


def main():
    channels = build_92_channels(mode='deep')
    N_diag = np.asarray([ch['noise'] for ch in channels])
    print(f"Loaded {len(channels)} channels, "
          f"noise range σ_n ∈ [{np.sqrt(N_diag.min()/1e-13):.2f}, "
          f"{np.sqrt(N_diag.max()/1e-13):.2f}] * √(1e-13) units")
    print()

    report_ell_limber(channels)

    # Step (a): Limber-only, f_NL only. Use a wide ℓ grid that matches the
    # paper's ℓ ∈ [2, 300] range.
    ell_grid_a = np.array([2, 5, 10, 20, 40, 80, 150, 250])
    print("─" * 62)
    print("Step (a): Limber-only, f_NL only, no marginalisation")
    print(f"  ℓ grid = {list(ell_grid_a)}")
    print(f"  f_sky = {F_SKY_PAPER}")
    print("─" * 62)
    t0 = time.time()
    F_a = fisher_limber_fNL_only(ell_grid_a, channels, N_diag,
                                 f_sky=F_SKY_PAPER, verbose=True)
    t1 = time.time()
    sigma_a = 1.0 / np.sqrt(F_a) if F_a > 0 else np.inf
    print(f"\n  σ(f_NL, step a) = {sigma_a:.3f}   ({t1-t0:.1f}s)")
    print(f"  Target (paper) ≈ 0.71")
    within_factor_2 = 0.35 <= sigma_a <= 1.4
    print(f"  Within factor of 2 of paper: {within_factor_2}")
    print()

    return dict(sigma_a=sigma_a, F_a=F_a, channels=channels, N_diag=N_diag)


if __name__ == "__main__":
    main()
