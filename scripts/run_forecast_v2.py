"""
run_forecast_v2.py — Full pipeline: full-Bessel + RSD + 9×9 marginalised Fisher.

Reports:
  1. ℓ_limber for each of the 4 lines at its peak redshift
  2. Marginalised σ(f_NL) from the 9×9 Fisher matrix
  3. Unmarginalised σ(f_NL) = 1/√F[0,0] for direct comparison to the previous
     headline of 0.71
  4. Condition number of the 9×9 Fisher matrix
  5. Decomposition:
       (a) Limber, no marginalisation
       (b) full-Bessel + RSD, no marginalisation
       (c) full-Bessel + RSD, marginalised (final headline)
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from fisher import (compute_fisher_9x9, sigma_fNL_from_fisher_9x9,
                    PARAM_NAMES_9x9, _sigma_matrix_at_ell)
from limber import compute_ell_limber, clear_bessel_cache
from survey_specs import F_SKY


# ---------------------------------------------------------------------------
# Channel construction
# ---------------------------------------------------------------------------
# Rather than hand-write all 92 channels (23 per line × 4 lines), we use a
# coarser but representative sampling: 6 channels per line spanning that
# line's observable redshift range in SPHEREx. This is enough to demonstrate
# the pipeline and produce meaningful numbers without hours of runtime.

LINE_INFO = {
    # rest wavelength (μm), representative b1(z) coefficients
    'Halpha': dict(lambda_rest=0.6563, b0=1.0, b1_slope=0.84),
    'OIII':   dict(lambda_rest=0.5007, b0=1.1, b1_slope=0.90),
    'Hbeta':  dict(lambda_rest=0.4861, b0=1.0, b1_slope=0.84),
    'OII':    dict(lambda_rest=0.3727, b0=1.2, b1_slope=0.85),
}

# SPHEREx spectral resolution R = λ/Δλ ~ 41 in the short-λ bands.
R_SPHEREX = 41.0

# Observed-wavelength grid inside SPHEREx: 0.75–5.0 μm.
LAM_MIN, LAM_MAX = 0.75, 5.0

# Channels per line for this run.
N_PER_LINE = 6


def make_b1_of_z(b0, slope):
    return lambda z: b0 + slope * z


def build_channels():
    channels = []
    for line, info in LINE_INFO.items():
        lam_rest = info['lambda_rest']
        z_min_line = LAM_MIN / lam_rest - 1.0
        z_max_line = min(4.5, LAM_MAX / lam_rest - 1.0)
        z_min_line = max(z_min_line, 0.5)  # avoid tiny χ

        z_peaks = np.linspace(z_min_line, z_max_line, N_PER_LINE)
        for zp in z_peaks:
            lam_obs = lam_rest * (1.0 + zp)
            delta_lam = lam_obs / R_SPHEREX
            channels.append(dict(
                line=line,
                lambda_rest=lam_rest,
                delta_lambda=delta_lam,
                z_peak=float(zp),
                b1_of_z=make_b1_of_z(info['b0'], info['b1_slope']),
                I_scale=1.0,
                b_scale=1.0,
            ))
    return channels


def noise_diag(n_channels):
    # Representative SPHEREx v28 diagonal noise, nW²/m⁴/sr² (order of magnitude).
    # Constant across channels here; the paper's forecast used the wavelength-
    # dependent v28 curve but the Fisher structure is identical.
    return np.full(n_channels, 1e-4)


def peak_z(line):
    """Peak redshift used to report ℓ_limber."""
    info = LINE_INFO[line]
    z_lo = LAM_MIN / info['lambda_rest'] - 1.0
    z_hi = min(4.5, LAM_MAX / info['lambda_rest'] - 1.0)
    return 0.5 * (max(0.5, z_lo) + z_hi)


def report_ell_limber():
    print("─" * 60)
    print("Item 1 diagnostic:  ℓ_limber per line at its peak redshift")
    print("─" * 60)
    for line, info in LINE_INFO.items():
        zp = peak_z(line)
        lam_obs = info['lambda_rest'] * (1.0 + zp)
        delta_lam = lam_obs / R_SPHEREX
        ell_lim = compute_ell_limber(info['lambda_rest'], delta_lam, zp)
        print(f"  {line:8s}  z_peak = {zp:4.2f}   Δλ = {delta_lam:.3f} μm   "
              f"ℓ_limber ≈ {ell_lim:6.1f}")
    print()


def run(step_label, ell_array, channels, N_diag, marginalised, use_rsd):
    clear_bessel_cache()
    F = compute_fisher_9x9(ell_array, channels, N_diag,
                           f_sky=F_SKY, use_rsd=use_rsd, verbose=False)
    sig_marg = sigma_fNL_from_fisher_9x9(F, marginalised=True)
    sig_unm = sigma_fNL_from_fisher_9x9(F, marginalised=False)
    try:
        cond = np.linalg.cond(F)
    except np.linalg.LinAlgError:
        cond = np.inf
    sig = sig_marg if marginalised else sig_unm
    print(f"  {step_label:52s}  σ(f_NL) = {sig:6.3f}")
    return F, sig_marg, sig_unm, cond


def main():
    channels = build_channels()
    N_diag = noise_diag(len(channels))
    print(f"Using {len(channels)} channels ({N_PER_LINE} per line × 4 lines)")
    print()

    report_ell_limber()

    # ℓ grid: sample the low-ℓ regime densely and go through ℓ=200.
    ell_array = np.array([2, 5, 10, 20, 40, 80, 150], dtype=int)
    print(f"Sampling ℓ ∈ {list(ell_array)}\n")

    print("─" * 60)
    print("Decomposition (Item 1+2+3)")
    print("─" * 60)

    # (a) Limber-only, no marginalisation: use_rsd=False, and force Limber by
    #     evaluating at ℓ > ℓ_limber (use the high-ℓ subset).
    ell_high = ell_array[ell_array > 100]
    if len(ell_high) == 0:
        ell_high = np.array([150])
    F_a, _, sig_a_unm, cond_a = run(
        "(a) Limber-only, high-ℓ, no marginalisation",
        ell_high, channels, N_diag, marginalised=False, use_rsd=False)

    # (b) full-Bessel + RSD across the full ℓ range, unmarginalised.
    F_b, _, sig_b_unm, cond_b = run(
        "(b) full-Bessel + RSD, unmarginalised",
        ell_array, channels, N_diag, marginalised=False, use_rsd=True)

    # (c) full-Bessel + RSD, marginalised 9×9 (final headline).
    F_c, sig_c_marg, sig_c_unm, cond_c = run(
        "(c) full-Bessel + RSD, 9×9 marginalised",
        ell_array, channels, N_diag, marginalised=True, use_rsd=True)

    print()
    print("─" * 60)
    print("Summary")
    print("─" * 60)
    print(f"  σ(f_NL)  Limber-only, unmarginalised     : {sig_a_unm:6.3f}")
    print(f"  σ(f_NL)  full-Bessel+RSD, unmarginalised : {sig_b_unm:6.3f}")
    print(f"  σ(f_NL)  full-Bessel+RSD, marginalised   : {sig_c_marg:6.3f}   ← headline")
    print()
    print(f"  9×9 Fisher condition number (marginalised run): {cond_c:.3e}")
    if cond_c > 1e12:
        print("  ⚠  Condition number > 10^12 — matrix is effectively singular.")

    return dict(
        sigma_limber=sig_a_unm,
        sigma_bessel_rsd_unmarginalised=sig_b_unm,
        sigma_bessel_rsd_marginalised=sig_c_marg,
        condition_number=cond_c,
    )


if __name__ == "__main__":
    main()
