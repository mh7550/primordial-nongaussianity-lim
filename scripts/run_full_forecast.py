"""
run_full_forecast.py — End-to-end 92-channel LIM Fisher forecast in intensity units.

Runs Steps (a)–(e) of the Pullen refactor with the true intensity-space
C_ℓ backend. All numbers are computed on the fly; nothing is hardcoded.

  (a)  Limber-only C_ℓ, f_NL only, no marginalisation.
       Physically reasonable range: 0.5–2.0. Halt otherwise.
  (b)  Full-Bessel + RSD, f_NL only, no marginalisation.
  (c)  Full-Bessel + RSD, 9×9 marginalised — new headline.
  (d)  Condition number of the 9×9 Fisher matrix.
  (e)  ℓ_min robustness scan (Table 1) with the full pipeline.
"""
import numpy as np
import sys, os, time
import pickle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from lim_channels import build_92_channels, LINE_ORDER
from limber import (compute_lim_cls_matrix, compute_ell_limber,
                    clear_intensity_cache)


F_SKY = 0.60
DELTA_AB = 0.01     # ±1% for A_i, B_i
DELTA_FNL = 0.1     # step for f_NL derivative
FNL_FID = 1.0       # non-zero fiducial to avoid k^-2 divergence at f_NL=0
ELL_GRID = np.array([2, 5, 10, 20, 40, 80, 150, 250])

PARAM_NAMES = [
    'fNL',
    'A_Halpha', 'B_Halpha',
    'A_OIII',   'B_OIII',
    'A_Hbeta',  'B_Hbeta',
    'A_OII',    'B_OII',
]


def sigma_matrix(ell, channels, N_diag, fNL, use_rsd, use_bessel):
    C = compute_lim_cls_matrix(ell, channels, fNL=fNL,
                               use_bessel_below_limber=use_bessel,
                               use_rsd=use_rsd)
    return C + np.diag(N_diag)


def _apply_AB(channels, A_vec, B_vec):
    """Return a channel list with the given A_i, B_i multipliers applied."""
    out = []
    for ch in channels:
        c = dict(ch)
        c['I_scale'] = A_vec[ch['line']]
        c['b_scale'] = B_vec[ch['line']]
        out.append(c)
    return out


def fisher_at_ell(ell, channels, N_diag, f_sky, use_rsd, use_bessel,
                  parameters='fNL'):
    """Fisher information matrix at a single ℓ.

    parameters : 'fNL' → 1×1 information; '9x9' → full nuisance block.
    """
    A0 = {line: 1.0 for line in LINE_ORDER}
    B0 = A0.copy()

    def _sigma(A_vec, B_vec, fNL):
        chs = _apply_AB(channels, A_vec, B_vec)
        return sigma_matrix(ell, chs, N_diag, fNL, use_rsd, use_bessel)

    Sigma = _sigma(A0, B0, FNL_FID)
    try:
        Sigma_inv = np.linalg.inv(Sigma)
    except np.linalg.LinAlgError:
        raise RuntimeError(f"Σ_ℓ singular at ℓ={ell}")

    weight = (2.0 * ell + 1.0) * f_sky / 2.0

    if parameters == 'fNL':
        Sp = _sigma(A0, B0, FNL_FID + DELTA_FNL)
        Sm = _sigma(A0, B0, FNL_FID - DELTA_FNL)
        dS = (Sp - Sm) / (2.0 * DELTA_FNL)
        M = Sigma_inv @ dS
        return np.array([[weight * np.trace(M @ M)]])

    # 9×9 case
    n = 9
    dS_list = []
    # ∂/∂f_NL
    Sp = _sigma(A0, B0, FNL_FID + DELTA_FNL)
    Sm = _sigma(A0, B0, FNL_FID - DELTA_FNL)
    dS_list.append((Sp - Sm) / (2.0 * DELTA_FNL))
    # ∂/∂A_i, ∂/∂B_i
    for line in LINE_ORDER:
        Ap = A0.copy(); Ap[line] = 1.0 + DELTA_AB
        Am = A0.copy(); Am[line] = 1.0 - DELTA_AB
        Sp = _sigma(Ap, B0, FNL_FID)
        Sm = _sigma(Am, B0, FNL_FID)
        dS_list.append((Sp - Sm) / (2.0 * DELTA_AB))
        Bp = B0.copy(); Bp[line] = 1.0 + DELTA_AB
        Bm = B0.copy(); Bm[line] = 1.0 - DELTA_AB
        Sp = _sigma(A0, Bp, FNL_FID)
        Sm = _sigma(A0, Bm, FNL_FID)
        dS_list.append((Sp - Sm) / (2.0 * DELTA_AB))

    M_list = [Sigma_inv @ dS for dS in dS_list]
    F = np.zeros((n, n))
    for a in range(n):
        for b in range(a, n):
            val = weight * np.trace(M_list[a] @ M_list[b])
            F[a, b] = val
            F[b, a] = val
    return F


def total_fisher(channels, N_diag, ell_array, use_rsd, use_bessel,
                 parameters='fNL', f_sky=F_SKY, verbose=True):
    n = 9 if parameters == '9x9' else 1
    F = np.zeros((n, n))
    for ell in ell_array:
        clear_intensity_cache()  # cache is z-quantized, safe to clear
        t0 = time.time()
        F += fisher_at_ell(int(ell), channels, N_diag, f_sky,
                           use_rsd, use_bessel, parameters=parameters)
        if verbose:
            sig = np.sqrt(1.0 / F[0, 0]) if F[0, 0] > 0 else np.inf
            print(f"    ℓ={ell:>4d}: {time.time()-t0:5.1f}s, "
                  f"σ_unm(f_NL) running = {sig:.3f}")
    return F


def sigma_marginalised(F):
    cov = np.linalg.pinv(F)
    return float(np.sqrt(cov[0, 0]))


def sigma_unmarginalised(F):
    return float(1.0 / np.sqrt(F[0, 0]))


def report_ell_limber(channels):
    print("─" * 62)
    print("Step (d-diag): ℓ_limber per line at peak z")
    print("─" * 62)
    for line in LINE_ORDER:
        subset = [ch for ch in channels if ch['line'] == line]
        ch = subset[len(subset) // 2]
        ell_lim = compute_ell_limber(ch['lambda_rest'], ch['delta_lambda'],
                                     ch['z_peak'])
        print(f"  {line:8s}  z_peak = {ch['z_peak']:4.2f}  "
              f"Δλ = {ch['delta_lambda']:.4f} μm  "
              f"ℓ_limber ≈ {ell_lim:5.1f}")
    print()


def main():
    channels = build_92_channels(mode='deep')
    N_diag = np.asarray([ch['noise'] for ch in channels])
    print(f"Loaded {len(channels)} channels "
          f"({sum(1 for c in channels if c['line']=='Halpha')} per line × 4 lines)")
    print(f"f_sky = {F_SKY}")
    print(f"ℓ grid = {list(ELL_GRID)}")
    print()

    report_ell_limber(channels)

    # ---- STEP (a) — Limber-only, f_NL only ------------------------------
    print("─" * 62)
    print("Step (a): Limber-only C_ℓ, f_NL only, no marginalisation")
    print("─" * 62)
    t0 = time.time()
    F_a = total_fisher(channels, N_diag, ELL_GRID,
                       use_rsd=False, use_bessel=False, parameters='fNL')
    sigma_a = sigma_unmarginalised(F_a)
    print(f"\n  σ(f_NL, step a) = {sigma_a:.3f}   ({time.time()-t0:.1f}s)")
    print(f"  Physically reasonable range: [0.5, 2.0]")
    in_range = 0.5 <= sigma_a <= 2.0
    print(f"  In range: {in_range}")
    if not in_range:
        print("\n  ⚠  STEP (a) OUTSIDE PHYSICALLY REASONABLE RANGE — halting.")
        return dict(sigma_a=sigma_a, halted=True)
    print()

    # ---- STEP (b) — full Bessel + RSD, f_NL only -------------------------
    print("─" * 62)
    print("Step (b): full-Bessel + RSD, f_NL only, no marginalisation")
    print("─" * 62)
    t0 = time.time()
    F_b = total_fisher(channels, N_diag, ELL_GRID,
                       use_rsd=True, use_bessel=True, parameters='fNL')
    sigma_b = sigma_unmarginalised(F_b)
    print(f"\n  σ(f_NL, step b) = {sigma_b:.3f}   ({time.time()-t0:.1f}s)")
    print(f"  Shift vs step (a):  {sigma_b - sigma_a:+.3f}   "
          f"({100*(sigma_b/sigma_a - 1):+.1f}%)")
    print()

    # ---- STEP (c) — 9×9 marginalised ------------------------------------
    print("─" * 62)
    print("Step (c): full-Bessel + RSD, 9×9 marginalised")
    print("─" * 62)
    t0 = time.time()
    F_c = total_fisher(channels, N_diag, ELL_GRID,
                       use_rsd=True, use_bessel=True, parameters='9x9')
    print()

    # Step (d) — condition number BEFORE inversion.
    cond = np.linalg.cond(F_c)
    print(f"  Step (d) — Condition number of the 9×9 Fisher: {cond:.3e}")
    if cond > 1e12:
        print("  ⚠  Condition number > 10¹² — matrix is effectively singular. "
              "Halting before inversion.")
        return dict(F_c=F_c, cond=cond, halted=True)
    sig_c_marg = sigma_marginalised(F_c)
    sig_c_unm = sigma_unmarginalised(F_c)
    print(f"  σ(f_NL, unmarginalised, step c) = {sig_c_unm:.3f}")
    print(f"  σ(f_NL,   marginalised, step c) = {sig_c_marg:.3f}   ← NEW HEADLINE")
    print(f"  Marginalisation penalty: ×{sig_c_marg/sig_c_unm:.2f}")
    print(f"  ({time.time()-t0:.1f}s total)")
    print()

    # ---- STEP (e) — ℓ_min robustness scan --------------------------------
    print("─" * 62)
    print("Step (e): ℓ_min robustness scan (updated Table 1)")
    print("─" * 62)
    ell_min_scan = [2, 10, 20, 30, 40, 50]
    ellmin_results = {}
    for ell_min in ell_min_scan:
        ell_sub = ELL_GRID[ELL_GRID >= ell_min]
        F_sub = total_fisher(channels, N_diag, ell_sub,
                             use_rsd=True, use_bessel=True,
                             parameters='9x9', verbose=False)
        sig = sigma_marginalised(F_sub)
        ellmin_results[ell_min] = sig
        print(f"  ℓ_min = {ell_min:>2d}  →  σ(f_NL) = {sig:.3f}")
    print()

    # ---- Summary ---------------------------------------------------------
    print("─" * 62)
    print("SUMMARY")
    print("─" * 62)
    print(f"  Step (a) Limber, unmarg.                     : {sigma_a:.3f}")
    print(f"  Step (b) Bessel+RSD, unmarg.                 : {sigma_b:.3f}")
    print(f"  Step (c) Bessel+RSD, marginalised (headline) : {sig_c_marg:.3f}")
    print(f"           Bessel+RSD, unmarg.                 : {sig_c_unm:.3f}")
    print(f"  Step (d) 9×9 condition number                : {cond:.3e}")
    print(f"  Step (e) ℓ_min scan: {ellmin_results}")
    print()
    print(f"  Improvement factors vs Planck (5.1):")
    print(f"    Step (a): {5.1/sigma_a:.2f}×")
    print(f"    Step (b): {5.1/sigma_b:.2f}×")
    print(f"    Step (c) marg.: {5.1/sig_c_marg:.2f}×")

    # Save results for the paper update.
    out_path = os.path.join(os.path.dirname(__file__), '..',
                            'data', 'full_forecast_results.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump({
            'sigma_a': sigma_a, 'sigma_b': sigma_b,
            'sigma_c_marg': sig_c_marg, 'sigma_c_unm': sig_c_unm,
            'condition_number': cond,
            'ellmin_scan': ellmin_results,
            'F_c': F_c, 'ell_grid': ELL_GRID,
        }, f)
    print(f"\n  Results saved: {out_path}")
    return dict(sigma_a=sigma_a, sigma_b=sigma_b,
                sigma_c_marg=sig_c_marg, sigma_c_unm=sig_c_unm,
                cond=cond, ellmin=ellmin_results)


if __name__ == "__main__":
    main()
