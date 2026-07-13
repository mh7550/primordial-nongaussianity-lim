"""
run_decomposition.py — Physics-driven decomposition of σ(f_NL).

Runs Steps (b)–(e) as a monotonic derivation.  Before each measurement
the driver states the physical mechanism the step introduces and the
magnitude of change predicted from that mechanism, then reports what
the pipeline actually produces.  No tuning; no calibration to a target.

Reference: Step (a) — full 92×92 Limber, f_NL only, no marginalisation
           σ(f_NL, a) = 8.94   (already computed and confirmed in prior run)
"""
import numpy as np
import sys, os, time, pickle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from lim_channels import build_92_channels, LINE_ORDER
from limber import compute_lim_cls_matrix, clear_intensity_cache


F_SKY = 0.60
ELL_GRID = np.array([2, 5, 10, 20, 40, 80, 150, 250])
ELL_BESSEL_MAX = 20             # per user spec: Bessel at ℓ < 20
FNL_FID = 1.0
DELTA_FNL = 0.1
DELTA_AB = 0.01
PLANCK_SIGMA = 5.1

SIGMA_A = 8.935   # from previous run — the anchor for factor comparisons


# ---------------------------------------------------------------------------
# Fisher primitives
# ---------------------------------------------------------------------------

def sigma_matrix(ell, channels, N_diag, fNL, use_rsd, use_bessel,
                 diagonal_only=False, ell_bessel_max=None):
    C = compute_lim_cls_matrix(
        ell, channels, fNL=fNL,
        use_bessel_below_limber=use_bessel, use_rsd=use_rsd,
        diagonal_only=diagonal_only, ell_bessel_max=ell_bessel_max,
    )
    if diagonal_only:
        # Keep only the diagonal — no cross-channel information.
        C = np.diag(np.diag(C))
    return C + np.diag(N_diag)


def _apply_AB(channels, A_vec, B_vec):
    out = []
    for ch in channels:
        c = dict(ch)
        c['I_scale'] = A_vec[ch['line']]
        c['b_scale'] = B_vec[ch['line']]
        out.append(c)
    return out


def fisher_fNL_at_ell(ell, channels, N_diag, use_rsd, use_bessel,
                     diagonal_only=False, ell_bessel_max=None):
    """1×1 Fisher information on f_NL at multipole ℓ."""
    S = sigma_matrix(ell, channels, N_diag, FNL_FID, use_rsd, use_bessel,
                     diagonal_only, ell_bessel_max)
    Sp = sigma_matrix(ell, channels, N_diag, FNL_FID + DELTA_FNL,
                      use_rsd, use_bessel, diagonal_only, ell_bessel_max)
    Sm = sigma_matrix(ell, channels, N_diag, FNL_FID - DELTA_FNL,
                      use_rsd, use_bessel, diagonal_only, ell_bessel_max)
    dS = (Sp - Sm) / (2.0 * DELTA_FNL)
    Sinv = np.linalg.inv(S)
    weight = (2.0 * ell + 1.0) * F_SKY / 2.0
    M = Sinv @ dS
    return weight * float(np.trace(M @ M))


def fisher_9x9_at_ell(ell, channels, N_diag, use_rsd, use_bessel,
                      ell_bessel_max=None):
    """9×9 Fisher: f_NL + 8 (A_i, B_i) nuisance parameters."""
    A0 = {l: 1.0 for l in LINE_ORDER}
    B0 = A0.copy()

    def _sig(A, B, fNL):
        chs = _apply_AB(channels, A, B)
        return sigma_matrix(ell, chs, N_diag, fNL, use_rsd, use_bessel,
                            diagonal_only=False,
                            ell_bessel_max=ell_bessel_max)

    S = _sig(A0, B0, FNL_FID)
    try:
        Sinv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        raise RuntimeError(f"Σ_ℓ singular at ℓ={ell}")

    dS_list = []
    # f_NL
    Sp = _sig(A0, B0, FNL_FID + DELTA_FNL)
    Sm = _sig(A0, B0, FNL_FID - DELTA_FNL)
    dS_list.append((Sp - Sm) / (2.0 * DELTA_FNL))
    # A_i, B_i per line
    for line in LINE_ORDER:
        Ap = A0.copy(); Ap[line] = 1.0 + DELTA_AB
        Am = A0.copy(); Am[line] = 1.0 - DELTA_AB
        dS_list.append((_sig(Ap, B0, FNL_FID) - _sig(Am, B0, FNL_FID))
                       / (2.0 * DELTA_AB))
        Bp = B0.copy(); Bp[line] = 1.0 + DELTA_AB
        Bm = B0.copy(); Bm[line] = 1.0 - DELTA_AB
        dS_list.append((_sig(A0, Bp, FNL_FID) - _sig(A0, Bm, FNL_FID))
                       / (2.0 * DELTA_AB))

    M_list = [Sinv @ dS for dS in dS_list]
    weight = (2.0 * ell + 1.0) * F_SKY / 2.0
    F = np.zeros((9, 9))
    for a in range(9):
        for b in range(a, 9):
            val = weight * np.trace(M_list[a] @ M_list[b])
            F[a, b] = val
            F[b, a] = val
    return F


def total_fisher_fNL(channels, N_diag, use_rsd, use_bessel,
                     diagonal_only=False, ell_bessel_max=None,
                     label="run"):
    F = 0.0
    for ell in ELL_GRID:
        clear_intensity_cache()
        t0 = time.time()
        F += fisher_fNL_at_ell(int(ell), channels, N_diag, use_rsd,
                               use_bessel, diagonal_only, ell_bessel_max)
        sig = 1.0/np.sqrt(F) if F > 0 else np.inf
        print(f"    ℓ={ell:>4d}: {time.time()-t0:5.1f}s  σ_run = {sig:.3f}")
    return 1.0 / np.sqrt(F), F


def total_fisher_9x9(channels, N_diag, use_rsd, use_bessel, ell_bessel_max):
    F = np.zeros((9, 9))
    for ell in ELL_GRID:
        clear_intensity_cache()
        t0 = time.time()
        F += fisher_9x9_at_ell(int(ell), channels, N_diag,
                               use_rsd, use_bessel, ell_bessel_max)
        cov = np.linalg.pinv(F)
        sig = float(np.sqrt(cov[0, 0])) if cov[0, 0] > 0 else np.inf
        print(f"    ℓ={ell:>4d}: {time.time()-t0:5.1f}s  σ_marg = {sig:.3f}")
    return F


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def hdr(title):
    print()
    print("═" * 70)
    print(title)
    print("═" * 70)


def predict(text):
    print(f"[predict]  {text}")


def report_shift(label, sigma_new, sigma_prev, prev_label):
    factor = sigma_prev / sigma_new
    if factor >= 1:
        direction = f"IMPROVEMENT ×{factor:.2f}"
    else:
        direction = f"DEGRADATION ×{1/factor:.2f}"
    print(f"[measure]  σ({label}) = {sigma_new:.3f}   "
          f"({direction} vs σ({prev_label}) = {sigma_prev:.3f})")
    return factor


def check_prediction(actual_factor, predicted_lo, predicted_hi, step_name):
    """Halt if actual is > 3× outside the predicted range."""
    if actual_factor < predicted_lo / 3 or actual_factor > predicted_hi * 3:
        print(f"[!! diagnose !!]  step {step_name} shift {actual_factor:.2f} "
              f"is >3× outside predicted [{predicted_lo:.2f}, {predicted_hi:.2f}]")
        return False
    return True


def main():
    channels = build_92_channels(mode='deep')
    N_diag = np.asarray([ch['noise'] for ch in channels])
    print(f"92 channels loaded. f_sky = {F_SKY}. ℓ grid = {list(ELL_GRID)}.")
    print(f"Bessel used for ℓ ≤ {ELL_BESSEL_MAX} (per user spec).")

    results = {'sigma_a': SIGMA_A}

    # -----------------------------------------------------------------
    # Step (b) — diagonal-only 92-channel Fisher, Limber
    # -----------------------------------------------------------------
    hdr("Step (b) — sum over 92 auto-C_ℓ  |  no cross-power  |  Limber")
    print(
        "Mechanism: 92 independent single-tracer Fishers summed. Removes all\n"
        "off-diagonal covariance information that step (a) had. Since (a)\n"
        "already included cross-power via σ_z = 0.12 window overlap, dropping\n"
        "it should DEGRADE the constraint, not improve it."
    )
    predict("Δσ ≈ +5% to +25%  (i.e. factor 0.8–0.95 vs step (a))")
    t0 = time.time()
    sigma_b, F_b = total_fisher_fNL(channels, N_diag,
                                    use_rsd=False, use_bessel=False,
                                    diagonal_only=True, label="b")
    print(f"    total {time.time()-t0:.1f}s")
    f_b = report_shift("b", sigma_b, SIGMA_A, "a")
    ok = check_prediction(f_b, 0.80, 0.95, "b")
    results['sigma_b'] = sigma_b
    if not ok:
        return results

    # -----------------------------------------------------------------
    # Step (c) — add Bessel at ℓ<20 and RSD
    # -----------------------------------------------------------------
    hdr("Step (c) — diagonal-only  |  Bessel at ℓ<20  |  Kaiser RSD")
    print(
        "Mechanism: Bessel captures the exact projection at low ℓ, where the\n"
        "k^-2 PNG signal peaks. RSD replaces b_i(z) → b_i(z) + f(z) in the\n"
        "Bessel transfer function, adding coherent signal power.\n"
        "Both effects act at ℓ ≲ 20 only."
    )
    predict("Bessel: ~+10–30% info at ℓ<20   RSD: ~+10–30%   combined ×1.1–1.5")
    t0 = time.time()
    sigma_c, F_c = total_fisher_fNL(channels, N_diag,
                                    use_rsd=True, use_bessel=True,
                                    diagonal_only=True,
                                    ell_bessel_max=ELL_BESSEL_MAX, label="c")
    print(f"    total {time.time()-t0:.1f}s")
    f_c = report_shift("c", sigma_c, sigma_b, "b")
    ok = check_prediction(f_c, 1.10, 1.50, "c")
    results['sigma_c'] = sigma_c
    if not ok:
        return results

    # -----------------------------------------------------------------
    # Step (d) — restore full 92×92 off-diagonal cross-power
    # -----------------------------------------------------------------
    hdr("Step (d) — FULL 92×92  |  Bessel at ℓ<20  |  RSD")
    print(
        "Mechanism: reinstate the σ_z-window off-diagonal terms. This is the\n"
        "LIM multi-tracer benefit — different lines at the SAME physical z\n"
        "and different channels of the SAME line at nearby z contribute\n"
        "correlated signal. Cosmic-variance cancellation in the Fisher trace."
    )
    predict("×1.1–1.4  (Seljak-style multi-tracer at LIM S/N)")
    t0 = time.time()
    sigma_d, F_d = total_fisher_fNL(channels, N_diag,
                                    use_rsd=True, use_bessel=True,
                                    diagonal_only=False,
                                    ell_bessel_max=ELL_BESSEL_MAX, label="d")
    print(f"    total {time.time()-t0:.1f}s")
    f_d = report_shift("d", sigma_d, sigma_c, "c")
    ok = check_prediction(f_d, 1.05, 1.60, "d")
    results['sigma_d'] = sigma_d
    if not ok:
        return results

    # -----------------------------------------------------------------
    # Step (e) — 9×9 marginalisation over A_i, B_i for all 4 lines
    # -----------------------------------------------------------------
    hdr("Step (e) — FULL + Bessel + RSD  |  9×9 marginalised (A_i, B_i)")
    print(
        "Mechanism: 8 line-astrophysics nuisance parameters — A_i (intensity\n"
        "amplitude) and B_i (bias amplitude) per line. Without RSD these\n"
        "are almost perfectly degenerate; RSD's f(z) term breaks the\n"
        "degeneracy so marginalisation costs less than a naive count would\n"
        "predict. Cosmological params (n_s, σ_8) not marginalised — the paper\n"
        "arguments in Sec. V.D that they are constrained at percent level.\n"
    )
    predict("×1.5–3.0 degradation (RSD helps; naive 9-param would be worse)")
    t0 = time.time()
    F9 = total_fisher_9x9(channels, N_diag,
                          use_rsd=True, use_bessel=True,
                          ell_bessel_max=ELL_BESSEL_MAX)
    print(f"    total {time.time()-t0:.1f}s")
    cond = np.linalg.cond(F9)
    print(f"    9×9 Fisher condition number: {cond:.3e}")
    if cond > 1e12:
        print(f"    ⚠ condition > 10^12 — halting before inversion.")
        results['condition'] = cond
        return results
    cov = np.linalg.pinv(F9)
    sigma_e_marg = float(np.sqrt(cov[0, 0]))
    sigma_e_unm  = float(1.0 / np.sqrt(F9[0, 0]))
    f_e = report_shift("e (marg)", sigma_e_marg, sigma_d, "d")
    ok = check_prediction(1.0/f_e, 1.20, 4.00, "e")   # e degrades relative to d
    results['sigma_e_unm'] = sigma_e_unm
    results['sigma_e_marg'] = sigma_e_marg
    results['condition'] = cond
    if not ok:
        return results

    # -----------------------------------------------------------------
    # Decomposition table
    # -----------------------------------------------------------------
    hdr("DECOMPOSITION  σ(f_NL)   with running factor improvement over Planck 5.1")
    print(f"  {'step':<70}  {'σ(f_NL)':>8}  {'vs Planck':>10}")
    rows = [
        ("(a) full 92×92 Limber, unmarginalised",           SIGMA_A),
        ("(b) 92 auto-C_ℓ only, Limber",                    sigma_b),
        ("(c) 92 auto-C_ℓ, Bessel(ℓ<20) + RSD",             sigma_c),
        ("(d) full 92×92, Bessel(ℓ<20) + RSD",              sigma_d),
        ("(e) same as (d), 9×9 marginalised (A_i, B_i)",    sigma_e_marg),
    ]
    for label, sig in rows:
        r = PLANCK_SIGMA / sig
        print(f"  {label:<70}  {sig:8.3f}  {r:9.2f}×")
    print()
    print(f"  Planck 2018 (5.1) reference — sub-unity σ discriminates single- vs multi-field")
    if sigma_e_marg < 1.0:
        print(f"  σ(e) = {sigma_e_marg:.3f} < 1  → crosses the multi-field threshold")
    else:
        print(f"  σ(e) = {sigma_e_marg:.3f} ≥ 1  → does NOT cross the multi-field threshold")

    # Persist for the paper.
    out = os.path.join(os.path.dirname(__file__), '..',
                       'data', 'decomposition_results.pkl')
    with open(out, 'wb') as f:
        pickle.dump(results, f)
    print(f"\n  Results pickled: {out}")
    return results


if __name__ == "__main__":
    main()
