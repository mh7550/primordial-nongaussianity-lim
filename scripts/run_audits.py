"""
run_audits.py — Physical-input audits for σ(f_NL) = 11.9.

Runs three internally-consistent configurations of Step (d) and reports
σ(f_NL) at each. No tuning; no target.

Audit 1: σ_z window width
    baseline:       σ_z = 0.12 (paper text, matches SPHEREx sample-4 photo-z)
    spectral:       σ_z(ν) = Δλ_ν / λ_rest,ν  (per-channel spectral width)
    → shift attributable to using the true LIM window vs the photo-z-style width

Audit 2: internally-consistent noise / f_sky pairs
    deep:           v28 deep-field noise, f_sky = 0.0048  (200 deg²)
    all-sky:        v28 all-sky   noise,  f_sky = 0.60    (24 000 deg²)
    → separates the mixed 'deep-noise-at-full-sky-f_sky' setup used above

Marginalisation adds +3% (established) — omitted here for time. Add ×1.03
if you want the full 9×9 headline.
"""
import numpy as np
import sys, os, time, pickle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from lim_channels import build_92_channels, LINE_ORDER
from limber import compute_lim_cls_matrix, clear_intensity_cache


ELL_GRID = np.array([2, 5, 10, 20, 40, 80, 150, 250])
ELL_BESSEL_MAX = 20
FNL_FID = 1.0
DELTA_FNL = 0.1
PLANCK_SIGMA = 5.1


def add_spectral_sigma_z(channels):
    """Set ch['sigma_z'] = Δλ / λ_rest for every channel (the LIM window)."""
    for ch in channels:
        ch['sigma_z'] = ch['delta_lambda'] / ch['lambda_rest']
    return channels


def run_step_d(channels, N_diag, f_sky, label):
    print(f"\n─── {label}  f_sky = {f_sky}  ───")
    F = 0.0
    for ell in ELL_GRID:
        clear_intensity_cache()
        t0 = time.time()
        S  = compute_lim_cls_matrix(int(ell), channels, fNL=FNL_FID,
                                    use_bessel_below_limber=True,
                                    use_rsd=True,
                                    ell_bessel_max=ELL_BESSEL_MAX) + np.diag(N_diag)
        Sp = compute_lim_cls_matrix(int(ell), channels, fNL=FNL_FID + DELTA_FNL,
                                    use_bessel_below_limber=True,
                                    use_rsd=True,
                                    ell_bessel_max=ELL_BESSEL_MAX) + np.diag(N_diag)
        Sm = compute_lim_cls_matrix(int(ell), channels, fNL=FNL_FID - DELTA_FNL,
                                    use_bessel_below_limber=True,
                                    use_rsd=True,
                                    ell_bessel_max=ELL_BESSEL_MAX) + np.diag(N_diag)
        dS = (Sp - Sm) / (2.0 * DELTA_FNL)
        Sinv = np.linalg.inv(S)
        M = Sinv @ dS
        F += (2.0 * ell + 1.0) * f_sky / 2.0 * float(np.trace(M @ M))
        sig = 1.0 / np.sqrt(F) if F > 0 else np.inf
        print(f"    ℓ={ell:>4d}: {time.time()-t0:5.1f}s   σ_run = {sig:.3f}")
    return 1.0 / np.sqrt(F)


def diag_sigma_z(channels):
    print("\nChannel σ_z summary (spectral):")
    lines = {}
    for ch in channels:
        lines.setdefault(ch['line'], []).append(ch['sigma_z'])
    for line, vals in lines.items():
        v = np.asarray(vals)
        print(f"    {line:8s}  min={v.min():.4f}  mean={v.mean():.4f}  max={v.max():.4f}")


def main():
    print("=" * 72)
    print("AUDIT 1 — σ_z window width")
    print("=" * 72)

    # Reference: 0.12 window, deep noise, f_sky=0.60 (as we've been running).
    ch_012 = build_92_channels(mode='deep')
    for ch in ch_012:
        ch['sigma_z'] = 0.12   # explicit
    N_012 = np.asarray([ch['noise'] for ch in ch_012])
    sigma_012 = run_step_d(ch_012, N_012, f_sky=0.60,
                           label="σ_z=0.12  deep-noise  f_sky=0.60  (baseline)")

    # Spectral σ_z: Δλ/λ_rest per channel.
    ch_spec = build_92_channels(mode='deep')
    ch_spec = add_spectral_sigma_z(ch_spec)
    N_spec = np.asarray([ch['noise'] for ch in ch_spec])
    diag_sigma_z(ch_spec)
    sigma_spec = run_step_d(ch_spec, N_spec, f_sky=0.60,
                            label="σ_z=Δλ/λ_rest  deep-noise  f_sky=0.60")

    print()
    print(f"[audit-1]  σ(baseline, σ_z=0.12)  = {sigma_012:.3f}")
    print(f"[audit-1]  σ(spectral, per-ch)    = {sigma_spec:.3f}")
    print(f"[audit-1]  shift factor           = {sigma_012/sigma_spec:.2f}× "
          f"({'tighter' if sigma_spec<sigma_012 else 'looser'})")

    print()
    print("=" * 72)
    print("AUDIT 2 — internally-consistent noise / f_sky pairs")
    print("=" * 72)
    # 2a: deep noise, f_sky = 0.0048
    ch_2a = build_92_channels(mode='deep')
    N_2a = np.asarray([ch['noise'] for ch in ch_2a])
    sigma_2a = run_step_d(ch_2a, N_2a, f_sky=0.0048,
                          label="v28 deep, f_sky=0.0048  (200 deg², consistent)")

    # 2b: all-sky noise, f_sky = 0.60
    ch_2b = build_92_channels(mode='all-sky')
    N_2b = np.asarray([ch['noise'] for ch in ch_2b])
    sigma_2b = run_step_d(ch_2b, N_2b, f_sky=0.60,
                          label="v28 all-sky, f_sky=0.60  (24 000 deg², consistent)")

    print()
    print(f"[audit-2]  σ(deep, f_sky=0.0048)     = {sigma_2a:.3f}")
    print(f"[audit-2]  σ(all-sky, f_sky=0.60)    = {sigma_2b:.3f}")
    print(f"[audit-2]  σ(mixed baseline, from A1) = {sigma_012:.3f}")

    print()
    print("=" * 72)
    print("SUMMARY  (step (d), unmarginalised; +3% for marginalisation)")
    print("=" * 72)
    print(f"  {'configuration':<55}  {'σ(f_NL)':>8}  {'vs Planck':>10}")
    rows = [
        ("A1 baseline  σ_z=0.12, deep noise, f_sky=0.60  (mixed)",   sigma_012),
        ("A1 spectral σ_z, deep noise, f_sky=0.60",                  sigma_spec),
        ("A2a  deep noise, f_sky=0.0048  (self-consistent deep)",    sigma_2a),
        ("A2b  all-sky noise, f_sky=0.60  (self-consistent all-sky)", sigma_2b),
    ]
    for label, sig in rows:
        r = PLANCK_SIGMA / sig
        print(f"  {label:<55}  {sig:8.3f}  {r:9.2f}×")

    out = os.path.join(os.path.dirname(__file__), '..', 'data',
                       'audit_results.pkl')
    with open(out, 'wb') as f:
        pickle.dump({
            'sigma_baseline_012': sigma_012,
            'sigma_spectral':     sigma_spec,
            'sigma_deep':         sigma_2a,
            'sigma_allsky':       sigma_2b,
        }, f)
    print(f"\n  Results pickled: {out}")


if __name__ == "__main__":
    main()
