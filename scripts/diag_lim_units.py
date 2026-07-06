"""
diag_lim_units.py — Unit-consistency diagnostic for the LIM C_ℓ backend.

Per Step 2 of the Pullen refactor: before running any Fisher, print
the signal and noise for a representative channel (Hα auto-channel
near z = 1) at ℓ = 10, and check the S/N per mode is O(1)–O(a few).

If S/N is ~10⁶ or ~10⁻⁶, there is still a units mismatch — do not proceed.
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from lim_channels import build_92_channels
from limber import (compute_lim_cl_limber, compute_lim_cl_bessel,
                    compute_ell_limber)


def main():
    channels = build_92_channels(mode='deep')

    # Locate an Hα channel near z ≈ 1.
    ha = [ch for ch in channels if ch['line'] == 'Halpha']
    z_targets = np.abs(np.asarray([ch['z_peak'] for ch in ha]) - 1.0)
    ch = ha[int(np.argmin(z_targets))]

    ell = 10
    print("─" * 66)
    print("Step-2 units diagnostic (Hα auto-channel near z ≈ 1, ℓ = 10)")
    print("─" * 66)
    print(f"  channel line       : {ch['line']}")
    print(f"  channel z_peak     : {ch['z_peak']:.3f}")
    print(f"  channel λ_obs      : {ch['lambda_obs']:.3f} μm")
    print(f"  channel Δλ         : {ch['delta_lambda']:.4f} μm")
    print(f"  σ_n (v28)          : {ch['sigma_n']:.3f} nW/m²/sr")
    print()

    ell_lim = compute_ell_limber(ch['lambda_rest'], ch['delta_lambda'],
                                 ch['z_peak'])
    print(f"  ℓ_limber for this channel: {ell_lim:.1f}")
    print(f"  ℓ = {ell} is  {'BELOW' if ell <= ell_lim else 'ABOVE'} ℓ_limber "
          f"→ default backend is {'Bessel' if ell <= ell_lim else 'Limber'}")
    print()

    # Signal via Limber path (fNL = 0 to focus on gaussian signal).
    cl_limber = compute_lim_cl_limber(ell, ch, ch, fNL=0.0)
    # Signal via Bessel path (no RSD, matched conditions).
    cl_bessel = compute_lim_cl_bessel(ell, ch, ch, fNL=0.0, use_rsd=False)

    N = ch['noise']
    print(f"  Signal C_ℓ (Limber)  : {cl_limber:.3e}  (nW/m²/sr)²")
    print(f"  Signal C_ℓ (Bessel)  : {cl_bessel:.3e}  (nW/m²/sr)²")
    print(f"  Noise  N_ℓ            : {N:.3e}  (nW/m²/sr)²")
    print()
    print(f"  S/N  Limber   = C/N = {cl_limber/N:.3e}")
    print(f"  S/N  Bessel   = C/N = {cl_bessel/N:.3e}")
    print()

    # Physical reasonableness check.
    ratios = [cl_limber / N, cl_bessel / N]
    r_max = max(ratios)
    r_min = min(ratios)
    ok = 1e-3 < r_max < 1e5 and 1e-3 < r_min < 1e5
    print(f"  Verdict: {'PASS — S/N is physically reasonable' if ok else 'FAIL — likely units mismatch, DO NOT PROCEED'}")
    return ok


if __name__ == "__main__":
    main()
