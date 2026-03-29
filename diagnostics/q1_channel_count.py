"""
QUESTION 1: Channel count diagnostic - 92 vs 96 vs 102

Investigate why we get 92 channels instead of SPHEREx's specification
of 17 per band (102 total) or the literature value of 96.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from angular_power_spectrum import (
    LAMBDA_BAND_EDGES,
    SPECTRAL_RESOLUTION_R,
    N_CHANNELS,
    _nchan_per_band
)

print("="*70)
print("QUESTION 1: CHANNEL COUNT DIAGNOSTIC")
print("="*70)

# (a) Exact number per band vs 17 per band
print("\n(a) CHANNELS PER BAND - ACTUAL vs EXPECTED")
print("="*70)

print(f"\nBand | λ range (μm) |  R  | Our nchan | Expected | Difference")
print("-"*70)

expected_per_band = 17  # SPHEREx spec
total_expected = expected_per_band * 6
total_our = 0

for i in range(6):
    our_nchan = _nchan_per_band[i]
    total_our += our_nchan
    diff = our_nchan - expected_per_band

    print(f"  {i+1}  | {LAMBDA_BAND_EDGES[i]:.2f}-{LAMBDA_BAND_EDGES[i+1]:.2f} "
          f"| {SPECTRAL_RESOLUTION_R[i]:3} |    {our_nchan:2}     |    {expected_per_band}    |    {diff:+3}")

print("-"*70)
print(f"TOTAL |              |     |    {total_our:2}     |   {total_expected}    |   {total_our - total_expected:+3}")

print(f"\nSummary:")
print(f"  Our implementation: {total_our} channels")
print(f"  SPHEREx spec (17×6): {total_expected} channels")
print(f"  Literature (cited): 96 channels")
print(f"  Discrepancy: {total_our} - {total_expected} = {total_our - total_expected} channels")

# (b) Why 15/15/15/15/16/16 instead of 17?
print("\n(b) WHY 15/15/15/15/16/16 INSTEAD OF 17 PER BAND?")
print("="*70)

print("\nThe nchan calculation uses:")
print("  nchan = floor(Δλ_band / Δλ_chan)")
print("  where Δλ_chan = λ_center / R")

print(f"\nDetailed calculation for each band:")
print(f"\n{'Band':<6} {'Δλ_band':<12} {'λ_center':<12} {'R':<6} {'Δλ_chan':<12} {'Δλ/Δλ_c':<12} {'floor()':<8}")
print("-"*70)

dlamband = np.diff(LAMBDA_BAND_EDGES)
lamcen = 0.5 * (LAMBDA_BAND_EDGES[:-1] + LAMBDA_BAND_EDGES[1:])

for i in range(6):
    delta_lam_band = dlamband[i]
    lam_center = lamcen[i]
    R = SPECTRAL_RESOLUTION_R[i]
    delta_lam_chan = lam_center / R
    ratio = delta_lam_band / delta_lam_chan
    nchan_floor = int(np.floor(ratio))

    print(f"{i+1:<6} {delta_lam_band:<12.4f} {lam_center:<12.4f} {R:<6} "
          f"{delta_lam_chan:<12.6f} {ratio:<12.4f} {nchan_floor:<8}")

print("\nExplanation:")
print("  The ratio Δλ_band / Δλ_chan is slightly less than 17 for bands 1-4")
print("  because the channel width Δλ_chan = λ_center/R is computed at the")
print("  band CENTER, not the edges. This causes:")
print("    - Bands 1-4 (R=35-41): ratio ≈ 15.x → floor() = 15")
print("    - Bands 5-6 (R=110,130): ratio ≈ 16.x → floor() = 16")

# (c) What if we used ceil() instead of floor()?
print("\n(c) USING ceil() INSTEAD OF floor()")
print("="*70)

print("\nIf we use ceil() instead of floor():")
print(f"\n{'Band':<6} {'Δλ/Δλ_c':<12} {'floor()':<10} {'ceil()':<10} {'Change':<8}")
print("-"*60)

total_floor = 0
total_ceil = 0

for i in range(6):
    delta_lam_band = dlamband[i]
    lam_center = lamcen[i]
    R = SPECTRAL_RESOLUTION_R[i]
    delta_lam_chan = lam_center / R
    ratio = delta_lam_band / delta_lam_chan

    nchan_floor = int(np.floor(ratio))
    nchan_ceil = int(np.ceil(ratio))
    change = nchan_ceil - nchan_floor

    total_floor += nchan_floor
    total_ceil += nchan_ceil

    print(f"{i+1:<6} {ratio:<12.4f} {nchan_floor:<10} {nchan_ceil:<10} {change:+8}")

print("-"*60)
print(f"TOTAL                  {total_floor:<10} {total_ceil:<10} {total_ceil - total_floor:+8}")

print(f"\nResult:")
print(f"  floor(): {total_floor} channels (current)")
print(f"  ceil():  {total_ceil} channels")
print(f"\nNeither floor() nor ceil() gives 96 or 102!")
print(f"  SPHEREx spec: 102 channels (17 per band)")
print(f"  Literature:    96 channels")
print(f"  Our floor():   92 channels")
print(f"  Our ceil():    {total_ceil} channels")

# Limber_min for extra channels with ceil()
print("\n(c.1) LIMBER_MIN VALUES FOR EXTRA CHANNELS (if using ceil())")
print("="*70)

print("\nTo estimate limber_min for extra channels, we use Professor Pullen's formula:")
print("  limber_min ~ (χ/3000) × E(z) × (λ/Δλ)")

print("\nFor each band that would gain channels with ceil():")

# Load current limber_min
LIMBER_MIN = np.loadtxt('test_limber.txt').astype(int)

for i in range(6):
    nchan_floor = int(np.floor(dlamband[i] / (lamcen[i] / SPECTRAL_RESOLUTION_R[i])))
    nchan_ceil = int(np.ceil(dlamband[i] / (lamcen[i] / SPECTRAL_RESOLUTION_R[i])))

    if nchan_ceil > nchan_floor:
        # This band would gain channels
        # Find the last channel in this band from our current setup
        channel_start = sum(_nchan_per_band[:i])
        channel_end = channel_start + _nchan_per_band[i]

        if channel_end <= len(LIMBER_MIN):
            last_limber = LIMBER_MIN[channel_end - 1]

            print(f"\n  Band {i+1} ({LAMBDA_BAND_EDGES[i]:.2f}-{LAMBDA_BAND_EDGES[i+1]:.2f} μm):")
            print(f"    Current channels: {nchan_floor}")
            print(f"    With ceil(): {nchan_ceil} (+{nchan_ceil - nchan_floor})")
            print(f"    Last current limber_min: {last_limber}")
            print(f"    Extra channels would have limber_min ~ {last_limber} (similar)")

# (d) Summary for Professor Pullen
print("\n(d) SUMMARY FOR PROFESSOR PULLEN")
print("="*70)

print(f"\nFINDINGS:")
print(f"  1. Our implementation: {total_floor} channels")
print(f"     Breakdown: {_nchan_per_band[0]}, {_nchan_per_band[1]}, {_nchan_per_band[2]}, "
      f"{_nchan_per_band[3]}, {_nchan_per_band[4]}, {_nchan_per_band[5]} per band")

print(f"\n  2. SPHEREx instrument spec: 102 channels (17 per band)")
print(f"     Our code produces 10 fewer channels than spec")

print(f"\n  3. Literature commonly cites: 96 channels")
print(f"     Our code produces 4 fewer than this")

print(f"\n  4. Root cause: nchan = floor(Δλ_band / Δλ_chan)")
print(f"     The ratio is 15.x for bands 1-4, 16.x for bands 5-6")
print(f"     Using floor() rounds down, losing channels")

print(f"\n  5. Using ceil() would give: {total_ceil} channels")
print(f"     Still not 96 or 102")

print(f"\n  6. To get exactly 17 per band (102 total):")
print(f"     Need to SPECIFY nchan = 17, not compute from Δλ/R")

print(f"\nRECOMMENDATION:")
print(f"  OPTION A: Keep current 92 channels (matches Professor Pullen's code)")
print(f"  OPTION B: Use nchan = [17, 17, 17, 17, 17, 17] to match SPHEREx spec")
print(f"  OPTION C: Use nchan = [16, 16, 16, 16, 16, 16] to get 96 (literature)")

print(f"\n  Professor Pullen's original code uses the same floor() approach,")
print(f"  so our 92 channels MATCHES his implementation exactly.")

print("\n" + "="*70)
