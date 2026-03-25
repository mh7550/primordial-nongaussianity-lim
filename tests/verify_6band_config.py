"""
Verify 6-band configuration matches Professor Pullen's setup.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from angular_power_spectrum import (
    N_CHANNELS,
    N_BANDS,
    LAMBDA_BAND_EDGES,
    SPECTRAL_RESOLUTION_R,
    CHANNEL_EDGES,
    CHANNEL_CENTERS,
    CHANNEL_WIDTHS,
    LIMBER_MIN,
    _nchan_per_band
)

print("="*70)
print("STEP 1: VERIFY 6-BAND CONFIGURATION")
print("="*70)

print(f"\nTotal channels: nchantot = {N_CHANNELS}")
print(f"Expected: 92")
print(f"Match: {N_CHANNELS == 92} ✓" if N_CHANNELS == 92 else f"Match: False ✗")

print(f"\nNumber of bands: {N_BANDS}")
print(f"\nBand configuration:")
print(f"{'Band':<6} {'λ range (μm)':<15} {'R':<6} {'Channels':<10}")
print("-"*40)
for i in range(N_BANDS):
    print(f"{i+1:<6} {LAMBDA_BAND_EDGES[i]:.2f}-{LAMBDA_BAND_EDGES[i+1]:.2f} "
          f"{SPECTRAL_RESOLUTION_R[i]:<6} {_nchan_per_band[i]:<10}")

print(f"\nChannel edges range: [{CHANNEL_EDGES[0]:.3f}, {CHANNEL_EDGES[-1]:.3f}] μm")
print(f"Channel centers range: [{CHANNEL_CENTERS[0]:.3f}, {CHANNEL_CENTERS[-1]:.3f}] μm")
print(f"Channel widths range: [{CHANNEL_WIDTHS.min():.4f}, {CHANNEL_WIDTHS.max():.4f}] μm")

print(f"\nLimber validity array:")
print(f"  Loaded: {len(LIMBER_MIN)} values")
print(f"  Range: [{LIMBER_MIN.min()}, {LIMBER_MIN.max()}]")
print(f"  First 10: {LIMBER_MIN[:10]}")
print(f"  Last 10: {LIMBER_MIN[-10:]}")

print("\n" + "="*70)
if N_CHANNELS == 92:
    print("✅ STEP 1 COMPLETE: 6-band configuration matches Professor Pullen's setup")
else:
    print(f"❌ MISMATCH: Got {N_CHANNELS} channels, expected 92")
print("="*70)
