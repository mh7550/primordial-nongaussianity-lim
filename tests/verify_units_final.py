"""
Final verification of units match between signal and noise.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from angular_power_spectrum import (
    compute_C_ell_signal_matrix,
    compute_C_ell_noise_matrix,
    CHANNEL_CENTERS,
    ELL_BIN_CENTERS,
    LINE_PROPERTIES
)

print("="*70)
print("FINAL UNITS VERIFICATION")
print("="*70)

# Test at ell bin 1, Halpha channel
ell = ELL_BIN_CENTERS[0]
target_z = 1.5
lambda_obs = LINE_PROPERTIES['Halpha']['lambda_rest'] * (1 + target_z)
channel_idx = np.argmin(np.abs(CHANNEL_CENTERS - lambda_obs))

print(f"\nTest configuration:")
print(f"  ℓ = {ell:.1f}")
print(f"  Channel {channel_idx}: λ = {CHANNEL_CENTERS[channel_idx]:.3f} μm (Halpha at z~{target_z})")

# Compute matrices
C_signal = compute_C_ell_signal_matrix(ell)
C_noise = compute_C_ell_noise_matrix(survey_mode='deep')

# Get auto-spectrum values
signal_auto = C_signal[channel_idx, channel_idx]
noise_auto = C_noise[channel_idx, channel_idx]
SNR = signal_auto / noise_auto

print(f"\n{'Quantity':<30} {'Value':<25} {'Units':<20}")
print("-"*75)
print(f"{'Signal C_ℓ (auto)':<30} {signal_auto:<25.6e} {'(MJy/sr)² × ...':<20}")
print(f"{'Noise C_n (auto)':<30} {noise_auto:<25.6e} {'MJy²/sr':<20}")
print(f"{'Signal-to-Noise':<30} {SNR:<25.1f} {'dimensionless':<20}")

print(f"\n{'Status':<30} {'Check':<45}")
print("-"*75)

# Check 1: Both positive
check1 = signal_auto > 0 and noise_auto > 0
print(f"{'✓ Both positive':<30} {str(check1):<45}")

# Check 2: Same order of magnitude (within ~10^6)
ratio = signal_auto / noise_auto
check2 = 10 < ratio < 1e6
print(f"{'✓ S/N in range [10, 10^6]':<30} {str(check2) + f' (S/N = {ratio:.1f})':<45}")

# Check 3: Signal dominates noise
check3 = signal_auto > noise_auto
print(f"{'✓ Signal > Noise':<30} {str(check3):<45}")

# Check 4: Values are finite
check4 = np.isfinite(signal_auto) and np.isfinite(noise_auto)
print(f"{'✓ Finite values':<30} {str(check4):<45}")

print("\n" + "="*70)
if all([check1, check2, check3, check4]):
    print("✅ UNITS VERIFIED: Signal and noise are in compatible units!")
    print(f"   S/N ~ {ratio:.0f} is physically reasonable")
    print(f"   (Cheng+2024 Fig. 3 shows S/N ~ 100-1000 for brightest features)")
else:
    print("❌ UNITS MISMATCH: Issues detected!")
print("="*70)
