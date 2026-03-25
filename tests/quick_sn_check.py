"""
Quick S/N check after channel width fix.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from angular_power_spectrum import (
    _compute_C_ell_limber_pair,
    compute_C_ell_noise_matrix,
    CHANNEL_CENTERS,
    CHANNEL_WIDTHS,
    ELL_BIN_CENTERS,
    LIMBER_MIN,
    LINE_PROPERTIES
)

print("="*70)
print("QUICK S/N CHECK (Limber only, for speed)")
print("="*70)

# Use a low-R channel where Limber is valid at ℓ=100
ell = 100.0
channel_idx = 10  # Band 1, limber_min = 44 < 100
line = 'Halpha'

print(f"\nTest configuration:")
print(f"  ℓ = {ell}")
print(f"  Channel {channel_idx}: λ = {CHANNEL_CENTERS[channel_idx]:.3f} μm")
print(f"  Channel width: Δλ = {CHANNEL_WIDTHS[channel_idx]:.4f} μm")
print(f"  limber_min = {LIMBER_MIN[channel_idx]} (< ℓ, so Limber valid)")

# Compute signal using Limber (fast)
C_signal = _compute_C_ell_limber_pair(ell, channel_idx, line, channel_idx, line)

# Compute noise
C_noise_matrix = compute_C_ell_noise_matrix(survey_mode='deep')
C_noise = C_noise_matrix[channel_idx, channel_idx]

# S/N
SNR = C_signal / C_noise

print(f"\nResults:")
print(f"  C_ℓ,signal = {C_signal:.6e}")
print(f"  C_n = {C_noise:.6e}")
print(f"  S/N = {SNR:.1f}")

print(f"\n{'Status':<30} {'Value':<30}")
print("-"*60)
print(f"{'Expected S/N range':<30} {'100 - 1000':<30}")
print(f"{'Our S/N':<30} {f'{SNR:.0f}':<30}")

if 10 < SNR < 10000:
    print(f"{'Assessment':<30} {'✓ REASONABLE':<30}")
    if 100 < SNR < 1000:
        print(f"{'Match with Cheng+2024':<30} {'✓ EXCELLENT':<30}")
    else:
        print(f"{'Match with Cheng+2024':<30} {'⚠ Close but not perfect':<30}")
else:
    print(f"{'Assessment':<30} {'✗ OUT OF RANGE':<30}")

print("="*70)
