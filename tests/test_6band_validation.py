"""
STEP 5: Focused validation for 6-band configuration.

Tests key properties after upgrade to 92 channels with Bessel/Limber switching.
Designed to run quickly by testing only essential features.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from angular_power_spectrum import (
    N_CHANNELS,
    ELL_BIN_CENTERS,
    _compute_C_ell_limber_pair,
    compute_C_ell_noise_matrix,
    LIMBER_MIN,
    CHANNEL_CENTERS
)

print("="*70)
print("STEP 5: 6-BAND CONFIGURATION VALIDATION (FAST)")
print("="*70)

test_results = []

# Test 1: Channel count
print("\nTest 1: Channel Count")
print(f"  N_CHANNELS = {N_CHANNELS}")
print(f"  Expected: 92")
test1_pass = (N_CHANNELS == 92)
print(f"  Status: {'✓ PASS' if test1_pass else '✗ FAIL'}")
test_results.append(test1_pass)

# Test 2: Limber validity array loaded
print("\nTest 2: Limber Validity Array")
print(f"  LIMBER_MIN length: {len(LIMBER_MIN)}")
print(f"  Range: [{LIMBER_MIN.min()}, {LIMBER_MIN.max()}]")
test2_pass = (len(LIMBER_MIN) == N_CHANNELS and LIMBER_MIN.min() > 0)
print(f"  Status: {'✓ PASS' if test2_pass else '✗ FAIL'}")
test_results.append(test2_pass)

# Test 3: Signal positivity (Limber only, for speed)
print("\nTest 3: Signal Positivity (Limber channels)")
ell = 100.0
limber_channels = np.where(LIMBER_MIN <= ell)[0][:5]  # First 5 Limber channels
all_positive = True
for ch in limber_channels:
    C = _compute_C_ell_limber_pair(ell, ch, 'Halpha', ch, 'Halpha')
    if C <= 0:
        all_positive = False
        break
test3_pass = all_positive
print(f"  Tested {len(limber_channels)} channels")
print(f"  All positive: {all_positive}")
print(f"  Status: {'✓ PASS' if test3_pass else '✗ FAIL'}")
test_results.append(test3_pass)

# Test 4: Noise matrix diagonal structure
print("\nTest 4: Noise Matrix Structure")
C_n = compute_C_ell_noise_matrix(survey_mode='deep')
off_diag = C_n - np.diag(np.diag(C_n))
max_off_diag = np.max(np.abs(off_diag))
test4_pass = (max_off_diag == 0.0)
print(f"  Shape: {C_n.shape}")
print(f"  Max off-diagonal: {max_off_diag:.2e}")
print(f"  Diagonal: {np.diag(C_n).min():.6e} to {np.diag(C_n).max():.6e}")
print(f"  Status: {'✓ PASS' if test4_pass else '✗ FAIL'}")
test_results.append(test4_pass)

# Test 5: S/N in reasonable range
print("\nTest 5: Signal-to-Noise Ratio")
# Use channel 10 (limber_min = 44 < 100)
ch_test = 10
C_signal = _compute_C_ell_limber_pair(ell, ch_test, 'Halpha', ch_test, 'Halpha')
C_noise = C_n[ch_test, ch_test]
SNR = C_signal / C_noise
test5_pass = (10 < SNR < 10000)
print(f"  Channel {ch_test} (λ = {CHANNEL_CENTERS[ch_test]:.3f} μm)")
print(f"  C_signal = {C_signal:.6e}")
print(f"  C_noise = {C_noise:.6e}")
print(f"  S/N = {SNR:.1f}")
print(f"  Expected range: [10, 10000]")
print(f"  Cheng+2024 range: [100, 1000]")
print(f"  Status: {'✓ PASS' if test5_pass else '✗ FAIL'}")
test_results.append(test5_pass)

# Test 6: High-R channels have high limber_min
print("\nTest 6: High-R Channels (Bands 5-6)")
# Channels 76-91 are bands 5-6
high_R_channels = range(76, 92)
high_R_limber = LIMBER_MIN[76:92]
all_high = np.all(high_R_limber > 350)
test6_pass = all_high
print(f"  Channels 76-91 (bands 5-6, R=110/130)")
print(f"  limber_min range: [{high_R_limber.min()}, {high_R_limber.max()}]")
print(f"  All > 350: {all_high}")
print(f"  Status: {'✓ PASS' if test6_pass else '✗ FAIL'}")
test_results.append(test6_pass)

# Summary
print("\n" + "="*70)
print("VALIDATION SUMMARY")
print("="*70)
passed = sum(test_results)
total = len(test_results)
print(f"\nTests passed: {passed}/{total}")

if passed == total:
    print("\n✅ ALL TESTS PASSED")
    print("\n6-band configuration validated:")
    print(f"  • {N_CHANNELS} channels ✓")
    print(f"  • Limber validity array loaded ✓")
    print(f"  • Signal positive ✓")
    print(f"  • Noise diagonal ✓")
    print(f"  • S/N ~ {SNR:.0f} (within expected range) ✓")
    print(f"  • High-R channels identified ✓")
else:
    print(f"\n❌ {total - passed} TEST(S) FAILED")
    sys.exit(1)

print("="*70)
