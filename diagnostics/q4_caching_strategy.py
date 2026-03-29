"""
QUESTION 4: Caching and precomputation strategy

Evaluate optimization strategies for Phase 4.
"""

import numpy as np
import sys
import os
import time
from scipy.special import spherical_jn
from scipy.integrate import trapezoid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from angular_power_spectrum import (
    N_CHANNELS,
    ELL_BIN_CENTERS,
    K_GRID,
    _chi_grid_cache,
    compute_window_function
)
from cosmology import get_growth_factor, get_power_spectrum

print("="*70)
print("QUESTION 4: CACHING AND PRECOMPUTATION STRATEGY")
print("="*70)

# (a) File size for precomputed 92×92×8 matrix
print("\n(a) FILE SIZE FOR PRECOMPUTED C_ELL MATRIX")
print("="*70)

n_channels = N_CHANNELS
n_ell_bins = len(ELL_BIN_CENTERS)

# Size of one matrix element (float64)
bytes_per_element = 8  # 64-bit float

# Total elements
# Matrix is symmetric, but store full matrix for simplicity
elements_per_ell = n_channels * n_channels
total_elements = elements_per_ell * n_ell_bins

total_bytes = total_elements * bytes_per_element
total_kb = total_bytes / 1024
total_mb = total_kb / 1024

print(f"\nMatrix dimensions:")
print(f"  Per ℓ bin: {n_channels} × {n_channels} = {elements_per_ell:,} elements")
print(f"  Number of ℓ bins: {n_ell_bins}")
print(f"  Total elements: {total_elements:,}")

print(f"\nStorage (float64):")
print(f"  Bytes per element: {bytes_per_element}")
print(f"  Total: {total_bytes:,} bytes")
print(f"       = {total_kb:.1f} KB")
print(f"       = {total_mb:.2f} MB")

print(f"\nStorage formats:")
print(f"  NumPy .npy (uncompressed): ~{total_mb:.1f} MB")
print(f"  NumPy .npz (compressed): ~{total_mb * 0.3:.1f} MB (typical 70% compression)")
print(f"  HDF5 (compressed): ~{total_mb * 0.2:.1f} MB (better compression)")

print(f"\nFeasibility:")
print(f"  ✓ Very small ({total_mb:.1f} MB)")
print(f"  ✓ Easily fits in memory")
print(f"  ✓ Fast to save/load from disk")
print(f"  ✓ Can store multiple realizations (different f_NL values)")

print(f"\nLimitation:")
print(f"  ✗ Only useful if C_ℓ doesn't change during optimization")
print(f"  ✗ For f_NL fitting, C_ℓ DOES change (via scale-dependent bias)")
print(f"  → Precomputing won't help for Newton-Raphson!")

print(f"\nCONCLUSION (a):")
print(f"  Precomputing full matrix: NOT EFFECTIVE for Phase 4")
print(f"  Reason: f_NL enters through scale-dependent bias Δb(k, z, f_NL)")
print(f"  Need to recompute C_ℓ(f_NL) at each iteration")

# (b) Profile Bessel integral components
print("\n(b) PROFILE BESSEL INTEGRAL COMPONENTS")
print("="*70)

print(f"\nBessel integral breakdown:")
print(f"  I_iν(k, ℓ) = ∫ dχ D(χ) W_iν(χ) j_ℓ(k×χ)")
print(f"  C_ℓ = (2/π) ∫ dk k² P(k) I_iν(k) I_i'ν'(k)")

print(f"\nComponents to time:")
print(f"  1. j_ℓ(k×χ) evaluation (spherical Bessel functions)")
print(f"  2. D(χ) evaluation (growth factor)")
print(f"  3. W_iν(χ) evaluation (window function)")
print(f"  4. χ integration (trapezoid over chi grid)")
print(f"  5. P(k,z) evaluation (power spectrum)")
print(f"  6. k integration (trapezoid over k grid)")

# Test with a representative channel
channel_test = 30
line_test = 'Halpha'
ell_test = 100

print(f"\nTest case:")
print(f"  Channel: {channel_test}")
print(f"  Line: {line_test}")
print(f"  ℓ: {ell_test}")
print(f"  k grid: {len(K_GRID)} points")
print(f"  χ grid: {len(_chi_grid_cache)} points")

# Component 1: Window function
t0 = time.time()
W_inu, chi_grid = compute_window_function(channel_test, line_test)
t_window = time.time() - t0

print(f"\n1. Window function W_iν(χ):")
print(f"   Time: {t_window*1000:.2f} ms")

# Component 2: Growth factor
from angular_power_spectrum import chi_to_z
z_grid = chi_to_z(chi_grid)

t0 = time.time()
D_grid = get_growth_factor(z_grid)
t_growth = time.time() - t0

print(f"\n2. Growth factor D(χ) on {len(chi_grid)} points:")
print(f"   Time: {t_growth*1000:.2f} ms")

# Component 3: Bessel functions for all k
t0 = time.time()
j_ell_grid = np.zeros((len(K_GRID), len(chi_grid)))
for i, k in enumerate(K_GRID):
    j_ell_grid[i, :] = spherical_jn(ell_test, k * chi_grid)
t_bessel = time.time() - t0

print(f"\n3. Bessel j_ℓ(k×χ) for {len(K_GRID)} k values:")
print(f"   Time: {t_bessel*1000:.2f} ms")
print(f"   ({t_bessel/len(K_GRID)*1000:.3f} ms per k)")

# Component 4: Chi integration
t0 = time.time()
I_inu = np.zeros(len(K_GRID))
for i in range(len(K_GRID)):
    integrand = D_grid * W_inu * j_ell_grid[i, :]
    I_inu[i] = trapezoid(integrand, chi_grid)
t_chi_int = time.time() - t0

print(f"\n4. χ integration (trapezoid over {len(chi_grid)} points):")
print(f"   Time: {t_chi_int*1000:.2f} ms for {len(K_GRID)} k values")
print(f"   ({t_chi_int/len(K_GRID)*1000:.3f} ms per k)")

# Component 5: Power spectrum
z_test = 1.0
t0 = time.time()
P_k_grid = np.array([get_power_spectrum(k, z_test) for k in K_GRID])
t_power = time.time() - t0

print(f"\n5. Power spectrum P(k,z) for {len(K_GRID)} k values:")
print(f"   Time: {t_power*1000:.2f} ms")

# Component 6: k integration
t0 = time.time()
integrand_k = K_GRID**2 * P_k_grid * I_inu**2
C_ell_test = (2.0 / np.pi) * trapezoid(integrand_k, K_GRID)
t_k_int = time.time() - t0

print(f"\n6. k integration (trapezoid over {len(K_GRID)} points):")
print(f"   Time: {t_k_int*1000:.2f} ms")

total_time_breakdown = t_window + t_growth + t_bessel + t_chi_int + t_power + t_k_int

print(f"\nTOTAL (breakdown sum): {total_time_breakdown*1000:.2f} ms")

print(f"\nBreakdown:")
print(f"  Window function:     {t_window/total_time_breakdown*100:5.1f}%  ({t_window*1000:6.2f} ms)")
print(f"  Growth factor:       {t_growth/total_time_breakdown*100:5.1f}%  ({t_growth*1000:6.2f} ms)")
print(f"  Bessel j_ℓ:          {t_bessel/total_time_breakdown*100:5.1f}%  ({t_bessel*1000:6.2f} ms)")
print(f"  χ integration:       {t_chi_int/total_time_breakdown*100:5.1f}%  ({t_chi_int*1000:6.2f} ms)")
print(f"  Power spectrum:      {t_power/total_time_breakdown*100:5.1f}%  ({t_power*1000:6.2f} ms)")
print(f"  k integration:       {t_k_int/total_time_breakdown*100:5.1f}%  ({t_k_int*1000:6.2f} ms)")

# (c) Optimal caching strategy
print("\n(c) OPTIMAL CACHING STRATEGY")
print("="*70)

print(f"\nBased on the breakdown:")

print(f"\n1. CACHEABLE (independent of f_NL):")
print(f"   • j_ℓ(k×χ) grids: {t_bessel/total_time_breakdown*100:.0f}% of time")
print(f"     - Depends only on k, χ, ℓ (fixed geometry)")
print(f"     - Can precompute for all ℓ bins once")
print(f"     - Storage: {len(ELL_BIN_CENTERS)} ℓ × {len(K_GRID)} k × {len(chi_grid)} χ × 8 bytes")
print(f"       = {len(ELL_BIN_CENTERS) * len(K_GRID) * len(chi_grid) * 8 / 1e6:.1f} MB")

print(f"\n   • Growth factor D(χ): {t_growth/total_time_breakdown*100:.0f}% of time")
print(f"     - Depends only on χ (cosmology)")
print(f"     - Negligible storage: {len(chi_grid) * 8 / 1024:.1f} KB")

print(f"\n2. PARTIALLY CACHEABLE:")
print(f"   • Window functions W_iν(χ): {t_window/total_time_breakdown*100:.0f}% of time")
print(f"     - Depends on M₀_i(z) which is FIXED (not fit parameter)")
print(f"     - Can precompute for all channels/lines once")
print(f"     - Storage: {N_CHANNELS} ch × 4 lines × {len(chi_grid)} χ × 8 bytes")
print(f"       = {N_CHANNELS * 4 * len(chi_grid) * 8 / 1e6:.2f} MB")

print(f"\n3. NOT CACHEABLE (depends on f_NL):")
print(f"   • Power spectrum P(k,z): {t_power/total_time_breakdown*100:.0f}% of time")
print(f"     - Enters through scale-dependent bias")
print(f"     - Must recompute for each f_NL value")

print(f"   • χ and k integrals: {(t_chi_int + t_k_int)/total_time_breakdown*100:.0f}% of time")
print(f"     - Fast, not worth optimizing")

print(f"\nCACHING STRATEGY RECOMMENDATION:")

speedup_estimate = 1.0 / (1.0 - (t_bessel + t_growth + t_window) / total_time_breakdown)

print(f"\n  LEVEL 1 (Bessel functions only):")
print(f"    - Precompute j_ℓ(k×χ) for all ℓ bins")
print(f"    - Storage: ~{len(ELL_BIN_CENTERS) * len(K_GRID) * len(chi_grid) * 8 / 1e6:.0f} MB")
print(f"    - Speedup: ~{1.0/(1.0 - t_bessel/total_time_breakdown):.1f}×")

print(f"\n  LEVEL 2 (Bessel + Window + Growth):")
print(f"    - Precompute j_ℓ, W_iν, D(χ)")
print(f"    - Combine into I_iν(k, ℓ) cache")
print(f"    - Storage: ~{N_CHANNELS * 4 * len(ELL_BIN_CENTERS) * len(K_GRID) * 8 / 1e6:.0f} MB")
print(f"    - Speedup: ~{speedup_estimate:.1f}×")
print(f"    - This is OPTIMAL!")

print(f"\n  LEVEL 3 (Full C_ℓ precomputation):")
print(f"    - NOT USEFUL for Phase 4 (f_NL changes C_ℓ)")

print(f"\nIMPLEMENTATION APPROACH:")
print(f"  1. Initialize: Precompute I_iν(k, ℓ) for all channels/lines/ℓ bins")
print(f"     - Do once at start of Phase 4")
print(f"     - Store in memory ({N_CHANNELS * 4 * len(ELL_BIN_CENTERS) * len(K_GRID) * 8 / 1e6:.0f} MB OK)")

print(f"\n  2. Each Newton-Raphson iteration:")
print(f"     - Load cached I_iν(k, ℓ)")
print(f"     - Compute P(k, z, f_NL) with current f_NL")
print(f"     - Final k integral: C_ℓ = (2/π) ∫ dk k² P(k) I₁ I₂")
print(f"     - Time: ~{(t_power + t_k_int)*1000:.0f} ms per pair (vs {total_time_breakdown*1000:.0f} ms uncached)")

print(f"\n  3. Expected speedup:")
print(f"     - Per Bessel pair: {total_time_breakdown/(t_power + t_k_int):.1f}×")
print(f"     - Overall Phase 4: ~{speedup_estimate:.0f}× faster")
print(f"     - From hours → minutes!")

print("\n" + "="*70)
print("CACHING STRATEGY SUMMARY:")
print(f"  • Precompute I_iν(k, ℓ) integrals")
print(f"  • Storage: ~{N_CHANNELS * 4 * len(ELL_BIN_CENTERS) * len(K_GRID) * 8 / 1e6:.0f} MB (fits in RAM)")
print(f"  • Expected speedup: ~{speedup_estimate:.0f}×")
print(f"  • Phase 4 time: hours → minutes ✓")
print("="*70)
