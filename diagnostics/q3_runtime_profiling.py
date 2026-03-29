"""
QUESTION 3: Runtime profiling for C_ell matrix computation

Profile actual runtime for computing 92×92 matrices and estimate Phase 4 cost.
"""

import numpy as np
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from angular_power_spectrum import (
    compute_C_ell_signal_matrix,
    N_CHANNELS,
    ELL_BIN_CENTERS,
    ELL_BIN_EDGES,
    LIMBER_MIN
)

print("="*70)
print("QUESTION 3: RUNTIME PROFILING")
print("="*70)

# (a) Time full 92×92 matrix at ell bin 1
print("\n(a) TIME FULL 92×92 MATRIX AT ELL BIN 1")
print("="*70)

ell_bin_1 = ELL_BIN_CENTERS[0]
print(f"\nℓ bin 1: ℓ ∈ [{ELL_BIN_EDGES[0]:.1f}, {ELL_BIN_EDGES[1]:.1f}]")
print(f"  Center: ℓ = {ell_bin_1:.1f}")

print(f"\nComputing 92×92 C_ℓ matrix...")
print(f"  (This will take several minutes due to Bessel integrals)")

t_start = time.time()
C_ell_1 = compute_C_ell_signal_matrix(ell_bin_1)
t_end = time.time()

runtime_bin1 = t_end - t_start

print(f"\nRUNTIME:")
print(f"  Wall-clock time: {runtime_bin1:.1f} seconds = {runtime_bin1/60:.2f} minutes")
print(f"  Matrix size: {C_ell_1.shape}")
print(f"  Total elements: {C_ell_1.size}")
print(f"  Time per element: {runtime_bin1 / C_ell_1.size * 1000:.1f} ms")

# (b) Break down Limber vs Bessel at each ell bin
print("\n(b) LIMBER VS BESSEL BREAKDOWN AT EACH ELL BIN")
print("="*70)

print(f"\nFor each ℓ bin, count how many of the {N_CHANNELS*(N_CHANNELS+1)//2} unique")
print(f"channel pairs use Limber vs Bessel (using symmetry):")

print(f"\n{'ℓ bin':<8} {'ℓ range':<20} {'ℓ center':<10} {'Limber pairs':<15} {'Bessel pairs':<15} {'% Bessel':<10}")
print("-"*85)

total_pairs = N_CHANNELS * (N_CHANNELS + 1) // 2  # Upper triangle including diagonal

for i, ell_center in enumerate(ELL_BIN_CENTERS):
    ell_min = ELL_BIN_EDGES[i]
    ell_max = ELL_BIN_EDGES[i+1]

    # Count how many channels require Bessel at this ell
    n_bessel_channels = np.sum(LIMBER_MIN > ell_center)
    n_limber_channels = N_CHANNELS - n_bessel_channels

    # Count pairs
    # A pair uses Bessel if EITHER channel requires it
    # A pair uses Limber if BOTH channels allow it

    limber_pairs = 0
    bessel_pairs = 0

    for ch1 in range(N_CHANNELS):
        for ch2 in range(ch1, N_CHANNELS):  # Upper triangle
            # Use Limber if ell > limber_min for BOTH
            if (ell_center > LIMBER_MIN[ch1]) and (ell_center > LIMBER_MIN[ch2]):
                limber_pairs += 1
            else:
                bessel_pairs += 1

    percent_bessel = 100.0 * bessel_pairs / total_pairs

    print(f"{i+1:<8} [{ell_min:5.1f}, {ell_max:5.1f}] {ell_center:<10.1f} "
          f"{limber_pairs:<15} {bessel_pairs:<15} {percent_bessel:<10.1f}")

print("-"*85)
print(f"Total unique pairs per ℓ bin: {total_pairs}")

# (c) Estimate total runtime for all 8 ell bins
print("\n(c) ESTIMATE TOTAL RUNTIME FOR ALL 8 ELL BINS")
print("="*70)

print(f"\nUsing ℓ bin 1 as representative:")
print(f"  Runtime for 1 bin: {runtime_bin1:.1f} s")

# More accurate estimate: different bins have different Bessel fractions
print(f"\nRefinement: account for varying Bessel fraction per bin")
print(f"  Assume: Limber ~60 ms, Bessel ~320 ms per pair")
print(f"  (Based on earlier timing tests)")

limber_time_per_pair = 0.06  # seconds
bessel_time_per_pair = 0.32  # seconds

total_estimated = 0

print(f"\n{'ℓ bin':<8} {'Limber':<12} {'Bessel':<12} {'Est. time (s)':<15} {'Est. time (min)':<15}")
print("-"*65)

for i, ell_center in enumerate(ELL_BIN_CENTERS):
    limber_pairs = 0
    bessel_pairs = 0

    for ch1 in range(N_CHANNELS):
        for ch2 in range(ch1, N_CHANNELS):
            if (ell_center > LIMBER_MIN[ch1]) and (ell_center > LIMBER_MIN[ch2]):
                limber_pairs += 1
            else:
                bessel_pairs += 1

    # Symmetry: off-diagonal computed twice but counted once above
    # Total computations = diagonal + 2×off_diagonal
    # For matrix computation, we compute upper triangle then copy
    # So time = limber_pairs * t_limber + bessel_pairs * t_bessel

    est_time = limber_pairs * limber_time_per_pair + bessel_pairs * bessel_time_per_pair
    total_estimated += est_time

    print(f"{i+1:<8} {limber_pairs:<12} {bessel_pairs:<12} {est_time:<15.1f} {est_time/60:<15.2f}")

print("-"*65)
print(f"TOTAL                                       {total_estimated:<15.1f} {total_estimated/60:<15.2f}")

print(f"\nSummary:")
print(f"  Measured (ℓ bin 1): {runtime_bin1:.1f} s")
print(f"  Estimated (8 bins): {total_estimated:.1f} s = {total_estimated/60:.1f} min")
print(f"  Estimated per bin: {total_estimated/8:.1f} s")

# (d) Phase 4 Newton-Raphson requirements
print("\n(d) PHASE 4 NEWTON-RAPHSON REQUIREMENTS")
print("="*70)

print(f"\nNewton-Raphson optimization for f_NL constraint:")
print(f"  Each iteration requires:")
print(f"    1. Compute C_ℓ(f_NL) at current f_NL value")
print(f"    2. Compute numerical derivatives ∂C_ℓ/∂f_NL")
print(f"       (typically via finite differences: 2 evaluations)")

print(f"\nTypical iteration count:")
print(f"  Conservative estimate: 10-20 iterations to convergence")
print(f"  Optimistic estimate: 5-10 iterations")

n_iter_conservative = 15
n_iter_optimistic = 7

print(f"\nC_ℓ evaluations per iteration:")
print(f"  - 1 for C_ℓ(f_NL)")
print(f"  - 2 for derivatives (f_NL ± δf_NL)")
print(f"  = 3 full C_ℓ computations per iteration")

evals_per_iter = 3

print(f"\nTotal C_ℓ evaluations:")
print(f"  Conservative ({n_iter_conservative} iter): {n_iter_conservative * evals_per_iter} evaluations")
print(f"  Optimistic ({n_iter_optimistic} iter): {n_iter_optimistic * evals_per_iter} evaluations")

# (e) Total Phase 4 runtime estimate
print("\n(e) TOTAL PHASE 4 RUNTIME ESTIMATE (NO CACHING)")
print("="*70)

time_per_full_matrix = total_estimated  # seconds for all 8 ell bins

runtime_conservative = n_iter_conservative * evals_per_iter * time_per_full_matrix
runtime_optimistic = n_iter_optimistic * evals_per_iter * time_per_full_matrix

print(f"\nTime for 1 full C_ℓ computation (all 8 ℓ bins):")
print(f"  {time_per_full_matrix:.1f} seconds = {time_per_full_matrix/60:.1f} minutes")

print(f"\nPhase 4 total runtime WITHOUT caching:")
print(f"  Conservative ({n_iter_conservative} iter × {evals_per_iter} evals):")
print(f"    {runtime_conservative:.1f} s = {runtime_conservative/60:.1f} min = {runtime_conservative/3600:.2f} hours")

print(f"\n  Optimistic ({n_iter_optimistic} iter × {evals_per_iter} evals):")
print(f"    {runtime_optimistic:.1f} s = {runtime_optimistic/60:.1f} min = {runtime_optimistic/3600:.2f} hours")

print(f"\nConclusion:")
if runtime_conservative > 3600:
    print(f"  ⚠️  Without caching, Phase 4 could take {runtime_conservative/3600:.1f}+ hours")
    print(f"  STRONG RECOMMENDATION: Implement caching strategy!")
else:
    print(f"  Runtime is manageable but caching would still help")

print("\n" + "="*70)
print("PROFILING COMPLETE")
print("  • Full 92×92 matrix: ~{:.0f} seconds per ℓ bin".format(time_per_full_matrix/8))
print("  • All 8 ℓ bins: ~{:.0f} minutes".format(total_estimated/60))
print("  • Phase 4 (no cache): ~{:.1f}-{:.1f} hours".format(
    runtime_optimistic/3600, runtime_conservative/3600))
print("="*70)
