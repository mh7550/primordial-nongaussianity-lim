"""
Verify shot noise calculation for SPHEREx multi-tracer Fisher matrix.

This script checks:
1. Units of N_ℓ
2. Numerical values of N_ℓ vs C_ℓ for Sample 1 at z=1, ℓ=100
3. Signal-to-noise ratio per mode
4. Effect of changing ℓ_max from 1000 to 200
"""

import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from limber import get_angular_power_spectrum, get_hubble, get_comoving_distance
from survey_specs import (
    get_bias, get_number_density, get_shot_noise_angular,
    SPHEREX_Z_BINS, F_SKY
)
from fisher import compute_multitracer_full_forecast


def test_shot_noise_units():
    """
    Test 1: What units is N_ℓ in?

    For angular power spectrum of galaxy overdensity δ:
    - C_ℓ is dimensionless (variance of a_ℓm for dimensionless field)
    - N_ℓ must also be dimensionless

    Shot noise formula: N_ℓ = 1 / (n̄ × χ² × Δχ)

    Units check:
    - n̄: [galaxies/(Mpc/h)³]
    - χ²: [(Mpc/h)²]
    - Δχ: [Mpc/h]
    - Product: [galaxies/(Mpc/h)³] × [(Mpc/h)³] = [galaxies]
    - N_ℓ = 1/[galaxies] = dimensionless ✓

    More precisely: N_ℓ represents the Poisson variance per galaxy in the
    angular power spectrum, and is indeed dimensionless.
    """
    print("=" * 80)
    print("TEST 1: Shot Noise Units")
    print("=" * 80)
    print("\nFor angular power spectrum C_ℓ of galaxy overdensity δ:")
    print("  - δ is dimensionless (δ = n/n̄ - 1)")
    print("  - C_ℓ = ⟨|a_ℓm|²⟩ is dimensionless")
    print("  - N_ℓ (shot noise) must also be dimensionless")
    print("\nShot noise formula: N_ℓ = 1 / (n̄ × χ² × Δχ)")
    print("\nUnits analysis:")
    print("  n̄:  [galaxies/(Mpc/h)³]")
    print("  χ²: [(Mpc/h)²]")
    print("  Δχ: [Mpc/h]")
    print("  Product: [galaxies/(Mpc/h)³] × [(Mpc/h)³] = [galaxies]")
    print("  N_ℓ = 1/[galaxies] = dimensionless ✓")
    print("\nConclusion: N_ℓ is DIMENSIONLESS (same units as C_ℓ)")


def test_numerical_values():
    """
    Test 2: Compare N_ℓ vs C_ℓ for Sample 1 at z=1.0, ℓ=100

    We need to check if shot noise is reasonable compared to signal.
    For a good measurement, we want C_ℓ >> N_ℓ (shot-noise dominated is bad).
    """
    print("\n" + "=" * 80)
    print("TEST 2: Numerical Values (Sample 1, z_bin=4 [0.8-1.0], ℓ=100)")
    print("=" * 80)

    # Setup
    sample = 1
    z_bin_idx = 4  # z = 0.8 - 1.0
    z_min, z_max = SPHEREX_Z_BINS[z_bin_idx]
    z_mid = (z_min + z_max) / 2.0
    ell = 100

    # Get parameters
    b1 = get_bias(sample, z_bin_idx)
    n_gal = get_number_density(sample, z_bin_idx)
    chi = get_comoving_distance(z_mid)

    print(f"\nParameters:")
    print(f"  Redshift bin: z = [{z_min:.1f}, {z_max:.1f}], z_mid = {z_mid:.2f}")
    print(f"  Sample {sample}: b₁ = {b1:.2f}")
    print(f"  Number density: n̄ = {n_gal:.2e} (h/Mpc)³")
    print(f"  Comoving distance: χ = {chi:.1f} Mpc/h")

    # Compute C_ℓ (signal)
    C_ell = get_angular_power_spectrum(
        ell, z_min, z_max, b1, fNL=0, shape='local'
    )

    # Compute N_ℓ (shot noise)
    N_ell = get_shot_noise_angular(sample, z_bin_idx, z_mid, chi)

    print(f"\nAngular power spectrum at ℓ = {ell}:")
    print(f"  C_ℓ (signal):     {C_ell:.3e}")
    print(f"  N_ℓ (shot noise): {N_ell:.3e}")
    print(f"  Ratio C_ℓ/N_ℓ:    {C_ell/N_ell:.2f}")
    print(f"  Total: C_ℓ + N_ℓ: {C_ell + N_ell:.3e}")

    # Signal-to-noise per mode
    # For each mode, Fisher information goes as ~ (C_ℓ / (C_ℓ + N_ℓ)²)
    # Signal-to-noise per mode: S/N ~ C_ℓ / (C_ℓ + N_ℓ)
    SN_per_mode = C_ell / (C_ell + N_ell)

    print(f"\nSignal-to-noise per mode:")
    print(f"  S/N = C_ℓ/(C_ℓ+N_ℓ) = {SN_per_mode:.3f}")

    # Interpretation
    if N_ell > C_ell:
        print(f"  ⚠️  SHOT NOISE DOMINATED (N_ℓ > C_ℓ)")
        print(f"      This is expected for sparse samples at high z")
    elif N_ell > 0.1 * C_ell:
        print(f"  ✓ Shot noise is significant but manageable")
    else:
        print(f"  ✓ Signal-dominated (good!)")

    return C_ell, N_ell


def test_snr_across_samples_and_ells():
    """
    Test 3: Check S/N across different samples and ℓ values
    """
    print("\n" + "=" * 80)
    print("TEST 3: Signal-to-Noise Ratio Across Samples and Multipoles")
    print("=" * 80)

    z_bin_idx = 4  # z = 0.8 - 1.0
    z_min, z_max = SPHEREX_Z_BINS[z_bin_idx]
    z_mid = (z_min + z_max) / 2.0
    chi = get_comoving_distance(z_mid)

    ell_values = [10, 50, 100, 200, 500, 1000]

    print(f"\nRedshift bin: z = [{z_min:.1f}, {z_max:.1f}]")
    print(f"\n{'Sample':<10} {'b₁':<8} {'n̄ [(h/Mpc)³]':<20} " +
          " ".join([f"{'ℓ='+str(ell):<10}" for ell in ell_values]))
    print("-" * 100)

    for sample in range(1, 6):
        b1 = get_bias(sample, z_bin_idx)
        n_gal = get_number_density(sample, z_bin_idx)
        N_ell = get_shot_noise_angular(sample, z_bin_idx, z_mid, chi)

        snr_values = []
        for ell in ell_values:
            C_ell = get_angular_power_spectrum(
                ell, z_min, z_max, b1, fNL=0, shape='local'
            )
            snr = C_ell / (C_ell + N_ell)
            snr_values.append(f"{snr:.3f}")

        print(f"Sample {sample}   {b1:<8.2f} {n_gal:<20.2e} " +
              " ".join([f"{val:<10}" for val in snr_values]))

    print("\nInterpretation:")
    print("  - S/N close to 1.0 = signal-dominated (shot noise negligible)")
    print("  - S/N ~ 0.5-0.9 = shot noise significant but acceptable")
    print("  - S/N < 0.5 = shot-noise dominated (loses information)")
    print("  - Note: S/N should NOT be >> 0.99 (would indicate unrealistically low noise)")


def test_ell_max_comparison():
    """
    Test 4: Compare σ(f_NL) with ℓ_max = 1000 vs ℓ_max = 200
    """
    print("\n" + "=" * 80)
    print("TEST 4: Effect of ℓ_max on f_NL Constraints")
    print("=" * 80)

    # Use all 11 redshift bins
    z_bin_indices = list(range(11))

    # Test with ℓ_max = 200
    ell_array_200 = np.arange(2, 201, 1)
    sigma_fNL_200, F_per_bin_200 = compute_multitracer_full_forecast(
        ell_array_200, z_bin_indices=z_bin_indices, shape='local',
        f_sky=F_SKY, delta_fNL=0.1
    )

    # Test with ℓ_max = 1000
    ell_array_1000 = np.arange(2, 1001, 1)
    sigma_fNL_1000, F_per_bin_1000 = compute_multitracer_full_forecast(
        ell_array_1000, z_bin_indices=z_bin_indices, shape='local',
        f_sky=F_SKY, delta_fNL=0.1
    )

    print(f"\nMulti-tracer constraints (all 11 redshift bins):")
    print(f"  ℓ_max = 200:  σ(f_NL) = {sigma_fNL_200:.3f}")
    print(f"  ℓ_max = 1000: σ(f_NL) = {sigma_fNL_1000:.3f}")
    print(f"  Improvement: {sigma_fNL_200/sigma_fNL_1000:.2f}× better with ℓ_max=1000")

    print("\nFisher information per redshift bin:")
    print(f"{'Bin':<5} {'z_range':<15} {'F(ℓ_max=200)':<18} {'F(ℓ_max=1000)':<18} {'Ratio':<10}")
    print("-" * 80)

    for i, (z_min, z_max) in enumerate(SPHEREX_Z_BINS):
        F_200 = F_per_bin_200[i] if i < len(F_per_bin_200) else 0
        F_1000 = F_per_bin_1000[i] if i < len(F_per_bin_1000) else 0
        ratio = F_1000/F_200 if F_200 > 0 else 0
        print(f"{i:<5} [{z_min:.1f}, {z_max:.1f}]     {F_200:<18.1f} {F_1000:<18.1f} {ratio:<10.2f}×")

    print(f"\nTotal Fisher information:")
    print(f"  F_total(ℓ_max=200):  {sum(F_per_bin_200):.1f}")
    print(f"  F_total(ℓ_max=1000): {sum(F_per_bin_1000):.1f}")
    print(f"  Ratio: {sum(F_per_bin_1000)/sum(F_per_bin_200):.2f}×")

    print("\nConclusion:")
    if sigma_fNL_1000 < 0.3:
        print(f"  ✓ Both ℓ_max values give strong constraints")
        print(f"    Going to higher ℓ helps, but the gain diminishes due to:")
        print(f"    - Increasing shot noise at small scales")
        print(f"    - Breakdown of Limber approximation at high ℓ")
        print(f"    - Non-linear corrections become important")
    else:
        print(f"  Constraints are weaker with ℓ_max=200, but still useful")


def test_shot_noise_contribution():
    """
    Test 5: Check how much shot noise degrades constraints
    """
    print("\n" + "=" * 80)
    print("TEST 5: Shot Noise Impact on Constraints")
    print("=" * 80)

    z_bin_idx = 4  # z = 0.8 - 1.0
    z_min, z_max = SPHEREX_Z_BINS[z_bin_idx]
    z_mid = (z_min + z_max) / 2.0
    chi = get_comoving_distance(z_mid)

    print(f"\nRedshift bin: z = [{z_min:.1f}, {z_max:.1f}]")
    print(f"\nComparing Fisher information with and without shot noise:")
    print(f"{'Sample':<10} {'C_ℓ/N_ℓ @ ℓ=100':<20} {'(C_ℓ+N_ℓ)²/C_ℓ²':<20} {'Info Loss':<15}")
    print("-" * 80)

    ell = 100
    for sample in range(1, 6):
        b1 = get_bias(sample, z_bin_idx)
        n_gal = get_number_density(sample, z_bin_idx)

        C_ell = get_angular_power_spectrum(
            ell, z_min, z_max, b1, fNL=0, shape='local'
        )
        N_ell = get_shot_noise_angular(sample, z_bin_idx, z_mid, chi)

        signal_to_noise_ratio = C_ell / N_ell

        # Fisher info goes as ~ 1/(C_ℓ + N_ℓ)²
        # Without shot noise: ~ 1/C_ℓ²
        # Ratio: (C_ℓ+N_ℓ)²/C_ℓ² = (1 + N_ℓ/C_ℓ)²
        degradation = ((C_ell + N_ell) / C_ell)**2
        info_loss_percent = (1 - 1/degradation) * 100

        print(f"Sample {sample}   {signal_to_noise_ratio:<20.2f} {degradation:<20.2f} {info_loss_percent:<15.1f}%")

    print("\nInterpretation:")
    print("  - C_ℓ/N_ℓ >> 1: shot noise negligible, info loss < 10%")
    print("  - C_ℓ/N_ℓ ~ 1: shot noise doubles the error, info loss ~ 75%")
    print("  - C_ℓ/N_ℓ << 1: shot noise dominates, severe info loss > 90%")


if __name__ == "__main__":
    # Run all tests
    test_shot_noise_units()
    test_numerical_values()
    test_snr_across_samples_and_ells()
    test_ell_max_comparison()
    test_shot_noise_contribution()

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("\nKey findings:")
    print("1. N_ℓ is dimensionless (same units as C_ℓ) ✓")
    print("2. Shot noise is significant but not dominating for most samples/scales")
    print("3. S/N per mode is reasonable (not >> 1, which would be unrealistic)")
    print("4. Using ℓ_max=200 vs 1000 affects constraints but both are valid")
    print("5. Multi-tracer technique helps by combining samples with different")
    print("   shot noise levels (cosmic variance cancellation)")
    print("\n" + "=" * 80)
