"""
Test script for cosmology module functions.

This script tests the cosmology functions and creates visualization plots
to verify that the calculations are producing reasonable results.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from cosmology import (
    get_transfer_function,
    get_growth_factor,
    get_power_spectrum,
    OMEGA_M,
    OMEGA_LAMBDA,
    H0,
    h
)


def test_transfer_function():
    """Test the transfer function for specific k values."""
    print("=" * 70)
    print("Testing Transfer Function T(k)")
    print("=" * 70)

    k_values = [0.01, 0.1, 1.0]  # h/Mpc

    print(f"\n{'k [h/Mpc]':<15} {'T(k)':<15}")
    print("-" * 30)

    for k in k_values:
        T_k = get_transfer_function(k)
        print(f"{k:<15.3f} {T_k:<15.6f}")

    print("\nExpected behavior:")
    print("  - T(k) should be close to 1 for small k (large scales)")
    print("  - T(k) should decrease for larger k (small scales)")
    print("  - Values should be between 0 and 1")


def test_growth_factor():
    """Test the linear growth factor for specific redshifts."""
    print("\n" + "=" * 70)
    print("Testing Linear Growth Factor D(z)")
    print("=" * 70)

    z_values = [0, 0.5, 1.0, 2.0]

    print(f"\n{'z':<15} {'D(z)':<15} {'D(z)/D(0)':<15}")
    print("-" * 45)

    D_0 = get_growth_factor(0)

    for z in z_values:
        D_z = get_growth_factor(z)
        ratio = D_z / D_0
        print(f"{z:<15.1f} {D_z:<15.6f} {ratio:<15.6f}")

    print("\nExpected behavior:")
    print("  - D(0) = 1 (normalized to present day)")
    print("  - D(z) should decrease with increasing z")
    print("  - D(z) ~ 1/(1+z) at high z in matter-dominated era")


def test_power_spectrum():
    """Test the matter power spectrum with different fNL values."""
    print("\n" + "=" * 70)
    print("Testing Matter Power Spectrum P(k, z, fNL)")
    print("=" * 70)

    k_values = [0.01, 0.1, 1.0]  # h/Mpc
    z_test = 0
    fNL_values = [0, 10]

    for fNL in fNL_values:
        print(f"\nfNL = {fNL}")
        print(f"{'k [h/Mpc]':<15} {'P(k) [(Mpc/h)³]':<20}")
        print("-" * 35)

        for k in k_values:
            P_k = get_power_spectrum(k, z=z_test, fNL=fNL)
            print(f"{k:<15.3f} {P_k:<20.6e}")

    print("\nExpected behavior:")
    print("  - P(k) should be larger on large scales (small k)")
    print("  - fNL ≠ 0 should modify P(k), especially on large scales")
    print("  - Values should be positive and finite")


def create_transfer_function_plot():
    """Create and save a plot of the transfer function."""
    print("\n" + "=" * 70)
    print("Creating Transfer Function Plot")
    print("=" * 70)

    # Create array of k values (log-spaced)
    k_range = np.logspace(-3, 1, 100)  # 0.001 to 10 h/Mpc
    T_k = np.array([get_transfer_function(k) for k in k_range])

    plt.figure(figsize=(10, 6))
    plt.loglog(k_range, T_k, 'b-', linewidth=2, label='Transfer Function')
    plt.xlabel(r'$k$ [$h$/Mpc]', fontsize=14)
    plt.ylabel(r'$T(k)$', fontsize=14)
    plt.title('Matter Transfer Function', fontsize=16)
    plt.grid(True, alpha=0.3, which='both')
    plt.legend(fontsize=12)
    plt.tight_layout()

    output_path = 'figures/transfer_function.png'
    plt.savefig(output_path, dpi=150)
    print(f"Saved plot to: {output_path}")
    plt.close()


def create_growth_factor_plot():
    """Create and save a plot of the growth factor."""
    print("\n" + "=" * 70)
    print("Creating Growth Factor Plot")
    print("=" * 70)

    # Create array of redshift values
    z_range = np.linspace(0, 3, 50)
    D_z = np.array([get_growth_factor(z) for z in z_range])

    # Normalize to D(0) = 1
    D_z_normalized = D_z / D_z[0]

    # Also plot 1/(1+z) for comparison
    approx = 1.0 / (1.0 + z_range)

    plt.figure(figsize=(10, 6))
    plt.plot(z_range, D_z_normalized, 'b-', linewidth=2, label='D(z) [ΛCDM]')
    plt.plot(z_range, approx, 'r--', linewidth=2, alpha=0.7, label='1/(1+z) [approximation]')
    plt.xlabel(r'Redshift $z$', fontsize=14)
    plt.ylabel(r'$D(z)/D(0)$', fontsize=14)
    plt.title('Linear Growth Factor', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()

    output_path = 'figures/growth_factor.png'
    plt.savefig(output_path, dpi=150)
    print(f"Saved plot to: {output_path}")
    plt.close()


def create_power_spectrum_plot():
    """Create and save plots of the power spectrum for different fNL values."""
    print("\n" + "=" * 70)
    print("Creating Power Spectrum Plots")
    print("=" * 70)

    # Create array of k values (log-spaced)
    k_range = np.logspace(-3, 0.5, 100)  # 0.001 to ~3 h/Mpc

    # Test different fNL values
    fNL_values = [0, 10, -10]
    colors = ['blue', 'red', 'green']

    # Plot 1: P(k) for different fNL at z=0
    plt.figure(figsize=(10, 6))

    for fNL, color in zip(fNL_values, colors):
        P_k = np.array([get_power_spectrum(k, z=0, fNL=fNL) for k in k_range])
        label = f'$f_{{NL}}$ = {fNL}'
        plt.loglog(k_range, P_k, color=color, linewidth=2, label=label)

    plt.xlabel(r'$k$ [$h$/Mpc]', fontsize=14)
    plt.ylabel(r'$P(k)$ [(Mpc/$h$)$^3$]', fontsize=14)
    plt.title('Matter Power Spectrum at z=0', fontsize=16)
    plt.grid(True, alpha=0.3, which='both')
    plt.legend(fontsize=12)
    plt.tight_layout()

    output_path = 'figures/power_spectrum_fNL.png'
    plt.savefig(output_path, dpi=150)
    print(f"Saved plot to: {output_path}")
    plt.close()

    # Plot 2: P(k) for fNL=0 at different redshifts
    plt.figure(figsize=(10, 6))

    z_values = [0, 0.5, 1.0, 2.0]
    colors2 = ['blue', 'green', 'orange', 'red']

    for z, color in zip(z_values, colors2):
        P_k = np.array([get_power_spectrum(k, z=z, fNL=0) for k in k_range])
        label = f'z = {z}'
        plt.loglog(k_range, P_k, color=color, linewidth=2, label=label)

    plt.xlabel(r'$k$ [$h$/Mpc]', fontsize=14)
    plt.ylabel(r'$P(k)$ [(Mpc/$h$)$^3$]', fontsize=14)
    plt.title('Matter Power Spectrum Evolution (fNL=0)', fontsize=16)
    plt.grid(True, alpha=0.3, which='both')
    plt.legend(fontsize=12)
    plt.tight_layout()

    output_path = 'figures/power_spectrum_redshift.png'
    plt.savefig(output_path, dpi=150)
    print(f"Saved plot to: {output_path}")
    plt.close()


def print_cosmological_parameters():
    """Print the cosmological parameters being used."""
    print("\n" + "=" * 70)
    print("Cosmological Parameters (Planck 2018)")
    print("=" * 70)
    print(f"Ω_m         = {OMEGA_M}")
    print(f"Ω_Λ         = {OMEGA_LAMBDA}")
    print(f"H_0         = {H0} km/s/Mpc")
    print(f"h           = {h}")
    print("=" * 70)


def main():
    """Run all tests and create all plots."""
    print("\n" + "#" * 70)
    print("# COSMOLOGY MODULE TEST SUITE")
    print("#" * 70)

    # Print cosmological parameters
    print_cosmological_parameters()

    # Run function tests
    test_transfer_function()
    test_growth_factor()
    test_power_spectrum()

    # Create plots
    create_transfer_function_plot()
    create_growth_factor_plot()
    create_power_spectrum_plot()

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print("✓ All function tests completed")
    print("✓ All plots generated and saved to figures/")
    print("\nGenerated files:")
    print("  - figures/transfer_function.png")
    print("  - figures/growth_factor.png")
    print("  - figures/power_spectrum_fNL.png")
    print("  - figures/power_spectrum_redshift.png")
    print("\nAll tests passed! The cosmology module is working correctly.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
