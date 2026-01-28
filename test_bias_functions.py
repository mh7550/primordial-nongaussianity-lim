"""
Test suite for bias functions module.

This script validates the implementation of scale-dependent bias functions
for primordial non-Gaussianity and generates comparison plots.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from bias_functions import (
    delta_b_local,
    delta_b_equilateral,
    delta_b_orthogonal,
    get_total_bias
)
from cosmology import get_growth_factor


def test_local_scaling():
    """
    Test 1: Validate that local PNG shows characteristic 1/(k²*T(k)) scaling.

    Expected: Δb(k=0.01)/Δb(k=0.1) ≈ 100 * T(0.1)/T(0.01) ≈ 85-95
    The transfer function T(k) causes deviation from pure k^(-2).
    """
    print("=" * 70)
    print("TEST 1: Local PNG 1/(k²*T(k)) Scaling")
    print("=" * 70)

    from cosmology import get_transfer_function

    k_small = 0.01  # h/Mpc
    k_large = 0.1   # h/Mpc
    z = 0.0
    fNL = 10.0
    b1 = 2.0

    delta_b_small = delta_b_local(k_small, z, fNL, b1)
    delta_b_large = delta_b_local(k_large, z, fNL, b1)

    T_small = get_transfer_function(k_small)
    T_large = get_transfer_function(k_large)

    ratio_observed = delta_b_small / delta_b_large
    ratio_pure_k2 = (k_large / k_small)**2  # 100 for pure k^(-2)
    ratio_expected = ratio_pure_k2 * (T_large / T_small)  # Include T(k) effect

    print(f"T(k={k_small}) = {T_small:.4f}")
    print(f"T(k={k_large}) = {T_large:.4f}")
    print(f"Δb_local(k={k_small} h/Mpc) = {delta_b_small:.6f}")
    print(f"Δb_local(k={k_large} h/Mpc) = {delta_b_large:.6f}")
    print(f"Ratio Δb(0.01)/Δb(0.1) = {ratio_observed:.2f}")
    print(f"Expected for pure k^(-2) = {ratio_pure_k2:.2f}")
    print(f"Expected with T(k) effect = {ratio_expected:.2f}")

    # Allow 5% tolerance - should match closely with T(k) included
    assert np.abs(ratio_observed - ratio_expected) / ratio_expected < 0.05, \
        f"Local scaling test failed: ratio {ratio_observed:.2f} not close to {ratio_expected:.2f}"

    print("✓ Local PNG shows correct 1/(k²*T(k)) scaling\n")


def test_equilateral_weak_scaling():
    """
    Test 2: Validate that equilateral PNG shows much weaker scaling than local.

    Expected: Ratio << 100
    """
    print("=" * 70)
    print("TEST 2: Equilateral PNG Weak Scaling")
    print("=" * 70)

    k_small = 0.01  # h/Mpc
    k_large = 0.1   # h/Mpc
    z = 0.0
    fNL = 10.0
    b1 = 2.0

    delta_b_small = delta_b_equilateral(k_small, z, fNL, b1)
    delta_b_large = delta_b_equilateral(k_large, z, fNL, b1)

    ratio = delta_b_small / delta_b_large if delta_b_large != 0 else np.inf

    print(f"Δb_equil(k={k_small} h/Mpc) = {delta_b_small:.6f}")
    print(f"Δb_equil(k={k_large} h/Mpc) = {delta_b_large:.6f}")
    print(f"Ratio Δb(0.01)/Δb(0.1) = {ratio:.2f}")
    print(f"Local ratio for comparison: 100")

    # Equilateral should have much weaker scaling than local
    assert ratio < 50, \
        f"Equilateral scaling too strong: ratio {ratio:.2f} should be << 100"

    print("✓ Equilateral PNG shows weaker scale dependence than local\n")


def test_orthogonal_intermediate_scaling():
    """
    Test 3: Validate that orthogonal PNG has intermediate scale dependence.

    Expected: Local ratio > Orthogonal ratio > Equilateral ratio
    """
    print("=" * 70)
    print("TEST 3: Orthogonal PNG Intermediate Scaling")
    print("=" * 70)

    k_small = 0.01
    k_large = 0.1
    z = 0.0
    fNL = 10.0
    b1 = 2.0

    # Compute ratios for all three shapes
    ratio_local = delta_b_local(k_small, z, fNL, b1) / \
                  delta_b_local(k_large, z, fNL, b1)

    ratio_equil = delta_b_equilateral(k_small, z, fNL, b1) / \
                  delta_b_equilateral(k_large, z, fNL, b1)

    ratio_ortho = delta_b_orthogonal(k_small, z, fNL, b1) / \
                  delta_b_orthogonal(k_large, z, fNL, b1)

    print(f"Local ratio:       {ratio_local:.2f}")
    print(f"Orthogonal ratio:  {ratio_ortho:.2f}")
    print(f"Equilateral ratio: {ratio_equil:.2f}")

    # Orthogonal should be intermediate
    assert ratio_equil < ratio_ortho < ratio_local, \
        "Orthogonal scaling not intermediate between local and equilateral"

    print("✓ Orthogonal PNG shows intermediate scale dependence\n")


def test_fNL_zero():
    """
    Test 4: All bias functions should return 0 when fNL = 0.
    """
    print("=" * 70)
    print("TEST 4: Zero fNL Returns Zero Bias")
    print("=" * 70)

    k_test = np.array([0.01, 0.1])
    z = 0.0
    fNL = 0.0
    b1 = 2.0

    delta_b_loc = delta_b_local(k_test, z, fNL, b1)
    delta_b_equ = delta_b_equilateral(k_test, z, fNL, b1)
    delta_b_ort = delta_b_orthogonal(k_test, z, fNL, b1)

    print(f"Local bias (fNL=0): {delta_b_loc}")
    print(f"Equilateral bias (fNL=0): {delta_b_equ}")
    print(f"Orthogonal bias (fNL=0): {delta_b_ort}")

    assert np.allclose(delta_b_loc, 0.0, atol=1e-10), \
        "Local bias not zero when fNL=0"
    assert np.allclose(delta_b_equ, 0.0, atol=1e-10), \
        "Equilateral bias not zero when fNL=0"
    assert np.allclose(delta_b_ort, 0.0, atol=1e-10), \
        "Orthogonal bias not zero when fNL=0"

    print("✓ All bias functions return 0 when fNL = 0\n")


def test_b1_unity():
    """
    Test 5: All bias functions should return 0 when b1 = 1.

    Unbiased tracers (b1 = 1) trace the matter distribution exactly,
    so they should have no PNG signature.
    """
    print("=" * 70)
    print("TEST 5: Unbiased Tracers (b1=1) Have No PNG Signal")
    print("=" * 70)

    k_test = np.array([0.01, 0.1])
    z = 0.0
    fNL = 10.0
    b1 = 1.0  # Unbiased tracer

    delta_b_loc = delta_b_local(k_test, z, fNL, b1)
    delta_b_equ = delta_b_equilateral(k_test, z, fNL, b1)
    delta_b_ort = delta_b_orthogonal(k_test, z, fNL, b1)

    print(f"Local bias (b1=1): {delta_b_loc}")
    print(f"Equilateral bias (b1=1): {delta_b_equ}")
    print(f"Orthogonal bias (b1=1): {delta_b_ort}")

    assert np.allclose(delta_b_loc, 0.0, atol=1e-10), \
        "Local bias not zero when b1=1"
    assert np.allclose(delta_b_equ, 0.0, atol=1e-10), \
        "Equilateral bias not zero when b1=1"
    assert np.allclose(delta_b_ort, 0.0, atol=1e-10), \
        "Orthogonal bias not zero when b1=1"

    print("✓ Unbiased tracers (b1=1) have no PNG signal\n")


def test_redshift_evolution():
    """
    Test 6: Validate redshift evolution at multiple redshifts.
    """
    print("=" * 70)
    print("TEST 6: Redshift Evolution")
    print("=" * 70)

    k = 0.01  # h/Mpc
    z_array = np.array([0, 1, 2])
    fNL = 10.0
    b1 = 2.0

    print(f"Testing at k = {k} h/Mpc, fNL = {fNL}, b1 = {b1}")
    print("\nLocal bias vs redshift:")
    for z in z_array:
        delta_b = delta_b_local(k, z, fNL, b1)
        print(f"  z = {z}: Δb_local = {delta_b:.6f}")

    # Bias should increase with redshift (1/D(z) dependence)
    # D(z) decreases with increasing z, so Δb increases
    delta_b_z0 = delta_b_local(k, 0.0, fNL, b1)
    delta_b_z2 = delta_b_local(k, 2.0, fNL, b1)

    print(f"\nRatio Δb(z=2)/Δb(z=0) = {delta_b_z2/delta_b_z0:.2f}")

    # With corrected D(z) that DECREASES with z:
    # Δb ∝ 1/D(z) INCREASES at higher z
    # This is the CORRECT physics: earlier times (higher z) have less growth
    # so the PNG bias signal is amplified
    print("Expected: Δb(z=2) > Δb(z=0) because D(z) decreases with z")
    print(f"         D(z=0) = {get_growth_factor(0.0):.4f}, D(z=2) = {get_growth_factor(2.0):.4f}")

    assert delta_b_z2 > delta_b_z0, \
        "Bias should increase with z because Δb ∝ 1/D(z) and D(z) decreases"

    print("✓ Bias correctly increases with redshift (Δb ∝ 1/D(z))\n")


def test_total_bias():
    """
    Test 7: Validate get_total_bias function.
    """
    print("=" * 70)
    print("TEST 7: Total Bias Function")
    print("=" * 70)

    k = 0.01
    z = 0.0
    fNL = 10.0
    b1 = 2.0

    # Test all shapes
    b_tot_local = get_total_bias(k, z, fNL, b1, shape='local')
    b_tot_equil = get_total_bias(k, z, fNL, b1, shape='equilateral')
    b_tot_ortho = get_total_bias(k, z, fNL, b1, shape='orthogonal')

    # Manual calculation for local
    delta_b_manual = delta_b_local(k, z, fNL, b1)
    b_tot_manual = b1 + delta_b_manual

    print(f"Total bias (local): {b_tot_local:.6f}")
    print(f"Total bias (equilateral): {b_tot_equil:.6f}")
    print(f"Total bias (orthogonal): {b_tot_ortho:.6f}")
    print(f"Manual calculation (local): {b_tot_manual:.6f}")

    assert np.isclose(b_tot_local, b_tot_manual), \
        "get_total_bias doesn't match manual calculation"

    # Test invalid shape
    try:
        get_total_bias(k, z, fNL, b1, shape='invalid')
        assert False, "Should raise ValueError for invalid shape"
    except ValueError:
        print("\n✓ Correctly raises ValueError for invalid shape")

    print("✓ get_total_bias function works correctly\n")


def generate_comparison_plot():
    """
    Generate comparison plot of all three bias shapes.

    Saves to: figures/bias_comparison_shapes.png
    """
    print("=" * 70)
    print("GENERATING PLOT 1: Bias Comparison (All Shapes)")
    print("=" * 70)

    # Create figures directory if it doesn't exist
    figures_dir = Path(__file__).parent / 'figures'
    figures_dir.mkdir(exist_ok=True)

    # Parameters
    k = np.logspace(-3, 0, 100)  # 0.001 to 1 h/Mpc
    z = 0.0
    fNL = 10.0
    b1 = 2.0

    # Compute bias for all shapes
    delta_b_loc = delta_b_local(k, z, fNL, b1)
    delta_b_equ = delta_b_equilateral(k, z, fNL, b1)
    delta_b_ort = delta_b_orthogonal(k, z, fNL, b1)

    # Create plot
    plt.figure(figsize=(10, 7))

    plt.loglog(k, delta_b_loc, 'b-', linewidth=2.5, label='Local', zorder=3)
    plt.loglog(k, delta_b_equ, 'r--', linewidth=2.5, label='Equilateral', zorder=2)
    plt.loglog(k, delta_b_ort, 'g-.', linewidth=2.5, label='Orthogonal', zorder=2)

    # Add reference line showing 1/k² scaling at large scales
    # Use k range where T(k) ≈ 1 (k < 0.05)
    k_ref = np.logspace(-3, -1.3, 20)  # k from 0.001 to ~0.05
    # Normalize to match local bias at k=0.01
    k_norm = 0.01
    idx_norm = np.argmin(np.abs(k - k_norm))
    delta_ref = delta_b_loc[idx_norm] * (k_norm / k_ref)**2
    plt.loglog(k_ref, delta_ref, 'k:', linewidth=2, alpha=0.6,
               label=r'$\propto k^{-2}$ (reference)', zorder=1)

    plt.xlabel(r'$k$ [h/Mpc]', fontsize=14)
    plt.ylabel(r'$\Delta b(k, z)$', fontsize=14)
    plt.title(f'Scale-Dependent Bias: PNG Shape Comparison\n' +
              f'$f_{{\\rm NL}} = {fNL}$, $b_1 = {b1}$, $z = {z}$',
              fontsize=15)
    plt.legend(fontsize=12, loc='best')
    plt.grid(True, alpha=0.3, which='both')

    # Add annotation about local bias scaling
    plt.text(0.05, 0.25,
             r'Local: $\Delta b \propto 1/(k^2 T(k))$' + '\n' +
             r'Parallel to $k^{-2}$ at large scales' + '\n' +
             r'Steeper than $k^{-2}$ at small scales',
             transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

    plt.tight_layout()

    output_path = figures_dir / 'bias_comparison_shapes.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def generate_fNL_variation_plot():
    """
    Generate plot showing local bias for different fNL values.

    Saves to: figures/bias_local_fNL.png
    """
    print("=" * 70)
    print("GENERATING PLOT 2: Local Bias fNL Variation")
    print("=" * 70)

    figures_dir = Path(__file__).parent / 'figures'

    # Parameters
    k = np.logspace(-3, 0, 100)
    z = 0.0
    b1 = 2.0
    fNL_values = [-10, 10]  # Removed 0 since it can't be shown on log scale

    # Create plot
    plt.figure(figsize=(10, 7))

    colors = ['blue', 'red']
    linestyles = ['--', '-']

    for fNL, color, ls in zip(fNL_values, colors, linestyles):
        delta_b = delta_b_local(k, z, fNL, b1)
        # Sign of delta_b matches sign of fNL
        label = f'$f_{{\\rm NL}}^{{\\rm loc}} = {fNL:+d}$'

        # Plot actual signed values (not absolute value)
        # Negative values will be shown as dashed to distinguish
        if fNL > 0:
            plt.loglog(k, delta_b, color=color, linestyle=ls,
                       linewidth=2.5, label=label)
        else:
            # For negative fNL, plot absolute value with dashed line
            plt.loglog(k, -delta_b, color=color, linestyle=ls,
                       linewidth=2.5, label=label)

    plt.xlabel(r'$k$ [h/Mpc]', fontsize=14)
    plt.ylabel(r'$|\Delta b_{\rm local}(k, z)|$', fontsize=14)
    plt.title(f'Local-Type PNG Bias vs $f_{{\\rm NL}}$\n' +
              f'$b_1 = {b1}$, $z = {z}$ (Note: $f_{{\\rm NL}}=0$ gives $\\Delta b=0$)',
              fontsize=13)
    plt.legend(fontsize=12, loc='best')
    plt.grid(True, alpha=0.3, which='both')

    # Add text note explaining that fNL=0 is not shown
    plt.text(0.05, 0.95,
             'Note: Both curves have same magnitude\n(Δb ∝ fNL)',
             transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    output_path = figures_dir / 'bias_local_fNL.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    print(f"Note: Both fNL=±10 curves should overlap (same magnitude)")
    plt.close()


def generate_redshift_evolution_plot():
    """
    Generate plot showing local bias at different redshifts.

    Saves to: figures/bias_redshift_evolution.png
    """
    print("=" * 70)
    print("GENERATING PLOT 3: Redshift Evolution")
    print("=" * 70)

    figures_dir = Path(__file__).parent / 'figures'

    # Parameters
    k = np.logspace(-3, 0, 100)
    fNL = 10.0
    b1 = 2.0
    z_values = [0, 1, 2, 3]

    # Create plot
    plt.figure(figsize=(10, 7))

    # Use colors that go from dark (z=0) to bright (z=3) to show evolution clearly
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(z_values)))

    # Plot in reverse order so z=3 is on top in the legend
    for i, (z, color) in enumerate(zip(z_values, colors)):
        delta_b = delta_b_local(k, z, fNL, b1)
        plt.loglog(k, delta_b, color=color, linewidth=2.5,
                   label=f'$z = {z}$', zorder=len(z_values)-i)

    plt.xlabel(r'$k$ [h/Mpc]', fontsize=14)
    plt.ylabel(r'$\Delta b_{\rm local}(k, z)$', fontsize=14)
    plt.title(f'Local PNG Bias: Redshift Evolution\n' +
              f'$f_{{\\rm NL}}^{{\\rm loc}} = {fNL}$, $b_1 = {b1}$',
              fontsize=15)
    plt.legend(fontsize=12, loc='best')
    plt.grid(True, alpha=0.3, which='both')

    # Add annotation with actual observed behavior
    # Check which direction the curves go
    delta_b_z0 = delta_b_local(0.01, 0, fNL, b1)
    delta_b_z3 = delta_b_local(0.01, 3, fNL, b1)

    if delta_b_z3 > delta_b_z0:
        annotation = r'$\Delta b \propto 1/D(z)$ increases at higher $z$'
    else:
        annotation = r'$\Delta b \propto 1/D(z)$ with $D(z)$ from cosmology module'

    plt.text(0.05, 0.05,
             annotation,
             transform=plt.gca().transAxes,
             fontsize=11, style='italic',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    output_path = figures_dir / 'bias_redshift_evolution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")

    # Print diagnostic info
    print(f"Diagnostic: Δb(k=0.01, z=0) = {delta_b_z0:.6f}")
    print(f"Diagnostic: Δb(k=0.01, z=3) = {delta_b_z3:.6f}")
    print(f"Ratio Δb(z=3)/Δb(z=0) = {delta_b_z3/delta_b_z0:.2f}")

    plt.close()


def print_diagnostic_values():
    """Print diagnostic values for verification."""
    print("=" * 70)
    print("DIAGNOSTIC VALUES")
    print("=" * 70)

    from cosmology import get_transfer_function

    k_values = [0.001, 0.01, 0.1]
    z = 0.0
    fNL = 10.0
    b1 = 2.0

    print(f"\nParameters: fNL = {fNL}, b1 = {b1}, z = {z}")
    print(f"\n{'k [h/Mpc]':<12} {'T(k)':<10} {'Δb_local':<12} {'Units Check'}")
    print("-" * 70)

    for k in k_values:
        T_k = get_transfer_function(k)
        delta_b = delta_b_local(k, z, fNL, b1)
        print(f"{k:<12.3f} {T_k:<10.4f} {delta_b:<12.6f}")

    # Verify units
    print("\nUnits verification:")
    print(f"  Ωm = {0.3111} (dimensionless)")
    print(f"  H0 = {67.66} km/s/Mpc")
    print(f"  c = {299792.458} km/s")
    print(f"  H0²/c² = {(67.66**2)/(299792.458**2):.6e} Mpc^-2")
    print(f"  3*Ωm*H0²/c² = {3*0.3111*(67.66**2)/(299792.458**2):.6e} Mpc^-2")
    print(f"  k = 0.01 h/Mpc → k² = {0.01**2} (h/Mpc)^2")

    # Check magnitude
    print(f"\nMagnitude check:")
    print(f"  At k=0.01, Δb ≈ {delta_b_local(0.01, 0, 10, 2):.4f}")
    print(f"  This is << b1={b1}, as expected (small correction)")

    print()


def run_all_tests():
    """Run all validation tests."""
    print("\n" + "=" * 70)
    print("BIAS FUNCTIONS TEST SUITE")
    print("=" * 70 + "\n")

    # Print diagnostic values first
    print_diagnostic_values()

    # Run validation tests
    test_local_scaling()
    test_equilateral_weak_scaling()
    test_orthogonal_intermediate_scaling()
    test_fNL_zero()
    test_b1_unity()
    test_redshift_evolution()
    test_total_bias()

    # Generate plots
    generate_comparison_plot()
    generate_fNL_variation_plot()
    generate_redshift_evolution_plot()

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED ✓")
    print("=" * 70)
    print("\nGenerated plots:")
    print("  1. figures/bias_comparison_shapes.png")
    print("  2. figures/bias_local_fNL.png")
    print("  3. figures/bias_redshift_evolution.png")
    print()


if __name__ == "__main__":
    run_all_tests()
