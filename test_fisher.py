#!/usr/bin/env python
"""
Comprehensive tests for Fisher matrix forecasting module.

This script validates the Fisher matrix implementation for primordial
non-Gaussianity constraints and generates validation plots.

Expected Results:
- χ(z=1) ≈ 3300 Mpc/h for Planck cosmology
- C_ℓ should decrease with ℓ and be positive
- Fisher matrix should be symmetric and positive definite
- For SPHEREx-like survey: σ(f_NL^local) ~ 5-10
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.limber import get_comoving_distance, get_hubble, get_angular_power_spectrum
from src.fisher import compute_fisher_matrix, get_constraints, compute_constraints_vs_ell_max
from src.survey_specs import get_noise_power_spectrum_simple, F_SKY


def test_comoving_distance():
    """Test that comoving distance gives expected values."""
    print("=" * 70)
    print("TEST 1: Comoving Distance")
    print("=" * 70)

    z_test = np.array([0.0, 0.5, 1.0, 2.0, 3.0])
    chi_test = np.array([get_comoving_distance(z) for z in z_test])

    print(f"\n{'z':<10} {'χ(z) [Mpc/h]':<20}")
    print("-" * 30)
    for z, chi in zip(z_test, chi_test):
        print(f"{z:<10.1f} {chi:<20.1f}")

    # Check χ(z=1) ≈ 3300 Mpc/h for Planck cosmology
    chi_z1 = chi_test[2]
    expected = 3300.0
    relative_error = abs(chi_z1 - expected) / expected

    print(f"\n✓ Expected: χ(z=1) ≈ {expected:.0f} Mpc/h")
    print(f"  Computed: χ(z=1) = {chi_z1:.1f} Mpc/h")
    print(f"  Relative error: {relative_error*100:.1f}%")

    assert relative_error < 0.05, f"χ(z=1) too far from expected value"
    print("✓ PASS: Comoving distance test")
    return True


def test_angular_power_spectrum():
    """Test angular power spectrum properties."""
    print("\n" + "=" * 70)
    print("TEST 2: Angular Power Spectrum Properties")
    print("=" * 70)

    ell = np.logspace(1, 3, 20)
    z_min, z_max = 0.5, 1.5
    b1 = 2.0

    # Compute C_ℓ for different fNL values
    C_ell_minus10 = get_angular_power_spectrum(ell, z_min, z_max, b1, fNL=-10, shape='local')
    C_ell_0 = get_angular_power_spectrum(ell, z_min, z_max, b1, fNL=0, shape='local')
    C_ell_plus10 = get_angular_power_spectrum(ell, z_min, z_max, b1, fNL=10, shape='local')

    print(f"\nTesting C_ℓ for z ∈ [{z_min}, {z_max}], b₁ = {b1}")
    print(f"{'ℓ':<10} {'C_ℓ(fNL=-10)':<20} {'C_ℓ(fNL=0)':<20} {'C_ℓ(fNL=+10)':<20}")
    print("-" * 70)

    for i in [0, 5, 10, 15, 19]:
        print(f"{ell[i]:<10.0f} {C_ell_minus10[i]:<20.6e} {C_ell_0[i]:<20.6e} {C_ell_plus10[i]:<20.6e}")

    # Test 1: All C_ℓ should be positive
    assert np.all(C_ell_0 > 0), "C_ℓ should be positive"
    print("\n✓ PASS: All C_ℓ > 0")

    # Test 2: C_ℓ should decrease at high ℓ
    assert C_ell_0[-1] < C_ell_0[0], "C_ℓ should decrease at high ℓ"
    print("✓ PASS: C_ℓ decreases with ℓ at high ℓ")

    # Test 3: Positive fNL should increase C_ℓ
    assert np.all(C_ell_plus10 > C_ell_0), "Positive fNL should increase C_ℓ"
    print("✓ PASS: fNL=+10 increases C_ℓ relative to fNL=0")

    # Test 4: Negative fNL should decrease C_ℓ at low ℓ
    # (At high ℓ the effect is smaller)
    assert C_ell_minus10[0] < C_ell_0[0], "Negative fNL should decrease C_ℓ at low ℓ"
    print("✓ PASS: fNL=-10 decreases C_ℓ at low ℓ")

    return True


def test_noise_power_spectrum():
    """Test noise power spectrum."""
    print("\n" + "=" * 70)
    print("TEST 3: Noise Power Spectrum")
    print("=" * 70)

    ell = np.array([10, 100, 1000])
    z = 1.0

    N_ell_full = get_noise_power_spectrum_simple(ell, z, survey_mode='full')
    N_ell_deep = get_noise_power_spectrum_simple(ell, z, survey_mode='deep')

    print(f"\nAt z = {z}:")
    print(f"  Full survey: N_ℓ = {N_ell_full[0]:.2e} (nW/m²/sr)²")
    print(f"  Deep fields: N_ℓ = {N_ell_deep[0]:.2e} (nW/m²/sr)²")
    print(f"  Improvement factor: {N_ell_full[0]/N_ell_deep[0]:.1f}×")

    # Test 1: Noise should be independent of ℓ
    assert np.allclose(N_ell_full, N_ell_full[0]), "Noise should be flat in ℓ"
    print("\n✓ PASS: Noise is constant across ℓ")

    # Test 2: Deep fields should have lower noise
    assert np.all(N_ell_deep < N_ell_full), "Deep fields should have lower noise"
    print("✓ PASS: Deep field noise < full survey noise")

    return True


def test_fisher_matrix():
    """Test Fisher matrix computation."""
    print("\n" + "=" * 70)
    print("TEST 4: Fisher Matrix Properties")
    print("=" * 70)

    # Single parameter, single redshift bin
    ell = np.logspace(1, 3, 20)
    z_bins = [(0.5, 1.5)]
    params = ['fNL_local']
    b1_values = [2.0]

    print(f"\nComputing Fisher matrix for:")
    print(f"  Parameters: {params}")
    print(f"  Redshift bins: {z_bins}")
    print(f"  ℓ range: {ell[0]:.0f} - {ell[-1]:.0f}")

    F, param_names = compute_fisher_matrix(
        ell, z_bins, params, b1_values=b1_values,
        fNL_fid=0.0, f_sky=F_SKY, survey_mode='full'
    )

    print(f"\nFisher matrix:")
    print(F)

    # Test 1: Fisher matrix should be symmetric
    assert np.allclose(F, F.T), "Fisher matrix should be symmetric"
    print("\n✓ PASS: Fisher matrix is symmetric")

    # Test 2: Fisher matrix should be positive definite
    eigenvalues = np.linalg.eigvals(F)
    assert np.all(eigenvalues > 0), "Fisher matrix should be positive definite"
    print(f"✓ PASS: Fisher matrix is positive definite (eigenvalues: {eigenvalues})")

    # Test 3: Get constraints
    constraints = get_constraints(F, param_names)
    sigma_fNL = constraints['fNL_local']

    print(f"\nConstraints:")
    print(f"  σ(fNL_local) = {sigma_fNL:.2f}")

    # Test 4: Constraint should be in expected range for SPHEREx-like survey
    assert 3 < sigma_fNL < 15, f"σ(fNL) = {sigma_fNL:.2f} outside expected range [3, 15]"
    print(f"✓ PASS: σ(fNL_local) in expected range for SPHEREx-like survey")

    return True


def test_multi_parameter_fisher():
    """Test multi-parameter Fisher matrix."""
    print("\n" + "=" * 70)
    print("TEST 5: Multi-Parameter Fisher Matrix")
    print("=" * 70)

    # Three parameters: local, equilateral, orthogonal
    ell = np.logspace(1, 2.5, 15)  # Fewer ℓ for speed
    z_bins = [(0.5, 1.5)]
    params = ['fNL_local', 'fNL_equilateral', 'fNL_orthogonal']
    b1_values = [2.0]

    print(f"\nComputing Fisher matrix for:")
    print(f"  Parameters: {params}")
    print(f"  ℓ range: {ell[0]:.0f} - {ell[-1]:.0f}")

    F, param_names = compute_fisher_matrix(
        ell, z_bins, params, b1_values=b1_values,
        fNL_fid=0.0, f_sky=F_SKY, survey_mode='full'
    )

    print(f"\nFisher matrix:")
    print(F)

    # Test symmetry
    assert np.allclose(F, F.T), "Multi-parameter Fisher should be symmetric"
    print("\n✓ PASS: Multi-parameter Fisher matrix is symmetric")

    # Get constraints
    constraints = get_constraints(F, param_names)

    print(f"\nMarginalized constraints:")
    for param, sigma in constraints.items():
        print(f"  σ({param}) = {sigma:.2f}")

    print("✓ PASS: Multi-parameter Fisher matrix computed successfully")

    return True


def generate_angular_power_spectrum_plot():
    """Generate figure: Angular power spectrum vs ℓ for different fNL."""
    print("\n" + "=" * 70)
    print("GENERATING FIGURE: angular_power_spectrum.png")
    print("=" * 70)

    ell = np.logspace(1, 3, 30)
    z_min, z_max = 0.5, 1.5
    b1 = 2.0

    # Compute C_ℓ for different fNL values
    fNL_values = [-10, 0, 10]
    C_ell_dict = {}

    for fNL in fNL_values:
        C_ell = get_angular_power_spectrum(ell, z_min, z_max, b1, fNL=fNL, shape='local')
        C_ell_dict[fNL] = C_ell
        print(f"  Computed C_ℓ for fNL = {fNL:+3d}")

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 7))

    for fNL in fNL_values:
        label = f'$f_{{\\rm NL}}^{{\\rm local}} = {fNL:+d}$' if fNL != 0 else '$f_{\\rm NL}^{\\rm local} = 0$ (Gaussian)'
        ax.loglog(ell, C_ell_dict[fNL], label=label, linewidth=2)

    ax.set_xlabel(r'Multipole $\ell$', fontsize=14)
    ax.set_ylabel(r'Angular Power Spectrum $C_\ell$ [(nW/m$^2$/sr)$^2$]', fontsize=14)
    ax.set_title(f'Angular Power Spectrum from Limber Approximation\n$z \\in [{z_min}, {z_max}]$, $b_1 = {b1}$',
                 fontsize=14)
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xlim(ell[0], ell[-1])

    # Add text box with key info
    textstr = 'SPHEREx-like survey\n$f_{\\rm sky} = 0.75$'
    ax.text(0.05, 0.05, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    # Save figure
    output_dir = Path('figures')
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'angular_power_spectrum.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")

    plt.close()
    return True


def generate_noise_comparison_plot():
    """Generate figure: Signal vs noise comparison."""
    print("\n" + "=" * 70)
    print("GENERATING FIGURE: noise_comparison.png")
    print("=" * 70)

    ell = np.logspace(1, 3, 30)
    z_min, z_max = 0.5, 1.5
    z_mid = (z_min + z_max) / 2.0
    b1 = 2.0

    # Compute signal
    C_ell_signal = get_angular_power_spectrum(ell, z_min, z_max, b1, fNL=0, shape='local')

    # Compute noise for different survey modes
    N_ell_full = get_noise_power_spectrum_simple(ell, z_mid, survey_mode='full')
    N_ell_deep = get_noise_power_spectrum_simple(ell, z_mid, survey_mode='deep')

    print(f"  Computed signal C_ℓ(fNL=0)")
    print(f"  Computed noise N_ℓ (full and deep)")

    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    # Top panel: Signal and noise
    ax1.loglog(ell, C_ell_signal, 'b-', linewidth=2.5, label='Signal $C_\\ell$ ($f_{\\rm NL}=0$)')
    ax1.loglog(ell, N_ell_full, 'r--', linewidth=2, label='Noise $N_\\ell$ (full survey)')
    ax1.loglog(ell, N_ell_deep, 'g:', linewidth=2, label='Noise $N_\\ell$ (deep fields)')

    ax1.set_xlabel(r'Multipole $\ell$', fontsize=14)
    ax1.set_ylabel(r'Power Spectrum [(nW/m$^2$/sr)$^2$]', fontsize=14)
    ax1.set_title('Signal vs Noise for SPHEREx Line Intensity Mapping', fontsize=14)
    ax1.legend(fontsize=12, loc='upper right')
    ax1.grid(True, alpha=0.3, which='both')

    # Bottom panel: Signal-to-noise ratio
    SNR_full = np.sqrt(C_ell_signal / N_ell_full)
    SNR_deep = np.sqrt(C_ell_signal / N_ell_deep)

    ax2.loglog(ell, SNR_full, 'r--', linewidth=2, label='S/N (full survey)')
    ax2.loglog(ell, SNR_deep, 'g:', linewidth=2, label='S/N (deep fields)')
    ax2.axhline(1, color='k', linestyle='-', linewidth=1, alpha=0.5, label='S/N = 1')

    ax2.set_xlabel(r'Multipole $\ell$', fontsize=14)
    ax2.set_ylabel(r'Signal-to-Noise Ratio (per mode)', fontsize=14)
    ax2.set_title('Signal-to-Noise Ratio', fontsize=14)
    ax2.legend(fontsize=12, loc='upper right')
    ax2.grid(True, alpha=0.3, which='both')

    plt.tight_layout()

    # Save figure
    output_dir = Path('figures')
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'noise_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")

    plt.close()
    return True


def generate_fisher_constraints_plot():
    """Generate figure: σ(fNL) vs ℓ_max."""
    print("\n" + "=" * 70)
    print("GENERATING FIGURE: fisher_constraints.png")
    print("=" * 70)

    ell_max_array = np.array([50, 100, 200, 300, 500, 700, 1000])
    z_bins = [(0.5, 1.5)]
    b1_values = [2.0]

    print(f"  Computing constraints vs ℓ_max...")
    print(f"  Testing ℓ_max values: {ell_max_array}")

    # Compute constraints for local PNG
    sigma_local = compute_constraints_vs_ell_max(
        ell_max_array, z_bins, 'fNL_local', b1_values=b1_values,
        ell_min=10, f_sky=F_SKY, survey_mode='full'
    )

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 7))

    ax.loglog(ell_max_array, sigma_local, 'bo-', linewidth=2.5, markersize=8,
              label='$\\sigma(f_{\\rm NL}^{\\rm local})$')

    # Add reference line: σ ∝ 1/sqrt(ℓ_max)
    # This is expected if adding more modes improves constraints
    reference = sigma_local[0] * np.sqrt(ell_max_array[0] / ell_max_array)
    ax.loglog(ell_max_array, reference, 'k--', linewidth=1.5, alpha=0.5,
              label=r'$\propto 1/\sqrt{\ell_{\rm max}}$ (reference)')

    ax.set_xlabel(r'Maximum Multipole $\ell_{\rm max}$', fontsize=14)
    ax.set_ylabel(r'$\sigma(f_{\rm NL}^{\rm local})$', fontsize=14)
    ax.set_title('Fisher Matrix Constraints vs Maximum Multipole\nSPHEREx-like Survey', fontsize=14)
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(True, alpha=0.3, which='both')

    # Add text box with key results
    textstr = f'$z \\in [0.5, 1.5]$, $b_1 = 2.0$\\n$f_{{\\rm sky}} = 0.75$\\n$\\sigma(f_{{\\rm NL}}) = {sigma_local[-1]:.2f}$ at $\\ell_{{\\rm max}} = 1000$'
    ax.text(0.05, 0.05, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    # Save figure
    output_dir = Path('figures')
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'fisher_constraints.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")

    plt.close()
    return True


def print_forecast_summary():
    """Print summary of σ(fNL) forecasts for SPHEREx."""
    print("\n" + "=" * 70)
    print("FISHER MATRIX FORECAST SUMMARY")
    print("=" * 70)

    # Single redshift bin
    ell = np.logspace(1, 3, 25)
    z_bins = [(0.5, 1.5)]
    b1_values = [2.0]

    print(f"\nSingle redshift bin: z ∈ [0.5, 1.5]")
    print(f"Linear bias: b₁ = 2.0")
    print(f"Multipole range: ℓ ∈ [10, 1000]")
    print(f"Sky fraction: f_sky = {F_SKY}")
    print("-" * 70)

    # Local PNG
    F_local, _ = compute_fisher_matrix(
        ell, z_bins, ['fNL_local'], b1_values=b1_values,
        f_sky=F_SKY, survey_mode='full'
    )
    constraints_local = get_constraints(F_local, ['fNL_local'])

    print(f"\nLocal PNG:")
    print(f"  σ(fNL_local) = {constraints_local['fNL_local']:.2f}")

    # Compare with Planck 2018
    planck_sigma = 4.7  # Planck 2018: fNL = -0.9 ± 4.7
    print(f"\n  Comparison with Planck 2018: σ(fNL) = {planck_sigma}")
    print(f"  SPHEREx/Planck ratio: {constraints_local['fNL_local']/planck_sigma:.2f}")

    # Multiple redshift bins
    print("\n" + "-" * 70)
    print("Multiple redshift bins:")
    z_bins_multi = [(0.5, 1.0), (1.0, 1.5), (1.5, 2.0)]
    b1_values_multi = [2.0, 2.5, 3.0]

    print(f"  Bins: {z_bins_multi}")
    print(f"  Biases: {b1_values_multi}")

    F_multi, _ = compute_fisher_matrix(
        ell, z_bins_multi, ['fNL_local'], b1_values=b1_values_multi,
        f_sky=F_SKY, survey_mode='full'
    )
    constraints_multi = get_constraints(F_multi, ['fNL_local'])

    print(f"\n  σ(fNL_local) = {constraints_multi['fNL_local']:.2f}")
    print(f"  Improvement over single bin: {constraints_local['fNL_local']/constraints_multi['fNL_local']:.2f}×")

    print("\n" + "=" * 70)
    print("Note: These are idealized forecasts assuming:")
    print("  - Limber approximation is valid")
    print("  - Linear bias model is accurate")
    print("  - No foreground contamination")
    print("  - No systematic errors")
    print("Real constraints will be weaker due to these effects.")
    print("=" * 70)


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("COMPREHENSIVE FISHER MATRIX VALIDATION TESTS")
    print("=" * 70)

    # Run all tests
    all_passed = True

    try:
        all_passed &= test_comoving_distance()
        all_passed &= test_angular_power_spectrum()
        all_passed &= test_noise_power_spectrum()
        all_passed &= test_fisher_matrix()
        all_passed &= test_multi_parameter_fisher()

        # Generate figures
        all_passed &= generate_angular_power_spectrum_plot()
        all_passed &= generate_noise_comparison_plot()
        all_passed &= generate_fisher_constraints_plot()

        # Print forecast summary
        print_forecast_summary()

        if all_passed:
            print("\n" + "=" * 70)
            print("✓ ALL TESTS PASSED")
            print("=" * 70)
        else:
            print("\n" + "=" * 70)
            print("✗ SOME TESTS FAILED")
            print("=" * 70)

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    print("\n")
