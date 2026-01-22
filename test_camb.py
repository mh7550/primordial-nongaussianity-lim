#!/usr/bin/env python3
"""
Test script to verify CAMB installation and functionality.
Uses Planck 2018 cosmological parameters to generate matter power spectrum.
"""

import numpy as np
import matplotlib.pyplot as plt
import camb
from camb import model

def test_camb():
    """Test CAMB by generating matter power spectrum with Planck 2018 parameters."""

    print("=" * 60)
    print("CAMB Test Script - Primordial Non-Gaussianity LIM Project")
    print("=" * 60)
    print()

    # Set up cosmological parameters (Planck 2018)
    print("Setting up cosmological parameters (Planck 2018)...")
    pars = camb.CAMBparams()

    # Planck 2018 base parameters
    h = 0.6736          # Hubble parameter
    H0 = h * 100        # Hubble constant in km/s/Mpc
    ombh2 = 0.02237     # Baryon density
    omch2 = 0.1200      # Cold dark matter density
    omk = 0.0           # Curvature
    tau = 0.0544        # Optical depth to reionization
    As = 2.1e-9         # Scalar amplitude
    ns = 0.9649         # Scalar spectral index

    # Set parameters
    pars.set_cosmology(
        H0=H0,
        ombh2=ombh2,
        omch2=omch2,
        omk=omk,
        tau=tau
    )

    # Set initial power spectrum
    pars.InitPower.set_params(As=As, ns=ns)

    # Set up matter power spectrum calculation
    pars.set_matter_power(redshifts=[0.0], kmax=10.0)

    # Non-linear corrections (Halofit)
    pars.NonLinear = model.NonLinear_both

    print(f"  H0 = {H0:.2f} km/s/Mpc")
    print(f"  Ωb h² = {ombh2:.5f}")
    print(f"  Ωc h² = {omch2:.4f}")
    print(f"  τ = {tau:.4f}")
    print(f"  As = {As:.2e}")
    print(f"  ns = {ns:.4f}")
    print()

    # Run CAMB
    print("Running CAMB...")
    results = camb.get_results(pars)
    print("✓ CAMB calculation completed successfully!")
    print()

    # Get matter power spectrum
    print("Extracting matter power spectrum...")
    kh, z, pk = results.get_matter_power_spectrum(
        minkh=1e-4,
        maxkh=10,
        npoints=200
    )

    # Get linear and nonlinear power spectra separately
    pars.NonLinear = model.NonLinear_none
    results_linear = camb.get_results(pars)
    kh_lin, z_lin, pk_lin = results_linear.get_matter_power_spectrum(
        minkh=1e-4,
        maxkh=10,
        npoints=200
    )

    # Extract P(k) at z=0
    pk_z0 = pk[0, :]  # Nonlinear
    pk_lin_z0 = pk_lin[0, :]  # Linear

    print(f"✓ Power spectrum calculated for {len(kh)} k values")
    print()

    # Print some sample values
    print("Sample P(k) values at z=0:")
    print("-" * 60)
    print(f"{'k [h/Mpc]':<15} {'P(k) Linear':<20} {'P(k) Nonlinear':<20}")
    print("-" * 60)

    sample_indices = [0, len(kh)//4, len(kh)//2, 3*len(kh)//4, -1]
    for idx in sample_indices:
        print(f"{kh[idx]:<15.4e} {pk_lin_z0[idx]:<20.4e} {pk_z0[idx]:<20.4e}")
    print("-" * 60)
    print()

    # Calculate sigma_8
    sigma8 = results.get_sigma8()
    print(f"σ₈(z=0) = {sigma8[0]:.4f}")
    print()

    # Create plot
    print("Creating plot...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    # Plot 1: P(k) vs k
    ax1.loglog(kh, pk_lin_z0, 'b-', label='Linear', linewidth=2)
    ax1.loglog(kh, pk_z0, 'r-', label='Nonlinear (Halofit)', linewidth=2)
    ax1.set_xlabel(r'$k$ [h/Mpc]', fontsize=12)
    ax1.set_ylabel(r'$P(k)$ [(Mpc/h)³]', fontsize=12)
    ax1.set_title('Matter Power Spectrum at z=0 (Planck 2018)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)

    # Plot 2: Ratio of nonlinear to linear
    ratio = pk_z0 / pk_lin_z0
    ax2.semilogx(kh, ratio, 'g-', linewidth=2)
    ax2.axhline(y=1, color='k', linestyle='--', alpha=0.5)
    ax2.set_xlabel(r'$k$ [h/Mpc]', fontsize=12)
    ax2.set_ylabel(r'$P_{\rm nl}(k) / P_{\rm lin}(k)$', fontsize=12)
    ax2.set_title('Nonlinear to Linear Ratio', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0.8, 3.0])

    plt.tight_layout()

    # Save plot
    output_file = 'camb_test_output.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Plot saved to: {output_file}")
    print()

    print("=" * 60)
    print("CAMB test completed successfully! ✓")
    print("=" * 60)

    # Show plot
    plt.show()

if __name__ == "__main__":
    test_camb()
