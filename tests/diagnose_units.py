"""
Diagnose units mismatch in C_ell calculation.

Compare signal vs noise units step by step.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from angular_power_spectrum import (
    compute_C_ell_signal_matrix,
    compute_C_ell_noise_matrix,
    get_window_function_A0,
    chi_to_z,
    CHANNEL_CENTERS,
    ELL_BIN_CENTERS
)
from lim_signal import (
    get_bias_weighted_luminosity_density,
    get_line_luminosity_density,
    get_halo_bias_simple,
    get_angular_diameter_distance,
    get_luminosity_distance,
    get_spherex_noise_at_wavelength,
    LINE_PROPERTIES
)
from cosmology import get_power_spectrum, get_comoving_distance

print("="*70)
print("UNITS DIAGNOSIS: Signal vs Noise")
print("="*70)

# Test at ell bin 1, Halpha channel
ell = ELL_BIN_CENTERS[0]
target_z = 1.5
lambda_obs = LINE_PROPERTIES['Halpha']['lambda_rest'] * (1 + target_z)
channel_idx = np.argmin(np.abs(CHANNEL_CENTERS - lambda_obs))

print(f"\nTest point:")
print(f"  ℓ = {ell:.1f}")
print(f"  Halpha at z ~ {target_z:.1f}")
print(f"  Channel {channel_idx}: λ = {CHANNEL_CENTERS[channel_idx]:.3f} μm")

# ============================================================================
# SIGNAL UNITS TRACE
# ============================================================================

print("\n" + "="*70)
print("SIGNAL UNITS TRACE")
print("="*70)

# Use typical overlap parameters
z_overlap = 1.5
chi_overlap = get_comoving_distance(z_overlap)

print(f"\n1. Bias-weighted luminosity density M_i(z):")
M_i = get_bias_weighted_luminosity_density(z_overlap, line='Halpha')
print(f"   M_i = {M_i:.6e} erg/s/Mpc³")

print(f"\n2. Geometric factor A₀(χ):")
A0 = get_window_function_A0(chi_overlap)
print(f"   A₀ = {A0:.6e} (dimensionless)")
print(f"   A₀² = {A0**2:.6e}")

print(f"\n3. Matter power spectrum P(k,z):")
k = (ell + 0.5) / chi_overlap
P_k = get_power_spectrum(k, z_overlap)
print(f"   k = {k:.6e} h/Mpc")
print(f"   P(k,z) = {P_k:.6e} (Mpc/h)³")

print(f"\n4. Geometric factors:")
delta_chi = 100.0  # Typical overlap width in Mpc
print(f"   Δχ/χ² = {delta_chi / chi_overlap**2:.6e} Mpc⁻¹")

print(f"\n5. Combined signal:")
nu_over_delta_nu = 10.0  # Typical value
signal_product = (nu_over_delta_nu**2 * (delta_chi / chi_overlap**2) *
                  A0**2 * M_i**2 * P_k)
print(f"   (ν/Δν)² × (Δχ/χ²) × A₀² × M_i² × P(k)")
print(f"   = {signal_product:.6e}")

print(f"\n   SIGNAL UNITS:")
print(f"   (dimensionless) × Mpc⁻¹ × (dimensionless) × (erg/s/Mpc³)² × (Mpc/h)³")
print(f"   = (erg/s)² / Mpc⁴ × (Mpc/h)³")
print(f"   = (erg/s)² / (Mpc × h³)")
print(f"   ⚠️  This is NOT in intensity units (MJy/sr)²!")

# ============================================================================
# NOISE UNITS TRACE
# ============================================================================

print("\n" + "="*70)
print("NOISE UNITS TRACE")
print("="*70)

print(f"\n1. SPHEREx surface brightness noise:")
sigma_n = get_spherex_noise_at_wavelength(CHANNEL_CENTERS[channel_idx], survey_mode='deep')
print(f"   σ_n = {sigma_n:.6e} MJy/sr")

print(f"\n2. Pixel solid angle:")
PIXEL_SIZE_ARCSEC = 6.2
Omega_pix = (PIXEL_SIZE_ARCSEC / 3600.0 * np.pi / 180.0)**2
print(f"   Ω_pix = {Omega_pix:.6e} sr")
print(f"   (for {PIXEL_SIZE_ARCSEC} arcsec pixels)")

print(f"\n3. Noise power:")
C_n = sigma_n**2 * Omega_pix
print(f"   C_n = σ_n² × Ω_pix")
print(f"   = {C_n:.6e}")

print(f"\n   NOISE UNITS:")
print(f"   (MJy/sr)² × sr = MJy² / sr")
print(f"   ✓ This is in intensity² units")

# ============================================================================
# ACTUAL COMPUTED VALUES
# ============================================================================

print("\n" + "="*70)
print("ACTUAL COMPUTED C_ℓ VALUES")
print("="*70)

C_signal_matrix = compute_C_ell_signal_matrix(ell)
C_noise_matrix = compute_C_ell_noise_matrix(survey_mode='deep')

C_signal = C_signal_matrix[channel_idx, channel_idx]
C_noise = C_noise_matrix[channel_idx, channel_idx]

print(f"\nAt ℓ = {ell:.1f}, channel {channel_idx}:")
print(f"  C_ℓ,signal = {C_signal:.6e}")
print(f"  C_n = {C_noise:.6e}")
print(f"  S/N = {C_signal / C_noise:.6e}")

print("\n" + "="*70)
print("DIAGNOSIS")
print("="*70)

print("\n⚠️  UNITS MISMATCH CONFIRMED:")
print("  • Signal: (erg/s)² / (Mpc × h³) — luminosity density units")
print("  • Noise: MJy² / sr — intensity units")
print("\n  These are incompatible! The signal needs conversion to intensity.")

print("\n📖 MISSING CONVERSION:")
print("  The window function should convert luminosity density to intensity.")
print("  The standard conversion is:")
print("    I_ν = c/(4π) × M₀_i(z) / (H(z) × (1+z))")
print("  where:")
print("    - c is speed of light")
print("    - H(z) is Hubble parameter")
print("    - (1+z) converts from emitted to observed frequency")
print("\n  This gives units:")
print("    (cm/s) × (erg/s/Mpc³) / ((km/s/Mpc) × 1)")
print("    = erg/(s × cm² × Hz × sr)")
print("    = intensity in CGS → convert to MJy/sr")

print("\n💡 FIX REQUIRED:")
print("  Add proper intensity conversion factor to window function")
print("  or to C_ℓ signal calculation.")
print("="*70)
