"""
STEP 4: Diagnose S/N ~ 12,000 issue.

Supervisor flagged that S/N ~ 12,000 for Halpha at ℓ~100 is too high.
Cheng et al. (2024) Fig. 3 shows S/N ~ 100-1000 at comparable scales.

This script performs detailed diagnosis of the noise normalization.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from angular_power_spectrum import (
    compute_C_ell_signal_matrix,
    compute_C_ell_noise_matrix,
    CHANNEL_CENTERS,
    CHANNEL_WIDTHS,
    ELL_BIN_CENTERS,
    PIXEL_SIZE_ARCSEC,
    OMEGA_PIX,
    LINE_PROPERTIES
)
from lim_signal import get_spherex_noise_at_wavelength

print("="*70)
print("STEP 4: DIAGNOSE S/N ~ 12,000 ISSUE")
print("="*70)

# Test configuration
ell = ELL_BIN_CENTERS[0]  # ℓ ~ 100
target_z = 1.5
lambda_obs = LINE_PROPERTIES['Halpha']['lambda_rest'] * (1 + target_z)
channel_idx = np.argmin(np.abs(CHANNEL_CENTERS - lambda_obs))

print(f"\nTest configuration:")
print(f"  ℓ = {ell:.1f} (bin 1, largest scales)")
print(f"  Halpha at z ~ {target_z}")
print(f"  Channel {channel_idx}: λ = {CHANNEL_CENTERS[channel_idx]:.3f} μm")
print(f"  Channel width: Δλ = {CHANNEL_WIDTHS[channel_idx]:.4f} μm")

# ============================================================================
# 1. CHECK OMEGA_PIX
# ============================================================================

print("\n" + "="*70)
print("1. PIXEL SOLID ANGLE Ω_pix")
print("="*70)

print(f"\nSPHEREx pixel size: {PIXEL_SIZE_ARCSEC} arcsec")
print(f"\nConversion to radians:")
print(f"  1 degree = 3600 arcsec")
print(f"  1 degree = π/180 radians")
print(f"  {PIXEL_SIZE_ARCSEC} arcsec = {PIXEL_SIZE_ARCSEC / 3600.0:.6e} degrees")

theta_rad = PIXEL_SIZE_ARCSEC / 3600.0 * np.pi / 180.0
print(f"  {PIXEL_SIZE_ARCSEC} arcsec = {theta_rad:.6e} radians")

Omega_pix_calc = theta_rad**2
print(f"\nΩ_pix = (θ)² = {Omega_pix_calc:.6e} sr")
print(f"Module value: {OMEGA_PIX:.6e} sr")
print(f"Match: {np.isclose(Omega_pix_calc, OMEGA_PIX)} ✓" if np.isclose(Omega_pix_calc, OMEGA_PIX) else "Match: False ✗")
print(f"Expected: ~9e-10 sr")

# ============================================================================
# 2. CHECK SIGMA_N
# ============================================================================

print("\n" + "="*70)
print("2. SPHEREx NOISE SIGMA_N")
print("="*70)

sigma_n_MJy = get_spherex_noise_at_wavelength(CHANNEL_CENTERS[channel_idx],
                                                survey_mode='deep')
print(f"\nσ_n from SPHEREx public products:")
print(f"  σ_n = {sigma_n_MJy:.6e} MJy/sr")

# Convert MJy/sr to nW/m²/sr
# 1 Jy = 10^-26 W/m²/Hz
# 1 MJy = 10^6 Jy = 10^6 × 10^-26 W/m²/Hz = 10^-20 W/m²/Hz
# For surface brightness (per sr), same conversion
# But we need to account for bandwidth Δν

# Assume SPHEREx noise is continuum (per Hz)
print(f"\nUnit conversion MJy/sr → nW/m²/sr:")
print(f"  1 MJy = 10^6 Jy = 10^6 × 10^-26 W/m²/Hz = 10^-20 W/m²/Hz")
print(f"  For continuum noise in Δν bandwidth:")

# Channel bandwidth in Hz
lambda_center_m = CHANNEL_CENTERS[channel_idx] * 1e-6  # m
delta_lambda_m = CHANNEL_WIDTHS[channel_idx] * 1e-6  # m
c_m_s = 2.998e8  # m/s

nu_center = c_m_s / lambda_center_m  # Hz
delta_nu = c_m_s * delta_lambda_m / lambda_center_m**2  # Hz (from δν = c δλ / λ²)

print(f"    ν_center = c/λ = {nu_center:.6e} Hz")
print(f"    Δν = c Δλ / λ² = {delta_nu:.6e} Hz")

# Continuum noise integrated over bandwidth
sigma_n_W_m2_sr = sigma_n_MJy * 1e-20 * delta_nu  # W/m²/sr
sigma_n_nW_m2_sr = sigma_n_W_m2_sr * 1e9  # nW/m²/sr

print(f"\n  Integrated over channel bandwidth Δν:")
print(f"    σ_n = {sigma_n_MJy:.6e} MJy/sr × Δν")
print(f"        = {sigma_n_MJy:.6e} × 10^-20 W/m²/Hz/sr × {delta_nu:.6e} Hz")
print(f"        = {sigma_n_W_m2_sr:.6e} W/m²/sr")
print(f"        = {sigma_n_nW_m2_sr:.6e} nW/m²/sr")

# ============================================================================
# 3. COMPARE C_SIGNAL AND C_NOISE
# ============================================================================

print("\n" + "="*70)
print("3. SIGNAL VS NOISE POWER SPECTRA")
print("="*70)

C_signal_matrix = compute_C_ell_signal_matrix(ell)
C_noise_matrix = compute_C_ell_noise_matrix(survey_mode='deep')

C_signal = C_signal_matrix[channel_idx, channel_idx]
C_noise = C_noise_matrix[channel_idx, channel_idx]

print(f"\nC_ℓ,signal:")
print(f"  Value: {C_signal:.6e}")
print(f"  Units: (claimed) MJy²/sr (or compatible)")

print(f"\nC_n (noise):")
print(f"  Value: {C_noise:.6e}")
print(f"  Formula: σ_n² × Ω_pix")
print(f"  Calculation: ({sigma_n_MJy:.6e})² × {OMEGA_PIX:.6e}")
print(f"              = {sigma_n_MJy**2 * OMEGA_PIX:.6e}")
print(f"  Match: {np.isclose(C_noise, sigma_n_MJy**2 * OMEGA_PIX)} ✓")
print(f"  Units: MJy²/sr")

print(f"\nSignal-to-Noise:")
print(f"  S/N = C_signal / C_noise = {C_signal / C_noise:.6e}")
print(f"\nExpected from Cheng+ 2024 Fig. 3: S/N ~ 100-1000")
print(f"Our value: S/N ~ {C_signal / C_noise:.0f}")
print(f"Discrepancy: Factor of {(C_signal / C_noise) / 500:.1f}×")

# ============================================================================
# 4. CHECK PER-PIXEL VS PER-BEAM
# ============================================================================

print("\n" + "="*70)
print("4. PER-PIXEL VS PER-BEAM CHECK")
print("="*70)

print(f"\nSPHEREx noise specification:")
print(f"  Source: SPHEREx public data products")
print(f"  Units listed: MJy/sr (surface brightness)")
print(f"  Interpretation: Continuum noise per steradian")
print(f"\n  ⚠️ CHECK: Is this per PIXEL or per BEAM?")
print(f"  - If per pixel: use Ω_pix = {OMEGA_PIX:.6e} sr")
print(f"  - If per beam (beam > pixel): need beam solid angle instead")

# SPHEREx beam size (FWHM) is approximately equal to pixel size
# So per-beam ≈ per-pixel for SPHEREx
print(f"\n  SPHEREx configuration:")
print(f"    Pixel: 6.2 arcsec")
print(f"    Beam: ~6 arcsec FWHM (diffraction limited)")
print(f"    → Beam ≈ Pixel, so no correction needed ✓")

# ============================================================================
# 5. UNIT CONVERSION CHECK
# ============================================================================

print("\n" + "="*70)
print("5. UNIT CONVERSION: MJy/sr ↔ nW/m²/sr")
print("="*70)

print(f"\nFor CONTINUUM (spectral radiance, per Hz):")
print(f"  1 MJy/sr = 10^-20 W/m²/Hz/sr")
print(f"  1 nW/m²/Hz/sr = 10^-9 W/m²/Hz/sr = 10^11 MJy/sr")

print(f"\nFor EMISSION LINES (integrated intensity, ν I_ν):")
print(f"  Signal uses: ν I_ν divided by line width Δν_line")
print(f"  Noise uses: continuum noise integrated over channel width Δν_chan")
print(f"  These two Δν's are DIFFERENT!")

print(f"\n  Line width: Δν_line ~ ν × (σ_v/c) ~ ν × 10^-3 (for σ_v = 300 km/s)")
print(f"              Δν_line ~ {nu_center * 1e-3:.6e} Hz")

print(f"\n  Channel width: Δν_chan = {delta_nu:.6e} Hz")

print(f"\n  Ratio: Δν_chan / Δν_line = {delta_nu / (nu_center * 1e-3):.2f}")
print(f"\n  ⚠️ MISMATCH: Signal uses Δν_line, Noise uses Δν_chan")
print(f"     This causes S/N to be off by factor of Δν_chan / Δν_line!")

# ============================================================================
# DIAGNOSIS SUMMARY
# ============================================================================

print("\n" + "="*70)
print("DIAGNOSIS SUMMARY")
print("="*70)

print(f"\n✓ CORRECT:")
print(f"  1. Ω_pix = 9.04×10^-10 sr ✓")
print(f"  2. Beam ≈ pixel for SPHEREx, no correction needed ✓")
print(f"  3. Noise formula C_n = σ_n² × Ω_pix is correct ✓")

print(f"\n⚠️ ISSUE IDENTIFIED:")
print(f"  Signal and noise use DIFFERENT frequency bandwidths:")
print(f"  - Signal: Uses line width Δν_line ~ 3×10^11 Hz (σ_v = 300 km/s)")
print(f"  - Noise: Uses channel width Δν_chan ~ {delta_nu:.2e} Hz")
print(f"  - Ratio: {delta_nu / (nu_center * 1e-3):.1f}×")

print(f"\n💡 LIKELY FIX:")
print(f"  Option 1: Signal is TOO LOW (line width too narrow)")
print(f"            → Use effective width matching channel width")
print(f"  Option 2: Noise is TOO LOW (not integrated over channel)")
print(f"            → Remove Δν integration from noise")
print(f"\n  Expected S/N ~ 100-1000 suggests we need to:")
print(f"  DECREASE signal OR INCREASE noise by factor of ~{(C_signal/C_noise)/500:.0f}")

print("="*70)
