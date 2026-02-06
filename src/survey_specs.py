"""
SPHEREx survey specifications and noise models.

This module provides survey parameters and noise calculations for the
SPHEREx (Spectro-Photometer for the History of the Universe, Epoch of
Reionization and Ices Explorer) mission.

References
----------
SPHEREx Public Products: https://github.com/SPHEREx/Public-products
Doré et al., arXiv:1412.4872 (2014) - SPHEREx mission concept
"""

import numpy as np
from pathlib import Path


# SPHEREx Survey Parameters
PIXEL_SIZE_ARCSEC = 6.2  # Pixel size in arcseconds
PIXEL_SIZE_RAD = PIXEL_SIZE_ARCSEC * np.pi / 180.0 / 3600.0  # Convert to radians
PIXEL_SOLID_ANGLE = PIXEL_SIZE_RAD**2  # Solid angle per pixel in steradians
F_SKY = 0.75  # Fraction of sky covered (full survey)
F_SKY_DEEP = 0.04  # Fraction of sky for deep fields

# Wavelength ranges for SPHEREx
LAMBDA_MIN = 0.75  # Minimum wavelength in μm
LAMBDA_MAX = 4.8   # Maximum wavelength in μm

# Spectral resolution
SPECTRAL_RESOLUTION = 41  # R = λ/Δλ for 0.75-3.8 μm
SPECTRAL_RESOLUTION_LONG = 135  # R = λ/Δλ for 3.8-4.8 μm

# Emission line wavelengths (rest-frame, in μm)
EMISSION_LINES = {
    'Halpha': 0.6563,  # Hα
    'Hbeta': 0.4861,   # Hβ
    'OIII_5007': 0.5007,  # [OIII] 5007Å
    'OII_3727': 0.3727,   # [OII] 3727Å doublet
}

# ============================================================================
# Official SPHEREx Galaxy Survey Parameters (v28 CBE)
# ============================================================================
# From: galaxy_density_v28_base_cbe.txt
# Reference: Doré et al. (2014), arXiv:1412.4872
#
# 5 galaxy samples defined by photo-z quality σ_z/(1+z):
# Sample 1: σ_z/(1+z) ≤ 0.003 (best photo-z, ~19M galaxies)
# Sample 2: σ_z/(1+z) = 0.003-0.01
# Sample 3: σ_z/(1+z) = 0.01-0.03
# Sample 4: σ_z/(1+z) = 0.03-0.1
# Sample 5: σ_z/(1+z) = 0.1-0.2 (worst photo-z, highest density)
# ============================================================================

# Redshift bin edges (11 bins from z=0 to z=4.6)
SPHEREX_Z_BINS = [
    (0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0),
    (1.0, 1.6), (1.6, 2.2), (2.2, 2.8), (2.8, 3.4), (3.4, 4.0), (4.0, 4.6)
]

# Linear galaxy bias b₁(z) for each sample and redshift bin
# Shape: SPHEREX_BIAS[sample_num][z_bin_idx]
# sample_num: 1-5, z_bin_idx: 0-10
SPHEREX_BIAS = {
    1: [1.3, 1.5, 1.8, 2.3, 2.1, 2.7, 3.6, 2.3, 3.2, 2.7, 3.8],  # Sample 1 (best photo-z)
    2: [1.2, 1.4, 1.6, 1.9, 2.3, 2.6, 3.4, 4.2, 4.3, 3.7, 4.6],  # Sample 2
    3: [1.0, 1.3, 1.5, 1.7, 1.9, 2.6, 3.0, 3.2, 3.5, 4.1, 5.0],  # Sample 3
    4: [0.98, 1.3, 1.4, 1.5, 1.7, 2.2, 3.6, 3.7, 2.7, 2.9, 5.0], # Sample 4
    5: [0.83, 1.2, 1.3, 1.4, 1.6, 2.1, 3.2, 4.2, 4.1, 4.5, 5.0], # Sample 5 (worst photo-z)
}

# Number density n(z) in (h/Mpc)³ comoving for each sample and redshift bin
# Shape: SPHEREX_NUMBER_DENSITY[sample_num][z_bin_idx]
# Used for shot noise: P_shot = 1/n̄
SPHEREX_NUMBER_DENSITY = {
    1: [9.97e-3, 4.11e-3, 5.01e-4, 7.05e-5, 3.16e-5, 1.64e-5, 3.59e-6, 8.07e-7, 1.84e-6, 1.50e-6, 1.13e-6],
    2: [1.23e-2, 8.56e-3, 2.82e-3, 9.37e-4, 4.30e-4, 5.00e-5, 8.03e-6, 3.83e-6, 3.28e-6, 1.07e-6, 6.79e-7],
    3: [1.34e-2, 8.57e-3, 3.62e-3, 2.94e-3, 2.04e-3, 2.12e-4, 6.97e-6, 2.02e-6, 1.43e-6, 1.93e-6, 6.79e-7],
    4: [2.29e-2, 1.29e-2, 5.35e-3, 4.95e-3, 4.15e-3, 7.96e-4, 7.75e-5, 7.87e-6, 2.46e-6, 1.93e-6, 1.36e-6],
    5: [1.49e-2, 7.52e-3, 3.27e-3, 2.50e-3, 1.83e-3, 7.34e-4, 2.53e-4, 5.41e-5, 2.99e-5, 9.41e-6, 2.04e-6],
}

# Photo-z error σ_z/(1+z) for each sample
SPHEREX_PHOTO_Z_ERROR = {
    1: 0.003,   # Best photo-z quality
    2: 0.01,    # 0.003-0.01 range, use midpoint
    3: 0.03,    # 0.01-0.03 range
    4: 0.1,     # 0.03-0.1 range
    5: 0.2,     # 0.1-0.2 range (worst quality)
}

# Number of samples and redshift bins
N_SAMPLES = 5
N_Z_BINS = 11


def wavelength_to_redshift(wavelength_obs, wavelength_rest):
    """
    Convert observed wavelength to redshift for a given emission line.

    Parameters
    ----------
    wavelength_obs : float
        Observed wavelength in μm
    wavelength_rest : float
        Rest-frame wavelength of emission line in μm

    Returns
    -------
    z : float
        Redshift
    """
    return (wavelength_obs / wavelength_rest) - 1.0


def redshift_to_wavelength(z, wavelength_rest):
    """
    Convert redshift to observed wavelength for a given emission line.

    Parameters
    ----------
    z : float
        Redshift
    wavelength_rest : float
        Rest-frame wavelength of emission line in μm

    Returns
    -------
    wavelength_obs : float
        Observed wavelength in μm
    """
    return wavelength_rest * (1.0 + z)


def get_noise_power_spectrum_simple(ell, z, emission_line='Halpha', survey_mode='full'):
    """
    Get simple noise power spectrum estimate for SPHEREx.

    This is a simplified model that doesn't require the actual sensitivity files.
    Uses typical SPHEREx sensitivities:
    - Full survey: ~5 × 10^-18 W/m²/sr per pixel
    - Deep fields: ~1 × 10^-18 W/m²/sr per pixel

    Parameters
    ----------
    ell : float or array_like
        Multipole moment(s)
    z : float
        Redshift of observation
    emission_line : str, optional
        Emission line to observe (default: 'Halpha')
    survey_mode : str, optional
        'full' for full-sky survey or 'deep' for deep fields

    Returns
    -------
    N_ell : float or array_like
        Noise power spectrum in (nW/m²/sr)²

    Notes
    -----
    The noise power spectrum is:

    N_ℓ = σ_pix² × Ω_pix

    where:
    - σ_pix is the pixel noise (1σ sensitivity)
    - Ω_pix is the solid angle per pixel
    """
    # Typical SPHEREx sensitivities (1σ per pixel)
    # Convert from W/m²/sr to nW/m²/sr
    if survey_mode == 'full':
        sigma_pix = 5.0e-18 * 1e9  # nW/m²/sr
    elif survey_mode == 'deep':
        sigma_pix = 1.0e-18 * 1e9  # nW/m²/sr
    else:
        raise ValueError(f"survey_mode must be 'full' or 'deep', got '{survey_mode}'")

    # Noise power spectrum
    # N_ℓ = σ²_pix × Ω_pix
    N_ell = sigma_pix**2 * PIXEL_SOLID_ANGLE

    # Make sure output shape matches input
    ell = np.asarray(ell)
    if ell.ndim == 0:
        return float(N_ell)
    else:
        return np.full_like(ell, N_ell, dtype=float)


def load_spherex_sensitivity(filename=None):
    """
    Load SPHEREx surface brightness sensitivity from file.

    Parameters
    ----------
    filename : str or Path, optional
        Path to Surface_Brightness_v28_base_cbe.txt file.
        If None, looks in standard locations.

    Returns
    -------
    data : dict
        Dictionary with keys:
        - 'wavelength': array of wavelengths in μm
        - 'sensitivity_full': full-sky noise in nW/m²/sr per pixel (1σ)
        - 'sensitivity_deep': deep-field noise in nW/m²/sr per pixel (1σ)

    Notes
    -----
    The sensitivity file can be downloaded from:
    https://github.com/SPHEREx/Public-products/blob/master/Surface_Brightness_v28_base_cbe.txt
    """
    if filename is None:
        # Try standard locations
        possible_paths = [
            Path('Surface_Brightness_v28_base_cbe.txt'),
            Path('data/Surface_Brightness_v28_base_cbe.txt'),
            Path('../Surface_Brightness_v28_base_cbe.txt'),
        ]

        for path in possible_paths:
            if path.exists():
                filename = path
                break

        if filename is None:
            raise FileNotFoundError(
                "Could not find Surface_Brightness_v28_base_cbe.txt. "
                "Please download from https://github.com/SPHEREx/Public-products "
                "or specify the path explicitly."
            )

    # Load the file
    # Expected format: wavelength (μm), full-sky noise, deep-field noise
    try:
        data_array = np.loadtxt(filename, comments='#')

        return {
            'wavelength': data_array[:, 0],
            'sensitivity_full': data_array[:, 1],
            'sensitivity_deep': data_array[:, 2] if data_array.shape[1] > 2 else None,
        }
    except Exception as e:
        raise IOError(f"Error loading sensitivity file: {e}")


def get_pixel_noise(z, emission_line='Halpha', survey_mode='full', sensitivity_data=None):
    """
    Get pixel noise for a given redshift and emission line.

    Parameters
    ----------
    z : float
        Redshift
    emission_line : str
        Emission line to observe
    survey_mode : str
        'full' or 'deep' survey mode
    sensitivity_data : dict, optional
        Output from load_spherex_sensitivity(). If None, uses simple model.

    Returns
    -------
    sigma_pix : float
        Pixel noise in nW/m²/sr
    """
    if sensitivity_data is None:
        # Use simple model
        if survey_mode == 'full':
            return 5.0e-18 * 1e9  # nW/m²/sr
        else:
            return 1.0e-18 * 1e9  # nW/m²/sr

    # Get observed wavelength
    lambda_rest = EMISSION_LINES.get(emission_line)
    if lambda_rest is None:
        raise ValueError(f"Unknown emission line: {emission_line}")

    lambda_obs = redshift_to_wavelength(z, lambda_rest)

    # Interpolate sensitivity at this wavelength
    wavelengths = sensitivity_data['wavelength']
    if survey_mode == 'full':
        sensitivities = sensitivity_data['sensitivity_full']
    elif survey_mode == 'deep':
        sensitivities = sensitivity_data['sensitivity_deep']
        if sensitivities is None:
            raise ValueError("Deep field sensitivities not available in data file")
    else:
        raise ValueError(f"survey_mode must be 'full' or 'deep'")

    # Interpolate
    sigma_pix = np.interp(lambda_obs, wavelengths, sensitivities)

    return sigma_pix


def get_bias(sample, z_bin_idx):
    """
    Get linear galaxy bias for a given sample and redshift bin.

    Parameters
    ----------
    sample : int
        Sample number (1-5)
    z_bin_idx : int
        Redshift bin index (0-10)

    Returns
    -------
    b1 : float
        Linear galaxy bias
    """
    if sample not in SPHEREX_BIAS:
        raise ValueError(f"Sample must be 1-5, got {sample}")
    if z_bin_idx < 0 or z_bin_idx >= N_Z_BINS:
        raise ValueError(f"z_bin_idx must be 0-{N_Z_BINS-1}, got {z_bin_idx}")

    return SPHEREX_BIAS[sample][z_bin_idx]


def get_number_density(sample, z_bin_idx):
    """
    Get number density for a given sample and redshift bin.

    Parameters
    ----------
    sample : int
        Sample number (1-5)
    z_bin_idx : int
        Redshift bin index (0-10)

    Returns
    -------
    n_gal : float
        Number density in (h/Mpc)³
    """
    if sample not in SPHEREX_NUMBER_DENSITY:
        raise ValueError(f"Sample must be 1-5, got {sample}")
    if z_bin_idx < 0 or z_bin_idx >= N_Z_BINS:
        raise ValueError(f"z_bin_idx must be 0-{N_Z_BINS-1}, got {z_bin_idx}")

    return SPHEREX_NUMBER_DENSITY[sample][z_bin_idx]


def get_photo_z_error(sample):
    """
    Get photo-z error σ_z/(1+z) for a given sample.

    Parameters
    ----------
    sample : int
        Sample number (1-5)

    Returns
    -------
    sigma_z : float
        Photo-z error σ_z/(1+z)
    """
    if sample not in SPHEREX_PHOTO_Z_ERROR:
        raise ValueError(f"Sample must be 1-5, got {sample}")

    return SPHEREX_PHOTO_Z_ERROR[sample]


def get_shot_noise_angular(sample, z_bin_idx, z_mid, comoving_distance):
    """
    Get shot noise for angular power spectrum.

    The shot noise in angular space is:
    N_ℓ = 1 / (n̄ × χ² × Δχ)

    where:
    - n̄ is the comoving number density in (h/Mpc)³
    - χ is the comoving distance to z_mid
    - Δχ is the radial width of the bin

    Parameters
    ----------
    sample : int
        Sample number (1-5)
    z_bin_idx : int
        Redshift bin index (0-10)
    z_mid : float
        Midpoint redshift of the bin
    comoving_distance : float
        Comoving distance to z_mid in Mpc/h

    Returns
    -------
    N_ell : float
        Shot noise power spectrum (independent of ℓ)
    """
    n_gal = get_number_density(sample, z_bin_idx)

    # Get bin width in redshift
    z_min, z_max = SPHEREX_Z_BINS[z_bin_idx]
    delta_z = z_max - z_min

    # Convert to comoving width (approximate)
    # Δχ ≈ c × Δz / H(z_mid)
    # For now, use a simple estimate
    # More accurate: integrate H(z) over the bin
    try:
        from .limber import get_hubble
    except ImportError:
        from limber import get_hubble
    H_z = get_hubble(z_mid)
    C_LIGHT = 299792.458  # km/s
    delta_chi = C_LIGHT * delta_z / H_z  # Mpc/h

    # Shot noise
    N_ell = 1.0 / (n_gal * comoving_distance**2 * delta_chi)

    return N_ell


def get_survey_info():
    """
    Return dictionary with SPHEREx survey information.

    Returns
    -------
    info : dict
        Survey specifications
    """
    return {
        'name': 'SPHEREx',
        'pixel_size_arcsec': PIXEL_SIZE_ARCSEC,
        'pixel_solid_angle_sr': PIXEL_SOLID_ANGLE,
        'f_sky_full': F_SKY,
        'f_sky_deep': F_SKY_DEEP,
        'wavelength_range_um': (LAMBDA_MIN, LAMBDA_MAX),
        'spectral_resolution': {
            'short': SPECTRAL_RESOLUTION,
            'long': SPECTRAL_RESOLUTION_LONG,
        },
        'emission_lines': EMISSION_LINES,
        'n_samples': N_SAMPLES,
        'n_z_bins': N_Z_BINS,
        'z_bins': SPHEREX_Z_BINS,
    }


if __name__ == "__main__":
    # Print survey information
    print("=" * 70)
    print("SPHEREx SURVEY SPECIFICATIONS")
    print("=" * 70)

    info = get_survey_info()
    print(f"\nMission: {info['name']}")
    print(f"Pixel size: {info['pixel_size_arcsec']:.1f} arcsec")
    print(f"Pixel solid angle: {info['pixel_solid_angle_sr']:.2e} sr")
    print(f"Sky coverage (full): {info['f_sky_full']*100:.0f}%")
    print(f"Sky coverage (deep): {info['f_sky_deep']*100:.1f}%")
    print(f"Wavelength range: {info['wavelength_range_um'][0]:.2f} - {info['wavelength_range_um'][1]:.1f} μm")
    print(f"Spectral resolution R: {info['spectral_resolution']['short']} (short λ), "
          f"{info['spectral_resolution']['long']} (long λ)")

    print("\nEmission lines:")
    for line, wavelength in info['emission_lines'].items():
        z_min = wavelength_to_redshift(LAMBDA_MIN, wavelength)
        z_max = wavelength_to_redshift(LAMBDA_MAX, wavelength)
        print(f"  {line:12s}: λ_rest = {wavelength:.4f} μm, "
              f"observable at z = {max(0, z_min):.2f} - {z_max:.2f}")

    # Test noise calculation
    print("\n" + "=" * 70)
    print("NOISE POWER SPECTRUM EXAMPLES")
    print("=" * 70)

    z_test = 1.0
    ell_test = np.array([10, 100, 1000])

    print(f"\nAt z = {z_test}, ℓ = {ell_test}:")
    print(f"Emission line: Halpha (λ_rest = {EMISSION_LINES['Halpha']:.4f} μm)")

    N_ell_full = get_noise_power_spectrum_simple(ell_test, z_test, survey_mode='full')
    N_ell_deep = get_noise_power_spectrum_simple(ell_test, z_test, survey_mode='deep')

    print(f"\nFull survey mode:")
    print(f"  N_ℓ = {N_ell_full[0]:.2e} (nW/m²/sr)² (constant for all ℓ)")

    print(f"\nDeep field mode:")
    print(f"  N_ℓ = {N_ell_deep[0]:.2e} (nW/m²/sr)² (constant for all ℓ)")
    print(f"  Improvement factor: {N_ell_full[0]/N_ell_deep[0]:.1f}×")

    # Print galaxy survey parameters
    print("\n" + "=" * 70)
    print("SPHEREX GALAXY SURVEY PARAMETERS (v28 CBE)")
    print("=" * 70)

    print(f"\nNumber of samples: {N_SAMPLES}")
    print(f"Number of redshift bins: {N_Z_BINS}")
    print(f"Redshift range: z = {SPHEREX_Z_BINS[0][0]:.1f} - {SPHEREX_Z_BINS[-1][1]:.1f}")

    print("\nGalaxy bias and number density per sample:")
    print(f"{'Sample':<8} {'σ_z/(1+z)':<12} {'Total Galaxies':<18} {'Avg Bias':<10}")
    print("-" * 70)

    for sample in range(1, N_SAMPLES + 1):
        sigma_z = get_photo_z_error(sample)
        biases = SPHEREX_BIAS[sample]
        densities = SPHEREX_NUMBER_DENSITY[sample]

        # Estimate total galaxies (very approximate)
        # This is a rough estimate, not exact
        total_gal_approx = sum(densities) * 1e6  # arbitrary normalization for display

        avg_bias = np.mean(biases)

        print(f"{sample:<8} {sigma_z:<12.3f} {total_gal_approx:>17.1e} {avg_bias:>9.2f}")

    print("\nRedshift bin details:")
    print(f"{'Bin':<5} {'z_range':<15} {'Sample 1 bias':<15} {'Sample 1 n(z)':<15}")
    print("-" * 70)

    for i, (z_min, z_max) in enumerate(SPHEREX_Z_BINS):
        b1 = get_bias(1, i)
        n_gal = get_number_density(1, i)
        print(f"{i:<5} [{z_min:.1f}, {z_max:.1f}]     {b1:<15.2f} {n_gal:<15.2e}")

    print("\n" + "=" * 70)
