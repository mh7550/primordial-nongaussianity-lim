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

    print("\n" + "=" * 70)
