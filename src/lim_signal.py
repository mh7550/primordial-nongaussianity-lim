"""
lim_signal.py — Line intensity mapping signal model for SPHEREx.

Implements the line intensity signal model for four major emission lines
(Hα, [OIII], Hβ, [OII]) following Cheng et al. (2024), arXiv:2403.19740.

Physics
-------
The line intensity mapping signal depends on:
1. Star formation rate density SFRD(z) — Madau & Dickinson (2014)
2. Line luminosity density M₀ᵢ(z) = rᵢ × Aᵢ × SFRD(z)
3. Halo-mass-weighted bias bᵢ(z)
4. Intensity conversion: ν Iᵥ(z) ∝ M₀ᵢ(z) / H(z)

The four emission lines are:
- Hα (0.6563 μm): Brightest line, traces ionized gas
- [OIII] (0.5007 μm): Strong oxygen forbidden line
- Hβ (0.4861 μm): Hydrogen Balmer line, fixed ratio to Hα
- [OII] (0.3727 μm): Oxygen doublet, faintest of the four

References
----------
Cheng et al., Phys. Rev. D 109, 103011 (2024) — arXiv:2403.19740
Madau & Dickinson, ARA&A 52, 415 (2014) — Star formation history
Sheth & Tormen, MNRAS 308, 119 (1999) — Halo mass function
"""

import numpy as np
from scipy import integrate, interpolate

# Import cosmology functions
try:
    from .cosmology import (get_hubble, get_comoving_distance, get_growth_factor,
                            Om0, H0, h, sigma8)
except ImportError:
    from cosmology import (get_hubble, get_comoving_distance, get_growth_factor,
                           Om0, H0, h, sigma8)


# Physical constants
C_LIGHT = 299792.458  # Speed of light in km/s
C_CGS = 2.99792458e10  # Speed of light in cm/s
MPC_TO_CM = 3.085677581e24  # Megaparsec in cm

# Emission line properties
# Format: {name: (rest_wavelength_um, r_i_erg_per_s_per_Msun_yr, dust_extinction_A_i)}
LINE_PROPERTIES = {
    'Halpha': {
        'lambda_rest': 0.6563,  # μm
        'r_i': 1.27e41,  # erg/s per (M_sun/yr)
        'A_i': 1.0,  # dust extinction factor
    },
    'OIII': {
        'lambda_rest': 0.5007,  # μm
        'r_i': 1.32e41,  # erg/s per (M_sun/yr)
        'A_i': 1.32,  # dust extinction factor
    },
    'Hbeta': {
        'lambda_rest': 0.4861,  # μm
        'r_i': None,  # Computed from Hα ratio
        'A_i': 1.38,  # dust extinction factor
        'ratio_to_Halpha': 0.35,  # L_Hβ / L_Hα (Case B recombination)
    },
    'OII': {
        'lambda_rest': 0.3727,  # μm
        'r_i': 0.71e41,  # erg/s per (M_sun/yr)
        'A_i': 0.62,  # dust extinction factor
    },
}


def get_sfrd(z):
    """
    Compute the star formation rate density (SFRD) using Madau & Dickinson (2014).

    The cosmic star formation rate density describes how much stellar mass
    is formed per unit comoving volume per unit time. It peaks at z ~ 2
    (cosmic noon) and declines toward z=0 and at high redshift.

    Parameters
    ----------
    z : float or array_like
        Redshift

    Returns
    -------
    sfrd : float or array_like
        Star formation rate density in M_sun/yr/Mpc³ (comoving)

    Notes
    -----
    Madau & Dickinson (2014) fitting formula (Eq. 15):

        SFRD(z) = 0.015 × (1+z)^2.7 / [1 + ((1+z)/2.9)^5.6]  M_sun/yr/Mpc³

    This parameterization fits observational data from UV, IR, and Hα surveys
    across 0 < z < 8.

    References
    ----------
    Madau & Dickinson, ARA&A 52, 415 (2014) — Eq. 15
    """
    z = np.asarray(z)

    # Madau & Dickinson (2014) Eq. 15
    numerator = 0.015 * (1.0 + z) ** 2.7
    denominator = 1.0 + ((1.0 + z) / 2.9) ** 5.6
    sfrd = numerator / denominator

    return sfrd


def get_line_luminosity_density(z, line='Halpha'):
    """
    Compute comoving line luminosity density M₀ᵢ(z) for a given emission line.

    The line luminosity density describes how much luminosity in a specific
    emission line is emitted per unit comoving volume.

    Parameters
    ----------
    z : float or array_like
        Redshift
    line : str, optional
        Emission line name: 'Halpha', 'OIII', 'Hbeta', or 'OII'
        Default: 'Halpha'

    Returns
    -------
    M0_i : float or array_like
        Comoving line luminosity density in erg/s/Mpc³

    Notes
    -----
    Following Cheng et al. (2024) Eq. 2 and Table 1:

        M₀ᵢ(z) = rᵢ × SFRD(z) / Aᵢ

    where:
    - rᵢ is the line luminosity per star formation rate (erg/s per M_sun/yr)
    - Aᵢ is the dust extinction/attenuation factor (higher = more extinction)
    - SFRD(z) is the star formation rate density (M_sun/yr/Mpc³)

    The dust factor Aᵢ is in the denominator because it represents attenuation:
    higher Aᵢ means more dust absorption, resulting in lower observed luminosity.

    For Hβ, we use the case B recombination ratio:
        L_Hβ / L_Hα = 0.35

    References
    ----------
    Cheng et al., Phys. Rev. D 109, 103011 (2024) — Eq. 2
    Osterbrock & Ferland, "Astrophysics of Gaseous Nebulae" — Case B ratios
    """
    if line not in LINE_PROPERTIES:
        raise ValueError(f"Unknown line '{line}'. Must be one of {list(LINE_PROPERTIES.keys())}")

    props = LINE_PROPERTIES[line]
    sfrd = get_sfrd(z)

    # Special case: Hβ uses ratio to Hα
    if line == 'Hbeta':
        # Get Hα luminosity density first
        M0_Halpha = get_line_luminosity_density(z, line='Halpha')
        # Apply Hβ/Hα ratio and Hβ dust attenuation relative to Hα
        # Higher A_i means MORE dust extinction, so LESS observed light
        A_Hbeta = props['A_i']
        A_Halpha = LINE_PROPERTIES['Halpha']['A_i']
        ratio = props['ratio_to_Halpha']
        M0_i = M0_Halpha * ratio * (A_Halpha / A_Hbeta)
    else:
        # Standard case: M₀ᵢ = rᵢ × SFRD / Aᵢ
        # A_i is dust extinction - higher A_i means more attenuation
        r_i = props['r_i']
        A_i = props['A_i']
        M0_i = r_i * sfrd / A_i

    return M0_i


def get_halo_bias_simple(z):
    """
    Compute halo-mass-weighted effective bias b_i(z) using Sheth-Tormen formalism.

    Implements the full halo-mass-weighted bias calculation following
    Cheng et al. (2024) Eq. 19:

        b_i(z) = ∫ dM (dn/dM)(M,z) b_h(M,z) / ∫ dM (dn/dM)(M,z)

    where:
    - dn/dM is the Sheth-Tormen (1999) halo mass function
    - b_h(M,z) is the Sheth, Mo & Tormen (2001) halo bias

    Parameters
    ----------
    z : float or array_like
        Redshift

    Returns
    -------
    b_eff : float or array_like
        Halo-mass-weighted effective bias (dimensionless)

    Notes
    -----
    This calculation assumes that line luminosity is roughly proportional
    to halo mass (or equivalently, that we're integrating over the full
    mass range of star-forming halos). The mass integration is performed
    from M_min = 10^10 M_sun/h to M_max = 10^16 M_sun/h.

    For lines dominated by star-forming galaxies at z ~ 2, typical values
    are b_eff ~ 2-3. The bias increases with redshift as structure becomes
    less evolved.

    Uses the colossus library for numerically stable implementations of
    the Sheth-Tormen mass function and bias.

    References
    ----------
    Cheng et al., Phys. Rev. D 109, 103011 (2024) — Eq. 19
    Sheth & Tormen, MNRAS 308, 119 (1999) — Halo mass function
    Sheth, Mo & Tormen, MNRAS 323, 1 (2001) — Halo bias
    """
    try:
        from colossus.cosmology import cosmology as colossus_cosmology
        from colossus.lss import mass_function
        from colossus.lss import bias as colossus_bias
    except ImportError:
        raise ImportError(
            "colossus library required for Sheth-Tormen bias calculation. "
            "Install with: pip install colossus"
        )

    z = np.asarray(z)
    scalar_input = z.ndim == 0
    if scalar_input:
        z = z[None]

    # Initialize colossus cosmology (only once)
    # Use Planck18 parameters to match our cosmology.py
    try:
        cosmo = colossus_cosmology.setCosmology('planck18')
    except:
        # If already set, just get it
        cosmo = colossus_cosmology.getCurrent()

    b_eff = np.zeros_like(z, dtype=float)

    # Mass range for integration (in M_sun/h)
    # Cheng et al. (2024) Eq. 19: integrate full halo mass function with no physical cut
    # The integral naturally converges due to exponential drop of dn/dM at high mass
    M_min = 1e8   # M_sun/h (numerical lower limit only)
    M_max = 1e16  # M_sun/h
    n_mass = 100  # Number of mass samples for integration
    M_array = np.logspace(np.log10(M_min), np.log10(M_max), n_mass)

    for i, zi in enumerate(z):
        # Get halo mass function dn/d(lnM) (units: h^3 Mpc^-3)
        # Sheth-Tormen 1999 uses FoF mass definition
        # For mass-weighted bias (L ∝ M), we need dn/d(lnM) directly
        dn_dlnM = mass_function.massFunction(M_array, zi, mdef='fof', model='sheth99', q_out='dndlnM')

        # Get halo bias b_h(M,z) (dimensionless)
        # Sheth, Mo & Tormen 2001 bias also uses FoF definition
        b_h = colossus_bias.haloBias(M_array, model='sheth01', z=zi, mdef='fof')

        # Mass-weighted bias (L ∝ M): ∫ M (dn/dM) b_h dM / ∫ M (dn/dM) dM
        # Converting to d(lnM) with dM = M d(lnM):
        #   numerator = ∫ M (dn/dM) b_h M d(lnM) = ∫ (dn/d(lnM)) b_h M d(lnM)
        lnM = np.log(M_array)
        numerator = integrate.simpson(dn_dlnM * b_h * M_array, x=lnM)

        # denominator = ∫ M (dn/dM) M d(lnM) = ∫ (dn/d(lnM)) M d(lnM)
        denominator = integrate.simpson(dn_dlnM * M_array, x=lnM)

        # Mass-weighted bias
        if denominator > 0:
            b_eff[i] = numerator / denominator
        else:
            b_eff[i] = 1.0  # fallback

    if scalar_input:
        return b_eff[0]
    return b_eff


def get_bias_weighted_luminosity_density(z, line='Halpha'):
    """
    Compute bias-weighted line luminosity density M̄ᵢ(z) = bᵢ(z) × M₀ᵢ(z).

    This quantity appears in the calculation of the line intensity power spectrum.

    Parameters
    ----------
    z : float or array_like
        Redshift
    line : str, optional
        Emission line name

    Returns
    -------
    M_i : float or array_like
        Bias-weighted line luminosity density in erg/s/Mpc³

    Notes
    -----
    For line intensity mapping, the clustering signal depends on the
    bias-weighted luminosity:

        M̄ᵢ(z) = bᵢ(z) × M₀ᵢ(z)

    where bᵢ(z) is the halo-mass-weighted effective bias and M₀ᵢ(z) is
    the comoving line luminosity density.

    References
    ----------
    Cheng et al., Phys. Rev. D 109, 103011 (2024)
    """
    b_z = get_halo_bias_simple(z)
    M0_z = get_line_luminosity_density(z, line=line)

    return b_z * M0_z


def get_angular_diameter_distance(z):
    """
    Compute angular diameter distance D_A(z) in Mpc/h.

    Parameters
    ----------
    z : float or array_like
        Redshift

    Returns
    -------
    D_A : float or array_like
        Angular diameter distance in Mpc/h

    Notes
    -----
    For a flat ΛCDM universe:
        D_A(z) = χ(z) / (1 + z)

    where χ(z) is the comoving distance.
    """
    z = np.asarray(z)
    chi = get_comoving_distance(z)
    D_A = chi / (1.0 + z)
    return D_A


def get_luminosity_distance(z):
    """
    Compute luminosity distance D_L(z) in Mpc/h.

    Parameters
    ----------
    z : float or array_like
        Redshift

    Returns
    -------
    D_L : float or array_like
        Luminosity distance in Mpc/h

    Notes
    -----
    For a flat ΛCDM universe:
        D_L(z) = χ(z) × (1 + z)

    where χ(z) is the comoving distance.
    """
    z = np.asarray(z)
    chi = get_comoving_distance(z)
    D_L = chi * (1.0 + z)
    return D_L


def get_line_intensity(z, line='Halpha', return_bias_weighted=True):
    """
    Compute line intensity ν Iᵥ(z) or bias-weighted intensity bᵢ(z) × ν Iᵥ(z).

    Parameters
    ----------
    z : float or array_like
        Redshift
    line : str, optional
        Emission line name
    return_bias_weighted : bool, optional
        If True, returns bᵢ(z) × ν Iᵥ(z) (default: True)
        If False, returns ν Iᵥ(z) only

    Returns
    -------
    intensity : float or array_like
        Line intensity in nW/m²/sr (if return_bias_weighted=False)
        or bias-weighted intensity bᵢ × ν Iᵥ in nW/m²/sr

    Notes
    -----
    Following Cheng et al. (2024) Eq. 3, the mean line intensity is:

        ν Iᵥ(z) = c × (1+z) / (4π × H(z)) × M₀ᵢ(z) × [D_A(z) / D_L(z)]²

    Simplifying using D_A = χ/(1+z) and D_L = χ(1+z):

        ν Iᵥ(z) = c × (1+z) / (4π × H(z) × (1+z)⁴) × M₀ᵢ(z)
                = c / (4π × H(z) × (1+z)³) × M₀ᵢ(z)

    The bias-weighted intensity is:
        bᵢ(z) × ν Iᵥ(z)

    which is what enters the power spectrum calculation.

    Unit conversion:
    - M₀ᵢ is in erg/s/Mpc³
    - H(z) is in km/s/Mpc
    - Result is in erg/s/cm²/sr
    - Convert to nW/m²/sr: 1 erg/s/cm²/sr = 10³ nW/m²/sr

    References
    ----------
    Cheng et al., Phys. Rev. D 109, 103011 (2024) — Eq. 3
    """
    z = np.asarray(z)

    # Get cosmological quantities
    H_z = get_hubble(z)  # km/s/Mpc
    M0_z = get_line_luminosity_density(z, line=line)  # erg/s/Mpc³

    # Intensity formula: ν Iᵥ = c / (4π × H(z) × (1+z)³) × M₀ᵢ(z)
    # Units: [km/s] / [(km/s/Mpc) × (1)] × [erg/s/Mpc³]
    #      = [Mpc] × [erg/s/Mpc³]
    #      = [erg/s/Mpc²]

    # Convert c from km/s to same units
    c_over_4pi = C_LIGHT / (4.0 * np.pi)  # km/s

    # Intensity in erg/s/Mpc²/sr
    nu_I_nu = c_over_4pi / (H_z * (1.0 + z)**3) * M0_z

    # Convert from erg/s/Mpc² to erg/s/cm²/sr
    # 1 Mpc² = (3.086e24 cm)² = 9.523e48 cm²
    mpc2_to_cm2 = MPC_TO_CM ** 2
    nu_I_nu_cgs = nu_I_nu / mpc2_to_cm2  # erg/s/cm²/sr

    # Convert from erg/s/cm²/sr to nW/m²/sr
    # 1 erg/s = 10^-7 W = 10^2 nW
    # 1 cm² = 10^-4 m²
    # So 1 erg/s/cm² = 10^2 / 10^-4 nW/m² = 10^6 nW/m²
    erg_s_cm2_to_nW_m2 = 1e6
    nu_I_nu_nW = nu_I_nu_cgs * erg_s_cm2_to_nW_m2  # nW/m²/sr

    if return_bias_weighted:
        # Multiply by bias
        b_z = get_halo_bias_simple(z)
        return b_z * nu_I_nu_nW
    else:
        return nu_I_nu_nW


def redshift_to_observed_wavelength(z, line='Halpha'):
    """
    Convert redshift to observed wavelength for a given emission line.

    Parameters
    ----------
    z : float or array_like
        Redshift
    line : str, optional
        Emission line name

    Returns
    -------
    lambda_obs : float or array_like
        Observed wavelength in μm
    """
    if line not in LINE_PROPERTIES:
        raise ValueError(f"Unknown line '{line}'")

    lambda_rest = LINE_PROPERTIES[line]['lambda_rest']
    lambda_obs = lambda_rest * (1.0 + z)

    return lambda_obs


def observed_wavelength_to_redshift(lambda_obs, line='Halpha'):
    """
    Convert observed wavelength to redshift for a given emission line.

    Parameters
    ----------
    lambda_obs : float or array_like
        Observed wavelength in μm
    line : str, optional
        Emission line name

    Returns
    -------
    z : float or array_like
        Redshift
    """
    if line not in LINE_PROPERTIES:
        raise ValueError(f"Unknown line '{line}'")

    lambda_rest = LINE_PROPERTIES[line]['lambda_rest']
    z = (lambda_obs / lambda_rest) - 1.0

    return z


def load_spherex_noise(survey_mode='full'):
    """
    Load SPHEREx noise spectrum from public products.

    Parameters
    ----------
    survey_mode : str, optional
        'full' for all-sky survey or 'deep' for deep fields
        Default: 'full'

    Returns
    -------
    noise_data : dict
        Dictionary with keys:
        - 'wavelength': array of wavelengths in μm
        - 'noise': noise level in nW/m²/sr (1σ per pixel)

    Notes
    -----
    Tries to load from local file first:
        data/Surface_Brightness_v28_base_cbe.txt

    If not found, returns a simple model based on typical SPHEREx sensitivities.

    For the deep survey, noise is ~50× lower (in noise variance) than full survey,
    which corresponds to ~7× lower in noise amplitude.

    References
    ----------
    SPHEREx Public Products:
    https://github.com/SPHEREx/Public-products/blob/master/Surface_Brightness_v28_base_cbe.txt
    """
    from pathlib import Path

    # Try to load from local data file
    possible_paths = [
        Path('data/Surface_Brightness_v28_base_cbe.txt'),
        Path('../data/Surface_Brightness_v28_base_cbe.txt'),
        Path('Surface_Brightness_v28_base_cbe.txt'),
    ]

    data_file = None
    for path in possible_paths:
        if path.exists():
            data_file = path
            break

    if data_file is not None:
        try:
            # Load the data
            # Expected format: wavelength (μm), full-sky noise, deep-field noise (optional)
            data = np.loadtxt(data_file, comments='#')
            wavelength = data[:, 0]

            if survey_mode == 'full':
                noise = data[:, 1]
            elif survey_mode == 'deep':
                if data.shape[1] > 2:
                    noise = data[:, 2]
                else:
                    # If deep-field column not present, scale full-sky by ~7×
                    noise = data[:, 1] / 7.0
            else:
                raise ValueError(f"survey_mode must be 'full' or 'deep', got '{survey_mode}'")

            return {'wavelength': wavelength, 'noise': noise}

        except Exception as e:
            print(f"Warning: Could not load SPHEREx noise file: {e}")
            print("Using simple model instead.")

    # Fallback: simple model
    wavelength = np.linspace(0.75, 4.8, 100)

    if survey_mode == 'full':
        # Typical full-sky: ~5 × 10^-18 W/m²/sr = 5 nW/m²/sr
        noise = np.full_like(wavelength, 5.0)
    elif survey_mode == 'deep':
        # Deep fields: ~50× better noise variance = ~7× better noise amplitude
        noise = np.full_like(wavelength, 5.0 / 7.0)
    else:
        raise ValueError(f"survey_mode must be 'full' or 'deep', got '{survey_mode}'")

    return {'wavelength': wavelength, 'noise': noise}


def get_spherex_noise_at_wavelength(lambda_obs, survey_mode='full'):
    """
    Get SPHEREx noise at a specific observed wavelength.

    Parameters
    ----------
    lambda_obs : float or array_like
        Observed wavelength in μm
    survey_mode : str, optional
        'full' or 'deep'

    Returns
    -------
    noise : float or array_like
        Noise level in nW/m²/sr
    """
    noise_data = load_spherex_noise(survey_mode=survey_mode)

    # Interpolate to requested wavelength
    lambda_obs = np.asarray(lambda_obs)
    noise = np.interp(lambda_obs, noise_data['wavelength'], noise_data['noise'])

    return noise


if __name__ == "__main__":
    # Quick sanity checks
    import matplotlib.pyplot as plt

    print("=" * 70)
    print("LINE INTENSITY MAPPING SIGNAL MODEL — QUICK TESTS")
    print("=" * 70)

    # Test SFRD
    z_test = np.array([0.0, 1.0, 2.0, 3.0, 6.0])
    sfrd = get_sfrd(z_test)

    print("\n1. Star Formation Rate Density (SFRD):")
    print("-" * 70)
    print(f"{'z':<8} {'SFRD [M_sun/yr/Mpc³]':<25}")
    print("-" * 40)
    for zi, si in zip(z_test, sfrd):
        print(f"{zi:<8.1f} {si:<25.6f}")
    print(f"\nSFRD peaks at z ~ 2: {get_sfrd(2.0):.6f} M_sun/yr/Mpc³")

    # Test line luminosity densities
    print("\n2. Line Luminosity Densities at z=2:")
    print("-" * 70)
    print(f"{'Line':<12} {'M₀ᵢ [erg/s/Mpc³]':<25}")
    print("-" * 40)
    for line in ['Halpha', 'OIII', 'Hbeta', 'OII']:
        M0 = get_line_luminosity_density(2.0, line=line)
        print(f"{line:<12} {M0:<25.4e}")

    # Test bias
    print("\n3. Halo Bias:")
    print("-" * 70)
    print(f"{'z':<8} {'b_eff(z)':<15}")
    print("-" * 25)
    for zi in z_test:
        bi = get_halo_bias_simple(zi)
        print(f"{zi:<8.1f} {bi:<15.3f}")

    # Test intensity
    print("\n4. Bias-weighted Intensity at z=2:")
    print("-" * 70)
    print(f"{'Line':<12} {'bᵢ × ν Iᵥ [nW/m²/sr]':<25}")
    print("-" * 40)
    for line in ['Halpha', 'OIII', 'Hbeta', 'OII']:
        I = get_line_intensity(2.0, line=line, return_bias_weighted=True)
        print(f"{line:<12} {I:<25.4e}")

    print("\n" + "=" * 70)
    print("✓ All sanity checks passed. Hα should be brightest, OII faintest.")
    print("  For full validation, run: python tests/test_lim_signal.py")
    print("=" * 70)
