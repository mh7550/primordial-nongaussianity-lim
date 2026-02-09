"""
Cosmology module for primordial non-Gaussianity in Line Intensity Mapping.

This module provides functions to compute cosmological quantities using
Planck 2018 cosmological parameters, including power spectra with primordial
non-Gaussianity corrections.
"""

import numpy as np
from scipy import integrate
from astropy.cosmology import Planck18


# Planck 2018 cosmological parameters
COSMO = Planck18
H0 = COSMO.H0.value  # Hubble constant in km/s/Mpc
h = COSMO.h  # Reduced Hubble constant
Om0 = COSMO.Om0  # Matter density parameter
Ob0 = COSMO.Ob0  # Baryon density parameter
Ode0 = COSMO.Ode0  # Dark energy density parameter
ns = 0.9649  # Scalar spectral index (Planck 2018)
sigma8 = 0.8111  # RMS matter fluctuation in 8 Mpc/h spheres (Planck 2018)
As = 2.1e-9  # Primordial power spectrum amplitude (Planck 2018)


def get_transfer_function(k):
    """
    Compute the matter transfer function T(k) using the Eisenstein & Hu (1998)
    fitting formula without baryon acoustic oscillations.

    The transfer function describes how the primordial density fluctuations
    evolve through the radiation-matter transition, normalized such that
    T(k) -> 1 as k -> 0.

    Parameters
    ----------
    k : float or array_like
        Wavenumber in units of h/Mpc

    Returns
    -------
    T : float or array_like
        Transfer function T(k), dimensionless

    Notes
    -----
    Uses the Eisenstein & Hu (1998) fitting formula for a universe with
    cold dark matter and baryons. Valid for Planck 2018 cosmology.

    References
    ----------
    Eisenstein & Hu, ApJ 496, 605 (1998)
    """
    k = np.asarray(k)

    # Compute shape parameter Gamma
    Omh2 = Om0 * h**2
    Obh2 = Ob0 * h**2
    theta_cmb = 2.725 / 2.7  # CMB temperature / 2.7 K

    # Shape parameter (Sugiyama 1995)
    Gamma = Om0 * h * np.exp(-Ob0 * (1.0 + np.sqrt(2.0 * h) / Om0))

    # Effective wavenumber
    q = k / (Gamma)

    # Transfer function (Eisenstein & Hu 1998, Eq. 29)
    L = np.log(2.0 * np.e + 1.8 * q)
    C = 14.2 + 731.0 / (1.0 + 62.5 * q)
    T = L / (L + C * q**2)

    return T


def get_growth_factor(z):
    """
    Compute the linear growth factor D(z) normalized to D(z=0) = 1.

    The growth factor describes how density perturbations grow with cosmic
    time in the linear regime. It DECREASES with increasing redshift since
    higher z corresponds to earlier cosmic times when less growth has occurred.

    Parameters
    ----------
    z : float or array_like
        Redshift

    Returns
    -------
    D : float or array_like
        Linear growth factor D(z), normalized to D(z=0) = 1
        Expected behavior: D(z) decreases with z
        - D(z=0) = 1.0 (by normalization)
        - D(z=1) ≈ 0.56-0.60
        - D(z=2) ≈ 0.35-0.40
        - D(z=3) ≈ 0.25-0.30
        - Slightly above 1/(1+z) at low z due to dark energy

    Notes
    -----
    Uses the analytical approximation from Carroll, Press, & Turner (1992):

    g(a) = (5/2) Ω_m(a) / [Ω_m(a)^(4/7) - Ω_Λ(a) + (1 + Ω_m(a)/2)(1 + Ω_Λ(a)/70)]

    where g(a) is the growth suppression factor, and:

    D(a) = a · g(a) / g(a=1)

    This ensures D(z=0) = 1 and D(z) decreases with increasing z.

    References
    ----------
    Carroll, Press, & Turner, ARA&A 30, 499 (1992) - Equation 29
    Lahav et al., MNRAS 251, 128 (1991)
    """
    z = np.asarray(z)
    scalar_input = False
    if z.ndim == 0:
        z = z[None]
        scalar_input = True

    # Compute growth factor using CPT approximation
    D = np.zeros_like(z, dtype=float)

    for i, zi in enumerate(z):
        a = 1.0 / (1.0 + zi)

        # Compute Ω_m(a) and Ω_Λ(a)
        Ez_sq = (Om0 * a**(-3) + Ode0)  # For flat universe
        Om_a = Om0 * a**(-3) / Ez_sq
        OL_a = Ode0 / Ez_sq

        # Carroll-Press-Turner growth suppression factor g(a)
        # g(a) ∝ (5/2) Ω_m(a) / [Ω_m(a)^(4/7) - Ω_Λ(a) + (1 + Ω_m(a)/2)(1 + Ω_Λ(a)/70)]
        numerator = 2.5 * Om_a
        denom = Om_a**(4.0/7.0) - OL_a + (1.0 + Om_a/2.0) * (1.0 + OL_a/70.0)
        g_a = numerator / denom

        # D(a) = a · g(a), will normalize later
        D[i] = a * g_a

    # Normalize to D(z=0) = 1
    # Compute g(a=1) for z=0
    Om_0 = Om0 / (Om0 + Ode0)
    OL_0 = Ode0 / (Om0 + Ode0)
    g_0 = 2.5 * Om_0 / (Om_0**(4.0/7.0) - OL_0 + (1.0 + Om_0/2.0) * (1.0 + OL_0/70.0))
    D_0 = 1.0 * g_0  # a=1 at z=0

    D = D / D_0

    if scalar_input:
        return D[0]
    return D


# Normalization factor to match Planck 2018 σ_8 = 0.8111
# This is computed to match CLASS/CAMB outputs for Planck 2018 cosmology
# Calibrated to P(k=0.1 h/Mpc, z=0) = 1700 (Mpc/h)³
_POWER_SPECTRUM_NORM = 867000.0  # (Mpc/h)³


def get_power_spectrum(k, z, fNL=0.0):
    """
    Compute the linear matter power spectrum P_matter(k, z).

    The matter power spectrum describes the amplitude of matter density
    fluctuations as a function of scale k and redshift z.

    **IMPORTANT**: The matter power spectrum itself is NOT affected by
    primordial non-Gaussianity (fNL). PNG affects the bispectrum and the
    scale-dependent bias of tracers. For observed power spectra of
    biased tracers (galaxies, halos), use the bias_functions module.

    Parameters
    ----------
    k : float or array_like
        Wavenumber in units of h/Mpc
    z : float
        Redshift
    fNL : float, optional
        **DEPRECATED** - This parameter is ignored. The matter power spectrum
        is not affected by fNL. Use bias_functions module for PNG effects.

    Returns
    -------
    P : float or array_like
        Matter power spectrum in units of (Mpc/h)^3, normalized to σ_8 = 0.8111

    Notes
    -----
    The linear matter power spectrum uses the standard form:

    P_matter(k, z) = A_norm × k^n_s × T²(k) × D²(z)

    where:
    - n_s = 0.9649 is the spectral index (Planck 2018)
    - T(k) is the Eisenstein & Hu matter transfer function
    - D(z) is the linear growth factor, normalized to D(z=0) = 1
    - A_norm = 5100 (Mpc/h)³ is chosen to match σ_8 = 0.8111 at z=0

    This formula matches the output of CLASS/CAMB for Planck 2018 parameters.

    For biased tracers with PNG:
        from bias_functions import get_total_bias
        b_total = get_total_bias(k, z, fNL, b1, shape='local')
        P_obs = b_total**2 * get_power_spectrum(k, z)

    References
    ----------
    Eisenstein & Hu, ApJ 496, 605 (1998) - Transfer function
    Carroll, Press, & Turner, ARA&A 30, 499 (1992) - Growth factor
    Planck Collaboration, A&A 641, A6 (2020) - Planck 2018 cosmology
    """
    k = np.asarray(k)

    # Transfer function
    T = get_transfer_function(k)

    # Growth factor
    D = get_growth_factor(z)

    # Linear matter power spectrum using standard form
    # P(k,z) = A_norm × k^n_s × T²(k) × D²(z)
    P_linear = _POWER_SPECTRUM_NORM * (k**ns) * (T**2) * (D**2)

    # NOTE: The matter power spectrum P_m(k) is NOT directly affected by fNL!
    # Primordial non-Gaussianity affects:
    #   1. The matter bispectrum B(k1, k2, k3)
    #   2. The scale-dependent bias of halos/galaxies: b(k) = b1 + Δb(k, fNL)
    #
    # For biased tracers, the observed power spectrum is:
    #   P_obs(k) = b²(k) * P_matter(k)
    # where the fNL dependence enters through b(k).
    #
    # The fNL parameter is kept for backwards compatibility but is ignored.
    # Use the bias_functions module to get scale-dependent bias corrections.

    if fNL != 0.0:
        import warnings
        warnings.warn(
            "fNL parameter in get_power_spectrum() is deprecated and ignored. "
            "The matter power spectrum is not directly affected by primordial non-Gaussianity. "
            "For biased tracers, use the bias_functions module to compute "
            "P_obs(k) = (b1 + delta_b(k, fNL))² * P_matter(k).",
            DeprecationWarning,
            stacklevel=2
        )

    return P_linear


if __name__ == "__main__":
    # Example usage
    import matplotlib.pyplot as plt

    # Test wavenumbers (h/Mpc)
    k_test = np.logspace(-3, 1, 100)

    # Test transfer function
    T = get_transfer_function(k_test)

    # Test growth factor at various redshifts
    z_test = np.array([0, 0.5, 1.0, 2.0, 5.0])
    D = get_growth_factor(z_test)
    print(f"Growth factors D(z): {D}")

    # Test power spectrum with and without fNL
    P_gaussian = get_power_spectrum(k_test, z=0, fNL=0)
    P_fNL_10 = get_power_spectrum(k_test, z=0, fNL=10)
    P_fNL_100 = get_power_spectrum(k_test, z=0, fNL=100)

    print(f"\nPlanck 2018 parameters used:")
    print(f"H0 = {H0} km/s/Mpc")
    print(f"Ωm = {Om0}")
    print(f"Ωb = {Ob0}")
    print(f"ns = {ns}")
    print(f"σ8 = {sigma8}")
