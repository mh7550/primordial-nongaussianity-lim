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
    time in the linear regime. It is computed by solving the growth equation
    for a flat ΛCDM universe.

    Parameters
    ----------
    z : float or array_like
        Redshift

    Returns
    -------
    D : float or array_like
        Linear growth factor D(z), normalized to D(z=0) = 1

    Notes
    -----
    The growth factor is computed using the integral solution for a flat
    ΛCDM cosmology:

    D(z) ∝ H(z) ∫_z^∞ (1+z')/(H(z'))^3 dz'

    where H(z) is the Hubble parameter.

    References
    ----------
    Carroll, Press, & Turner, ARA&A 30, 499 (1992)
    """
    z = np.asarray(z)
    scalar_input = False
    if z.ndim == 0:
        z = z[None]
        scalar_input = True

    def integrand(zp):
        """Integrand for growth factor calculation."""
        a = 1.0 / (1.0 + zp)
        Ez = COSMO.H(zp).value / H0
        return (1.0 + zp) / (a * Ez)**3

    # Compute growth factor for each redshift
    D = np.zeros_like(z, dtype=float)
    for i, zi in enumerate(z):
        # Integrate from zi to large redshift (z=1000 is effectively infinity)
        integral, _ = integrate.quad(integrand, zi, 1000.0)
        Ez = COSMO.H(zi).value / H0
        D[i] = Ez * integral

    # Normalize to D(z=0) = 1
    integral_0, _ = integrate.quad(integrand, 0.0, 1000.0)
    D0 = H0 / H0 * integral_0  # H(z=0) = H0
    D = D / D0

    if scalar_input:
        return D[0]
    return D


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
        Matter power spectrum in units of (Mpc/h)^3

    Notes
    -----
    The linear matter power spectrum is:

    P_matter(k, z) = (2π²/k³) * A_s * (k/k_pivot)^(n_s-1) * T²(k) * D²(z)

    where:
    - A_s = 2.1e-9 is the primordial power spectrum amplitude (Planck 2018)
    - n_s = 0.9649 is the spectral index
    - T(k) is the matter transfer function
    - D(z) is the linear growth factor, normalized to D(z=0) = 1
    - k_pivot = 0.05/h Mpc^-1 is the pivot scale

    For biased tracers with PNG:
        from bias_functions import get_total_bias
        b_total = get_total_bias(k, z, fNL, b1, shape='local')
        P_obs = b_total**2 * get_power_spectrum(k, z)

    References
    ----------
    Eisenstein & Hu, ApJ 496, 605 (1998) - Transfer function
    Carroll, Press, & Turner, ARA&A 30, 499 (1992) - Growth factor
    """
    k = np.asarray(k)

    # Pivot scale in Mpc^-1 (not h/Mpc)
    k_pivot = 0.05 / h  # Convert to h/Mpc

    # Primordial power spectrum: P_R(k) = As * (k/k_pivot)^(ns-1)
    P_primordial = As * (k / k_pivot)**(ns - 1.0)

    # Transfer function
    T = get_transfer_function(k)

    # Growth factor
    D = get_growth_factor(z)

    # Linear matter power spectrum (Gaussian)
    # P(k) = (2π²/k³) * P_R(k) * T²(k) * D²(z)
    P_linear = (2.0 * np.pi**2 / k**3) * P_primordial * T**2 * D**2

    # Normalize to sigma8 at z=0 if needed
    # For now, we use the direct calculation with Planck18 As

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
