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
    Compute the matter power spectrum P(k, z) with primordial non-Gaussianity.

    The power spectrum describes the amplitude of density fluctuations as a
    function of scale k and redshift z. This function includes corrections
    from local-type primordial non-Gaussianity parametrized by fNL.

    Parameters
    ----------
    k : float or array_like
        Wavenumber in units of h/Mpc
    z : float
        Redshift
    fNL : float, optional
        Local-type primordial non-Gaussianity parameter (default: 0.0)
        fNL = 0 corresponds to Gaussian initial conditions

    Returns
    -------
    P : float or array_like
        Matter power spectrum in units of (Mpc/h)^3

    Notes
    -----
    The power spectrum with primordial non-Gaussianity is given by:

    P(k, z) = P_Gaussian(k, z) * [1 + ΔP_fNL(k, z)]

    where P_Gaussian(k, z) = (2π²/k³) * As * (k/k_pivot)^(ns-1) * T²(k) * D²(z)

    and the fNL correction is:

    ΔP_fNL(k, z) = 2 * fNL * δc * M(k, z) * T(k) * D(z) / (k² * T(k_pivot) * D(z=0))

    where δc = 1.686 is the critical density for collapse, M(k,z) is the
    scale-dependent bias correction, and k_pivot = 0.05 Mpc^-1.

    References
    ----------
    Dalal et al., PRD 77, 123514 (2008)
    Desjacques et al., Phys. Rept. 733, 1 (2018)
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

    if fNL == 0.0:
        return P_linear

    # Apply fNL correction for primordial non-Gaussianity
    # The correction is scale-dependent and becomes important at large scales
    # ΔP/P ~ 2 * fNL * δc * (Ωm * H0²) / (k² * T(k) * c²)

    # Critical density for spherical collapse
    delta_c = 1.686

    # Scale-dependent correction (simplified form)
    # Full form requires transfer function at present day
    c_km_s = 299792.458  # Speed of light in km/s

    # Scale-dependent bias correction from fNL
    # This is a simplified version; full implementation would include
    # the exact Poisson equation solution
    T_pivot = get_transfer_function(k_pivot)
    D_0 = 1.0  # D(z=0) = 1 by normalization

    # fNL correction factor (scale-dependent)
    # Δ(k,z) = δc * (Ωm * H0² / k² / c²) * (T(k) / T_pivot) * (D(z) / D_0)
    prefactor = 3.0 * Om0 * H0**2 / c_km_s**2  # in (Mpc/h)^-2
    Delta_fNL = delta_c * prefactor / k**2 * (T / T_pivot) * (D / D_0)

    # Power spectrum with fNL correction
    # P(k,z) = P_Gaussian(k,z) * [1 + 2*fNL*Δ(k,z)]^2
    # For small fNL, this is approximately P_Gaussian * (1 + 4*fNL*Δ)
    correction_factor = (1.0 + 2.0 * fNL * Delta_fNL)**2

    P_fNL = P_linear * correction_factor

    return P_fNL


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
