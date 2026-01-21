"""
Cosmology module for primordial non-Gaussianity studies.

This module contains functions for computing cosmological quantities
relevant to studying primordial non-Gaussianity in large-scale structure.
"""

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d


# Cosmological parameters (Planck 2018)
OMEGA_M = 0.315  # Matter density parameter
OMEGA_B = 0.049  # Baryon density parameter
OMEGA_CDM = OMEGA_M - OMEGA_B  # Cold dark matter density parameter
OMEGA_LAMBDA = 1.0 - OMEGA_M  # Dark energy density parameter
H0 = 67.4  # Hubble constant in km/s/Mpc
h = H0 / 100.0  # Dimensionless Hubble parameter
N_S = 0.965  # Spectral index
SIGMA_8 = 0.811  # Amplitude of matter fluctuations
C_LIGHT = 299792.458  # Speed of light in km/s


def get_transfer_function(k):
    """
    Compute the matter transfer function using Eisenstein & Hu (1998) fitting formula.

    Parameters:
    -----------
    k : float or array-like
        Wavenumber in h/Mpc

    Returns:
    --------
    T : float or array
        Transfer function (dimensionless)
    """
    k = np.atleast_1d(k)

    # Eisenstein & Hu parameters
    theta_cmb = 2.728 / 2.7  # CMB temperature ratio
    omega_m = OMEGA_M * h**2
    omega_b = OMEGA_B * h**2

    # Shape parameter
    gamma = omega_m * h * np.exp(-OMEGA_B * (1.0 + np.sqrt(2.0 * h) / OMEGA_M))

    # Wavenumber in units of gamma
    q = k / (h * gamma)

    # CDM transfer function (Bardeen et al. 1986 approximation)
    L = np.log(2.0 * np.e + 1.8 * q)
    C = 14.2 + 731.0 / (1.0 + 62.5 * q)
    T = L / (L + C * q**2)

    return T if len(T) > 1 else T[0]


def get_growth_factor(z):
    """
    Compute the linear growth factor D(z) for a flat ΛCDM cosmology.
    Normalized to D(z=0) = 1.

    Parameters:
    -----------
    z : float or array-like
        Redshift

    Returns:
    --------
    D : float or array
        Linear growth factor (dimensionless)
    """
    z = np.atleast_1d(z)

    def integrand(a):
        """Integrand for growth factor calculation"""
        H_ratio = np.sqrt(OMEGA_M / a**3 + OMEGA_LAMBDA)
        return (a * H_ratio)**(-3)

    # Calculate growth factor for each redshift
    D = np.zeros_like(z, dtype=float)
    for i, zi in enumerate(z):
        a = 1.0 / (1.0 + zi)
        # Normalize at z=0
        integral_z, _ = quad(integrand, 0, a, limit=100)
        integral_0, _ = quad(integrand, 0, 1.0, limit=100)

        H_ratio = np.sqrt(OMEGA_M / a**3 + OMEGA_LAMBDA)
        H_ratio_0 = np.sqrt(OMEGA_M + OMEGA_LAMBDA)

        D[i] = (H_ratio / H_ratio_0) * (integral_z / integral_0)

    return D if len(D) > 1 else D[0]


def get_power_spectrum(k, z=0, fNL=0):
    """
    Compute the matter power spectrum including primordial non-Gaussianity effects.

    The power spectrum is modified by local-type primordial non-Gaussianity:
    P(k,z) = D²(z) * T²(k) * P_primordial(k) * [1 + ΔP_fNL(k,z)]

    where ΔP_fNL represents the scale-dependent bias from fNL.

    Parameters:
    -----------
    k : float or array-like
        Wavenumber in h/Mpc
    z : float, optional
        Redshift (default: 0)
    fNL : float, optional
        Local non-Gaussianity parameter (default: 0)

    Returns:
    --------
    P : float or array
        Matter power spectrum in (Mpc/h)³
    """
    k = np.atleast_1d(k)

    # Get transfer function and growth factor
    T_k = get_transfer_function(k)
    D_z = get_growth_factor(z)

    # Primordial power spectrum: P(k) = A_s * k^n_s
    # Normalize using sigma_8
    A_s = 2.0e-9  # Approximate amplitude
    k_pivot = 0.05  # Pivot scale in h/Mpc
    P_primordial = A_s * (k / k_pivot)**(N_S - 1.0)

    # Linear power spectrum
    P_linear = D_z**2 * T_k**2 * P_primordial * k**(N_S)

    # Scale-dependent bias from primordial non-Gaussianity
    # For local-type fNL, the correction is:
    # ΔP/P ≈ 2 * fNL * δ_c * Ω_m * H_0² * T(k) / (c² * k² * D(z))
    if fNL != 0:
        delta_c = 1.686  # Critical density for collapse
        a = 1.0 / (1.0 + z)

        # Scale-dependent bias correction
        # This is a simplified version; full calculation includes transfer function details
        k_h = k * h  # Convert to 1/Mpc
        fNL_correction = 2.0 * fNL * delta_c * OMEGA_M / (k_h**2 * D_z)

        # Apply correction (note: this is approximate for demonstration)
        P_fNL = P_linear * (1.0 + fNL_correction * T_k)
    else:
        P_fNL = P_linear

    # Normalize roughly to get realistic values
    normalization = 1e4
    P_fNL *= normalization

    return P_fNL if len(P_fNL) > 1 else P_fNL[0]


def get_hubble_parameter(z):
    """
    Compute the Hubble parameter H(z) in km/s/Mpc.

    Parameters:
    -----------
    z : float or array-like
        Redshift

    Returns:
    --------
    H : float or array
        Hubble parameter in km/s/Mpc
    """
    z = np.atleast_1d(z)
    H = H0 * np.sqrt(OMEGA_M * (1 + z)**3 + OMEGA_LAMBDA)
    return H if len(H) > 1 else H[0]


def get_comoving_distance(z):
    """
    Compute the comoving distance to redshift z in Mpc/h.

    Parameters:
    -----------
    z : float or array-like
        Redshift

    Returns:
    --------
    r : float or array
        Comoving distance in Mpc/h
    """
    z = np.atleast_1d(z)

    def integrand(zp):
        return C_LIGHT / get_hubble_parameter(zp)

    r = np.zeros_like(z, dtype=float)
    for i, zi in enumerate(z):
        if zi > 0:
            r[i], _ = quad(integrand, 0, zi, limit=100)
        else:
            r[i] = 0.0

    # Convert to Mpc/h
    r *= h

    return r if len(r) > 1 else r[0]
