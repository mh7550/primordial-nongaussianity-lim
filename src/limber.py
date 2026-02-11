"""
limber.py — Angular power spectra via the Limber approximation.

Computes the angular galaxy power spectrum C_ℓ (and cross-spectra C_ℓ^{ij})
for a photometric redshift bin using the extended Limber approximation.

Physics
-------
Under the Limber approximation (valid for ℓ ≳ 10), the angular power
spectrum for a tophat window in [z_min, z_max] is:

    C_ℓ = ∫ dz  [H(z)/c]  /χ²(z)  × [b(k_L, z)]²  P_m(k_L, z)  W²(z)

where:
  - k_L  = (ℓ + 1/2) / χ(z)  is the Limber wavenumber
  - b(k, z) = b_1 + Δb(k, z, f_NL)  is the total (scale-dependent) bias
  - W(z)  is the tophat window function, W = 1 for z ∈ [z_min, z_max]
  - χ(z)  is the comoving distance

For the cross-spectrum between tracers A and B:

    C_ℓ^{AB} = ∫ dz [H/c/χ²] b_A(k_L) b_B(k_L) P_m(k_L) W(z)

Cross-spectra have no shot noise; only auto-spectra add N_ℓ = 1/(n̄ χ² Δχ).

PNG signature
-------------
The scale-dependent bias Δb(k) ∝ k⁻² amplifies large-scale modes (low ℓ)
by orders of magnitude:
    C_ℓ(f_NL ≠ 0) / C_ℓ(f_NL = 0) − 1  ∝  f_NL² / ℓ⁴   (rough scaling)

References
----------
Limber, ApJ 117, 134 (1953) — Original projection equation
Kaiser, MNRAS 227, 1 (1987) — Modern angular power spectrum formulation
LoVerde & Afshordi, PRD 78, 123506 (2008) — Extended Limber for PNG
"""

import numpy as np
from scipy import integrate
from scipy.interpolate import interp1d

# Import from our modules
try:
    from .cosmology import get_growth_factor, get_power_spectrum, Om0, H0, Ode0
    from .bias_functions import get_total_bias
except ImportError:
    from cosmology import get_growth_factor, get_power_spectrum, Om0, H0, Ode0
    from bias_functions import get_total_bias


# Speed of light in km/s
C_LIGHT = 299792.458


def get_comoving_distance(z):
    """
    Compute comoving distance χ(z) in Mpc/h.

    The comoving distance is the distance light travels from redshift z to
    z=0, accounting for the expansion of the universe.

    Parameters
    ----------
    z : float or array_like
        Redshift

    Returns
    -------
    chi : float or array_like
        Comoving distance in Mpc/h

    Notes
    -----
    For a flat ΛCDM universe:

    χ(z) = (c/H₀) ∫₀^z dz'/E(z')

    where E(z) = H(z)/H₀ = sqrt(Ω_m(1+z)³ + Ω_Λ)

    For Planck 2018 cosmology, χ(z=1) ≈ 3300 Mpc/h
    """
    z = np.asarray(z)
    scalar_input = z.ndim == 0
    if scalar_input:
        z = z[None]

    chi = np.zeros_like(z, dtype=float)

    def integrand(zp):
        """Integrand: 1/E(z)"""
        E = np.sqrt(Om0 * (1 + zp)**3 + Ode0)
        return 1.0 / E

    for i, zi in enumerate(z):
        if zi > 0:
            integral, _ = integrate.quad(integrand, 0, zi, limit=100)
            chi[i] = (C_LIGHT / H0) * integral
        else:
            chi[i] = 0.0

    if scalar_input:
        return chi[0]
    return chi


def get_hubble(z):
    """
    Compute Hubble parameter H(z) in km/s/Mpc.

    Parameters
    ----------
    z : float or array_like
        Redshift

    Returns
    -------
    H : float or array_like
        Hubble parameter in km/s/Mpc

    Notes
    -----
    For flat ΛCDM:
    H(z) = H₀ × sqrt(Ω_m(1+z)³ + Ω_Λ)
    """
    z = np.asarray(z)
    E = np.sqrt(Om0 * (1 + z)**3 + Ode0)
    return H0 * E


def window_function_tophat(z, z_min, z_max):
    """
    Tophat window function for a redshift bin.

    Parameters
    ----------
    z : float or array_like
        Redshift(s)
    z_min : float
        Minimum redshift of bin
    z_max : float
        Maximum redshift of bin

    Returns
    -------
    W : float or array_like
        Window function value (normalized to integrate to 1)
    """
    z = np.asarray(z)
    W = np.where((z >= z_min) & (z <= z_max), 1.0 / (z_max - z_min), 0.0)
    return W


def get_angular_power_spectrum(ell, z_min, z_max, b1, fNL=0, shape='local',
                                window_type='tophat', n_z_samples=100):
    """
    Compute angular power spectrum C_ℓ using Limber approximation.

    This computes the angular power spectrum for a tracer with scale-dependent
    bias from primordial non-Gaussianity.

    Parameters
    ----------
    ell : float or array_like
        Multipole moment(s)
    z_min : float
        Minimum redshift of window
    z_max : float
        Maximum redshift of window
    b1 : float
        Linear bias parameter
    fNL : float, optional
        Primordial non-Gaussianity parameter (default: 0)
    shape : str, optional
        Shape of PNG: 'local', 'equilateral', or 'orthogonal' (default: 'local')
    window_type : str, optional
        Type of window function: 'tophat' (default)
    n_z_samples : int, optional
        Number of redshift samples for integration (default: 100)

    Returns
    -------
    C_ell : float or array_like
        Angular power spectrum in (nW/m²/sr)² or appropriate units

    Notes
    -----
    The Limber approximation gives:

    C_ℓ = ∫ dz [W(z)²/χ²(z)] × [1/H(z)] × P(k=(ℓ+1/2)/χ(z), z)

    where:
    - W(z) is the window function (redshift distribution)
    - χ(z) is the comoving distance
    - H(z) is the Hubble parameter
    - P(k,z) = b²(k,z) × P_matter(k,z) with b(k,z) = b₁ + Δb(k,z,fNL)

    IMPORTANT: We use k = (ℓ + 1/2)/χ(z), not just ℓ/χ(z)!
    """
    ell = np.asarray(ell)
    scalar_input = ell.ndim == 0
    if scalar_input:
        ell = ell[None]

    # Create redshift array for integration
    z_array = np.linspace(z_min, z_max, n_z_samples)

    # Pre-compute geometric quantities
    chi_array = np.array([get_comoving_distance(z) for z in z_array])
    H_array = np.array([get_hubble(z) for z in z_array])
    W_array = window_function_tophat(z_array, z_min, z_max)

    # Compute C_ℓ for each ell
    C_ell = np.zeros_like(ell, dtype=float)

    for i, ell_val in enumerate(ell):
        def integrand(z):
            """Limber integrand"""
            # Interpolate pre-computed values
            chi = np.interp(z, z_array, chi_array)
            H_z = np.interp(z, z_array, H_array)
            W = window_function_tophat(z, z_min, z_max)

            if chi == 0 or H_z == 0 or W == 0:
                return 0.0

            # Wavenumber: k = (ℓ + 1/2) / χ(z)
            # The +1/2 is important for accuracy!
            k = (ell_val + 0.5) / chi

            # Get total bias (includes PNG correction)
            b_total = get_total_bias(k, z, fNL, b1, shape=shape)

            # Get matter power spectrum
            P_matter = get_power_spectrum(k, z, fNL=0)

            # Observed power spectrum: P_obs = b²(k) × P_matter
            P_obs = b_total**2 * P_matter

            # Limber integrand: [W²/χ²] × [1/H] × P
            result = (W**2 / chi**2) * (1.0 / H_z) * P_obs

            return result

        # Perform integration over redshift
        try:
            integral, _ = integrate.quad(integrand, z_min, z_max, limit=100)
            C_ell[i] = integral
        except Exception as e:
            print(f"Warning: Integration failed for ell={ell_val}: {e}")
            C_ell[i] = 0.0

    if scalar_input:
        return C_ell[0]
    return C_ell


def get_cross_power_spectrum(ell, z_min, z_max, b1_A, b1_B,
                             fNL=0, shape='local', n_z_samples=100):
    """
    Compute cross-power spectrum C_ℓ^AB between two tracers.

    Parameters
    ----------
    ell : float or array_like
        Multipole moment(s)
    z_min : float
        Minimum redshift
    z_max : float
        Maximum redshift
    b1_A : float
        Linear bias of tracer A
    b1_B : float
        Linear bias of tracer B
    fNL : float, optional
        Primordial non-Gaussianity parameter
    shape : str, optional
        PNG shape
    n_z_samples : int, optional
        Number of redshift samples

    Returns
    -------
    C_ell_AB : float or array_like
        Cross-power spectrum

    Notes
    -----
    For cross-correlation between two galaxy samples:

    C_ℓ^AB = ∫ dz [W_A(z) W_B(z)] / [χ²(z) H(z)] × b_A(k,z) × b_B(k,z) × P_matter(k,z)

    where b_A(k,z) = b₁^A + Δb^A(k,z,fNL) and b_B(k,z) = b₁^B + Δb^B(k,z,fNL)

    For multi-tracer analysis, cross-spectra have NO shot noise (only auto-spectra do).
    This enables cosmic variance cancellation!
    """
    ell = np.asarray(ell)
    scalar_input = ell.ndim == 0
    if scalar_input:
        ell = ell[None]

    # Create redshift array
    z_array = np.linspace(z_min, z_max, n_z_samples)
    chi_array = np.array([get_comoving_distance(z) for z in z_array])
    H_array = np.array([get_hubble(z) for z in z_array])

    C_ell_AB = np.zeros_like(ell, dtype=float)

    for i, ell_val in enumerate(ell):
        def integrand(z):
            chi = np.interp(z, z_array, chi_array)
            H_z = np.interp(z, z_array, H_array)
            W_A = window_function_tophat(z, z_min, z_max)
            W_B = window_function_tophat(z, z_min, z_max)

            if chi == 0 or H_z == 0 or W_A == 0 or W_B == 0:
                return 0.0

            k = (ell_val + 0.5) / chi

            # Total biases for both tracers
            b_total_A = get_total_bias(k, z, fNL, b1_A, shape=shape)
            b_total_B = get_total_bias(k, z, fNL, b1_B, shape=shape)

            P_matter = get_power_spectrum(k, z, fNL=0)

            # Cross power: b_A × b_B × P_matter
            result = (W_A * W_B / chi**2) * (1.0 / H_z) * b_total_A * b_total_B * P_matter

            return result

        integral, _ = integrate.quad(integrand, z_min, z_max, limit=100)
        C_ell_AB[i] = integral

    if scalar_input:
        return C_ell_AB[0]
    return C_ell_AB


def compute_dCl_dfNL_cross(ell, z_min, z_max, b1_A, b1_B,
                           fNL_fid=0, shape='local', delta_fNL=0.1):
    """
    Compute derivative ∂C_ℓ^AB/∂f_NL for cross-power spectrum using finite differences.

    Parameters
    ----------
    ell : float or array_like
        Multipole moment(s)
    z_min : float
        Minimum redshift
    z_max : float
        Maximum redshift
    b1_A : float
        Linear bias of tracer A
    b1_B : float
        Linear bias of tracer B
    fNL_fid : float, optional
        Fiducial fNL value (default: 0)
    shape : str, optional
        PNG shape
    delta_fNL : float, optional
        Step size for finite difference (default: 0.1)

    Returns
    -------
    dCl_dfNL : float or array_like
        Derivative ∂C_ℓ^AB/∂f_NL

    Notes
    -----
    Uses centered finite difference:
    ∂C_ℓ/∂f_NL ≈ [C_ℓ(f_NL + δf) - C_ℓ(f_NL - δf)] / (2δf)

    For cross-spectra with PNG:
    P_AB(k,z,f_NL) = [b₁^A + Δb^A(k,z,f_NL)] × [b₁^B + Δb^B(k,z,f_NL)] × P_matter(k,z)

    At fiducial f_NL = 0:
    ∂P_AB/∂f_NL|_{f_NL=0} = [∂Δb^A/∂f_NL × b₁^B + b₁^A × ∂Δb^B/∂f_NL] × P_matter
    """
    C_plus = get_cross_power_spectrum(ell, z_min, z_max, b1_A, b1_B,
                                       fNL=fNL_fid + delta_fNL, shape=shape)
    C_minus = get_cross_power_spectrum(ell, z_min, z_max, b1_A, b1_B,
                                        fNL=fNL_fid - delta_fNL, shape=shape)

    dCl_dfNL = (C_plus - C_minus) / (2.0 * delta_fNL)

    return dCl_dfNL


def compute_dCl_dfNL_auto(ell, z_min, z_max, b1,
                          fNL_fid=0, shape='local', delta_fNL=0.1):
    """
    Compute derivative ∂C_ℓ/∂f_NL for auto-power spectrum.

    This is a convenience wrapper around compute_dCl_dfNL_cross for the
    case where both tracers are the same.

    Parameters
    ----------
    ell : float or array_like
        Multipole moment(s)
    z_min : float
        Minimum redshift
    z_max : float
        Maximum redshift
    b1 : float
        Linear bias
    fNL_fid : float, optional
        Fiducial fNL value
    shape : str, optional
        PNG shape
    delta_fNL : float, optional
        Step size for finite difference

    Returns
    -------
    dCl_dfNL : float or array_like
        Derivative ∂C_ℓ/∂f_NL
    """
    return compute_dCl_dfNL_cross(ell, z_min, z_max, b1, b1,
                                  fNL_fid=fNL_fid, shape=shape, delta_fNL=delta_fNL)


if __name__ == "__main__":
    # Test comoving distance
    print("=" * 70)
    print("LIMBER APPROXIMATION TESTS")
    print("=" * 70)

    print("\n1. Comoving Distance:")
    print("-" * 70)
    z_test = np.array([0.0, 0.5, 1.0, 2.0, 3.0])
    chi_test = get_comoving_distance(z_test)

    print(f"{'z':<8} {'χ(z) [Mpc/h]':<20}")
    print("-" * 30)
    for z, chi in zip(z_test, chi_test):
        print(f"{z:<8.1f} {chi:<20.1f}")

    print(f"\n✓ For Planck cosmology, χ(z=1) ≈ 3300 Mpc/h")
    print(f"  Computed: χ(z=1) = {chi_test[2]:.1f} Mpc/h")

    print("\n2. Hubble Parameter:")
    print("-" * 70)
    H_test = get_hubble(z_test)
    print(f"{'z':<8} {'H(z) [km/s/Mpc]':<20}")
    print("-" * 30)
    for z, H in zip(z_test, H_test):
        print(f"{z:<8.1f} {H:<20.2f}")

    print("\n3. Angular Power Spectrum:")
    print("-" * 70)
    ell = np.array([10, 50, 100, 500, 1000])
    z_min, z_max = 0.5, 1.5
    b1 = 2.0

    print(f"\nComputing C_ℓ for z ∈ [{z_min}, {z_max}], b₁ = {b1}")
    print(f"{'ℓ':<10} {'C_ℓ (fNL=0)':<20} {'C_ℓ (fNL=10)':<20}")
    print("-" * 50)

    C_ell_0 = get_angular_power_spectrum(ell, z_min, z_max, b1, fNL=0)
    C_ell_10 = get_angular_power_spectrum(ell, z_min, z_max, b1, fNL=10)

    for l, C0, C10 in zip(ell, C_ell_0, C_ell_10):
        print(f"{l:<10} {C0:<20.6e} {C10:<20.6e}")

    print("\nValidation:")
    print(f"✓ C_ℓ > 0 for all ℓ: {all(C_ell_0 > 0)}")
    print(f"✓ C_ℓ decreases with ℓ at high ℓ: {C_ell_0[-1] < C_ell_0[0]}")
    print(f"✓ fNL=10 increases C_ℓ: {all(C_ell_10 > C_ell_0)}")

    print("\n" + "=" * 70)
