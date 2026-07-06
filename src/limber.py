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
from scipy.special import spherical_jn

# Import from our modules
try:
    from .cosmology import (get_growth_factor, get_power_spectrum, Om0, H0, Ode0,
                            growth_rate)
    from .bias_functions import get_total_bias, delta_b_local
except ImportError:
    from cosmology import (get_growth_factor, get_power_spectrum, Om0, H0, Ode0,
                           growth_rate)
    from bias_functions import get_total_bias, delta_b_local


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


# ---------------------------------------------------------------------------
# Full-Bessel projection (Item 1 + Item 2)
# ---------------------------------------------------------------------------
#
# The Limber approximation is only accurate for ℓ ≳ ℓ_limber where
#
#     ℓ_limber = [r(z) / (3000 Mpc/h)] × E(z) × (λ_rest / Δλ)
#
# evaluated at the channel's peak redshift. At smaller ℓ we compute
#
#     C_ℓ^{ij} = (2/π) ∫ k² dk P(k) Δ_ℓ^i(k) Δ_ℓ^j(k)
#
# with the transfer function
#
#     Δ_ℓ^i(k) = ∫ dz W_i(z) [b_eff,i(z) + f_NL Δb_i(k,z)] D(z) j_ℓ(k r(z))
#
# where the Kaiser correction (Item 2) makes b_eff,i(z) = b_i(z) + f(z) with
# f(z) = Ω_m(z)^0.55 (see cosmology.growth_rate). The window W_i(z) is the
# top-hat channel profile in observed wavelength, i.e. z ∈ [z_min, z_max].
# ---------------------------------------------------------------------------


def compute_ell_limber(lambda_rest, delta_lambda, z_peak):
    """
    Compute the multipole above which the Limber approximation is accurate.

    Parameters
    ----------
    lambda_rest : float
        Rest-frame wavelength of the emission line in μm.
    delta_lambda : float
        Channel width in wavelength (observed frame) in μm.
    z_peak : float
        Peak (central) redshift of the channel = λ_obs / λ_rest − 1.

    Returns
    -------
    ell_limber : float
        Threshold multipole; use full Bessel for ℓ ≤ ell_limber.
    """
    r_z = get_comoving_distance(z_peak)  # Mpc/h
    E_z = np.sqrt(Om0 * (1.0 + z_peak) ** 3 + Ode0)
    return (r_z / 3000.0) * E_z * (lambda_rest / delta_lambda)


def _tophat_z_bounds(z_peak, lambda_rest, delta_lambda):
    """
    Convert a channel of width Δλ around observed wavelength λ_obs = λ_rest (1+z_peak)
    into a top-hat redshift window [z_min, z_max].
    """
    lam_obs = lambda_rest * (1.0 + z_peak)
    z_min = max(0.0, (lam_obs - 0.5 * delta_lambda) / lambda_rest - 1.0)
    z_max = (lam_obs + 0.5 * delta_lambda) / lambda_rest - 1.0
    return z_min, z_max


def _bessel_transfer(ell, k_grid, z_grid, chi_grid, D_grid,
                     b1_of_z, lambda_rest, delta_lambda, z_peak,
                     fNL, use_rsd):
    """
    Evaluate Δ_ℓ(k) on a k-grid for a single channel.

    Parameters
    ----------
    ell : int
        Multipole.
    k_grid : ndarray, shape (Nk,)
        Wavenumber grid in h/Mpc.
    z_grid, chi_grid, D_grid : ndarray, shape (Nz,)
        Redshift, comoving-distance, and linear-growth arrays inside the top-hat.
    b1_of_z : callable
        b1(z) for the channel's emission line.
    lambda_rest, delta_lambda, z_peak : float
        Channel geometry (unused inside the integrand but retained for signature).
    fNL : float
        Fiducial f_NL for the scale-dependent bias.
    use_rsd : bool
        If True, add Kaiser growth-rate term to the effective bias.

    Returns
    -------
    Delta : ndarray, shape (Nk,)
        Transfer function Δ_ℓ(k).
    """
    Nk = k_grid.size
    Nz = z_grid.size
    dz = z_grid[1] - z_grid[0]

    # Precompute z-dependent factors that do not depend on k.
    b1_arr = np.asarray([b1_of_z(z) for z in z_grid])
    if use_rsd:
        b_eff_arr = b1_arr + growth_rate(z_grid)
    else:
        b_eff_arr = b1_arr

    # Normalised top-hat window: W(z) = 1 / Δz within the bin.
    W_arr = np.full(Nz, 1.0 / (z_grid[-1] - z_grid[0]))

    Delta = np.zeros(Nk)
    for ik, k in enumerate(k_grid):
        # Δb_i(k,z) for each z (already contains D(z) factor internally).
        db_arr = np.asarray([delta_b_local(k, z, fNL, b1_arr[iz])
                             for iz, z in enumerate(z_grid)])
        # j_ℓ(k r(z))
        j_arr = spherical_jn(int(ell), k * chi_grid)
        integrand = W_arr * (b_eff_arr + db_arr) * D_grid * j_arr
        Delta[ik] = np.trapezoid(integrand, z_grid)

    return Delta


# Cache Δ_ℓ(k) arrays keyed by (line, ell, fNL_sign, use_rsd) to avoid
# recomputing the Bessel integrals repeatedly during Fisher assembly.
_BESSEL_CACHE = {}


def _bessel_cache_key(line_key, ell, fNL, use_rsd, b_scale, I_scale):
    # Round to keep small numerical perturbations from busting the cache.
    return (line_key, int(ell), round(fNL, 6), bool(use_rsd),
            round(b_scale, 8), round(I_scale, 8))


def compute_cl_bessel(ell, channel_i, channel_j, fNL=0.0, use_rsd=True,
                      k_min=1e-4, k_max=0.3, n_k=64, n_z=25):
    """
    Full-Bessel angular power spectrum between two LIM channels at multipole ℓ.

    Parameters
    ----------
    ell : int
        Multipole.
    channel_i, channel_j : dict
        Channel descriptors with keys:
            'line'          : line label used for caching,
            'lambda_rest'   : rest wavelength (μm),
            'delta_lambda'  : channel width (μm),
            'z_peak'        : central redshift,
            'b1_of_z'       : callable b1(z),
            'I_scale'       : multiplicative intensity nuisance (default 1),
            'b_scale'       : multiplicative bias nuisance (default 1).
    fNL : float
        Local f_NL used for Δb.
    use_rsd : bool
        Include Kaiser RSD correction (Item 2).
    k_min, k_max : float
        k-integration range in h/Mpc.
    n_k, n_z : int
        Grid sizes for k and z.
    """
    z_lo_i, z_hi_i = _tophat_z_bounds(channel_i['z_peak'],
                                      channel_i['lambda_rest'],
                                      channel_i['delta_lambda'])
    z_lo_j, z_hi_j = _tophat_z_bounds(channel_j['z_peak'],
                                      channel_j['lambda_rest'],
                                      channel_j['delta_lambda'])

    # Support each transfer function on its own top-hat.
    k_grid = np.logspace(np.log10(k_min), np.log10(k_max), n_k)

    def _delta_for_channel(ch, z_lo, z_hi):
        line = ch['line']
        b_scale = ch.get('b_scale', 1.0)
        I_scale = ch.get('I_scale', 1.0)
        key = _bessel_cache_key(line, ell, fNL, use_rsd, b_scale, I_scale)
        cached = _BESSEL_CACHE.get(key)
        if cached is not None:
            return cached

        z_grid = np.linspace(z_lo + 1e-6, z_hi, n_z)
        chi_grid = np.asarray([get_comoving_distance(z) for z in z_grid])
        D_grid = get_growth_factor(z_grid)

        base_b1 = ch['b1_of_z']
        b1_of_z_scaled = (lambda z: b_scale * base_b1(z))

        Delta = _bessel_transfer(ell, k_grid, z_grid, chi_grid, D_grid,
                                 b1_of_z_scaled,
                                 ch['lambda_rest'], ch['delta_lambda'],
                                 ch['z_peak'],
                                 fNL, use_rsd)
        Delta = I_scale * Delta
        _BESSEL_CACHE[key] = Delta
        return Delta

    Delta_i = _delta_for_channel(channel_i, z_lo_i, z_hi_i)
    Delta_j = _delta_for_channel(channel_j, z_lo_j, z_hi_j)

    P_grid = np.asarray([get_power_spectrum(k, z=0.0) for k in k_grid])
    integrand = (k_grid ** 2) * P_grid * Delta_i * Delta_j
    # (2/π) ∫ k² dk P(k) Δ_i(k) Δ_j(k), with the D(z) factor already in Δ.
    return (2.0 / np.pi) * np.trapezoid(integrand, k_grid)


def clear_bessel_cache():
    """Reset the Δ_ℓ(k) cache. Call between Fisher runs with different setups."""
    _BESSEL_CACHE.clear()


def compute_cls_full(ell, channel_i, channel_j, fNL=0.0, use_rsd=True,
                     validate=False):
    """
    Dispatch to the Limber or full-Bessel calculation based on ℓ_limber.

    Uses the Bessel integrator for ℓ ≤ ℓ_limber (evaluated at the peak
    redshift of the higher-ℓ channel) and the existing Limber implementation
    otherwise. When ``validate=True`` and ℓ ≈ 5 × ℓ_limber, both are computed
    and their agreement checked to within 5%.
    """
    # Use the max ℓ_limber across the two channels so we default to Bessel
    # whenever *either* channel would need it.
    ell_lim_i = compute_ell_limber(channel_i['lambda_rest'],
                                   channel_i['delta_lambda'],
                                   channel_i['z_peak'])
    ell_lim_j = compute_ell_limber(channel_j['lambda_rest'],
                                   channel_j['delta_lambda'],
                                   channel_j['z_peak'])
    ell_lim = max(ell_lim_i, ell_lim_j)

    if ell <= ell_lim:
        return compute_cl_bessel(ell, channel_i, channel_j,
                                 fNL=fNL, use_rsd=use_rsd)

    # High-ℓ Limber path. Reuse the existing cross-spectrum utility on the
    # geometric overlap of the two channels.
    z_lo_i, z_hi_i = _tophat_z_bounds(channel_i['z_peak'],
                                      channel_i['lambda_rest'],
                                      channel_i['delta_lambda'])
    z_lo_j, z_hi_j = _tophat_z_bounds(channel_j['z_peak'],
                                      channel_j['lambda_rest'],
                                      channel_j['delta_lambda'])
    z_lo = max(z_lo_i, z_lo_j)
    z_hi = min(z_hi_i, z_hi_j)
    if z_hi <= z_lo:
        return 0.0

    z_mid = 0.5 * (z_lo + z_hi)
    b1_i = channel_i['b_scale'] * channel_i['b1_of_z'](z_mid)
    b1_j = channel_j['b_scale'] * channel_j['b1_of_z'](z_mid)
    scale = channel_i['I_scale'] * channel_j['I_scale']
    cl_limber = scale * get_cross_power_spectrum(
        np.asarray([ell]), z_lo, z_hi, b1_i, b1_j, fNL=fNL, shape='local'
    )[0]

    if validate and ell >= 5 * ell_lim and ell_lim > 0:
        cl_bessel = compute_cl_bessel(ell, channel_i, channel_j,
                                      fNL=fNL, use_rsd=use_rsd)
        if cl_limber != 0:
            rel = abs(cl_bessel - cl_limber) / abs(cl_limber)
            if rel > 0.05:
                print(f"[compute_cls_full] WARNING: Bessel vs Limber disagree "
                      f"by {rel*100:.1f}% at ℓ={ell} for "
                      f"({channel_i['line']}, {channel_j['line']})")

    return cl_limber


# ---------------------------------------------------------------------------
# LIM intensity-space C_ℓ backend  (Step 1 of the Pullen refactor)
# ---------------------------------------------------------------------------
# Produces C_ℓ^{νν'} in (nW/m²/sr)² so that σ_n² Ω_pix on the diagonal is on
# the same units footing.  Formalism follows Cheng+2024 with a Gaussian
# redshift kernel of width σ_z = 0.12 per channel.
#
#     C_ℓ^{νν'} = [b_i(z̄) Ī_ν^i(z̄)] [b_j(z̄) Ī_ν'^j(z̄)]
#                × (H_h(z̄)/c) / χ_h²(z̄) × P_m(k=(ℓ+½)/χ_h, z̄) × W_{νν'}
#
# with the Gaussian-window overlap
#     W_{νν'} = 1/(2√π σ_z) × exp[-(z_ν − z_ν')² / (4 σ_z²)]
#
# Cross-line pairs (line_ν ≠ line_ν') pick up the same overlap factor, so a
# non-vanishing cross only occurs when the two channels see different lines
# at approximately the same physical redshift — the LIM analogue of the
# multi-tracer cosmic-variance cancellation.
# ---------------------------------------------------------------------------

# Lazy imports of lim_signal to avoid hard dependency on the LIM signal model
# for consumers that only need galaxy-clustering Limber. Keep the intensity
# lookup fast by caching values on a shared z-grid.
_LIM_SIGNAL_MOD = None
_INTENSITY_CACHE = {}   # keyed by (line, round(z, 4))


def _get_lim_signal():
    global _LIM_SIGNAL_MOD
    if _LIM_SIGNAL_MOD is None:
        try:
            from . import lim_signal as _m
        except ImportError:
            import lim_signal as _m
        _LIM_SIGNAL_MOD = _m
    return _LIM_SIGNAL_MOD


def _mean_intensity(line, z):
    """Ī_ν^i(z) in nW/m²/sr, mean line intensity WITHOUT the halo bias factor."""
    key = (line, round(float(z), 5))
    cached = _INTENSITY_CACHE.get(key)
    if cached is not None:
        return cached
    m = _get_lim_signal()
    val = float(m.get_line_intensity(float(z), line=line, return_bias_weighted=False))
    _INTENSITY_CACHE[key] = val
    return val


def clear_intensity_cache():
    _INTENSITY_CACHE.clear()


SIGMA_Z_KERNEL = 0.12  # Cheng+2024 / paper text


def _window_overlap(z_nu, z_nup, sigma_z=SIGMA_Z_KERNEL):
    """∫ W_ν(z) W_ν'(z) dz for two normalised Gaussians of width σ_z."""
    dz = z_nu - z_nup
    return (1.0 / (2.0 * np.sqrt(np.pi) * sigma_z)) * \
           np.exp(-(dz ** 2) / (4.0 * sigma_z ** 2))


def compute_lim_cl_limber(ell, ch_i, ch_j, fNL=0.0):
    """
    Limber-approximated intensity-space C_ℓ between two LIM channels.

    Units: (nW/m²/sr)². Assumes the Gaussian-window formalism above.
    """
    z_i = ch_i['z_peak']
    z_j = ch_j['z_peak']
    z_bar = 0.5 * (z_i + z_j)

    overlap = _window_overlap(z_i, z_j)
    # Numerically negligible: skip.
    if overlap < 1e-4 / (2.0 * np.sqrt(np.pi) * SIGMA_Z_KERNEL):
        return 0.0

    # Geometry at z̄
    chi_h = get_comoving_distance(z_bar)     # Mpc/h
    if chi_h <= 0:
        return 0.0
    try:
        from .cosmology import h as _h
    except ImportError:
        from cosmology import h as _h
    H_h = get_hubble(z_bar) * _h              # km/s/(Mpc/h)
    k_limber = max((ell + 0.5) / chi_h, 1e-4)  # h/Mpc
    P_m = get_power_spectrum(k_limber, z_bar)  # (Mpc/h)³

    # Bias-weighted intensities at z̄. B_scale, I_scale are nuisance params.
    b_i = ch_i['b_scale'] * ch_i['b1_of_z'](z_bar)
    b_j = ch_j['b_scale'] * ch_j['b1_of_z'](z_bar)
    I_i = ch_i['I_scale'] * _mean_intensity(ch_i['line'], z_bar)
    I_j = ch_j['I_scale'] * _mean_intensity(ch_j['line'], z_bar)

    # Scale-dependent bias contributions (local PNG).
    if fNL != 0.0:
        db_i = delta_b_local(k_limber, z_bar, fNL, b_i)
        db_j = delta_b_local(k_limber, z_bar, fNL, b_j)
        bI_i = (b_i + db_i) * I_i
        bI_j = (b_j + db_j) * I_j
    else:
        bI_i = b_i * I_i
        bI_j = b_j * I_j

    geom = (H_h / C_LIGHT) / (chi_h ** 2)   # (Mpc/h)⁻¹
    return bI_i * bI_j * geom * P_m * overlap


def compute_lim_cl_bessel(ell, ch_i, ch_j, fNL=0.0, use_rsd=True,
                          k_min=1e-4, k_max=0.3, n_k=48, n_z=25):
    """
    Full-Bessel intensity-space C_ℓ between two LIM channels.

    Adds the intensity multipliers Ī_ν^i(z) and Ī_ν'^j(z) inside the transfer
    functions and, if use_rsd=True, the Kaiser correction b → b + f(z).
    """
    # Support each channel on ±3σ_z of its Gaussian.
    z_lo_i = max(1e-3, ch_i['z_peak'] - 3.0 * SIGMA_Z_KERNEL)
    z_hi_i = ch_i['z_peak'] + 3.0 * SIGMA_Z_KERNEL
    z_lo_j = max(1e-3, ch_j['z_peak'] - 3.0 * SIGMA_Z_KERNEL)
    z_hi_j = ch_j['z_peak'] + 3.0 * SIGMA_Z_KERNEL

    k_grid = np.logspace(np.log10(k_min), np.log10(k_max), n_k)

    def _delta(ch, z_lo, z_hi):
        z_grid = np.linspace(z_lo, z_hi, n_z)
        chi_h = np.asarray([get_comoving_distance(z) for z in z_grid])
        D = get_growth_factor(z_grid)
        line = ch['line']
        b1_arr = np.asarray([ch['b_scale'] * ch['b1_of_z'](z) for z in z_grid])
        I_arr = np.asarray([ch['I_scale'] * _mean_intensity(line, z)
                            for z in z_grid])
        if use_rsd:
            try:
                from .cosmology import growth_rate as _growth_rate
            except ImportError:
                from cosmology import growth_rate as _growth_rate
            b_eff = b1_arr + _growth_rate(z_grid)
        else:
            b_eff = b1_arr

        # Normalised Gaussian window centered on the channel's z_peak.
        z_peak = ch['z_peak']
        W = (1.0 / (np.sqrt(2.0 * np.pi) * SIGMA_Z_KERNEL)) * \
            np.exp(-(z_grid - z_peak) ** 2 / (2.0 * SIGMA_Z_KERNEL ** 2))

        Delta = np.zeros_like(k_grid)
        for ik, k in enumerate(k_grid):
            db = np.asarray([delta_b_local(k, z, fNL, b1_arr[iz])
                             for iz, z in enumerate(z_grid)])
            j_l = spherical_jn(int(ell), k * chi_h)
            integrand = W * (b_eff + db) * I_arr * D * j_l
            Delta[ik] = np.trapezoid(integrand, z_grid)
        return Delta

    Delta_i = _delta(ch_i, z_lo_i, z_hi_i)
    Delta_j = _delta(ch_j, z_lo_j, z_hi_j)

    P0 = np.asarray([get_power_spectrum(k, z=0.0) for k in k_grid])
    integrand = (k_grid ** 2) * P0 * Delta_i * Delta_j
    return (2.0 / np.pi) * np.trapezoid(integrand, k_grid)


def _lim_bessel_delta_all(ell, channels, fNL, use_rsd,
                          k_min=1e-4, k_max=0.3, n_k=48, n_z=25):
    """
    Precompute Δ_ℓ(k) transfer functions for every channel at multipole ℓ.

    Returns
    -------
    Delta : ndarray shape (N_ch, n_k)
    k_grid : ndarray shape (n_k,)
    P0 : ndarray shape (n_k,) — matter P(k, z=0) on the k-grid.
    """
    try:
        from .cosmology import growth_rate as _growth_rate
    except ImportError:
        from cosmology import growth_rate as _growth_rate

    k_grid = np.logspace(np.log10(k_min), np.log10(k_max), n_k)
    P0 = np.asarray([get_power_spectrum(k, z=0.0) for k in k_grid])

    Delta = np.zeros((len(channels), n_k))
    for ich, ch in enumerate(channels):
        z_lo = max(1e-3, ch['z_peak'] - 3.0 * SIGMA_Z_KERNEL)
        z_hi = ch['z_peak'] + 3.0 * SIGMA_Z_KERNEL
        z_grid = np.linspace(z_lo, z_hi, n_z)
        chi_h = np.asarray([get_comoving_distance(z) for z in z_grid])
        D = get_growth_factor(z_grid)
        line = ch['line']
        b1_arr = np.asarray([ch['b_scale'] * ch['b1_of_z'](z) for z in z_grid])
        I_arr = np.asarray([ch['I_scale'] * _mean_intensity(line, z)
                            for z in z_grid])
        b_eff = b1_arr + _growth_rate(z_grid) if use_rsd else b1_arr
        W = (1.0 / (np.sqrt(2.0 * np.pi) * SIGMA_Z_KERNEL)) * \
            np.exp(-(z_grid - ch['z_peak']) ** 2 /
                   (2.0 * SIGMA_Z_KERNEL ** 2))
        # k-vectorised (using scalar loop for Δb since it has to be evaluated
        # for each (k, z), but j_ℓ is cheap once k*χ is built).
        for ik, k in enumerate(k_grid):
            db = np.asarray([delta_b_local(k, z, fNL, b1_arr[iz])
                             for iz, z in enumerate(z_grid)])
            j_l = spherical_jn(int(ell), k * chi_h)
            integrand = W * (b_eff + db) * I_arr * D * j_l
            Delta[ich, ik] = np.trapezoid(integrand, z_grid)
    return Delta, k_grid, P0


def _lim_C_matrix_from_delta(Delta, k_grid, P0):
    """C_ℓ^{νν'} = (2/π) ∫ k² dk P(k) Δ_ν(k) Δ_ν'(k) — vectorised."""
    weight = (k_grid ** 2) * P0
    # (N_ch, N_ch) matrix built as a Δ^T diag(w) Δ product with trapezoidal
    # weights on the k integration.
    dk = np.diff(k_grid)
    # trapezoidal weights on k
    w_trap = np.empty_like(k_grid)
    w_trap[1:-1] = 0.5 * (dk[:-1] + dk[1:])
    w_trap[0] = 0.5 * dk[0]
    w_trap[-1] = 0.5 * dk[-1]
    W = weight * w_trap
    # Δ * √W element-wise so that (Δ*√W) @ (Δ*√W)^T gives the integral.
    sw = np.sqrt(np.abs(W)) * np.sign(W)
    DW = Delta * sw
    return (2.0 / np.pi) * (DW @ DW.T)


def compute_lim_cls_matrix(ell, channels, fNL=0.0, use_bessel_below_limber=True,
                           use_rsd=True):
    """
    Assemble the N_ch × N_ch signal covariance C_ℓ (intensity units) at ℓ.

    Uses a vectorised Bessel path for the whole matrix when ℓ is below at
    least half the channels' ℓ_limber; otherwise it uses the per-pair Limber
    path. This keeps the 92×92 assembly under a few seconds per ℓ.
    """
    n = len(channels)
    ell_lim_arr = np.asarray([
        compute_ell_limber(ch['lambda_rest'], ch['delta_lambda'], ch['z_peak'])
        for ch in channels
    ])
    use_bessel = use_bessel_below_limber and (ell <= np.median(ell_lim_arr))

    if use_bessel:
        Delta, k_grid, P0 = _lim_bessel_delta_all(ell, channels, fNL, use_rsd)
        return _lim_C_matrix_from_delta(Delta, k_grid, P0)

    # Fast Limber matrix build: precompute per-channel intensities/geometry
    # at z_peak, then use outer products with the Gaussian-window overlap.
    z_peaks = np.asarray([ch['z_peak'] for ch in channels])
    # Pairwise mean z for the geometric factor.
    z_bar = 0.5 * (z_peaks[:, None] + z_peaks[None, :])
    overlap = _window_overlap(z_peaks[:, None], z_peaks[None, :])
    # Approximate geometry at each channel's own z_peak (cheap) and blend as
    # geometric mean per pair to keep the outer-product structure.
    chi_h_i = np.asarray([get_comoving_distance(z) for z in z_peaks])
    try:
        from .cosmology import h as _h
    except ImportError:
        from cosmology import h as _h
    H_h_i = np.asarray([get_hubble(z) * _h for z in z_peaks])
    b_i = np.asarray([ch['b_scale'] * ch['b1_of_z'](ch['z_peak'])
                      for ch in channels])
    I_i = np.asarray([ch['I_scale'] * _mean_intensity(ch['line'], ch['z_peak'])
                      for ch in channels])
    bI = b_i * I_i
    if fNL != 0.0:
        # Δb at (ν, ν') needs k = (ℓ+½)/χ evaluated on the pair — but the
        # per-channel geometric mean is a good approximation for near-diagonal
        # pairs where the overlap is non-negligible.
        pass

    # Geometry factor per pair, evaluated at √(χ_i χ_j) and √(H_i H_j).
    chi_pair = np.sqrt(np.outer(chi_h_i, chi_h_i))
    H_pair = np.sqrt(np.outer(H_h_i, H_h_i))
    k_pair = (ell + 0.5) / chi_pair
    k_pair = np.clip(k_pair, 1e-4, 10.0)
    z_flat = z_bar.ravel()
    k_flat = k_pair.ravel()
    P_flat = np.asarray([get_power_spectrum(k, z=z)
                         for k, z in zip(k_flat, z_flat)])
    P = P_flat.reshape(z_bar.shape)
    geom = (H_pair / C_LIGHT) / (chi_pair ** 2)

    # Scale-dependent bias correction (local PNG).
    if fNL != 0.0:
        db_pair = np.zeros_like(P)
        for i in range(n):
            for j in range(n):
                db_pair[i, j] = delta_b_local(k_pair[i, j], z_bar[i, j],
                                              fNL, b_i[i])
        # Symmetrise: Δb enters on both sides.
        bI_pair_i = (b_i[:, None] + db_pair) * I_i[:, None]
        bI_pair_j = (b_i[None, :] + db_pair.T) * I_i[None, :]
        C = bI_pair_i * bI_pair_j * geom * P * overlap
    else:
        C = np.outer(bI, bI) * geom * P * overlap
    return C


def compute_lim_sigma_matrix(ell, channels, N_diag, fNL=0.0,
                             use_bessel_below_limber=True, use_rsd=True):
    """Return Σ_ℓ = C_ℓ(signal, intensity units) + diag(N_ℓ)."""
    C = compute_lim_cls_matrix(ell, channels, fNL=fNL,
                               use_bessel_below_limber=use_bessel_below_limber,
                               use_rsd=use_rsd)
    return C + np.diag(N_diag)


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
