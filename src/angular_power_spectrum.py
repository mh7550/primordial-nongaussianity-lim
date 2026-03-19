"""
angular_power_spectrum.py — Angular power spectrum for SPHEREx LIM survey.

Implements the cross-frequency angular power spectrum matrix C_ℓ,νν' following
Cheng et al. (2024), arXiv:2403.19740. This module computes the signal and noise
contributions to the angular power spectrum for line intensity mapping with SPHEREx.

Survey Configuration
--------------------
- 64 spectral channels spanning 0.75 to 3.82 μm (first four SPHEREx bands)
- Channels equally spaced in log frequency (16 per band)
- 8 multipole bins spanning 50 < ℓ < 350
- Deep field: f_sky = 0.0048 (200 deg²)
- Multipole range motivated by:
  * ℓ_min = 50: SPHEREx 3.5° field of view
  * ℓ_max = 350: onset of nonlinear clustering at k ~ 0.2 h/Mpc at z=0.7

Physics
-------
The angular power spectrum consists of:

1. Window functions W_iν(χ) [Eq. 9]:
   Describes the redshift distribution of line i in channel ν

2. Signal C_ℓ,νν' [Eq. 13]:
   Limber approximation for cross-frequency angular power spectrum

3. Noise C^n_ℓ,νν' [Eq. 14]:
   Instrumental noise (diagonal in frequency)

4. Total: C_ℓ = C_ℓ,signal + C_ℓ,noise

References
----------
Cheng et al., Phys. Rev. D 109, 103011 (2024) — arXiv:2403.19740
"""

import numpy as np
from scipy import integrate, interpolate
import warnings

# Import from local modules
try:
    from .cosmology import get_power_spectrum, get_comoving_distance
    from .lim_signal import (
        LINE_PROPERTIES,
        get_line_luminosity_density,
        get_bias_weighted_luminosity_density,
        get_angular_diameter_distance,
        get_luminosity_distance,
        get_halo_bias_simple,
        get_spherex_noise_at_wavelength
    )
except ImportError:
    from cosmology import get_power_spectrum, get_comoving_distance
    from lim_signal import (
        LINE_PROPERTIES,
        get_line_luminosity_density,
        get_bias_weighted_luminosity_density,
        get_angular_diameter_distance,
        get_luminosity_distance,
        get_halo_bias_simple,
        get_spherex_noise_at_wavelength
    )


# ============================================================================
# SURVEY CONFIGURATION
# ============================================================================

# SPHEREx spectral channels
# 64 channels spanning 0.75 to 3.82 μm, equally spaced in log frequency
LAMBDA_MIN = 0.75  # μm
LAMBDA_MAX = 3.82  # μm
N_CHANNELS = 64
N_BANDS = 4
CHANNELS_PER_BAND = N_CHANNELS // N_BANDS  # 16 channels per band

# Generate channel edges (log-spaced in frequency → log-spaced in wavelength)
CHANNEL_EDGES = np.logspace(np.log10(LAMBDA_MIN), np.log10(LAMBDA_MAX), N_CHANNELS + 1)
CHANNEL_CENTERS = 0.5 * (CHANNEL_EDGES[:-1] + CHANNEL_EDGES[1:])
CHANNEL_WIDTHS = CHANNEL_EDGES[1:] - CHANNEL_EDGES[:-1]

# Multipole bins
# 8 bins spanning 50 < ℓ < 350 with approximately equal number of modes
ELL_MIN = 50
ELL_MAX = 350
N_ELL_BINS = 8

# Equal spacing in ℓ² gives equal number of modes per bin
ELL_SQUARED_EDGES = np.linspace(ELL_MIN**2, ELL_MAX**2, N_ELL_BINS + 1)
ELL_BIN_EDGES = np.sqrt(ELL_SQUARED_EDGES)
ELL_BIN_CENTERS = np.sqrt(0.5 * (ELL_SQUARED_EDGES[:-1] + ELL_SQUARED_EDGES[1:]))

# Sky coverage
F_SKY = 0.0048  # Deep field: 200 deg²

# SPHEREx pixel size
PIXEL_SIZE_ARCSEC = 6.2  # arcsec
OMEGA_PIX = (PIXEL_SIZE_ARCSEC / 3600.0 * np.pi / 180.0)**2  # sr

# Emission lines
EMISSION_LINES = ['Halpha', 'Hbeta', 'OIII', 'OII']


# ============================================================================
# COMOVING DISTANCE GRID
# ============================================================================

# Create fine grid in comoving distance for window function integration
# Covering z ~ 0.1 to 7 with at least 500 points
Z_MIN_GRID = 0.1
Z_MAX_GRID = 7.0
N_CHI_GRID = 1000

# Build chi grid (cached)
_z_grid_cache = np.linspace(Z_MIN_GRID, Z_MAX_GRID, N_CHI_GRID)
_chi_grid_cache = np.array([get_comoving_distance(z) for z in _z_grid_cache])

# Interpolator: chi -> z
_chi_to_z_interp = interpolate.interp1d(_chi_grid_cache, _z_grid_cache,
                                         kind='cubic', bounds_error=False,
                                         fill_value='extrapolate')


def chi_to_z(chi):
    """
    Convert comoving distance to redshift.

    Parameters
    ----------
    chi : float or array_like
        Comoving distance in Mpc

    Returns
    -------
    z : float or array_like
        Redshift
    """
    return _chi_to_z_interp(chi)


def z_to_chi(z):
    """
    Convert redshift to comoving distance.

    Parameters
    ----------
    z : float or array_like
        Redshift

    Returns
    -------
    chi : float or array_like
        Comoving distance in Mpc
    """
    return get_comoving_distance(z)


# ============================================================================
# WINDOW FUNCTIONS W_iν(χ)
# ============================================================================

def get_channel_redshift_range(channel_idx, line):
    """
    Get the redshift range where emission line falls within spectral channel.

    For a given channel ν with wavelength range [λ_min, λ_max] and an emission
    line with rest wavelength λ_rest, the line is observed in this channel when:

        λ_obs = λ_rest (1 + z) ∈ [λ_min, λ_max]

    This gives redshift range:
        z_min = λ_min / λ_rest - 1
        z_max = λ_max / λ_rest - 1

    Parameters
    ----------
    channel_idx : int
        Channel index (0 to 63)
    line : str
        Emission line name ('Halpha', 'OIII', 'Hbeta', 'OII')

    Returns
    -------
    z_min, z_max : float
        Redshift range where line falls in this channel
    """
    lambda_min = CHANNEL_EDGES[channel_idx]
    lambda_max = CHANNEL_EDGES[channel_idx + 1]
    lambda_rest = LINE_PROPERTIES[line]['lambda_rest']

    z_min = lambda_min / lambda_rest - 1.0
    z_max = lambda_max / lambda_rest - 1.0

    return z_min, z_max


def get_window_function_A0(chi):
    """
    Compute geometric factor A₀(χ) for window function.

    Following Cheng et al. (2024) Eq. 9:
        A₀(χ) = D_A²(χ) / (4π D_L²(χ))

    This factor accounts for the geometric dilution of light and the
    conversion between angular and comoving coordinates.

    Parameters
    ----------
    chi : float or array_like
        Comoving distance in Mpc

    Returns
    -------
    A0 : float or array_like
        Geometric factor (dimensionless)
    """
    z = chi_to_z(chi)
    D_A = get_angular_diameter_distance(z)
    D_L = get_luminosity_distance(z)
    A0 = D_A**2 / (4.0 * np.pi * D_L**2)
    return A0


def compute_window_function(channel_idx, line, chi_grid=None):
    """
    Compute window function W_iν(χ) for emission line i in channel ν.

    Following Cheng et al. (2024) Eq. 9:
        W_iν(χ) = (ν/Δν) × b_i(χ) × M₀_i(χ) × A₀(χ)

    for χ_min,iν < χ < χ_max,iν, and 0 otherwise.

    The window function describes the contribution of comoving distance χ
    to the observed intensity in channel ν for line i.

    Parameters
    ----------
    channel_idx : int
        Channel index (0 to 63)
    line : str
        Emission line name ('Halpha', 'OIII', 'Hbeta', 'OII')
    chi_grid : array_like, optional
        Comoving distance grid in Mpc. If None, uses default fine grid.

    Returns
    -------
    W_inu : array_like
        Window function values on chi_grid
    chi_grid : array_like
        Comoving distance grid used (same as input if provided)
    """
    if chi_grid is None:
        chi_grid = _chi_grid_cache.copy()

    # Get redshift range where line falls in this channel
    z_min, z_max = get_channel_redshift_range(channel_idx, line)

    # Convert to comoving distance range
    chi_min = z_to_chi(z_min)
    chi_max = z_to_chi(z_max)

    # Initialize window function
    W_inu = np.zeros_like(chi_grid)

    # Find chi values within range
    mask = (chi_grid >= chi_min) & (chi_grid <= chi_max)

    if not np.any(mask):
        # No overlap - return zeros
        return W_inu, chi_grid

    # Get redshifts for points in range
    z_in_range = chi_to_z(chi_grid[mask])

    # Compute components of window function
    # (ν/Δν) term: frequency / channel width (dimensionless normalization)
    lambda_center = CHANNEL_CENTERS[channel_idx]
    delta_lambda = CHANNEL_WIDTHS[channel_idx]
    nu_over_delta_nu = lambda_center / delta_lambda  # In wavelength space

    # b_i(z): halo bias
    b_i = get_halo_bias_simple(z_in_range)

    # M₀_i(z): line luminosity density in erg/s/Mpc³
    M0_i = get_line_luminosity_density(z_in_range, line=line)

    # A₀(χ): geometric factor
    A0 = get_window_function_A0(chi_grid[mask])

    # Combine: W_iν(χ) = (ν/Δν) × b_i × M₀_i × A₀
    W_inu[mask] = nu_over_delta_nu * b_i * M0_i * A0

    return W_inu, chi_grid


# ============================================================================
# ANGULAR POWER SPECTRUM — SIGNAL
# ============================================================================

def get_chi_overlap(channel_idx1, line1, channel_idx2, line2):
    """
    Find overlapping comoving distance range for two channel-line pairs.

    Returns the center and width of the overlap region in comoving distance.

    Parameters
    ----------
    channel_idx1, channel_idx2 : int
        Channel indices
    line1, line2 : str
        Emission line names

    Returns
    -------
    chi_overlap : float or None
        Center of overlapping region in Mpc (None if no overlap)
    delta_chi_overlap : float or None
        Width of overlapping region in Mpc (None if no overlap)
    """
    # Get redshift ranges for each channel-line pair
    z_min1, z_max1 = get_channel_redshift_range(channel_idx1, line1)
    z_min2, z_max2 = get_channel_redshift_range(channel_idx2, line2)

    # Find overlap in redshift
    z_overlap_min = max(z_min1, z_min2)
    z_overlap_max = min(z_max1, z_max2)

    if z_overlap_max <= z_overlap_min:
        # No overlap
        return None, None

    # Convert to comoving distance
    chi_min = z_to_chi(z_overlap_min)
    chi_max = z_to_chi(z_overlap_max)

    chi_overlap = 0.5 * (chi_min + chi_max)
    delta_chi_overlap = chi_max - chi_min

    return chi_overlap, delta_chi_overlap


def compute_C_ell_signal_pair(ell, channel_idx1, line1, channel_idx2, line2):
    """
    Compute signal angular power spectrum for a single (ν,i) × (ν',i') pair.

    Following Cheng et al. (2024) Eq. 13 (Limber approximation):

        C_ℓ,νν',ii' = (ν/Δν)(ν'/Δν') × (Δχ_overlap / χ²_overlap) ×
                      A₀²(χ_overlap) × M_i(χ_overlap) × M_i'(χ_overlap) ×
                      P((ℓ + 0.5) / χ_overlap, z_overlap)

    Parameters
    ----------
    ell : float
        Multipole moment
    channel_idx1, channel_idx2 : int
        Channel indices
    line1, line2 : str
        Emission line names

    Returns
    -------
    C_ell : float
        Angular power spectrum value (MJy²/sr²)
    """
    # Find overlap region
    chi_overlap, delta_chi_overlap = get_chi_overlap(channel_idx1, line1,
                                                       channel_idx2, line2)

    if chi_overlap is None:
        # No overlap
        return 0.0

    # Get redshift at overlap center
    z_overlap = chi_to_z(chi_overlap)

    # (ν/Δν) factors for each channel
    lambda1 = CHANNEL_CENTERS[channel_idx1]
    delta_lambda1 = CHANNEL_WIDTHS[channel_idx1]
    nu_over_delta_nu_1 = lambda1 / delta_lambda1

    lambda2 = CHANNEL_CENTERS[channel_idx2]
    delta_lambda2 = CHANNEL_WIDTHS[channel_idx2]
    nu_over_delta_nu_2 = lambda2 / delta_lambda2

    # Geometric factor A₀²(χ)
    A0 = get_window_function_A0(chi_overlap)
    A0_squared = A0**2

    # Bias-weighted luminosity densities M_i = b_i × M₀_i
    M_i_1 = get_bias_weighted_luminosity_density(z_overlap, line=line1)
    M_i_2 = get_bias_weighted_luminosity_density(z_overlap, line=line2)

    # Matter power spectrum at k = (ℓ + 0.5) / χ
    k = (ell + 0.5) / chi_overlap  # h/Mpc
    P_k = get_power_spectrum(k, z_overlap)  # (Mpc/h)³

    # Combine terms
    C_ell = (nu_over_delta_nu_1 * nu_over_delta_nu_2 *
             (delta_chi_overlap / chi_overlap**2) *
             A0_squared * M_i_1 * M_i_2 * P_k)

    return C_ell


def compute_C_ell_signal_matrix(ell):
    """
    Compute full 64×64 signal angular power spectrum matrix at multipole ℓ.

    Sums over all line pairs (i, i') including auto (i=i') and cross (i≠i').

    Parameters
    ----------
    ell : float
        Multipole moment

    Returns
    -------
    C_ell_matrix : ndarray, shape (64, 64)
        Signal angular power spectrum matrix (MJy²/sr²)
    """
    C_ell_matrix = np.zeros((N_CHANNELS, N_CHANNELS))

    # Loop over all channel pairs
    for nu1 in range(N_CHANNELS):
        for nu2 in range(nu1, N_CHANNELS):  # Use symmetry
            # Sum over all line pairs
            C_ell_sum = 0.0
            for line1 in EMISSION_LINES:
                for line2 in EMISSION_LINES:
                    C_ell_sum += compute_C_ell_signal_pair(ell, nu1, line1,
                                                            nu2, line2)

            # Fill matrix (symmetric)
            C_ell_matrix[nu1, nu2] = C_ell_sum
            if nu1 != nu2:
                C_ell_matrix[nu2, nu1] = C_ell_sum

    return C_ell_matrix


# ============================================================================
# ANGULAR POWER SPECTRUM — NOISE
# ============================================================================

def compute_C_ell_noise_matrix(survey_mode='deep'):
    """
    Compute noise angular power spectrum matrix.

    Following Cheng et al. (2024) Eq. 14:
        C^n_ℓ,νν' = σ_n²(ν) × Ω_pix × δ_K(ν,ν')

    Noise is diagonal in frequency (no cross-channel correlations).

    Parameters
    ----------
    survey_mode : str, optional
        SPHEREx survey mode: 'deep' (default) or 'full'

    Returns
    -------
    C_n_matrix : ndarray, shape (64, 64)
        Noise angular power spectrum matrix (MJy²/sr²)
    """
    C_n_matrix = np.zeros((N_CHANNELS, N_CHANNELS))

    # Fill diagonal elements
    for nu in range(N_CHANNELS):
        lambda_obs = CHANNEL_CENTERS[nu]
        # Get noise in MJy/sr
        sigma_n = get_spherex_noise_at_wavelength(lambda_obs, survey_mode=survey_mode)
        # Noise power: σ²(ν) × Ω_pix
        C_n_matrix[nu, nu] = sigma_n**2 * OMEGA_PIX

    return C_n_matrix


# ============================================================================
# TOTAL ANGULAR POWER SPECTRUM
# ============================================================================

def compute_C_ell_total_matrix(ell, survey_mode='deep'):
    """
    Compute total angular power spectrum matrix: signal + noise.

    Parameters
    ----------
    ell : float
        Multipole moment
    survey_mode : str, optional
        SPHEREx survey mode: 'deep' (default) or 'full'

    Returns
    -------
    C_ell_total : ndarray, shape (64, 64)
        Total angular power spectrum matrix (MJy²/sr²)
    """
    C_ell_signal = compute_C_ell_signal_matrix(ell)
    C_ell_noise = compute_C_ell_noise_matrix(survey_mode=survey_mode)
    C_ell_total = C_ell_signal + C_ell_noise

    return C_ell_total


# ============================================================================
# HIGH-LEVEL INTERFACE
# ============================================================================

def compute_all_ell_bins(survey_mode='deep', verbose=True):
    """
    Compute C_ℓ matrices for all 8 multipole bins.

    Parameters
    ----------
    survey_mode : str, optional
        SPHEREx survey mode: 'deep' (default) or 'full'
    verbose : bool, optional
        Print progress if True

    Returns
    -------
    results : dict
        Dictionary containing:
        - 'ell_centers': array of ℓ bin centers
        - 'ell_edges': array of ℓ bin edges
        - 'C_ell_signal': list of 64×64 signal matrices
        - 'C_ell_noise': 64×64 noise matrix (ℓ-independent)
        - 'C_ell_total': list of 64×64 total matrices
    """
    if verbose:
        print("Computing angular power spectra for 8 ℓ bins...")

    C_ell_signal_list = []
    C_ell_total_list = []

    # Compute noise once (ℓ-independent)
    C_ell_noise = compute_C_ell_noise_matrix(survey_mode=survey_mode)

    # Compute signal for each ℓ bin
    for i, ell_center in enumerate(ELL_BIN_CENTERS):
        if verbose:
            print(f"  ℓ bin {i+1}/8: ℓ = {ell_center:.1f}")

        C_signal = compute_C_ell_signal_matrix(ell_center)
        C_total = C_signal + C_ell_noise

        C_ell_signal_list.append(C_signal)
        C_ell_total_list.append(C_total)

    results = {
        'ell_centers': ELL_BIN_CENTERS,
        'ell_edges': ELL_BIN_EDGES,
        'C_ell_signal': C_ell_signal_list,
        'C_ell_noise': C_ell_noise,
        'C_ell_total': C_ell_total_list,
        'survey_mode': survey_mode
    }

    if verbose:
        print("Done!")

    return results
