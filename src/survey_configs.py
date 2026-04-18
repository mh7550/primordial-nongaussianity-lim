"""
survey_configs.py — SPHEREx survey configuration for deep-field vs all-sky comparison.

Implements the SurveyConfig class and Fisher matrix machinery for comparing
deep-field and all-sky survey strategies following Cheng et al. (2024) Section 3.

Key Trade-off:
--------------
Deep Field (200 deg²): f_sky = 0.0048, low noise, few modes
All-Sky (30,000 deg²): f_sky = 0.75, high noise (50× variance), many modes (~156×)

References
----------
Cheng et al., Phys. Rev. D 109, 103011 (2024) — Section 3, Figure 8
"""

import numpy as np

# Import from local modules
try:
    from .lim_signal import (
        get_line_luminosity_density,
        get_halo_bias_simple,
        get_sfrd,
        get_line_intensity,
        LINE_PROPERTIES
    )
    from .cosmology import get_hubble, get_power_spectrum, get_comoving_distance, h
except ImportError:
    from lim_signal import (
        get_line_luminosity_density,
        get_halo_bias_simple,
        get_sfrd,
        get_line_intensity,
        LINE_PROPERTIES
    )
    from cosmology import get_hubble, get_power_spectrum, get_comoving_distance, h


# SPHEREx 6-band channel configuration (92 channels total)
LAMBDA_BAND_EDGES = np.array([0.75, 1.10, 1.63, 2.42, 3.82, 4.42, 5.00])  # μm
SPECTRAL_RESOLUTION_R = np.array([41, 41, 41, 35, 110, 130])  # R per band

# Compute channel structure
_dlamband = np.diff(LAMBDA_BAND_EDGES)
_lamcen = 0.5 * (LAMBDA_BAND_EDGES[:-1] + LAMBDA_BAND_EDGES[1:])
_dlamchan = _lamcen / SPECTRAL_RESOLUTION_R
_nchan_per_band = np.floor(_dlamband / _dlamchan).astype(int)

N_CHANNELS = int(np.sum(_nchan_per_band))  # 92 channels

# Generate channel boundaries
_lamchan = np.array([LAMBDA_BAND_EDGES[0]])
for i in range(6):
    _lamchan_band = np.linspace(
        LAMBDA_BAND_EDGES[i] + _dlamband[i] / _nchan_per_band[i],
        LAMBDA_BAND_EDGES[i + 1],
        _nchan_per_band[i]
    )
    _lamchan = np.concatenate((_lamchan, _lamchan_band))

CHANNEL_EDGES = _lamchan
CHANNEL_CENTERS = 0.5 * (_lamchan[:-1] + _lamchan[1:])
CHANNEL_WIDTHS = _lamchan[1:] - _lamchan[:-1]

# SPHEREx pixel size
PIXEL_SIZE_ARCSEC = 6.2
PIXEL_SIZE_STERADIAN = (PIXEL_SIZE_ARCSEC / 206265.0)**2

# Emission lines
EMISSION_LINES = ['Halpha', 'OIII', 'Hbeta', 'OII']


class SurveyConfig:
    """
    SPHEREx survey configuration (deep-field or all-sky).

    Attributes
    ----------
    name : str
        'deep_field' or 'all_sky'
    f_sky : float
        Sky fraction (0.0048 for deep, 0.75 for all-sky)
    sigma_n : ndarray, shape (92,)
        Noise RMS per channel in MJy/sr
    omega_pix : float
        Pixel solid angle in steradians
    C_n : ndarray, shape (92,)
        Diagonal noise power spectrum: sigma_n² × omega_pix
    """

    def __init__(self, name, f_sky, sigma_n, omega_pix=None):
        self.name = name
        self.f_sky = f_sky
        self.sigma_n = np.asarray(sigma_n)
        self.omega_pix = omega_pix if omega_pix is not None else PIXEL_SIZE_STERADIAN
        self.C_n = self.sigma_n**2 * self.omega_pix

    def n_ell(self, ell_min, ell_max):
        """
        Number of modes in multipole bin [ell_min, ell_max].

        Returns f_sky × (ell_max² - ell_min²)
        """
        return self.f_sky * (ell_max**2 - ell_min**2)

    @staticmethod
    def get_config(name):
        """
        Factory method for standard configurations.

        Parameters
        ----------
        name : str
            'deep_field' or 'all_sky'

        Returns
        -------
        config : SurveyConfig
        """
        if name == 'deep_field':
            f_sky = 0.0048
            # Deep-field noise in nW/m²/sr (consistent with get_line_intensity units)
            sigma_n = np.full(N_CHANNELS, 0.018)
            return SurveyConfig('deep_field', f_sky, sigma_n)

        elif name == 'all_sky':
            f_sky = 0.75
            # All-sky noise variance is 50× higher than deep (sqrt(50)× in amplitude)
            sigma_n_deep = np.full(N_CHANNELS, 0.018)
            sigma_n = sigma_n_deep * np.sqrt(50.0)
            return SurveyConfig('all_sky', f_sky, sigma_n)

        else:
            raise ValueError(f"Unknown config: {name}")


def compute_signal_power_spectrum(z, line, ell_center=100.0):
    """
    Limber-approximated angular power spectrum C_ell for a line at redshift z.

    Uses the narrow-channel Limber formula:
        C_ell = I_bw² × (H_z_h/c) / chi_h² × P_m(k=ell/chi_h, z) × delta_z

    Parameters
    ----------
    z : float
        Redshift
    line : str
        Emission line name
    ell_center : float
        Central multipole for k_limber = ell/chi

    Returns
    -------
    C_signal : float
        Angular power spectrum amplitude in (nW/m²/sr)²
    """
    _c_light = 299792.458  # km/s

    I_bw = get_line_intensity(z, line=line, return_bias_weighted=True)  # nW/m²/sr
    chi = get_comoving_distance(z)       # Mpc
    chi_h = max(chi * h, 1.0)            # Mpc/h
    H_z = get_hubble(z)                  # km/s/Mpc
    H_z_h = H_z * h                      # km/s/(Mpc/h)

    k_limber = (ell_center + 0.5) / chi_h   # h/Mpc
    k_limber = max(k_limber, 1e-4)
    P_mat = get_power_spectrum(k_limber, z)  # (Mpc/h)³

    # Channel width in redshift at this line's observed wavelength
    lambda_rest = LINE_PROPERTIES[line]['lambda_rest']
    lambda_obs = lambda_rest * (1.0 + z)
    i_chan = int(np.argmin(np.abs(CHANNEL_CENTERS - lambda_obs)))
    delta_z = CHANNEL_WIDTHS[i_chan] / lambda_rest

    return I_bw**2 * (H_z_h / _c_light) / chi_h**2 * P_mat * delta_z


def compute_dCell_dMi(line, z_target, ell_bins, delta=0.02):
    """
    Compute derivative dC_ell/dM_i, localized to the single SPHEREx channel
    where this line appears at z_target.

    Parameters
    ----------
    line : str
        Emission line name
    z_target : float
        Target redshift
    ell_bins : array_like, shape (n_bins, 2)
        Array of (ell_min, ell_max) pairs
    delta : float
        Unused (kept for API compatibility; derivative is analytical)

    Returns
    -------
    dC_dMi : ndarray, shape (n_bins, N_CHANNELS, N_CHANNELS)
        Derivative of C_ell matrix (nonzero only at the target channel diagonal)
    M_i : float
        Bias-weighted luminosity density at z_target
    """
    n_bins = len(ell_bins)
    M_i = get_line_luminosity_density(z_target, line=line) * get_halo_bias_simple(z_target)
    dC_dMi = np.zeros((n_bins, N_CHANNELS, N_CHANNELS))

    # Identify the single channel where this line appears at z_target
    lambda_obs = LINE_PROPERTIES[line]['lambda_rest'] * (1.0 + z_target)
    if lambda_obs < CHANNEL_EDGES[0] or lambda_obs > CHANNEL_EDGES[-1]:
        return dC_dMi, M_i
    i_chan = int(np.argmin(np.abs(CHANNEL_CENTERS - lambda_obs)))

    # Analytical derivative: C_signal ∝ I_bw² ∝ M_i², so dC/dM_i = 2C/M_i
    if M_i > 0:
        for i_bin in range(n_bins):
            ell_center = 0.5 * (ell_bins[i_bin, 0] + ell_bins[i_bin, 1])
            C_sig = compute_signal_power_spectrum(z_target, line, ell_center)
            dC_dMi[i_bin, i_chan, i_chan] = 2.0 * C_sig / M_i

    return dC_dMi, M_i


def compute_fisher_matrix_diagonal(line, z_target, survey_config, ell_bins, delta=0.02):
    """
    Compute Fisher matrix diagonal F_ii(z) for bias-weighted luminosity.

    F_ii = (1/2) × sum_ell n_ell × (dC_ell/dM_i)² / (C_signal + C_n)²

    The perturbation is localized to the single SPHEREx channel where the line
    appears at z_target.  C_signal is computed via the Limber approximation.

    Parameters
    ----------
    line : str
        Emission line
    z_target : float
        Target redshift
    survey_config : SurveyConfig
        Survey configuration
    ell_bins : array_like, shape (n_bins, 2)
        Multipole bins
    delta : float
        Unused (kept for API compatibility)

    Returns
    -------
    F_ii : float
        Fisher matrix diagonal element
    M_i : float
        Bias-weighted luminosity density
    """
    M_i = get_line_luminosity_density(z_target, line=line) * get_halo_bias_simple(z_target)

    # Identify target channel
    lambda_obs = LINE_PROPERTIES[line]['lambda_rest'] * (1.0 + z_target)
    if lambda_obs < CHANNEL_EDGES[0] or lambda_obs > CHANNEL_EDGES[-1]:
        return 0.0, M_i
    i_chan = int(np.argmin(np.abs(CHANNEL_CENTERS - lambda_obs)))

    if M_i <= 0:
        return 0.0, M_i

    F_ii = 0.0
    for ell_min, ell_max in ell_bins:
        n_ell = survey_config.n_ell(ell_min, ell_max)
        ell_center = 0.5 * (ell_min + ell_max)

        P_sig = compute_signal_power_spectrum(z_target, line, ell_center)
        C_total = P_sig + survey_config.C_n[i_chan]

        if C_total > 0:
            # dC/dM_i = 2*P_sig/M_i  →  (dC/C)² = (2*P_sig/(M_i*C))²
            F_ii += 0.5 * n_ell * (2.0 * P_sig / (M_i * C_total))**2

    return F_ii, M_i


def compute_SNR_vs_redshift(survey_config, z_bins=None, ell_bins=None):
    """
    Compute S/N = M_i(z) × sqrt(F_ii(z)) for all lines vs redshift.

    Parameters
    ----------
    survey_config : SurveyConfig
        Survey configuration
    z_bins : array_like, optional
        Redshift bins (default: [0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
    ell_bins : array_like, optional
        Multipole bins (default: 2 bins for speed)

    Returns
    -------
    results : dict
        Dictionary with keys 'Halpha', 'OIII', 'Hbeta', 'OII'
        Each value is array of S/N vs z
    """
    if z_bins is None:
        z_bins = np.array([0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])

    if ell_bins is None:
        # Use 2 ell bins for speed
        ell_bins = np.array([[50, 150], [150, 300]])

    results = {}

    for line in EMISSION_LINES:
        SNR_vs_z = np.zeros(len(z_bins))

        for i, z in enumerate(z_bins):
            # Skip OII at z < 1
            if line == 'OII' and z < 1.0:
                SNR_vs_z[i] = 0.0
                continue

            F_ii, M_i = compute_fisher_matrix_diagonal(
                line, z, survey_config, ell_bins, delta=0.02
            )

            SNR_vs_z[i] = M_i * np.sqrt(F_ii) if F_ii > 0 else 0.0

        results[line] = SNR_vs_z

    return results


def compute_SNR_with_noise_scaling(line, z_target, alpha_values, ell_bins=None):
    """
    Compute S/N vs noise scaling factor alpha.

    Parameters
    ----------
    line : str
        Emission line
    z_target : float
        Target redshift
    alpha_values : array_like
        Noise variance scaling factors: sigma_n² / sigma_n_deep²
    ell_bins : array_like, optional
        Multipole bins

    Returns
    -------
    SNR_values : ndarray
        S/N at each alpha value
    """
    if ell_bins is None:
        ell_bins = np.array([[50, 150], [150, 300]])

    SNR_values = np.zeros(len(alpha_values))

    # Get baseline deep-field noise
    deep = SurveyConfig.get_config('deep_field')

    for i, alpha in enumerate(alpha_values):
        # Scale noise
        sigma_n_scaled = deep.sigma_n * np.sqrt(alpha)
        config_scaled = SurveyConfig(
            name=f'scaled_{alpha:.1f}',
            f_sky=deep.f_sky,  # Keep deep f_sky
            sigma_n=sigma_n_scaled
        )

        F_ii, M_i = compute_fisher_matrix_diagonal(
            line, z_target, config_scaled, ell_bins, delta=0.02
        )

        SNR_values[i] = M_i * np.sqrt(F_ii) if F_ii > 0 else 0.0

    return SNR_values


if __name__ == '__main__':
    print("survey_configs.py — SPHEREx survey configuration module")
    print(f"N_CHANNELS = {N_CHANNELS}")
    print(f"Channel range: {CHANNEL_CENTERS[0]:.3f} - {CHANNEL_CENTERS[-1]:.3f} μm")

    deep = SurveyConfig.get_config('deep_field')
    allsky = SurveyConfig.get_config('all_sky')

    print(f"\nDeep field: f_sky = {deep.f_sky}")
    print(f"All-sky: f_sky = {allsky.f_sky}")
    print(f"Noise ratio: {(allsky.sigma_n[0] / deep.sigma_n[0])**2:.1f}×")
