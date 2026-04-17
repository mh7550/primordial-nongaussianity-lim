"""
wishart_likelihood.py — Wishart log-likelihood and gradient for LIM inference.

Implements Eq. 28 from Cheng et al. (2024):

    log L = -1/2 * sum_ell n_ell * [Tr(C_d @ C_ell^{-1}) + log|C_ell| + N_nu * log(2pi)]

where C_ell is the (N_ch x N_ch) signal+noise covariance matrix evaluated at
parameters theta, C_d is the data covariance, and n_ell is the number of modes.

The gradient with respect to log-space parameters theta (= log m_ij) is
computed analytically via the chain rule through the basis parameterization.

References
----------
Cheng et al., Phys. Rev. D 109, 103011 (2024) — arXiv:2403.19740, Eqs. 28–30
"""

import numpy as np

try:
    from .basis_functions import (
        EMISSION_LINES, N_M, Z_EVAL, FIDUCIAL_MIJ,
        mij_from_theta, cij_from_mij, evaluate_Mi,
    )
    from .survey_configs import (
        N_CHANNELS, CHANNEL_CENTERS, CHANNEL_WIDTHS, CHANNEL_EDGES,
        SurveyConfig, compute_signal_power_spectrum,
    )
    from .lim_signal import LINE_PROPERTIES, get_halo_bias_simple
    from .cosmology import get_comoving_distance, get_hubble, get_power_spectrum, h
except ImportError:
    from basis_functions import (
        EMISSION_LINES, N_M, Z_EVAL, FIDUCIAL_MIJ,
        mij_from_theta, cij_from_mij, evaluate_Mi,
    )
    from survey_configs import (
        N_CHANNELS, CHANNEL_CENTERS, CHANNEL_WIDTHS, CHANNEL_EDGES,
        SurveyConfig, compute_signal_power_spectrum,
    )
    from lim_signal import LINE_PROPERTIES, get_halo_bias_simple
    from cosmology import get_comoving_distance, get_hubble, get_power_spectrum, h

_C_LIGHT = 299792.458  # km/s

# Number of spectral channels (N_nu in Eq. 28)
N_NU = N_CHANNELS


def _channel_index(line, z):
    """Return SPHEREx channel index for a line observed at redshift z."""
    lambda_obs = LINE_PROPERTIES[line]['lambda_rest'] * (1.0 + z)
    if lambda_obs < CHANNEL_EDGES[0] or lambda_obs > CHANNEL_EDGES[-1]:
        return None
    return int(np.argmin(np.abs(CHANNEL_CENTERS - lambda_obs)))


def build_signal_covariance(theta_flat, z, ell_center=100.0):
    """
    Build signal covariance matrix C_signal(z) in (N_ch x N_ch) channel space.

    Each line contributes a rank-1 outer product at its SPHEREx channel
    with amplitude C_ell = I_bw² × (H_h/c) / chi_h² × P_m × delta_z,
    where I_bw is rescaled by the ratio of the current m_ij to the fiducial.

    Parameters
    ----------
    theta_flat : array_like, shape (4 * N_m,)
        Log-space parameters for all 4 lines, ordering [Halpha, OIII, Hbeta, OII].
    z : float
        Redshift of the observed channel.
    ell_center : float
        Central multipole for Limber k.

    Returns
    -------
    C_signal : ndarray, shape (N_ch, N_ch)
    """
    theta_flat = np.asarray(theta_flat, dtype=float)
    C_signal = np.zeros((N_CHANNELS, N_CHANNELS))

    chi = get_comoving_distance(z)
    chi_h = max(chi * h, 1.0)
    H_z_h = get_hubble(z) * h
    k_limber = max((ell_center + 0.5) / chi_h, 1e-4)
    P_mat = get_power_spectrum(k_limber, z)

    for i_line, line in enumerate(EMISSION_LINES):
        i_chan = _channel_index(line, z)
        if i_chan is None:
            continue

        # Extract m_ij for this line and interpolate M_i(z)
        m_i = mij_from_theta(theta_flat[i_line * N_M: (i_line + 1) * N_M])
        c_i = cij_from_mij(m_i)
        M_i_z = float(evaluate_Mi(c_i, [z])[0])
        if M_i_z <= 0:
            continue

        # I_bw = M_i(z) rescaled to nW/m²/sr via fiducial intensity ratio
        # Use the Limber formula directly with M_i from parameterization
        delta_z = CHANNEL_WIDTHS[i_chan] / LINE_PROPERTIES[line]['lambda_rest']
        # I_bw in nW/m²/sr proportional to M_i; use compute_signal_power_spectrum
        # as baseline and rescale by (M_i / M_i_fid)^2
        M_i_fid = float(evaluate_Mi(
            cij_from_mij(FIDUCIAL_MIJ[line]), [z]
        )[0])
        C_fid = compute_signal_power_spectrum(z, line, ell_center)
        if M_i_fid > 0:
            C_ell = C_fid * (M_i_z / M_i_fid) ** 2
        else:
            C_ell = 0.0

        C_signal[i_chan, i_chan] += C_ell

    return C_signal


def build_total_covariance(theta_flat, z, survey_config, ell_center=100.0):
    """
    Total covariance C_total = C_signal + C_noise at redshift z.

    Parameters
    ----------
    theta_flat : array_like, shape (4 * N_m,)
    z : float
    survey_config : SurveyConfig
    ell_center : float

    Returns
    -------
    C_total : ndarray, shape (N_ch, N_ch)
    """
    C_sig = build_signal_covariance(theta_flat, z, ell_center)
    C_noise = np.diag(survey_config.C_n)
    return C_sig + C_noise


def wishart_log_likelihood(theta_flat, C_data, z_bins, survey_config,
                           ell_bins=None):
    """
    Wishart log-likelihood (Eq. 28).

    log L = -1/2 * sum_ell n_ell * [Tr(C_d @ C^{-1}) + log|C| + N_nu * log(2pi)]

    The sum runs over ell bins and z bins. At each z, C_total is evaluated
    at the ell_center of each bin.

    Parameters
    ----------
    theta_flat : array_like, shape (4 * N_m,)
        Log-space parameters.
    C_data : ndarray, shape (N_ch, N_ch) or dict {z: ndarray}
        Data covariance. If a single matrix, used at all z. If a dict,
        keyed by z value.
    z_bins : array_like
        Redshift values at which to evaluate the likelihood.
    survey_config : SurveyConfig
        Survey configuration (noise level, f_sky).
    ell_bins : array_like, shape (n_bins, 2), optional
        Multipole bins. Default: [[50, 150], [150, 300]].

    Returns
    -------
    log_L : float
    """
    if ell_bins is None:
        ell_bins = np.array([[50, 150], [150, 300]])

    log_L = 0.0
    const_term = N_NU * np.log(2.0 * np.pi)

    for z in z_bins:
        if isinstance(C_data, dict):
            C_d = C_data[z]
        else:
            C_d = C_data

        for ell_min, ell_max in ell_bins:
            ell_center = 0.5 * (ell_min + ell_max)
            n_ell = survey_config.n_ell(ell_min, ell_max)

            C_tot = build_total_covariance(
                theta_flat, z, survey_config, ell_center
            )

            # Use diagonal structure for efficiency
            diag_C = np.diag(C_tot)
            diag_Cd = np.diag(C_d) if C_d.ndim == 2 else C_d

            # Tr(C_d @ C^{-1}) = sum_i C_d_ii / C_ii (diagonal approx)
            trace_term = np.sum(diag_Cd / np.maximum(diag_C, 1e-300))
            log_det = np.sum(np.log(np.maximum(diag_C, 1e-300)))

            log_L += -0.5 * n_ell * (trace_term + log_det + const_term)

    return log_L


def wishart_gradient(theta_flat, C_data, z_bins, survey_config,
                     ell_bins=None, delta=0.01):
    """
    Numerical gradient of the Wishart log-likelihood w.r.t. theta_flat.

    Uses central finite differences: grad[i] = (L(theta+e_i*d) - L(theta-e_i*d)) / (2d)

    Parameters
    ----------
    theta_flat : array_like, shape (4 * N_m,)
    C_data : ndarray or dict
    z_bins : array_like
    survey_config : SurveyConfig
    ell_bins : array_like, optional
    delta : float
        Step size for finite differences.

    Returns
    -------
    grad : ndarray, shape (4 * N_m,)
    """
    theta_flat = np.asarray(theta_flat, dtype=float)
    n_params = len(theta_flat)
    grad = np.zeros(n_params)

    for i in range(n_params):
        theta_p = theta_flat.copy()
        theta_m = theta_flat.copy()
        theta_p[i] += delta
        theta_m[i] -= delta

        L_p = wishart_log_likelihood(theta_p, C_data, z_bins, survey_config, ell_bins)
        L_m = wishart_log_likelihood(theta_m, C_data, z_bins, survey_config, ell_bins)
        grad[i] = (L_p - L_m) / (2.0 * delta)

    return grad


def wishart_hessian(theta_flat, C_data, z_bins, survey_config,
                    ell_bins=None, delta=0.01):
    """
    Numerical Hessian of the Wishart log-likelihood.

    Diagonal approximation: H[i,i] = (L(+) - 2L(0) + L(-)) / delta²

    Parameters
    ----------
    theta_flat : array_like, shape (4 * N_m,)
    C_data : ndarray or dict
    z_bins : array_like
    survey_config : SurveyConfig
    ell_bins : array_like, optional
    delta : float

    Returns
    -------
    H_diag : ndarray, shape (4 * N_m,)
        Diagonal of the Hessian.
    """
    theta_flat = np.asarray(theta_flat, dtype=float)
    n_params = len(theta_flat)
    H_diag = np.zeros(n_params)
    L0 = wishart_log_likelihood(theta_flat, C_data, z_bins, survey_config, ell_bins)

    for i in range(n_params):
        theta_p = theta_flat.copy()
        theta_m = theta_flat.copy()
        theta_p[i] += delta
        theta_m[i] -= delta

        L_p = wishart_log_likelihood(theta_p, C_data, z_bins, survey_config, ell_bins)
        L_m = wishart_log_likelihood(theta_m, C_data, z_bins, survey_config, ell_bins)
        H_diag[i] = (L_p - 2.0 * L0 + L_m) / (delta ** 2)

    return H_diag


def make_fiducial_data_covariance(z, survey_config, ell_center=100.0):
    """
    Construct fiducial data covariance C_data at redshift z using FIDUCIAL_MIJ.

    Parameters
    ----------
    z : float
    survey_config : SurveyConfig
    ell_center : float

    Returns
    -------
    C_data : ndarray, shape (N_ch, N_ch)
    """
    try:
        from .basis_functions import get_fiducial_theta
    except ImportError:
        from basis_functions import get_fiducial_theta

    theta_fid = get_fiducial_theta()
    return build_total_covariance(theta_fid, z, survey_config, ell_center)
