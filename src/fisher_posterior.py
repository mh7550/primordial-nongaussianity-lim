"""
fisher_posterior.py — Fisher posterior and parameter constraints for LIM inference.

Implements Eq. 31 from Cheng et al. (2024):

    F_ab = (1/2) * sum_ell n_ell * Tr[C^{-1} dC/dtheta_a * C^{-1} dC/dtheta_b]

where theta_ab are the log-space basis parameters and C is the total covariance.
The posterior covariance is Sigma = F^{-1}, giving 1-sigma constraints
sigma_a = sqrt(F^{-1}_{aa}).

References
----------
Cheng et al., Phys. Rev. D 109, 103011 (2024) — arXiv:2403.19740, Eq. 31
"""

import numpy as np

try:
    from .basis_functions import (
        EMISSION_LINES, N_M, Z_EVAL, FIDUCIAL_MIJ,
        get_fiducial_theta, mij_from_theta, cij_from_mij, evaluate_Mi,
    )
    from .wishart_likelihood import (
        build_total_covariance, build_signal_covariance,
    )
    from .survey_configs import (
        N_CHANNELS, SurveyConfig, compute_signal_power_spectrum,
        CHANNEL_CENTERS, LINE_PROPERTIES,
    )
except ImportError:
    from basis_functions import (
        EMISSION_LINES, N_M, Z_EVAL, FIDUCIAL_MIJ,
        get_fiducial_theta, mij_from_theta, cij_from_mij, evaluate_Mi,
    )
    from wishart_likelihood import (
        build_total_covariance, build_signal_covariance,
    )
    from survey_configs import (
        N_CHANNELS, SurveyConfig, compute_signal_power_spectrum,
        CHANNEL_CENTERS, LINE_PROPERTIES,
    )


def dC_dtheta(theta_flat, param_idx, z, survey_config,
              ell_center=100.0, delta=0.01):
    """
    Numerical derivative dC_total/dtheta_a at parameter index param_idx.

    Uses central finite differences in log-space parameter theta_a.

    Parameters
    ----------
    theta_flat : ndarray, shape (4*N_m,)
    param_idx : int
        Index of the parameter to differentiate.
    z : float
    survey_config : SurveyConfig
    ell_center : float
    delta : float
        Finite-difference step.

    Returns
    -------
    dC : ndarray, shape (N_ch, N_ch)
        Derivative matrix (only diagonal entries are nonzero).
    """
    theta_p = theta_flat.copy()
    theta_m = theta_flat.copy()
    theta_p[param_idx] += delta
    theta_m[param_idx] -= delta

    C_p = build_signal_covariance(theta_p, z, ell_center)
    C_m = build_signal_covariance(theta_m, z, ell_center)
    return (C_p - C_m) / (2.0 * delta)


def compute_fisher_matrix(theta_flat, z_bins, survey_config,
                          ell_bins=None, fd_delta=0.01):
    """
    Full (4*N_m) x (4*N_m) Fisher information matrix F_ab (Eq. 31).

    F_ab = (1/2) * sum_{ell, z} n_ell * Tr[C^{-1} dC/dtheta_a * C^{-1} dC/dtheta_b]

    For diagonal C this simplifies to:
        F_ab = (1/2) * sum_i (dC_ii/dtheta_a) * (dC_ii/dtheta_b) / C_ii^2

    Parameters
    ----------
    theta_flat : array_like, shape (4*N_m,)
        Log-space parameters at which to evaluate (typically fiducial).
    z_bins : array_like
        Redshift values.
    survey_config : SurveyConfig
    ell_bins : array_like, shape (n_bins, 2), optional
    fd_delta : float
        Finite-difference step for dC/dtheta.

    Returns
    -------
    F : ndarray, shape (4*N_m, 4*N_m)
        Fisher information matrix.
    """
    if ell_bins is None:
        ell_bins = np.array([[50, 150], [150, 300]])

    theta_flat = np.asarray(theta_flat, dtype=float)
    n_params = len(theta_flat)
    F = np.zeros((n_params, n_params))

    for z in z_bins:
        for ell_min, ell_max in ell_bins:
            ell_center = 0.5 * (ell_min + ell_max)
            n_ell = survey_config.n_ell(ell_min, ell_max)

            C_tot = build_total_covariance(theta_flat, z, survey_config, ell_center)
            C_diag = np.diag(C_tot)
            C_diag_sq = np.maximum(C_diag, 1e-300) ** 2

            # Precompute dC_ii/dtheta_a for all params (diagonal entries only)
            dC_diags = np.zeros((n_params, N_CHANNELS))
            for a in range(n_params):
                dC = dC_dtheta(theta_flat, a, z, survey_config, ell_center, delta=fd_delta)
                dC_diags[a] = np.diag(dC)

            # F_ab = (1/2) * n_ell * sum_i dC_ii/da * dC_ii/db / C_ii^2
            F += 0.5 * n_ell * np.dot(
                dC_diags / C_diag_sq,
                dC_diags.T
            )

    return F


def compute_posterior_covariance(F):
    """
    Posterior covariance Sigma = F^{-1}.

    Parameters
    ----------
    F : ndarray, shape (n_params, n_params)

    Returns
    -------
    Sigma : ndarray, shape (n_params, n_params)
        Posterior covariance (inverse Fisher matrix).
    """
    try:
        Sigma = np.linalg.inv(F)
    except np.linalg.LinAlgError:
        Sigma = np.linalg.pinv(F)
    return Sigma


def compute_parameter_constraints(F):
    """
    1-sigma marginalized constraints sigma_a = sqrt(Sigma_aa) = 1/sqrt(F_aa).

    Also returns the correlation matrix rho_ab = Sigma_ab / sqrt(Sigma_aa * Sigma_bb).

    Parameters
    ----------
    F : ndarray, shape (n_params, n_params)

    Returns
    -------
    sigma : ndarray, shape (n_params,)
        1-sigma constraints.
    rho : ndarray, shape (n_params, n_params)
        Correlation matrix.
    Sigma : ndarray, shape (n_params, n_params)
        Full posterior covariance.
    """
    Sigma = compute_posterior_covariance(F)
    sigma = np.sqrt(np.maximum(np.diag(Sigma), 0.0))

    # Correlation matrix
    outer_sigma = np.outer(sigma, sigma)
    with np.errstate(divide='ignore', invalid='ignore'):
        rho = np.where(outer_sigma > 0, Sigma / outer_sigma, 0.0)

    return sigma, rho, Sigma


def compute_snr_per_parameter(theta_flat, F, sigma_floor=1e-100):
    """
    Signal-to-noise ratio for each parameter: S/N_a = |theta_a| / sigma_a.

    Returns NaN for unconstrained parameters (sigma < sigma_floor).

    Parameters
    ----------
    theta_flat : ndarray, shape (n_params,)
    F : ndarray, shape (n_params, n_params)
    sigma_floor : float
        Threshold below which a parameter is considered unconstrained.

    Returns
    -------
    snr : ndarray, shape (n_params,)
        S/N values; NaN for unconstrained parameters.
    """
    sigma, _, _ = compute_parameter_constraints(F)
    constrained = sigma > sigma_floor
    snr = np.full(len(theta_flat), np.nan)
    snr[constrained] = np.abs(theta_flat[constrained]) / sigma[constrained]
    return snr


def summarize_constraints(theta_flat, F, label="Fisher constraints"):
    """
    Print a formatted summary of parameter constraints.

    Parameters
    ----------
    theta_flat : ndarray, shape (4*N_m,)
    F : ndarray, shape (4*N_m, 4*N_m)
    label : str
    """
    sigma, rho, Sigma = compute_parameter_constraints(F)
    n_params = len(theta_flat)

    print(f"\n{label}")
    print("=" * 60)
    print(f"{'Param':<12} {'theta_fid':>12} {'sigma':>12} {'S/N':>8}")
    print("-" * 60)

    for a in range(n_params):
        line_idx = a // N_M
        node_idx = a % N_M
        line = EMISSION_LINES[line_idx]
        snr = abs(theta_flat[a]) / max(sigma[a], 1e-300)
        print(f"{line}[{node_idx}]    {theta_flat[a]:>12.4f} {sigma[a]:>12.4f} {snr:>8.2f}")

    print("-" * 60)
    print(f"Median S/N: {np.median(np.abs(theta_flat)/np.maximum(sigma, 1e-300)):.2f}")
    print(f"Fraction S/N > 2: {np.mean(np.abs(theta_flat)/np.maximum(sigma, 1e-300) > 2):.2f}")


def compute_line_constraints(F, theta_flat=None):
    """
    Aggregate Fisher constraints per emission line.

    Returns the joint S/N and typical constraint for each of the 4 lines.

    Parameters
    ----------
    F : ndarray, shape (4*N_m, 4*N_m)
    theta_flat : ndarray or None

    Returns
    -------
    line_results : dict
        Keys: line names. Values: dict with 'sigma_mean', 'snr_mean', 'F_sub'.
    """
    sigma, _, _ = compute_parameter_constraints(F)
    line_results = {}

    for i_line, line in enumerate(EMISSION_LINES):
        sl = slice(i_line * N_M, (i_line + 1) * N_M)
        F_sub = F[sl, sl]
        sigma_line = sigma[sl]
        line_results[line] = {
            'sigma': sigma_line,
            'sigma_mean': np.mean(sigma_line),
            'F_sub': F_sub,
        }
        if theta_flat is not None:
            theta_line = theta_flat[sl]
            constrained = sigma_line > 1e-100
            snr_line = np.full(N_M, np.nan)
            snr_line[constrained] = np.abs(theta_line[constrained]) / sigma_line[constrained]
            line_results[line]['snr'] = snr_line
            line_results[line]['snr_mean'] = np.nanmean(snr_line)

    return line_results
