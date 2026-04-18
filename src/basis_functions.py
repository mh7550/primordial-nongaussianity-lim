"""
basis_functions.py — Piecewise-linear (ReLU) parameterization of M_i(z).

Implements the basis decomposition from Cheng et al. (2024) Section 5.1:

    M_i(z) = sum_{j=1}^{N_m} c_ij * M_hat_j(z)

where M_hat_j(z) = max(z - z_j, 0) are ReLU basis functions with
anchoring redshifts {z_j} = {-1, 0, 1, 2, 3, 4, 5}, N_m = 7.

The log-space parameterization theta_ij = log(m_ij) enforces positivity,
where m_ij = M_i(z_{j+1}) are the basis node values at z = {0,1,2,3,4,5,6}.

References
----------
Cheng et al., Phys. Rev. D 109, 103011 (2024) — arXiv:2403.19740, Section 5.1
"""

import numpy as np

try:
    from .lim_signal import (
        get_line_luminosity_density,
        get_halo_bias_simple,
        LINE_PROPERTIES,
    )
except ImportError:
    from lim_signal import (
        get_line_luminosity_density,
        get_halo_bias_simple,
        LINE_PROPERTIES,
    )


# Anchoring redshifts (N_m = 7)
Z_ANCHORS = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
N_M = len(Z_ANCHORS)  # 7

# Evaluation points: z_{j+1} for j=0..N_m-1
Z_EVAL = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

EMISSION_LINES = ['Halpha', 'OIII', 'Hbeta', 'OII']


def relu_basis(z, z_anchor):
    """
    Single ReLU basis function M_hat_j(z) = max(z - z_anchor, 0).

    Parameters
    ----------
    z : float or ndarray
    z_anchor : float

    Returns
    -------
    float or ndarray
    """
    return np.maximum(np.asarray(z, dtype=float) - z_anchor, 0.0)


def evaluate_Mi(c_i, z_values):
    """
    Evaluate M_i(z) = sum_j c_ij * M_hat_j(z) at arbitrary redshifts.

    Parameters
    ----------
    c_i : array_like, shape (N_m,)
        Basis coefficients for one emission line.
    z_values : array_like
        Redshifts at which to evaluate.

    Returns
    -------
    M_i : ndarray, shape (len(z_values),)
        Bias-weighted luminosity density in units of c_i (typically erg/s/Mpc³).
    """
    c_i = np.asarray(c_i, dtype=float)
    z_values = np.atleast_1d(np.asarray(z_values, dtype=float))
    M_i = np.zeros(len(z_values))
    for j, z_j in enumerate(Z_ANCHORS):
        M_i += c_i[j] * relu_basis(z_values, z_j)
    return M_i


def jacobian_cm():
    """
    Jacobian matrix J^cm mapping basis coefficients c_i to node values m_i.

    J[j', j] = M_hat_j(z_{j'+1}) = max(z_{j'+1} - z_j, 0)

    where z_{j'+1} are Z_EVAL = {0,1,2,3,4,5,6} (rows)
    and   z_j      are Z_ANCHORS = {-1,0,1,2,3,4,5} (columns).

    Returns
    -------
    J : ndarray, shape (N_m, N_m)
        Lower-triangular matrix with integer entries.
    """
    J = np.zeros((N_M, N_M))
    for j_prime in range(N_M):
        for j in range(N_M):
            J[j_prime, j] = relu_basis(Z_EVAL[j_prime], Z_ANCHORS[j])
    return J


# Pre-compute J^cm and its inverse once at import time
_J_CM = jacobian_cm()
_J_CM_INV = np.linalg.inv(_J_CM)


def mij_from_cij(c_i):
    """
    Convert basis coefficients to node values: m = J^cm @ c.

    m_ij = M_i(z_{j+1}) = sum_j c_ij * M_hat_j(z_{j+1})

    Parameters
    ----------
    c_i : array_like, shape (N_m,)

    Returns
    -------
    m_i : ndarray, shape (N_m,)
        Values of M_i at Z_EVAL.
    """
    return _J_CM @ np.asarray(c_i, dtype=float)


def cij_from_mij(m_i):
    """
    Convert node values to basis coefficients: c = (J^cm)^{-1} @ m.

    Parameters
    ----------
    m_i : array_like, shape (N_m,)
        Values of M_i at Z_EVAL = {0,1,2,3,4,5,6}.

    Returns
    -------
    c_i : ndarray, shape (N_m,)
        ReLU basis coefficients.
    """
    return _J_CM_INV @ np.asarray(m_i, dtype=float)


def theta_from_mij(m_i):
    """
    Log-space encoding: theta_ij = log(m_ij).

    Parameters
    ----------
    m_i : array_like, shape (N_m,)
        Positive node values.

    Returns
    -------
    theta_i : ndarray, shape (N_m,)
    """
    m_i = np.asarray(m_i, dtype=float)
    if np.any(m_i <= 0):
        raise ValueError("All m_ij must be positive for log-space encoding.")
    return np.log(m_i)


def mij_from_theta(theta_i):
    """
    Inverse log-space encoding: m_ij = exp(theta_ij).

    Parameters
    ----------
    theta_i : array_like, shape (N_m,)

    Returns
    -------
    m_i : ndarray, shape (N_m,)
    """
    return np.exp(np.asarray(theta_i, dtype=float))


def _compute_fiducial_mij():
    """Compute fiducial m_ij = b(z) × rho_L(z) at Z_EVAL for all 4 lines."""
    fiducial = {}
    for line in EMISSION_LINES:
        m_vals = np.zeros(N_M)
        for k, z in enumerate(Z_EVAL):
            if z == 0.0:
                # Avoid z=0 singularities; use z=0.01
                z_eval = 0.01
            else:
                z_eval = z
            rho_L = get_line_luminosity_density(z_eval, line=line)
            b = get_halo_bias_simple(z_eval)
            m_vals[k] = rho_L * b
        fiducial[line] = m_vals
    return fiducial


# Fiducial node values M_i(z_{j+1}) in erg/s/Mpc³
FIDUCIAL_MIJ = _compute_fiducial_mij()


def get_fiducial_theta():
    """
    Return flattened log-space fiducial parameters for all 4 lines.

    Returns
    -------
    theta_fid : ndarray, shape (4 * N_m,)
        Ordering: [Halpha_0..6, OIII_0..6, Hbeta_0..6, OII_0..6]
    """
    parts = []
    for line in EMISSION_LINES:
        parts.append(theta_from_mij(FIDUCIAL_MIJ[line]))
    return np.concatenate(parts)


def get_fiducial_cij():
    """
    Return fiducial ReLU coefficients for all 4 lines.

    Returns
    -------
    c_fid : dict
        Keys are line names; values are ndarray of shape (N_m,).
    """
    return {line: cij_from_mij(FIDUCIAL_MIJ[line]) for line in EMISSION_LINES}
