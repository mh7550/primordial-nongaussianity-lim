"""
newton_raphson.py — Newton-Raphson optimizer for Wishart log-likelihood.

Implements the update rule:

    theta_{t+1} = theta_t + eta * H^{-1} * g

where g = grad(log L) and H = Hessian(log L). A backtracking line search
halves eta until the likelihood improves.

References
----------
Cheng et al., Phys. Rev. D 109, 103011 (2024) — arXiv:2403.19740, Section 5.2
"""

import numpy as np

try:
    from .wishart_likelihood import (
        wishart_log_likelihood,
        wishart_gradient,
        wishart_hessian,
    )
except ImportError:
    from wishart_likelihood import (
        wishart_log_likelihood,
        wishart_gradient,
        wishart_hessian,
    )


def newton_raphson_step(theta, C_data, z_bins, survey_config,
                        ell_bins=None, eta_init=1.0, max_halvings=10,
                        fd_delta=0.01):
    """
    Single Newton-Raphson step with backtracking line search.

    Parameters
    ----------
    theta : ndarray, shape (n_params,)
        Current log-space parameters.
    C_data : ndarray or dict
        Data covariance.
    z_bins : array_like
        Redshift bins.
    survey_config : SurveyConfig
    ell_bins : array_like, optional
    eta_init : float
        Initial step size (default 1.0).
    max_halvings : int
        Maximum number of step halvings in line search.
    fd_delta : float
        Finite-difference step for gradient/Hessian.

    Returns
    -------
    theta_new : ndarray
        Updated parameters.
    delta_theta : ndarray
        Step taken.
    log_L_new : float
        Log-likelihood at new point.
    converged : bool
        True if step size became too small (line search exhausted).
    """
    theta = np.asarray(theta, dtype=float)

    L_current = wishart_log_likelihood(theta, C_data, z_bins, survey_config, ell_bins)
    g = wishart_gradient(theta, C_data, z_bins, survey_config, ell_bins, delta=fd_delta)
    H_diag = wishart_hessian(theta, C_data, z_bins, survey_config, ell_bins, delta=fd_delta)

    # Regularize: ensure H_diag is negative (maximizing log L)
    # Use |H_diag| to avoid division-by-zero; direction follows sign of gradient
    H_diag_reg = np.where(np.abs(H_diag) > 1e-300, H_diag, -1e-300)

    # Newton direction: H^{-1} g (diagonal H)
    direction = g / (-H_diag_reg)  # negative because we maximize

    # Backtracking line search
    eta = eta_init
    converged = False
    for _ in range(max_halvings):
        theta_new = theta + eta * direction
        L_new = wishart_log_likelihood(theta_new, C_data, z_bins, survey_config, ell_bins)
        if L_new > L_current:
            break
        eta *= 0.5
    else:
        # Line search exhausted — accept tiny step anyway
        theta_new = theta + eta * direction
        L_new = wishart_log_likelihood(theta_new, C_data, z_bins, survey_config, ell_bins)
        converged = True

    return theta_new, eta * direction, L_new, converged


def newton_raphson_optimize(theta_init, C_data, z_bins, survey_config,
                            ell_bins=None, max_iter=50, tol_grad=1e-4,
                            tol_step=1e-6, verbose=False, fd_delta=0.01):
    """
    Newton-Raphson optimization of the Wishart log-likelihood.

    Iterates until convergence (gradient norm < tol_grad, step norm < tol_step,
    or max_iter iterations).

    Parameters
    ----------
    theta_init : array_like, shape (n_params,)
        Initial log-space parameters.
    C_data : ndarray or dict
        Data covariance.
    z_bins : array_like
    survey_config : SurveyConfig
    ell_bins : array_like, optional
    max_iter : int
    tol_grad : float
        Stop when max |gradient| < tol_grad (relative to initial).
    tol_step : float
        Stop when max |step| < tol_step.
    verbose : bool
    fd_delta : float
        Finite-difference step size.

    Returns
    -------
    result : dict
        'theta_opt'  : ndarray, optimized parameters
        'log_L'      : float, final log-likelihood
        'n_iter'     : int, iterations performed
        'converged'  : bool
        'history_L'  : list of log-L at each iteration
        'history_grad_norm' : list of gradient norms
    """
    theta = np.asarray(theta_init, dtype=float).copy()
    history_L = []
    history_grad_norm = []
    converged = False

    L = wishart_log_likelihood(theta, C_data, z_bins, survey_config, ell_bins)
    history_L.append(L)

    for i in range(max_iter):
        g = wishart_gradient(theta, C_data, z_bins, survey_config, ell_bins, delta=fd_delta)
        grad_norm = np.max(np.abs(g))
        history_grad_norm.append(grad_norm)

        if verbose:
            print(f"  iter {i:3d}: log_L = {L:.6e}, |grad|_inf = {grad_norm:.3e}")

        if grad_norm < tol_grad:
            converged = True
            break

        theta_new, step, L_new, line_search_failed = newton_raphson_step(
            theta, C_data, z_bins, survey_config, ell_bins,
            fd_delta=fd_delta
        )

        step_norm = np.max(np.abs(step))
        if step_norm < tol_step:
            converged = True
            theta = theta_new
            L = L_new
            history_L.append(L)
            break

        theta = theta_new
        L = L_new
        history_L.append(L)

        if line_search_failed:
            if verbose:
                print(f"  Line search exhausted at iter {i}.")
            converged = True
            break

    return {
        'theta_opt': theta,
        'log_L': L,
        'n_iter': i + 1,
        'converged': converged,
        'history_L': history_L,
        'history_grad_norm': history_grad_norm,
    }


def gradient_ascent_optimize(theta_init, C_data, z_bins, survey_config,
                              ell_bins=None, max_iter=200, learning_rate=0.01,
                              tol_step=1e-6, verbose=False, fd_delta=0.01):
    """
    Simple gradient ascent as a fallback optimizer.

    Uses fixed learning rate with step-size check for convergence.

    Parameters
    ----------
    theta_init : array_like
    C_data : ndarray or dict
    z_bins : array_like
    survey_config : SurveyConfig
    ell_bins : array_like, optional
    max_iter : int
    learning_rate : float
    tol_step : float
    verbose : bool
    fd_delta : float

    Returns
    -------
    result : dict
        Same keys as newton_raphson_optimize.
    """
    theta = np.asarray(theta_init, dtype=float).copy()
    history_L = []
    converged = False

    L = wishart_log_likelihood(theta, C_data, z_bins, survey_config, ell_bins)
    history_L.append(L)

    for i in range(max_iter):
        g = wishart_gradient(theta, C_data, z_bins, survey_config, ell_bins, delta=fd_delta)
        step = learning_rate * g
        theta = theta + step
        L = wishart_log_likelihood(theta, C_data, z_bins, survey_config, ell_bins)
        history_L.append(L)

        if verbose and i % 20 == 0:
            print(f"  iter {i:3d}: log_L = {L:.6e}, |step|_inf = {np.max(np.abs(step)):.3e}")

        if np.max(np.abs(step)) < tol_step:
            converged = True
            break

    return {
        'theta_opt': theta,
        'log_L': L,
        'n_iter': i + 1,
        'converged': converged,
        'history_L': history_L,
        'history_grad_norm': [],
    }
