"""
test_phase4_inference.py — 10 validation tests for the Phase 4 Bayesian inference framework.

Tests cover:
 1. ReLU basis function values and Jacobian structure
 2. Round-trip c <-> m <-> theta conversion
 3. Jacobian invertibility and identity recovery
 4. Fiducial M_i(z) peaks at cosmic noon (z ~ 2)
 5. Wishart log-likelihood is finite and changes with parameters
 6. Wishart gradient is nonzero at non-fiducial parameters
 7. Newton-Raphson step increases log-likelihood
 8. Fisher matrix is positive semi-definite
 9. 1-sigma constraints are positive for constrained parameters
10. Constrained parameter S/N > 1 (signal detectable with fiducial parameters)
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from basis_functions import (
    relu_basis, evaluate_Mi, jacobian_cm, cij_from_mij, mij_from_cij,
    theta_from_mij, mij_from_theta, FIDUCIAL_MIJ, get_fiducial_theta,
    Z_ANCHORS, Z_EVAL, N_M, EMISSION_LINES,
)
from wishart_likelihood import (
    wishart_log_likelihood, wishart_gradient, make_fiducial_data_covariance,
    build_signal_covariance, build_total_covariance,
)
from newton_raphson import newton_raphson_step
from fisher_posterior import (
    compute_fisher_matrix, compute_parameter_constraints,
    compute_snr_per_parameter,
)
from survey_configs import SurveyConfig

# Common fixtures
DEEP = SurveyConfig.get_config('deep_field')
Z_TEST = np.array([1.5, 2.0])
ELL_BINS = np.array([[50, 150], [150, 300]])
THETA_FID = get_fiducial_theta()


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: ReLU basis functions
# ─────────────────────────────────────────────────────────────────────────────

def test_relu_basis_values():
    """ReLU(z, z_anchor) = max(z - z_anchor, 0) for scalar and array inputs."""
    assert relu_basis(2.0, 1.0) == pytest.approx(1.0)
    assert relu_basis(0.5, 1.0) == pytest.approx(0.0)
    assert relu_basis(1.0, 1.0) == pytest.approx(0.0)

    z_arr = np.array([0.0, 1.0, 2.0, 3.0])
    result = relu_basis(z_arr, 1.0)
    expected = np.array([0.0, 0.0, 1.0, 2.0])
    np.testing.assert_allclose(result, expected)


def test_jacobian_structure():
    """J^cm is (7x7), lower-triangular with integer entries, det != 0."""
    J = jacobian_cm()
    assert J.shape == (N_M, N_M)
    # Upper triangle (excluding diagonal) must be zero
    assert np.allclose(np.triu(J, 1), 0.0)
    # All entries are non-negative integers
    assert np.all(J >= 0)
    assert np.allclose(J, np.round(J))
    # Must be invertible
    assert abs(np.linalg.det(J)) > 0.1


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: Round-trip conversions
# ─────────────────────────────────────────────────────────────────────────────

def test_roundtrip_c_m():
    """c -> m -> c round-trip is exact."""
    c_test = np.array([1.0, -0.5, 0.3, 0.1, -0.1, 0.05, 0.02]) * 1e40
    m = mij_from_cij(c_test)
    c_back = cij_from_mij(m)
    np.testing.assert_allclose(c_back, c_test, rtol=1e-10)


def test_roundtrip_theta_m():
    """theta -> m -> theta round-trip is exact (log-space encoding)."""
    m_test = np.array([1.0, 2.0, 3.0, 2.5, 2.0, 1.5, 1.0]) * 1e39
    theta = theta_from_mij(m_test)
    m_back = mij_from_theta(theta)
    np.testing.assert_allclose(m_back, m_test, rtol=1e-14)


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: Jacobian inverse identity
# ─────────────────────────────────────────────────────────────────────────────

def test_jacobian_inverse_identity():
    """J^cm @ (J^cm)^{-1} = I to machine precision."""
    J = jacobian_cm()
    J_inv = np.linalg.inv(J)
    product = J @ J_inv
    np.testing.assert_allclose(product, np.eye(N_M), atol=1e-10)


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: Fiducial peaks at cosmic noon
# ─────────────────────────────────────────────────────────────────────────────

def test_fiducial_peaks_cosmic_noon():
    """All 4 lines have fiducial M_i(z) peaking between z=1 and z=3."""
    for line in EMISSION_LINES:
        m_vals = FIDUCIAL_MIJ[line]
        # m_vals evaluated at Z_EVAL = [0,1,2,3,4,5,6]
        peak_idx = np.argmax(m_vals)
        # Peak should be between z=1 (idx=1) and z=3 (idx=3)
        assert 1 <= peak_idx <= 3, (
            f"{line}: peak at Z_EVAL[{peak_idx}]={Z_EVAL[peak_idx]}, expected z in [1,3]"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: Wishart log-likelihood
# ─────────────────────────────────────────────────────────────────────────────

def test_wishart_likelihood_finite():
    """Wishart log-likelihood is finite at fiducial parameters."""
    C_data = {z: make_fiducial_data_covariance(z, DEEP) for z in Z_TEST}
    log_L = wishart_log_likelihood(THETA_FID, C_data, Z_TEST, DEEP, ELL_BINS)
    assert np.isfinite(log_L), f"log_L = {log_L}"


def test_wishart_likelihood_changes_with_theta():
    """Perturbing a node that overlaps z_bins changes the likelihood."""
    C_data = {z: make_fiducial_data_covariance(z, DEEP) for z in Z_TEST}
    L_fid = wishart_log_likelihood(THETA_FID, C_data, Z_TEST, DEEP, ELL_BINS)

    # theta[2] = log(m_{Halpha,2}) = log(M_Halpha(z=2)), which affects z=1.5 and z=2.0
    theta_pert = THETA_FID.copy()
    theta_pert[2] += 0.5
    L_pert = wishart_log_likelihood(theta_pert, C_data, Z_TEST, DEEP, ELL_BINS)
    assert L_pert != L_fid, "Likelihood unchanged after parameter perturbation"


# ─────────────────────────────────────────────────────────────────────────────
# Test 6: Wishart gradient
# ─────────────────────────────────────────────────────────────────────────────

def test_wishart_gradient_nonzero():
    """Gradient of log-L is nonzero at perturbed parameters."""
    C_data = {z: make_fiducial_data_covariance(z, DEEP) for z in Z_TEST}

    theta_pert = THETA_FID.copy()
    theta_pert[2] += 0.3

    grad = wishart_gradient(theta_pert, C_data, Z_TEST, DEEP, ELL_BINS, delta=0.05)
    assert np.any(grad != 0.0), "All gradient components are zero"
    assert np.all(np.isfinite(grad)), "Gradient contains non-finite values"


# ─────────────────────────────────────────────────────────────────────────────
# Test 7: Newton-Raphson step increases log-likelihood
# ─────────────────────────────────────────────────────────────────────────────

def test_nr_step_increases_logL():
    """A single Newton-Raphson step from a perturbed point increases log L."""
    C_data = {z: make_fiducial_data_covariance(z, DEEP) for z in Z_TEST}

    theta_pert = THETA_FID.copy()
    theta_pert[2] += 0.5

    L_before = wishart_log_likelihood(theta_pert, C_data, Z_TEST, DEEP, ELL_BINS)
    theta_new, step, L_after, _ = newton_raphson_step(
        theta_pert, C_data, Z_TEST, DEEP, ELL_BINS, fd_delta=0.05
    )
    assert L_after >= L_before, (
        f"NR step did not improve: L_before={L_before:.6e}, L_after={L_after:.6e}"
    )
    assert theta_new.shape == THETA_FID.shape


# ─────────────────────────────────────────────────────────────────────────────
# Test 8: Fisher matrix is positive semi-definite
# ─────────────────────────────────────────────────────────────────────────────

def test_fisher_matrix_psd():
    """Fisher information matrix has non-negative eigenvalues (PSD)."""
    F = compute_fisher_matrix(THETA_FID, Z_TEST, DEEP, ELL_BINS, fd_delta=0.05)
    assert F.shape == (28, 28)
    assert np.allclose(F, F.T, rtol=1e-5), "Fisher matrix not symmetric"

    eigenvalues = np.linalg.eigvalsh(F)
    # Allow small negative eigenvalues from numerical noise (< 1e-6 * max_eig)
    max_eig = np.max(np.abs(eigenvalues))
    assert np.all(eigenvalues >= -1e-6 * max_eig), (
        f"Fisher matrix has negative eigenvalues: {eigenvalues[eigenvalues < 0]}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 9: 1-sigma constraints are positive
# ─────────────────────────────────────────────────────────────────────────────

def test_posterior_constraints_positive():
    """Posterior 1-sigma constraints sigma_a > 0 for constrained parameters."""
    F = compute_fisher_matrix(THETA_FID, Z_TEST, DEEP, ELL_BINS, fd_delta=0.05)
    sigma, rho, Sigma = compute_parameter_constraints(F)

    # Check constrained parameters (non-negligible Fisher information)
    constrained = np.diag(F) > 1e-10 * np.max(np.diag(F))
    assert np.sum(constrained) >= 4, "Need at least 4 constrained parameters"
    assert np.all(sigma[constrained] > 0), "Some constrained sigma values are <= 0"

    # Correlation diagonal = 1 for constrained params
    np.testing.assert_allclose(np.diag(rho)[constrained], 1.0, atol=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# Test 10: Constrained parameter S/N > 1
# ─────────────────────────────────────────────────────────────────────────────

def test_snr_constrained_parameters():
    """Fiducial parameter values are detectable (S/N > 1) for constrained params."""
    F = compute_fisher_matrix(THETA_FID, Z_TEST, DEEP, ELL_BINS, fd_delta=0.05)
    snr = compute_snr_per_parameter(THETA_FID, F)

    # Constrained: F_aa > 1e-10 * max(F_aa) and snr is not NaN
    constrained = ~np.isnan(snr)
    # Further filter to well-constrained (not numerical artifacts)
    well_constrained = constrained & (snr < 1e10)

    assert np.sum(well_constrained) >= 4, (
        f"Need >= 4 well-constrained parameters; got {np.sum(well_constrained)}"
    )
    assert np.all(snr[well_constrained] > 1.0), (
        f"Some constrained parameters have S/N <= 1: {snr[well_constrained]}"
    )
