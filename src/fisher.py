"""
Fisher matrix forecasting for primordial non-Gaussianity constraints.

This module implements the Fisher matrix formalism to forecast constraints
on f_NL parameters from angular power spectrum measurements.

References
----------
Tegmark et al., PRD 55, 5895 (1997) - Fisher matrix for cosmology
Sefusatti & Komatsu, PRD 76, 083004 (2007) - Fisher for PNG
LoVerde & Afshordi, PRD 78, 123506 (2008) - Multi-tracer PNG
"""

import numpy as np
from scipy import linalg

# Import from our modules
try:
    from .limber import (get_angular_power_spectrum, get_cross_power_spectrum,
                         compute_dCl_dfNL_cross, compute_dCl_dfNL_auto,
                         get_comoving_distance, get_hubble)
    from .survey_specs import (get_noise_power_spectrum_simple, F_SKY,
                               get_bias, get_number_density, SPHEREX_Z_BINS,
                               N_SAMPLES, N_Z_BINS, get_shot_noise_angular)
except ImportError:
    from limber import (get_angular_power_spectrum, get_cross_power_spectrum,
                        compute_dCl_dfNL_cross, compute_dCl_dfNL_auto,
                        get_comoving_distance, get_hubble)
    from survey_specs import (get_noise_power_spectrum_simple, F_SKY,
                              get_bias, get_number_density, SPHEREX_Z_BINS,
                              N_SAMPLES, N_Z_BINS, get_shot_noise_angular)


def compute_dCl_dfNL(ell, z_min, z_max, b1, fNL_fid, shape='local', delta_fNL=0.1):
    """
    Compute derivative ∂C_ℓ/∂f_NL numerically using finite differences.

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
    fNL_fid : float
        Fiducial f_NL value (usually 0)
    shape : str, optional
        PNG shape: 'local', 'equilateral', 'orthogonal'
    delta_fNL : float, optional
        Step size for numerical derivative (default: 0.1)

    Returns
    -------
    dCl_dfNL : float or array_like
        Derivative of C_ℓ with respect to f_NL

    Notes
    -----
    Uses centered finite difference:
    ∂C_ℓ/∂f_NL ≈ [C_ℓ(f_NL + δ) - C_ℓ(f_NL - δ)] / (2δ)

    For local PNG, since Δb ∝ f_NL, the derivative can also be computed
    analytically, but numerical differentiation is more general.
    """
    # C_ℓ at fNL_fid + delta_fNL
    C_ell_plus = get_angular_power_spectrum(
        ell, z_min, z_max, b1, fNL=fNL_fid + delta_fNL, shape=shape
    )

    # C_ℓ at fNL_fid - delta_fNL
    C_ell_minus = get_angular_power_spectrum(
        ell, z_min, z_max, b1, fNL=fNL_fid - delta_fNL, shape=shape
    )

    # Centered difference
    dCl_dfNL = (C_ell_plus - C_ell_minus) / (2.0 * delta_fNL)

    return dCl_dfNL


def compute_fisher_element(ell, z_min, z_max, b1, fNL_fid,
                          shape_i, shape_j, f_sky=F_SKY,
                          survey_mode='full', delta_fNL=0.1):
    """
    Compute a single Fisher matrix element F_ij.

    Parameters
    ----------
    ell : array_like
        Array of multipole moments
    z_min : float
        Minimum redshift
    z_max : float
        Maximum redshift
    b1 : float
        Linear bias
    fNL_fid : float
        Fiducial f_NL
    shape_i : str
        PNG shape for parameter i
    shape_j : str
        PNG shape for parameter j
    f_sky : float, optional
        Sky fraction (default: 0.75 for SPHEREx)
    survey_mode : str, optional
        'full' or 'deep' survey mode
    delta_fNL : float, optional
        Step size for derivatives

    Returns
    -------
    F_ij : float
        Fisher matrix element

    Notes
    -----
    The Fisher matrix element is:

    F_ij = Σ_ℓ (2ℓ + 1) f_sky / 2 × [∂C_ℓ/∂p_i] [∂C_ℓ/∂p_j] / [C_ℓ + N_ℓ]²

    The (2ℓ + 1) factor counts the number of independent m modes at
    each ℓ, and f_sky accounts for partial sky coverage.
    """
    # Get fiducial C_ℓ
    C_ell_fid = get_angular_power_spectrum(
        ell, z_min, z_max, b1, fNL=fNL_fid, shape='local'
    )

    # Get noise N_ℓ
    z_mid = (z_min + z_max) / 2.0
    N_ell = get_noise_power_spectrum_simple(ell, z_mid, survey_mode=survey_mode)

    # Compute derivatives
    dCl_dpi = compute_dCl_dfNL(ell, z_min, z_max, b1, fNL_fid,
                                shape=shape_i, delta_fNL=delta_fNL)
    dCl_dpj = compute_dCl_dfNL(ell, z_min, z_max, b1, fNL_fid,
                                shape=shape_j, delta_fNL=delta_fNL)

    # Fisher matrix element
    # F_ij = Σ_ℓ (2ℓ+1) f_sky/2 × [∂C_ℓ/∂p_i][∂C_ℓ/∂p_j] / [C_ℓ + N_ℓ]²
    F_ij = 0.0
    for i, ell_val in enumerate(ell):
        # Total covariance: signal + noise
        cov = C_ell_fid[i] + N_ell[i]

        if cov > 0:
            # Fisher contribution from this ℓ
            weight = (2.0 * ell_val + 1.0) * f_sky / 2.0
            F_ij += weight * dCl_dpi[i] * dCl_dpj[i] / cov**2

    return F_ij


def compute_fisher_matrix(ell_array, z_bins, params, b1_values=None,
                         fNL_fid=0.0, f_sky=F_SKY, survey_mode='full',
                         delta_fNL=0.1):
    """
    Compute full Fisher matrix for multiple parameters and redshift bins.

    Parameters
    ----------
    ell_array : array_like
        Array of multipole moments to sum over
    z_bins : list of tuples
        List of (z_min, z_max) redshift bins
    params : list of str
        List of parameter names: 'fNL_local', 'fNL_equilateral', 'fNL_orthogonal'
    b1_values : list of float, optional
        Linear bias for each redshift bin. If None, uses b1=2.0 for all bins.
    fNL_fid : float, optional
        Fiducial f_NL value (default: 0, assume Gaussian)
    f_sky : float, optional
        Sky fraction
    survey_mode : str, optional
        'full' or 'deep'
    delta_fNL : float, optional
        Step size for derivatives

    Returns
    -------
    fisher_matrix : ndarray
        Fisher matrix, shape (n_params, n_params)
    param_names : list
        List of parameter names

    Notes
    -----
    For multiple redshift bins, we sum the Fisher matrices:
    F_total = Σ_bins F_bin

    This assumes the bins are independent (no cross-correlations).
    For multi-tracer analysis, this should be extended to include
    cross-power spectra.
    """
    n_params = len(params)
    n_bins = len(z_bins)

    if b1_values is None:
        b1_values = [2.0] * n_bins

    # Map parameter names to shapes
    shape_map = {
        'fNL_local': 'local',
        'fNL_equilateral': 'equilateral',
        'fNL_orthogonal': 'orthogonal',
    }

    # Initialize Fisher matrix
    F = np.zeros((n_params, n_params))

    # Sum over redshift bins
    for bin_idx, (z_min, z_max) in enumerate(z_bins):
        b1 = b1_values[bin_idx]

        print(f"Computing Fisher for z bin [{z_min:.2f}, {z_max:.2f}], b1={b1:.2f}...")

        # Compute Fisher elements for this bin
        for i, param_i in enumerate(params):
            for j, param_j in enumerate(params):
                # Only compute upper triangle (Fisher is symmetric)
                if i <= j:
                    shape_i = shape_map.get(param_i, 'local')
                    shape_j = shape_map.get(param_j, 'local')

                    F_ij = compute_fisher_element(
                        ell_array, z_min, z_max, b1, fNL_fid,
                        shape_i, shape_j, f_sky, survey_mode, delta_fNL
                    )

                    F[i, j] += F_ij
                    if i != j:
                        F[j, i] += F_ij  # Symmetry

        print(f"  Done. Fisher diagonal: {np.diag(F)}")

    return F, params


def get_constraints(fisher_matrix, param_names):
    """
    Compute parameter constraints from Fisher matrix.

    Parameters
    ----------
    fisher_matrix : ndarray
        Fisher matrix
    param_names : list of str
        Parameter names

    Returns
    -------
    constraints : dict
        Dictionary with parameter names as keys and σ(param) as values

    Notes
    -----
    The parameter uncertainties are given by:
    σ(p_i) = sqrt((F^{-1})_ii)

    where F^{-1} is the inverse Fisher matrix (covariance matrix).

    For marginalized constraints, we use the diagonal elements of F^{-1}.
    For conditional constraints (fixing other parameters), we'd use 1/sqrt(F_ii).
    """
    # Check if Fisher matrix is singular
    try:
        # Invert Fisher matrix to get covariance
        cov_matrix = linalg.inv(fisher_matrix)

        # Extract 1σ uncertainties
        constraints = {}
        for i, param in enumerate(param_names):
            sigma = np.sqrt(cov_matrix[i, i])
            constraints[param] = sigma

        return constraints

    except linalg.LinAlgError:
        print("Warning: Fisher matrix is singular or nearly singular!")
        print("This may indicate insufficient information to constrain parameters.")

        # Return infinite uncertainties
        return {param: np.inf for param in param_names}


def compute_constraints_vs_ell_max(ell_max_array, z_bins, param, b1_values=None,
                                   ell_min=10, f_sky=F_SKY, survey_mode='full'):
    """
    Compute how constraints improve with increasing ℓ_max.

    This shows the information content as a function of scale.

    Parameters
    ----------
    ell_max_array : array_like
        Array of maximum ℓ values to test
    z_bins : list of tuples
        Redshift bins
    param : str
        Parameter name (e.g., 'fNL_local')
    b1_values : list of float, optional
        Linear biases
    ell_min : float, optional
        Minimum ℓ (default: 10)
    f_sky : float, optional
        Sky fraction
    survey_mode : str, optional
        Survey mode

    Returns
    -------
    sigma_array : array_like
        Array of σ(parameter) for each ℓ_max
    """
    sigma_array = np.zeros_like(ell_max_array, dtype=float)

    for i, ell_max in enumerate(ell_max_array):
        # Create ℓ array
        ell_array = np.logspace(np.log10(ell_min), np.log10(ell_max), 30)

        # Compute Fisher matrix
        F, param_names = compute_fisher_matrix(
            ell_array, z_bins, [param], b1_values=b1_values,
            f_sky=f_sky, survey_mode=survey_mode
        )

        # Get constraints
        constraints = get_constraints(F, param_names)
        sigma_array[i] = constraints[param]

        print(f"ℓ_max = {ell_max:.0f}: σ({param}) = {sigma_array[i]:.2f}")

    return sigma_array


def compute_multitracer_fisher(ell_array, z_bin_idx, fNL_fid=0, shape='local',
                                f_sky=F_SKY, delta_fNL=0.1):
    """
    Compute multi-tracer Fisher matrix for f_NL using all 5 SPHEREx samples.

    This function sums the Fisher information from all 5 galaxy samples.
    The multi-tracer benefit comes from having multiple independent measurements
    with different biases, which helps break degeneracies and reduce cosmic variance.

    Parameters
    ----------
    ell_array : array_like
        Array of multipole moments
    z_bin_idx : int
        Redshift bin index (0-10)
    fNL_fid : float, optional
        Fiducial f_NL value (default: 0)
    shape : str, optional
        PNG shape: 'local', 'equilateral', 'orthogonal'
    f_sky : float, optional
        Sky fraction
    delta_fNL : float, optional
        Step size for derivatives

    Returns
    -------
    F_fNL : float
        Fisher information on f_NL from this redshift bin

    Notes
    -----
    Simplified multi-tracer approach:
    - Sum Fisher matrices from all 5 samples (auto-spectra)
    - Each sample has different bias and number density
    - This captures the multi-tracer benefit of having independent measurements

    Full multi-tracer (not yet implemented):
    - Would use full 15×15 covariance matrix (5 auto + 10 cross spectra)
    - Cross-spectra have NO shot noise → cosmic variance cancellation
    - Requires proper covariance matrix inversion
    """
    # Get redshift bin edges
    z_min, z_max = SPHEREX_Z_BINS[z_bin_idx]

    # Get bias for all 5 samples
    biases = [get_bias(s, z_bin_idx) for s in range(1, N_SAMPLES + 1)]

    # Initialize total Fisher information
    F_total = 0.0

    # Sum Fisher from each sample
    for sample_idx in range(N_SAMPLES):
        sample_num = sample_idx + 1
        b1 = biases[sample_idx]

        # Compute Fisher for this sample using existing single-tracer function
        # This properly handles shot noise, cosmic variance, etc.
        F_sample = compute_fisher_element(
            ell_array, z_min, z_max, b1, fNL_fid,
            shape_i=shape, shape_j=shape,
            f_sky=f_sky, survey_mode='full', delta_fNL=delta_fNL
        )

        F_total += F_sample

    return F_total


def compute_multitracer_full_forecast(ell_array, z_bin_indices=None, shape='local',
                                      f_sky=F_SKY, delta_fNL=0.1):
    """
    Compute full multi-tracer forecast summing over multiple redshift bins.

    Parameters
    ----------
    ell_array : array_like
        Array of multipole moments
    z_bin_indices : list of int, optional
        List of redshift bin indices to include. If None, uses all bins (0-10).
    shape : str, optional
        PNG shape
    f_sky : float, optional
        Sky fraction
    delta_fNL : float, optional
        Step size for derivatives

    Returns
    -------
    sigma_fNL : float
        1σ constraint on f_NL
    F_per_bin : dict
        Fisher information from each redshift bin

    Notes
    -----
    The total Fisher information is the sum over all redshift bins:
    F_total = Σ_bins F_bin

    This assumes bins are independent (no cross-bin correlations).
    """
    if z_bin_indices is None:
        z_bin_indices = list(range(N_Z_BINS))

    F_total = 0.0
    F_per_bin = {}

    print("=" * 70)
    print(f"MULTI-TRACER FISHER FORECAST ({shape.upper()} PNG)")
    print("=" * 70)
    print(f"\nUsing {len(z_bin_indices)} redshift bins, {N_SAMPLES} samples per bin")
    print(f"ℓ range: {ell_array[0]:.0f} - {ell_array[-1]:.0f}")
    print(f"f_sky = {f_sky:.2f}")
    print("\n" + "-" * 70)

    for z_idx in z_bin_indices:
        z_min, z_max = SPHEREX_Z_BINS[z_idx]

        # Compute Fisher for this bin
        F_bin = compute_multitracer_fisher(
            ell_array, z_idx, fNL_fid=0, shape=shape,
            f_sky=f_sky, delta_fNL=delta_fNL
        )

        F_total += F_bin
        F_per_bin[z_idx] = F_bin

        # Compute cumulative constraint
        sigma_cumulative = 1.0 / np.sqrt(F_total) if F_total > 0 else np.inf

        print(f"Bin {z_idx} ([{z_min:.1f}, {z_max:.1f}]): "
              f"F = {F_bin:.3e}, σ_cumulative = {sigma_cumulative:.2f}")

    print("-" * 70)

    # Final constraint
    if F_total > 0:
        sigma_fNL = 1.0 / np.sqrt(F_total)
    else:
        sigma_fNL = np.inf

    print(f"\nFINAL MULTI-TRACER CONSTRAINT: σ(f_NL^{shape}) = {sigma_fNL:.2f}")
    print("=" * 70)

    return sigma_fNL, F_per_bin


def compute_single_sample_forecast(ell_array, sample_num, z_bin_indices=None,
                                    shape='local', f_sky=F_SKY):
    """
    Compute forecast using a SINGLE sample (for comparison with multi-tracer).

    This uses only auto-power spectra from one sample, with shot noise.

    Parameters
    ----------
    ell_array : array_like
        Array of multipole moments
    sample_num : int
        Sample number (1-5)
    z_bin_indices : list of int, optional
        Redshift bin indices
    shape : str, optional
        PNG shape
    f_sky : float, optional
        Sky fraction

    Returns
    -------
    sigma_fNL : float
        1σ constraint on f_NL
    """
    if z_bin_indices is None:
        z_bin_indices = list(range(N_Z_BINS))

    # Build z_bins and b1_values for single-tracer function
    z_bins = [SPHEREX_Z_BINS[i] for i in z_bin_indices]
    b1_values = [get_bias(sample_num, i) for i in z_bin_indices]

    # Use existing single-tracer function
    F, param_names = compute_fisher_matrix(
        ell_array, z_bins, [f'fNL_{shape}'],
        b1_values=b1_values, f_sky=f_sky, survey_mode='full'
    )

    constraints = get_constraints(F, param_names)
    sigma_fNL = constraints[f'fNL_{shape}']

    return sigma_fNL


if __name__ == "__main__":
    print("=" * 70)
    print("FISHER MATRIX TESTS")
    print("=" * 70)

    # Simple test case
    print("\n1. Single redshift bin, local PNG:")
    print("-" * 70)

    z_bins = [(0.5, 1.5)]
    b1_values = [2.0]

    # ℓ range
    ell = np.logspace(1, 3, 20)  # ℓ from 10 to 1000

    # Compute Fisher for local PNG only
    F, param_names = compute_fisher_matrix(
        ell, z_bins, ['fNL_local'],
        b1_values=b1_values,
        fNL_fid=0.0,
        f_sky=0.75,
        survey_mode='full'
    )

    print(f"\nFisher matrix:")
    print(F)

    # Get constraints
    constraints = get_constraints(F, param_names)

    print(f"\nConstraints:")
    for param, sigma in constraints.items():
        print(f"  σ({param}) = {sigma:.2f}")

    # Test Fisher matrix properties
    print(f"\nValidation:")
    print(f"  ✓ Positive definite: {np.all(np.linalg.eigvals(F) > 0)}")
    print(f"  ✓ Symmetric: {np.allclose(F, F.T)}")

    print("\n" + "=" * 70)
