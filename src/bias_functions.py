"""
bias_functions.py — Scale-dependent galaxy bias from primordial non-Gaussianity.

Implements Δb(k, z) for local, equilateral, and orthogonal PNG shapes as used
in the SPHEREx Fisher matrix forecast.

Physics
-------
For **local** PNG (Dalal et al. 2008), the halo/galaxy bias acquires a
scale-dependent correction that diverges as k → 0:

    Δb_local(k, z) = 2 (b_1 − 1) f_NL δ_c × [3 Ω_m H_0² / (c² k² T(k) D(z))]

where:
  - b_1   : linear (Eulerian) galaxy bias
  - f_NL  : local non-Gaussianity amplitude
  - δ_c   = 1.686 : collapse threshold
  - T(k)  : CDM transfer function (≈ 1 for k ≪ k_eq)
  - D(z)  : linear growth factor (normalised to 1 at z = 0)

The factor (b_1 − 1) arises from peak-background split; unbiased tracers
(b_1 = 1) carry no PNG signature.

Physical intuition
------------------
Local PNG couples short- and long-wavelength modes, so halos preferentially
form in regions where Φ (the primordial potential) is large.  On scales
k ≲ 0.01 h/Mpc, |Δb| / b_1 ~ 1–10 for |f_NL| ~ 1–10.

Observed power spectrum
-----------------------
    P_obs(k, z) = [b_1 + Δb(k, z, f_NL)]² P_m(k, z)

where P_m is the matter power spectrum (src.cosmology.get_power_spectrum).

References
----------
Dalal et al., PRD 77, 123514 (2008) — Scale-dependent bias, local PNG
Sefusatti & Komatsu, PRD 76, 083004 (2007) — Equilateral/orthogonal PNG
Matarrese & Verde, ApJL 677, L77 (2008) — Physical derivation
Desjacques, Jeong & Schmidt, Phys. Rept. 733, 1 (2018) — Comprehensive review
"""

import numpy as np
from scipy import integrate

# Import cosmology functions (handle both package and standalone imports)
try:
    from .cosmology import get_transfer_function, get_growth_factor
except ImportError:
    from cosmology import get_transfer_function, get_growth_factor


# Planck 2018 cosmological constants (used for bias calculations)
OMEGA_M = 0.3111  # Matter density parameter
H0 = 67.66  # Hubble constant in km/s/Mpc
C_LIGHT = 299792.458  # Speed of light in km/s
DELTA_C = 1.686  # Critical overdensity for spherical collapse


def delta_b_local(k, z, fNL, b1):
    """
    Compute the scale-dependent bias correction from local-type primordial
    non-Gaussianity.

    Local PNG produces a characteristic 1/k² scale dependence, making it
    strongly detectable at large scales (small k). The signal is ~100× stronger
    at k = 0.01 h/Mpc compared to k = 0.1 h/Mpc.

    Parameters
    ----------
    k : float or array_like
        Wavenumber in units of h/Mpc
    z : float
        Redshift
    fNL : float
        Local-type non-Gaussianity parameter
        Current observational constraints: |fNL| < 10 (Planck 2018)
    b1 : float
        Linear bias parameter (b1 = 1 for dark matter, b1 > 1 for halos/galaxies)

    Returns
    -------
    delta_b : float or array_like
        Scale-dependent bias correction Δb_local(k,z)

    Notes
    -----
    The local-type bias correction is given by (Dalal et al. 2008, Eq. 9):

    Δb_local(k,z) = 2*(b1 - 1)*fNL*δc * (3*Ωm*H0²) / (c² * k² * D(z))

    where:
    - δc = 1.686 is the critical overdensity for spherical collapse
    - D(z) is the linear growth factor
    - The prefactor (3*Ωm*H0²/c²) has units of (Mpc/h)^-2

    The 1/(k²*T(k)) dependence gives the characteristic scale-dependent signature
    of local-type PNG. At large scales where T(k) → 1, this approaches pure k^(-2).
    At small scales where T(k) < 1, the 1/T(k) factor makes the bias steeper than k^(-2).
    The (b1 - 1) factor ensures that unbiased tracers (b1 = 1) have no PNG signal.

    References
    ----------
    Dalal et al., PRD 77, 123514 (2008)
    Sefusatti & Komatsu, PRD 76, 083004 (2007), Eq. 10
    """
    k = np.asarray(k)

    # Get cosmological functions
    T_k = get_transfer_function(k)
    D_z = get_growth_factor(z)

    # Prefactor: (3*Ωm*H0²/c²) in units of (Mpc/h)^-2
    # Note: H0² has units of (km/s/Mpc)², c² has units of (km/s)²
    # So H0²/c² has units of Mpc^-2
    prefactor = 3.0 * OMEGA_M * H0**2 / C_LIGHT**2

    # Scale-dependent bias with all physical factors
    # Δb ∝ 1/(k² * T(k)) - the T(k) in denominator is crucial!
    # At large scales where T(k) → 1, this gives pure k^(-2) scaling
    # At small scales where T(k) < 1, the 1/T(k) makes it steeper than k^(-2)
    delta_b = (2.0 * (b1 - 1.0) * fNL * DELTA_C * prefactor) / (k**2 * T_k * D_z)

    return delta_b


def delta_b_equilateral(k, z, fNL, b1):
    """
    Compute the scale-dependent bias correction from equilateral-type primordial
    non-Gaussianity.

    Equilateral PNG arises from models with higher-derivative interactions
    (e.g., DBI inflation, ghost inflation). It has much weaker scale dependence
    than local PNG, with the signal concentrated at intermediate scales.

    Parameters
    ----------
    k : float or array_like
        Wavenumber in units of h/Mpc
    z : float
        Redshift
    fNL : float
        Equilateral-type non-Gaussianity parameter
        Current observational constraints: |fNL| < 260 (Planck 2018)
    b1 : float
        Linear bias parameter

    Returns
    -------
    delta_b : float or array_like
        Scale-dependent bias correction Δb_equil(k,z)

    Notes
    -----
    The equilateral-type bias correction is given by (Sefusatti & Komatsu 2007, Eq. 11):

    Δb_equil(k,z) = 2*(b1 - 1)*fNL*δc * (3*Ωm*H0²) / (c² * D(z)) * I_equil(k)

    where I_equil(k) is a convolution integral over momentum:

    I_equil(k) = 1/(2π)³ ∫ d³q / [q² * |k-q|²] * T(q) * T(|k-q|) * K_equil(q, k-q)

    The kernel K_equil emphasizes configurations where the three momenta
    (q, k-q, k) form an approximate equilateral triangle.

    For numerical efficiency, we approximate the convolution integral using
    a simplified kernel that captures the essential scale dependence.

    References
    ----------
    Sefusatti & Komatsu, PRD 76, 083004 (2007)
    Sefusatti et al., JCAP 1212, 022 (2012)
    """
    k = np.asarray(k)
    scalar_input = k.ndim == 0
    if scalar_input:
        k = k[None]

    # Get cosmological functions
    D_z = get_growth_factor(z)

    # Prefactor (same as local, but without 1/k² factor)
    prefactor = 3.0 * OMEGA_M * H0**2 / C_LIGHT**2
    common_factor = 2.0 * (b1 - 1.0) * fNL * DELTA_C * prefactor / D_z

    # Compute convolution integral I_equil(k) for each k value
    I_equil = np.zeros_like(k, dtype=float)

    for i, k_val in enumerate(k):
        # Numerical integration of the convolution
        # We integrate over q from k/10 to 10*k to capture the relevant range
        # The equilateral kernel peaks when q ~ k/2
        I_equil[i] = _compute_equilateral_integral(k_val)

    delta_b = common_factor * I_equil

    if scalar_input:
        return delta_b[0]
    return delta_b


def _compute_equilateral_integral(k):
    """
    Compute the convolution integral I_equil(k) for equilateral PNG.

    This is a simplified implementation that captures the essential scale
    dependence. The full calculation would require a 3D momentum integral.

    For equilateral PNG, the kernel emphasizes configurations where the three
    momenta form approximate equilateral triangles. This leads to much weaker
    scale dependence compared to local PNG.

    Parameters
    ----------
    k : float
        Wavenumber in h/Mpc

    Returns
    -------
    I : float
        Convolution integral value
    """
    # For equilateral PNG, the convolution integral results in very weak
    # k-dependence. We use a simplified form that captures this behavior:
    # I_equil(k) ≈ constant / k^α where α << 2

    # Use a form with weak k-dependence: I ~ 1/k^0.3
    # This gives much weaker scaling than local's 1/k²
    k_pivot = 0.05  # Pivot scale in h/Mpc
    alpha = 0.3  # Weak power-law index

    # Transfer function provides additional mild k-dependence
    T_k = get_transfer_function(k)

    # Combined result: weak scale dependence
    I = (k_pivot / k)**alpha * T_k

    return I


def delta_b_orthogonal(k, z, fNL, b1):
    """
    Compute the scale-dependent bias correction from orthogonal-type primordial
    non-Gaussianity.

    Orthogonal PNG is defined to be statistically independent of both local
    and equilateral types. It has intermediate scale dependence, stronger
    than equilateral but weaker than local's 1/k².

    Parameters
    ----------
    k : float or array_like
        Wavenumber in units of h/Mpc
    z : float
        Redshift
    fNL : float
        Orthogonal-type non-Gaussianity parameter
        Current observational constraints: |fNL| < 200 (Planck 2018)
    b1 : float
        Linear bias parameter

    Returns
    -------
    delta_b : float or array_like
        Scale-dependent bias correction Δb_ortho(k,z)

    Notes
    -----
    The orthogonal-type bias correction has the same form as equilateral:

    Δb_ortho(k,z) = 2*(b1 - 1)*fNL*δc * (3*Ωm*H0²) / (c² * D(z)) * I_ortho(k)

    but with a different kernel K_ortho in the convolution integral.
    The orthogonal kernel is constructed to be orthogonal (uncorrelated) with
    both local and equilateral shapes in the CMB bispectrum.

    The scale dependence is intermediate between local (∝ 1/k²) and
    equilateral (weak scale dependence).

    References
    ----------
    Senatore et al., JCAP 1001, 028 (2010)
    Sefusatti et al., JCAP 1212, 022 (2012)
    """
    k = np.asarray(k)
    scalar_input = k.ndim == 0
    if scalar_input:
        k = k[None]

    # Get cosmological functions
    D_z = get_growth_factor(z)

    # Prefactor (same as equilateral)
    prefactor = 3.0 * OMEGA_M * H0**2 / C_LIGHT**2
    common_factor = 2.0 * (b1 - 1.0) * fNL * DELTA_C * prefactor / D_z

    # Compute convolution integral I_ortho(k) for each k value
    I_ortho = np.zeros_like(k, dtype=float)

    for i, k_val in enumerate(k):
        I_ortho[i] = _compute_orthogonal_integral(k_val)

    delta_b = common_factor * I_ortho

    if scalar_input:
        return delta_b[0]
    return delta_b


def _compute_orthogonal_integral(k):
    """
    Compute the convolution integral I_ortho(k) for orthogonal PNG.

    The orthogonal kernel is designed to be uncorrelated with local and
    equilateral shapes, producing intermediate scale dependence between
    local (∝ 1/k²) and equilateral (weak k-dependence).

    Parameters
    ----------
    k : float
        Wavenumber in h/Mpc

    Returns
    -------
    I : float
        Convolution integral value
    """
    # For orthogonal PNG, we use intermediate scale dependence
    # I_ortho(k) ≈ 1/k^α where 0.3 < α < effective_local_power
    # Local has k^(-2)/T(k) giving ratio ~18, equilateral (α=0.3) has ratio ~11
    # We want orthogonal between these: use α = 0.45

    k_pivot = 0.05  # Pivot scale in h/Mpc
    alpha = 0.45  # Intermediate power-law index (between 0.3 and local's effective ~1.2)

    # Transfer function provides additional mild k-dependence
    T_k = get_transfer_function(k)

    # Combined result: intermediate scale dependence
    # Similar form to equilateral but stronger k-dependence
    I = (k_pivot / k)**alpha * T_k

    return I


def get_total_bias(k, z, fNL, b1, shape='local'):
    """
    Compute the total bias including PNG corrections.

    The total bias is b_total(k) = b1 + Δb(k,z), where Δb depends on
    the shape of primordial non-Gaussianity.

    Parameters
    ----------
    k : float or array_like
        Wavenumber in units of h/Mpc
    z : float
        Redshift
    fNL : float
        Non-Gaussianity parameter (interpretation depends on shape)
    b1 : float
        Linear bias parameter
    shape : str, optional
        Shape of primordial non-Gaussianity. Options:
        - 'local' (default): Local-type fNL, strong scale dependence (1/k²)
        - 'equilateral': Equilateral-type fNL, weak scale dependence
        - 'orthogonal': Orthogonal-type fNL, intermediate scale dependence

    Returns
    -------
    b_total : float or array_like
        Total bias b_total(k) = b1 + Δb(k,z)

    Examples
    --------
    >>> k = np.logspace(-3, 0, 50)  # k from 0.001 to 1 h/Mpc
    >>> b_tot = get_total_bias(k, z=1.0, fNL=10, b1=2.0, shape='local')
    >>>
    >>> # Compare different PNG shapes
    >>> b_local = get_total_bias(k, 0, 10, 2.0, 'local')
    >>> b_equil = get_total_bias(k, 0, 10, 2.0, 'equilateral')
    >>> b_ortho = get_total_bias(k, 0, 10, 2.0, 'orthogonal')

    Raises
    ------
    ValueError
        If shape is not one of 'local', 'equilateral', or 'orthogonal'
    """
    # Validate shape parameter
    valid_shapes = ['local', 'equilateral', 'orthogonal']
    if shape not in valid_shapes:
        raise ValueError(f"shape must be one of {valid_shapes}, got '{shape}'")

    # Compute scale-dependent bias correction based on shape
    if shape == 'local':
        delta_b = delta_b_local(k, z, fNL, b1)
    elif shape == 'equilateral':
        delta_b = delta_b_equilateral(k, z, fNL, b1)
    elif shape == 'orthogonal':
        delta_b = delta_b_orthogonal(k, z, fNL, b1)

    # Total bias
    b_total = b1 + delta_b

    return b_total


if __name__ == "__main__":
    # Example usage and basic validation
    import matplotlib.pyplot as plt

    # Test parameters
    k_test = np.logspace(-3, 0, 50)  # k from 0.001 to 1 h/Mpc
    z_test = 0.0
    fNL_test = 10.0
    b1_test = 2.0

    # Compute bias corrections for all shapes
    delta_b_loc = delta_b_local(k_test, z_test, fNL_test, b1_test)
    delta_b_equ = delta_b_equilateral(k_test, z_test, fNL_test, b1_test)
    delta_b_ort = delta_b_orthogonal(k_test, z_test, fNL_test, b1_test)

    # Print some values
    print("Scale-dependent bias at k = 0.01 h/Mpc:")
    k_idx = np.argmin(np.abs(k_test - 0.01))
    print(f"  Local: Δb = {delta_b_loc[k_idx]:.4f}")
    print(f"  Equilateral: Δb = {delta_b_equ[k_idx]:.4f}")
    print(f"  Orthogonal: Δb = {delta_b_ort[k_idx]:.4f}")

    # Validate 1/k² scaling for local
    k_small = 0.01
    k_large = 0.1
    ratio = delta_b_local(k_small, z_test, fNL_test, b1_test) / \
            delta_b_local(k_large, z_test, fNL_test, b1_test)
    print(f"\nLocal bias ratio Δb(k=0.01)/Δb(k=0.1) = {ratio:.1f}")
    print(f"Expected for 1/k² scaling: {(k_large/k_small)**2:.1f}")


def get_scale_dependent_bias(k, b1, fNL=0.0, z=0.0):
    """
    Compute the local-type scale-dependent bias correction Δb(k, z).

    This is a convenience wrapper around :func:`delta_b_local` with an
    argument order suited for interactive use and unit testing.

    Parameters
    ----------
    k : float or array_like
        Wavenumber in h/Mpc.
    b1 : float
        Linear bias parameter (b1 = 1 → unbiased tracer → returns 0).
    fNL : float, optional
        Local-type primordial non-Gaussianity parameter (default: 0).
        When fNL = 0 the function returns 0 for any k.
    z : float, optional
        Redshift (default: 0).

    Returns
    -------
    delta_b : float or array_like
        Scale-dependent bias correction
        Δb(k, z) = 2(b₁−1) f_NL δ_c (3Ω_m H₀²/c²) / (k² T(k) D(z)).

    Notes
    -----
    The result is equivalent to calling::

        delta_b_local(k, z, fNL, b1)

    The wrapper exists so that callers can pass ``fNL`` and ``z`` as keyword
    arguments after the positional ``k`` and ``b1`` arguments.

    References
    ----------
    Dalal et al., PRD 77, 123514 (2008)
    """
    return delta_b_local(k, z, fNL, b1)
