"""
Comprehensive validation test suite for primordial non-Gaussianity in LIM.

Tests cover:
  - Cosmological calculations (Hubble parameter, comoving distance, growth factor,
    transfer function, matter power spectrum)
  - Scale-dependent bias functions from local primordial non-Gaussianity
  - Angular power spectra via the Limber approximation
  - Galaxy shot noise for SPHEREx survey samples
  - Fisher matrix forecasting machinery

Run from the repository root:
    python -m pytest tests/test_validation_suite.py -v
    python tests/test_validation_suite.py          # via unittest runner
"""

import sys
import unittest
import numpy as np

# Allow ``from src.module import ...`` when the test is run from repo root.
sys.path.insert(0, '.')

from src.cosmology import (get_hubble, get_comoving_distance, get_growth_factor,
                            get_transfer_function, get_power_spectrum)
from src.limber import (get_angular_power_spectrum, get_cross_power_spectrum)
from src.survey_specs import (get_shot_noise_angular, get_bias, get_number_density,
                               SPHEREX_Z_BINS, N_SAMPLES, N_Z_BINS, F_SKY)
from src.fisher import (compute_fisher_element, get_constraints,
                         compute_multitracer_fisher, compute_single_sample_forecast)
from src.bias_functions import get_scale_dependent_bias


# ---------------------------------------------------------------------------
# Helper constants reused across multiple test classes
# ---------------------------------------------------------------------------
_Z_BIN_IDX = 4                                    # z = [0.8, 1.0]
_Z_MIN, _Z_MAX = SPHEREX_Z_BINS[_Z_BIN_IDX]       # (0.8, 1.0)
_Z_MID = (_Z_MIN + _Z_MAX) / 2.0                  # 0.9
_CHI_MID = get_comoving_distance(_Z_MID)           # ~3144 Mpc/h


# ===========================================================================
# TestCosmology
# ===========================================================================

class TestCosmology(unittest.TestCase):
    """Tests for core cosmological functions in src.cosmology."""

    # ------------------------------------------------------------------ H(z)
    def test_hubble_z0(self):
        """H(z=0) must equal H₀ = 67.66 km/s/Mpc within 1%.

        For a flat ΛCDM cosmology H(0) = H₀ × sqrt(Ωm + ΩΛ).  Because
        Planck 2018 is nearly (but not exactly) flat, the computed value is
        within 0.1% of H₀ = 67.66 km/s/Mpc.
        """
        H = get_hubble(0.0)
        self.assertAlmostEqual(
            H, 67.66, delta=67.66 * 0.01,
            msg=f"H(z=0) = {H:.4f} km/s/Mpc, expected 67.66 ± 1%"
        )

    # -------------------------------------------------------------- chi(z=1)
    def test_comoving_distance_z1(self):
        """Comoving distance at z=1 must lie in [3200, 3500] Mpc/h."""
        chi = get_comoving_distance(1.0)
        self.assertGreater(
            chi, 3200.0,
            msg=f"chi(z=1) = {chi:.1f} Mpc/h is below the expected lower bound 3200"
        )
        self.assertLess(
            chi, 3500.0,
            msg=f"chi(z=1) = {chi:.1f} Mpc/h is above the expected upper bound 3500"
        )

    # ---------------------------------------------------------- growth factor
    def test_growth_factor_normalized(self):
        """Growth factor D(z=0) must be normalized to 1.0 within 1%."""
        D = get_growth_factor(0.0)
        self.assertAlmostEqual(
            D, 1.0, delta=0.01,
            msg=f"D(z=0) = {D:.6f}, expected 1.0 ± 0.01"
        )

    def test_growth_factor_decreasing(self):
        """Growth factor must decrease monotonically: D(0) > D(0.5) > D(1) > D(2)."""
        z_vals = [0.0, 0.5, 1.0, 2.0]
        D_vals = [get_growth_factor(z) for z in z_vals]
        for i in range(len(D_vals) - 1):
            self.assertGreater(
                D_vals[i], D_vals[i + 1],
                msg=(
                    f"Growth factor not decreasing: "
                    f"D(z={z_vals[i]}) = {D_vals[i]:.5f}, "
                    f"D(z={z_vals[i+1]}) = {D_vals[i+1]:.5f}"
                )
            )

    # --------------------------------------------------- transfer function
    def test_transfer_function_large_scale(self):
        """Transfer function T(k=0.001 h/Mpc) must be close to 1 (within 2%).

        On very large scales the primordial power transfers to the matter
        field with negligible suppression, so T(k) → 1 as k → 0.  At
        k = 0.001 h/Mpc the EH98 formula gives T ≈ 0.989, which satisfies
        |T − 1| < 0.02.
        """
        T = get_transfer_function(0.001)
        self.assertAlmostEqual(
            T, 1.0, delta=0.02,
            msg=f"T(k=0.001) = {T:.5f}, expected ≈ 1.0 (within 2%)"
        )

    def test_transfer_function_small_scale(self):
        """Transfer function at k=10 h/Mpc must show strong suppression: T < 0.1."""
        T = get_transfer_function(10.0)
        self.assertLess(
            T, 0.1,
            msg=f"T(k=10) = {T:.6f}, expected < 0.1 (significant small-scale suppression)"
        )

    # ------------------------------------------------------- power spectrum
    def test_power_spectrum_positive(self):
        """P(k) > 0 for all k in {0.001, 0.01, 0.1, 1.0} h/Mpc at z=0."""
        k = np.array([0.001, 0.01, 0.1, 1.0])
        P = get_power_spectrum(k, z=0.0)
        self.assertTrue(
            np.all(P > 0),
            msg=f"P(k) has non-positive values: {P}"
        )

    def test_power_spectrum_peak(self):
        """P(k) must peak near the matter–radiation equality scale, k ∈ [0.01, 0.05] h/Mpc."""
        k = np.logspace(-3, 0, 200)
        P = get_power_spectrum(k, z=0.0)
        k_peak = k[np.argmax(P)]
        self.assertGreater(
            k_peak, 0.01,
            msg=f"P(k) peak at k={k_peak:.4f} h/Mpc is below 0.01"
        )
        self.assertLess(
            k_peak, 0.05,
            msg=f"P(k) peak at k={k_peak:.4f} h/Mpc is above 0.05"
        )


# ===========================================================================
# TestBiasFunctions
# ===========================================================================

class TestBiasFunctions(unittest.TestCase):
    """Tests for the local-type scale-dependent bias in src.bias_functions."""

    def setUp(self):
        """Pre-compute commonly used parameters."""
        self.k_arr = np.array([0.001, 0.002, 0.005, 0.01])
        self.b1 = 2.0
        self.z = 1.0

    # --------------------------------------------------- zero-fNL behaviour
    def test_bias_zero_fnl(self):
        """Scale-dependent bias Δb = 0 for all k when fNL = 0."""
        result = get_scale_dependent_bias(self.k_arr, self.b1, fNL=0.0, z=self.z)
        np.testing.assert_allclose(
            result, 0.0, atol=1e-30,
            err_msg="Δb ≠ 0 when fNL = 0; the bias must be proportional to fNL"
        )

    # ------------------------------------------------ unbiased-tracer limit
    def test_bias_zero_b1_minus_1(self):
        """Δb = 0 for all k when b1 = 1 (the factor (b1 − 1) vanishes)."""
        result = get_scale_dependent_bias(self.k_arr, b1=1.0, fNL=10.0, z=self.z)
        np.testing.assert_allclose(
            result, 0.0, atol=1e-30,
            err_msg="Δb ≠ 0 when b1 = 1; unbiased tracers carry no PNG signature"
        )

    # ------------------------------------------------------- k^-2 scaling
    def test_bias_k_scaling(self):
        """At large scales (k ≪ k_eq), local bias scales approximately as k^{-2}.

        Using k₁ = 0.001 and k₂ = 0.002 h/Mpc where T(k) ≈ 1, the ratio
        Δb(k₁)/Δb(k₂) should be close to (k₂/k₁)² = 4.  The tolerance is
        15% to account for the mild deviation of T(k) from unity.
        """
        k1, k2 = 0.001, 0.002
        db1 = get_scale_dependent_bias(k1, self.b1, fNL=10.0, z=self.z)
        db2 = get_scale_dependent_bias(k2, self.b1, fNL=10.0, z=self.z)
        ratio_observed = db1 / db2
        ratio_k2 = (k2 / k1) ** 2          # = 4 for pure k^{-2}
        # Check relative deviation from ideal k^{-2} is within 15%
        self.assertAlmostEqual(
            ratio_observed / ratio_k2, 1.0, delta=0.15,
            msg=(
                f"k^{{-2}} ratio test: observed Δb(k₁)/Δb(k₂) = {ratio_observed:.4f}, "
                f"expected ≈ {ratio_k2:.1f} (within 15%)"
            )
        )

    # ---------------------------------------- sign for positive fNL / b1>1
    def test_bias_positive_fnl(self):
        """Δb > 0 at large scales for fNL > 0 and b1 > 1."""
        k = 0.002          # large scale
        db = get_scale_dependent_bias(k, b1=2.0, fNL=10.0, z=self.z)
        self.assertGreater(
            db, 0.0,
            msg=f"Δb = {db:.4e} ≤ 0 for fNL=10, b1=2 at k=0.002 h/Mpc"
        )

    # ----------------------------------------------- redshift dependence
    def test_bias_redshift_scaling(self):
        """Δb increases with redshift because D(z) is in the denominator.

        The local-type bias is
            Δb(k,z) ∝ 1 / D(z)
        Since the growth factor D(z) decreases with increasing z (D(0)=1,
        D(2)≈0.42), the bias correction increases at higher redshift.
        This test verifies: Δb(z=0.5) < Δb(z=1.0) < Δb(z=2.0).
        """
        k = 0.002
        fNL = 10.0
        db_z05 = get_scale_dependent_bias(k, self.b1, fNL=fNL, z=0.5)
        db_z10 = get_scale_dependent_bias(k, self.b1, fNL=fNL, z=1.0)
        db_z20 = get_scale_dependent_bias(k, self.b1, fNL=fNL, z=2.0)
        self.assertLess(
            db_z05, db_z10,
            msg=(
                f"Expected Δb(z=0.5) < Δb(z=1.0) but got "
                f"{db_z05:.6f} ≥ {db_z10:.6f}"
            )
        )
        self.assertLess(
            db_z10, db_z20,
            msg=(
                f"Expected Δb(z=1.0) < Δb(z=2.0) but got "
                f"{db_z10:.6f} ≥ {db_z20:.6f}"
            )
        )


# ===========================================================================
# TestAngularPowerSpectrum
# ===========================================================================

class TestAngularPowerSpectrum(unittest.TestCase):
    """Tests for angular power spectra from the Limber approximation."""

    # Use a small number of redshift samples so that each scipy.quad call
    # stays fast; the interpolation accuracy is sufficient for sign/ratio tests.
    _NZ = 25

    def setUp(self):
        """Pre-compute redshift bin and bias used by all angular PS tests."""
        self.z_min = _Z_MIN    # 0.8
        self.z_max = _Z_MAX    # 1.0
        self.b1 = 2.0

    # ------------------------------------------------- positivity
    def test_cl_positive(self):
        """C_ℓ > 0 at ell = 10, 50, 100, 500 for a Gaussian sky (fNL=0)."""
        ell = np.array([10.0, 50.0, 100.0, 500.0])
        C = get_angular_power_spectrum(
            ell, self.z_min, self.z_max, self.b1,
            fNL=0, n_z_samples=self._NZ
        )
        self.assertTrue(
            np.all(C > 0),
            msg=f"Non-positive C_ell values encountered: {C}"
        )

    # ---------------------------------------- large-scale fNL enhancement
    def test_cl_increases_with_fnl(self):
        """C_ℓ(fNL=10) > C_ℓ(fNL=0) at ell=10 due to large-scale bias enhancement.

        At ell = 10 the Limber wavenumber is k ≈ 10.5/χ ≈ 0.003 h/Mpc.
        The local-type PNG scale-dependent bias Δb ∝ 1/k² is very large here,
        so the fNL=10 power spectrum substantially exceeds the fNL=0 case.
        """
        ell = np.array([10.0])
        C0  = get_angular_power_spectrum(
            ell, self.z_min, self.z_max, self.b1,
            fNL=0, n_z_samples=self._NZ
        )
        C10 = get_angular_power_spectrum(
            ell, self.z_min, self.z_max, self.b1,
            fNL=10, n_z_samples=self._NZ
        )
        self.assertGreater(
            C10[0], C0[0],
            msg=(
                f"C_ell(fNL=10) = {C10[0]:.4e} not greater than "
                f"C_ell(fNL=0) = {C0[0]:.4e} at ell=10"
            )
        )

    # ---------------------------------------- small-scale fNL suppression
    def test_cl_fnl_effect_small_scales(self):
        """PNG effect on C_ℓ is < 1% at ell=500 (scale-dependent bias is suppressed).

        At ell = 500, k ≈ 0.16 h/Mpc.  The local-type Δb ∝ 1/k² is much
        smaller than b₁, so the fractional change |C(10)−C(0)|/C(0) < 1%.
        """
        ell = np.array([500.0])
        C0  = get_angular_power_spectrum(
            ell, self.z_min, self.z_max, self.b1,
            fNL=0, n_z_samples=self._NZ
        )
        C10 = get_angular_power_spectrum(
            ell, self.z_min, self.z_max, self.b1,
            fNL=10, n_z_samples=self._NZ
        )
        relative_diff = abs(C10[0] - C0[0]) / C0[0]
        self.assertLess(
            relative_diff, 0.01,
            msg=(
                f"PNG fractional effect at ell=500 is {100*relative_diff:.3f}%, "
                f"expected < 1%"
            )
        )

    # --------------------------------- cross-spectrum equals auto-spectrum
    def test_cross_spectrum_equals_auto(self):
        """get_cross_power_spectrum(b1_A=b1_B=b1) equals get_angular_power_spectrum(b1).

        When both tracers have the same linear bias and the same redshift
        window the cross-power spectrum reduces to the auto-power spectrum.
        Both functions are evaluated with the same n_z_samples so that the
        numerical integration is identical.
        """
        ell = np.array([50.0, 100.0])
        C_auto = get_angular_power_spectrum(
            ell, self.z_min, self.z_max, self.b1,
            fNL=0, n_z_samples=self._NZ
        )
        C_cross = get_cross_power_spectrum(
            ell, self.z_min, self.z_max, self.b1, self.b1,
            fNL=0, n_z_samples=self._NZ
        )
        np.testing.assert_allclose(
            C_cross, C_auto, rtol=1e-5,
            err_msg=(
                "Cross-spectrum with identical biases differs from auto-spectrum "
                f"(max relative error = {np.max(np.abs(C_cross/C_auto - 1)):.2e})"
            )
        )

    # ----------------------------------------- symmetry under label swap
    def test_cross_spectrum_symmetric(self):
        """Cross-spectrum C_ij(b1_A, b1_B) = C_ij(b1_B, b1_A) under tracer exchange."""
        ell = np.array([50.0, 100.0])
        b1_A, b1_B = 1.5, 2.5
        C_AB = get_cross_power_spectrum(
            ell, self.z_min, self.z_max, b1_A, b1_B,
            fNL=0, n_z_samples=self._NZ
        )
        C_BA = get_cross_power_spectrum(
            ell, self.z_min, self.z_max, b1_B, b1_A,
            fNL=0, n_z_samples=self._NZ
        )
        np.testing.assert_allclose(
            C_AB, C_BA, rtol=1e-5,
            err_msg="Cross-spectrum not symmetric under exchange of tracer A and B"
        )


# ===========================================================================
# TestShotNoise
# ===========================================================================

class TestShotNoise(unittest.TestCase):
    """Tests for galaxy shot noise N_ℓ = 1/(n̄ χ² Δχ)."""

    def setUp(self):
        """Pre-compute redshift bin geometry shared across shot-noise tests."""
        self.z_bin_idx = _Z_BIN_IDX        # 4 → z = [0.8, 1.0]
        self.z_min = _Z_MIN
        self.z_max = _Z_MAX
        self.z_mid = _Z_MID
        self.chi   = _CHI_MID              # comoving distance to z_mid ≈ 3144 Mpc/h

    # ------------------------------------------------------- positivity
    def test_shot_noise_positive(self):
        """Shot noise N_ℓ > 0 for all 5 SPHEREx samples at z_bin_idx=4."""
        for s in range(1, N_SAMPLES + 1):
            N = get_shot_noise_angular(s, self.z_bin_idx, self.z_mid, self.chi)
            self.assertGreater(
                N, 0.0,
                msg=f"Shot noise for sample {s} is not positive: N_ell = {N:.4e}"
            )

    # ------------------------------------ sample 1 has highest shot noise
    def test_shot_noise_sample1_highest(self):
        """Sample 1 has the highest shot noise at z_bin_idx=4.

        Sample 1 contains the galaxies with the best photo-z quality, which
        are a highly-selected subset with the lowest comoving number density
        (n₁ ≈ 3.2×10⁻⁵ (h/Mpc)³).  Shot noise N_ℓ = 1/(n̄ χ² Δχ) is
        therefore largest for sample 1.
        """
        noises = [
            get_shot_noise_angular(s, self.z_bin_idx, self.z_mid, self.chi)
            for s in range(1, N_SAMPLES + 1)
        ]
        max_sample_idx = int(np.argmax(noises))   # 0-based → sample = idx+1
        self.assertEqual(
            max_sample_idx, 0,
            msg=(
                f"Expected sample 1 to have the highest shot noise, "
                f"but sample {max_sample_idx + 1} is highest. "
                f"Shot noises: {[f'{n:.3e}' for n in noises]}"
            )
        )

    # --------------------------------- explicit formula verification
    def test_shot_noise_formula(self):
        """N_ℓ = 1/(n × χ² × Δχ) verified analytically for Sample 1.

        The radial bin width is approximated as
            Δχ = (c/H(z_mid)) × Δz
        where Δz = z_max − z_min, consistent with the implementation.
        """
        C_LIGHT = 299792.458              # km/s
        n         = get_number_density(1, self.z_bin_idx)
        H_z       = get_hubble(self.z_mid)
        delta_z   = self.z_max - self.z_min
        delta_chi = C_LIGHT * delta_z / H_z      # Mpc/h

        expected = 1.0 / (n * self.chi ** 2 * delta_chi)
        result   = get_shot_noise_angular(1, self.z_bin_idx, self.z_mid, self.chi)

        self.assertAlmostEqual(
            result, expected, delta=expected * 1e-6,
            msg=(
                f"Shot noise formula mismatch for sample 1: "
                f"computed = {result:.6e}, expected = {expected:.6e}"
            )
        )

    # ------------------------------------ physical magnitude check
    def test_shot_noise_dimensionless(self):
        """Shot noise lies in a physically reasonable range: 1×10⁻¹⁰ < N_ℓ < 1.0."""
        for s in range(1, N_SAMPLES + 1):
            N = get_shot_noise_angular(s, self.z_bin_idx, self.z_mid, self.chi)
            self.assertLess(
                N, 1.0,
                msg=(
                    f"Shot noise for sample {s} is unreasonably large: "
                    f"N_ell = {N:.4e} (expected < 1.0)"
                )
            )
            self.assertGreater(
                N, 1e-10,
                msg=(
                    f"Shot noise for sample {s} is unreasonably small: "
                    f"N_ell = {N:.4e} (expected > 1e-10)"
                )
            )


# ===========================================================================
# TestFisherMatrix
# ===========================================================================

class TestFisherMatrix(unittest.TestCase):
    """Tests for Fisher matrix forecasting in src.fisher."""

    def setUp(self):
        """Pre-compute common inputs for Fisher tests.

        We use a minimal three-element ell array [10, 50, 100] so that each
        test completes in a few seconds while still exercising the code path.
        """
        self.z_bin_idx = _Z_BIN_IDX          # 4 → z = [0.8, 1.0]
        self.z_min     = _Z_MIN
        self.z_max     = _Z_MAX
        self.b1        = get_bias(1, self.z_bin_idx)   # 2.1 for sample 1
        self.ell_arr   = np.array([10.0, 50.0, 100.0])

        # Galaxy shot noise for sample 1 at this redshift bin
        self.N_ell = get_shot_noise_angular(
            1, self.z_bin_idx, _Z_MID, _CHI_MID
        )

    # ------------------------------------------------------- positivity
    def test_fisher_positive(self):
        """Fisher information F > 0 for z=[0.8,1.0], ell=[10,50,100] with shot noise."""
        F = compute_fisher_element(
            self.ell_arr, self.z_min, self.z_max, self.b1,
            fNL_fid=0.0,
            shape_i='local', shape_j='local',
            f_sky=F_SKY,
            N_ell_override=self.N_ell
        )
        self.assertGreater(
            F, 0.0,
            msg=f"Fisher element F = {F:.4e} is not positive"
        )

    # ------------------------------------------- linear f_sky scaling
    def test_fisher_fsky_scaling(self):
        """F(fsky=0.75) / F(fsky=0.375) = 2.0 within 1%.

        Each ell-mode contribution to the Fisher sum is proportional to f_sky,
        so the ratio of total Fisher values equals the ratio of sky fractions.
        """
        f_full = 0.75
        f_half = 0.375
        F_full = compute_fisher_element(
            self.ell_arr, self.z_min, self.z_max, self.b1,
            fNL_fid=0.0, shape_i='local', shape_j='local',
            f_sky=f_full, N_ell_override=self.N_ell
        )
        F_half = compute_fisher_element(
            self.ell_arr, self.z_min, self.z_max, self.b1,
            fNL_fid=0.0, shape_i='local', shape_j='local',
            f_sky=f_half, N_ell_override=self.N_ell
        )
        expected_ratio = f_full / f_half   # = 2.0
        actual_ratio   = F_full / F_half
        self.assertAlmostEqual(
            actual_ratio, expected_ratio, delta=expected_ratio * 0.01,
            msg=(
                f"Fisher f_sky scaling: "
                f"F(0.75)/F(0.375) = {actual_ratio:.5f}, expected {expected_ratio:.1f} ± 1%"
            )
        )

    # ------------------------------ multi-tracer beats every single sample
    def test_multitracer_better_than_single(self):
        """Multi-tracer σ(fNL) < σ_single for every one of the 5 SPHEREx samples.

        The multi-tracer technique combines cross-correlations between all
        N=5 tracers in the full N×N covariance matrix.  This is always at
        least as good as any single-tracer forecast; in practice the
        additional cross-spectrum information provides a substantial
        improvement via cosmic-variance cancellation.

        A two-element ell array is used here to keep the test fast (the full
        5×5 spectrum matrix is computed for each ell mode).
        """
        ell_fast = np.array([10.0, 50.0])

        F_multi    = compute_multitracer_fisher(
            ell_fast, self.z_bin_idx, fNL_fid=0, f_sky=F_SKY
        )
        sigma_multi = 1.0 / np.sqrt(F_multi)

        for s in range(1, N_SAMPLES + 1):
            sigma_single = compute_single_sample_forecast(
                ell_fast, s,
                z_bin_indices=[self.z_bin_idx],
                f_sky=F_SKY
            )
            self.assertLess(
                sigma_multi, sigma_single,
                msg=(
                    f"Multi-tracer σ = {sigma_multi:.4f} is NOT smaller than "
                    f"single-sample {s} σ = {sigma_single:.4f}"
                )
            )

    # --------------------------------- 1/sqrt(F) matches matrix inversion
    def test_sigma_from_inverse(self):
        """1/√F matches the constraint from inverting the 1×1 Fisher matrix.

        For a single parameter the covariance matrix is the scalar inverse
        of F, so get_constraints([[F]], ['fNL_local'])['fNL_local'] = 1/√F
        to machine precision.
        """
        F = compute_fisher_element(
            self.ell_arr, self.z_min, self.z_max, self.b1,
            fNL_fid=0.0, shape_i='local', shape_j='local',
            f_sky=F_SKY, N_ell_override=self.N_ell
        )
        sigma_direct  = 1.0 / np.sqrt(F)
        constraints   = get_constraints(np.array([[F]]), ['fNL_local'])
        sigma_inverse = constraints['fNL_local']
        self.assertAlmostEqual(
            sigma_direct, sigma_inverse, delta=sigma_direct * 1e-6,
            msg=(
                f"Direct σ = {sigma_direct:.8e} does not match "
                f"matrix-inversion σ = {sigma_inverse:.8e}"
            )
        )


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
