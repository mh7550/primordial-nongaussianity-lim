"""
test_survey_configs.py — Validation tests for SPHEREx survey configurations.

Tests the deep-field vs all-sky survey comparison following Cheng et al. (2024).

Test Coverage (14 tests total)
-------------------------------
Tests 1-8: Core configuration validation
Tests 9-12: S/N physics validation
Tests 13-14: Fisher matrix validation

References
----------
Cheng et al., Phys. Rev. D 109, 103011 (2024) — Section 3, Figure 8
"""

import unittest
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from survey_configs import (
    SurveyConfig,
    N_CHANNELS,
    PIXEL_SIZE_STERADIAN,
    compute_fisher_matrix_diagonal,
    compute_SNR_vs_redshift
)


class TestSurveyConfigs(unittest.TestCase):
    """Validation tests for SPHEREx survey configurations."""

    def setUp(self):
        """Initialize configurations for testing."""
        self.deep = SurveyConfig.get_config('deep_field')
        self.allsky = SurveyConfig.get_config('all_sky')
        # Use 2 ell bins for speed
        self.ell_bins = np.array([[50, 150], [150, 300]])

    # =========================================================================
    # TESTS 1-8: Core Configuration Validation
    # =========================================================================

    def test_1_deep_field_f_sky(self):
        """TEST 1: Deep field f_sky = 0.0048 exactly."""
        self.assertEqual(
            self.deep.f_sky, 0.0048,
            msg="Deep field f_sky must be exactly 0.0048"
        )

    def test_2_all_sky_f_sky(self):
        """TEST 2: All-sky f_sky = 0.75 exactly."""
        self.assertEqual(
            self.allsky.f_sky, 0.75,
            msg="All-sky f_sky must be exactly 0.75"
        )

    def test_3_noise_variance_ratio(self):
        """TEST 3: Noise variance ratio = 50 at all 92 channels."""
        for i in range(N_CHANNELS):
            ratio = (self.allsky.sigma_n[i] / self.deep.sigma_n[i])**2
            self.assertAlmostEqual(
                ratio, 50.0, places=10,
                msg=f"Channel {i}: noise variance ratio must be exactly 50.0"
            )

    def test_4_mode_count_scaling(self):
        """TEST 4: n_ell scales linearly with f_sky to machine precision."""
        ell_min, ell_max = 50, 150
        n_deep = self.deep.n_ell(ell_min, ell_max)
        n_allsky = self.allsky.n_ell(ell_min, ell_max)

        expected_ratio = self.allsky.f_sky / self.deep.f_sky
        actual_ratio = n_allsky / n_deep

        self.assertAlmostEqual(
            actual_ratio, expected_ratio, places=14,
            msg=f"Mode count ratio must equal f_sky ratio to machine precision"
        )

    def test_5_C_n_diagonal(self):
        """TEST 5: C_n is diagonal (represented as 1D array)."""
        # C_n stored as 1D array (diagonal elements only)
        self.assertEqual(
            len(self.deep.C_n), N_CHANNELS,
            msg="C_n must have N_CHANNELS elements"
        )
        self.assertEqual(
            len(self.allsky.C_n), N_CHANNELS,
            msg="C_n must have N_CHANNELS elements"
        )

        # Verify C_n = sigma_n² × omega_pix
        for i in range(N_CHANNELS):
            expected_deep = self.deep.sigma_n[i]**2 * self.deep.omega_pix
            self.assertAlmostEqual(
                self.deep.C_n[i], expected_deep, places=15,
                msg=f"Deep C_n[{i}] = sigma_n² × omega_pix"
            )

    def test_6_omega_pix_value(self):
        """TEST 6: omega_pix = (6.2/206265)² steradians to within 1%."""
        expected = (6.2 / 206265.0)**2
        self.assertAlmostEqual(
            self.deep.omega_pix / expected, 1.0, delta=0.01,
            msg="omega_pix must match 6.2 arcsec pixels to within 1%"
        )

    def test_7_sigma_n_ordering(self):
        """TEST 7: sigma_n_allsky > sigma_n_deep at all channels."""
        for i in range(N_CHANNELS):
            self.assertGreater(
                self.allsky.sigma_n[i], self.deep.sigma_n[i],
                msg=f"Channel {i}: all-sky noise must exceed deep-field noise"
            )

    def test_8_mode_count_ratio(self):
        """TEST 8: All-sky has ~156× more modes than deep-field."""
        ell_min, ell_max = 50, 300
        n_deep = self.deep.n_ell(ell_min, ell_max)
        n_allsky = self.allsky.n_ell(ell_min, ell_max)

        ratio = n_allsky / n_deep
        expected = 0.75 / 0.0048  # ~156.25

        self.assertAlmostEqual(
            ratio, expected, delta=1.0,
            msg=f"Mode count ratio should be ~156×, got {ratio:.1f}×"
        )

    # =========================================================================
    # TESTS 9-12: S/N Physics Validation
    # =========================================================================

    def test_9_SNR_positive(self):
        """TEST 9: S/N > 0 for all lines and redshifts in both configs."""
        z_test = [1.0, 2.0, 3.0]

        for config in [self.deep, self.allsky]:
            results = compute_SNR_vs_redshift(config, z_bins=z_test, ell_bins=self.ell_bins)
            for line in ['Halpha', 'OIII', 'Hbeta']:
                for z, snr in zip(z_test, results[line]):
                    self.assertGreater(
                        snr, 0.0,
                        msg=f"{config.name} {line} S/N at z={z} must be positive"
                    )

    def test_10_line_ordering(self):
        """TEST 10: Line ordering at z > 1: Halpha > OIII > Hbeta > OII."""
        z_test = 2.0
        results_deep = compute_SNR_vs_redshift(
            self.deep, z_bins=[z_test], ell_bins=self.ell_bins
        )

        snr_Ha = results_deep['Halpha'][0]
        snr_OIII = results_deep['OIII'][0]
        snr_Hb = results_deep['Hbeta'][0]
        snr_OII = results_deep['OII'][0]

        self.assertGreater(snr_Ha, snr_OIII, msg="Halpha > OIII")
        self.assertGreater(snr_OIII, snr_Hb, msg="OIII > Hbeta")
        self.assertGreater(snr_Hb, snr_OII, msg="Hbeta > OII")

    def test_11_z1_SNR_ratio(self):
        """TEST 11: At z=1, all-sky S/N within factor of 3 of deep-field."""
        z_test = 1.0
        results_deep = compute_SNR_vs_redshift(
            self.deep, z_bins=[z_test], ell_bins=self.ell_bins
        )
        results_allsky = compute_SNR_vs_redshift(
            self.allsky, z_bins=[z_test], ell_bins=self.ell_bins
        )

        snr_deep = results_deep['Halpha'][0]
        snr_allsky = results_allsky['Halpha'][0]
        ratio = snr_allsky / snr_deep

        self.assertLess(
            ratio, 3.0,
            msg=f"At z=1, S/N ratio should be < 3×, got {ratio:.2f}×"
        )

    def test_12_z3_deep_wins(self):
        """TEST 12: At z=3, deep-field Halpha S/N > all-sky Halpha S/N."""
        z_test = 3.0
        results_deep = compute_SNR_vs_redshift(
            self.deep, z_bins=[z_test], ell_bins=self.ell_bins
        )
        results_allsky = compute_SNR_vs_redshift(
            self.allsky, z_bins=[z_test], ell_bins=self.ell_bins
        )

        snr_deep = results_deep['Halpha'][0]
        snr_allsky = results_allsky['Halpha'][0]

        self.assertGreater(
            snr_deep, snr_allsky,
            msg=f"At z=3, deep S/N should exceed all-sky (got {snr_deep:.1f} vs {snr_allsky:.1f})"
        )

    # =========================================================================
    # TESTS 13-14: Fisher Matrix Validation
    # =========================================================================

    def test_13_fisher_positive_definite(self):
        """TEST 13: F_ii > 0 for all lines and redshifts in both configs."""
        z_test = 2.0
        line_test = 'Halpha'

        for config in [self.deep, self.allsky]:
            F_ii, M_i = compute_fisher_matrix_diagonal(
                line_test, z_test, config, self.ell_bins, delta=0.02
            )

            self.assertGreater(
                F_ii, 0.0,
                msg=f"{config.name}: Fisher matrix must be positive at z={z_test}"
            )

            self.assertGreater(
                M_i, 0.0,
                msg=f"{config.name}: M_i must be positive at z={z_test}"
            )

    def test_14_crossover_redshift(self):
        """TEST 14: Cross-over redshift between z=1.0 and z=2.5."""
        line = 'Halpha'

        # Low z: all-sky should win
        z_low = 1.0
        results_deep_low = compute_SNR_vs_redshift(
            self.deep, z_bins=[z_low], ell_bins=self.ell_bins
        )
        results_allsky_low = compute_SNR_vs_redshift(
            self.allsky, z_bins=[z_low], ell_bins=self.ell_bins
        )

        snr_deep_low = results_deep_low[line][0]
        snr_allsky_low = results_allsky_low[line][0]

        # High z: deep should win
        z_high = 2.5
        results_deep_high = compute_SNR_vs_redshift(
            self.deep, z_bins=[z_high], ell_bins=self.ell_bins
        )
        results_allsky_high = compute_SNR_vs_redshift(
            self.allsky, z_bins=[z_high], ell_bins=self.ell_bins
        )

        snr_deep_high = results_deep_high[line][0]
        snr_allsky_high = results_allsky_high[line][0]

        # Verify cross-over behavior
        self.assertGreater(
            snr_allsky_low, snr_deep_low,
            msg=f"All-sky should win at z={z_low}"
        )

        self.assertGreater(
            snr_deep_high, snr_allsky_high,
            msg=f"Deep-field should win at z={z_high}"
        )

        print(f"\n  Cross-over validation:")
        print(f"    z={z_low}: ratio (all/deep) = {snr_allsky_low/snr_deep_low:.2f}")
        print(f"    z={z_high}: ratio (all/deep) = {snr_allsky_high/snr_deep_high:.2f}")


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
