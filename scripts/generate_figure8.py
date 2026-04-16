"""
generate_figure8.py — Reproduce Cheng et al. (2024) Figure 8.

Generates 3-panel plot showing S/N vs noise level for deep-field and all-sky
SPHEREx configurations at z=1, 2, 3.

Output: figures/survey_comparison_fig8.png

Usage:
    python scripts/generate_figure8.py
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from survey_configs import (
    SurveyConfig,
    compute_SNR_with_noise_scaling,
    compute_SNR_vs_redshift
)


def generate_figure8(save_path='figures/survey_comparison_fig8.png'):
    """
    Generate Figure 8: Survey comparison at z = 1, 2, 3.

    Three panels showing S/N vs noise variance ratio for four emission lines.
    """
    print("\n" + "="*70)
    print("GENERATING FIGURE 8: SURVEY COMPARISON")
    print("="*70)

    # Redshift bins for three panels
    z_panels = np.array([1.0, 2.0, 3.0])

    # Emission lines and colors
    lines = ['Halpha', 'OIII', 'Hbeta', 'OII']
    colors = {'Halpha': 'red', 'OIII': 'blue', 'Hbeta': 'orange', 'OII': 'green'}
    labels = {'Halpha': r'H$\alpha$', 'OIII': r'[OIII]',
              'Hbeta': r'H$\beta$', 'OII': r'[OII]'}

    # Noise scaling values (20 log-spaced points)
    alpha_values = np.logspace(np.log10(0.1), np.log10(100), 20)

    # Use 2 ell bins for speed
    ell_bins = np.array([[50, 150], [150, 300]])

    print(f"\nComputing S/N vs noise scaling...")
    print(f"  Redshifts: {z_panels}")
    print(f"  Alpha range: [{alpha_values[0]:.2f}, {alpha_values[-1]:.1f}]")
    print(f"  Lines: {lines}")

    # Compute deep-field noise scan
    results_deep_scan = {}
    for line in lines:
        print(f"\n  {line}:")
        SNR_scan = np.zeros((len(z_panels), len(alpha_values)))

        for i_z, z in enumerate(z_panels):
            if line == 'OII' and z < 1.0:
                continue

            print(f"    z={z:.1f}...", end='')
            SNR_scan[i_z, :] = compute_SNR_with_noise_scaling(
                line, z, alpha_values, ell_bins=ell_bins
            )
            print(f" done")

        results_deep_scan[line] = SNR_scan

    # Compute all-sky points (alpha=50, f_sky=0.75)
    print(f"\n  Computing all-sky points (alpha=50, f_sky=0.75)...")
    allsky = SurveyConfig.get_config('all_sky')
    results_allsky = compute_SNR_vs_redshift(
        allsky, z_bins=z_panels, ell_bins=ell_bins
    )

    # Create figure
    print(f"\n  Creating figure...")
    fig, axes = plt.subplots(3, 1, figsize=(8, 10))

    # f_sky ratio for mode boost
    deep = SurveyConfig.get_config('deep_field')
    f_sky_ratio = allsky.f_sky / deep.f_sky

    for i_panel, (ax, z) in enumerate(zip(axes, z_panels)):
        # Plot deep-field scan curves
        for line in lines:
            if line == 'OII' and z < 1.0:
                continue

            SNR_curve = results_deep_scan[line][i_panel, :]

            ax.loglog(alpha_values, SNR_curve, '-', color=colors[line],
                     linewidth=2, label=labels[line] if i_panel == 0 else '')

        # Plot all-sky points with crosses
        for line in lines:
            if line == 'OII' and z < 1.0:
                continue

            SNR_allsky = results_allsky[line][i_panel]

            # All-sky marker at alpha=50
            ax.loglog(50.0, SNR_allsky, 'x', color=colors[line],
                     markersize=10, markeredgewidth=2.5)

        # Reference lines
        ax.axvline(1.0, color='grey', linestyle='-', linewidth=1, alpha=0.5, zorder=0)
        ax.axvline(50.0, color='grey', linestyle='--', linewidth=1, alpha=0.5, zorder=0)

        ax.axhline(1.0, color='grey', linestyle='-', linewidth=0.8, alpha=0.4, zorder=0)
        ax.axhline(3.0, color='grey', linestyle='--', linewidth=0.8, alpha=0.4, zorder=0)
        ax.axhline(10.0, color='grey', linestyle=':', linewidth=0.8, alpha=0.4, zorder=0)

        # Labels and formatting
        ax.set_xlabel(r'$\sigma_n^2 / (\sigma_n^{\rm deep})^2$', fontsize=11)
        if i_panel == 1:
            ax.set_ylabel(r'S/N', fontsize=12)

        ax.set_xlim(0.08, 120)
        ax.set_ylim(0.08, 400)

        # Panel label
        ax.text(0.95, 0.95, f'$z = {z:.1f}$',
               transform=ax.transAxes, fontsize=12,
               ha='right', va='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Legend only on first panel
        if i_panel == 0:
            ax.legend(loc='lower left', fontsize=10, framealpha=0.9)

        ax.grid(True, alpha=0.2, which='both')

    plt.tight_layout()

    # Create output directory
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Figure saved to: {save_path}")

    plt.close()

    return results_deep_scan, results_allsky


def print_tradeoff_summary(results_deep_scan, results_allsky):
    """Print trade-off summary table."""
    deep = SurveyConfig.get_config('deep_field')
    allsky = SurveyConfig.get_config('all_sky')

    ell_bins = np.array([[50, 150], [150, 300]])
    n_deep_bin1 = deep.n_ell(ell_bins[0, 0], ell_bins[0, 1])
    n_allsky_bin1 = allsky.n_ell(ell_bins[0, 0], ell_bins[0, 1])

    print("\n" + "="*70)
    print("SURVEY CONFIGURATION TRADE-OFF SUMMARY")
    print("="*70)

    print(f"\n{'Config':<15} {'f_sky':<10} {'N_modes':<18} {'noise var':<15}")
    print("-"*70)
    print(f"{'Deep field':<15} {deep.f_sky:<10.4f} {n_deep_bin1:<18.2e} {'1× (ref)':<15}")
    print(f"{'All-sky':<15} {allsky.f_sky:<10.2f} {n_allsky_bin1:<18.2e} {'50×':<15}")

    print(f"\n{'Ratio':<15} {allsky.f_sky/deep.f_sky:<10.1f}× "
          f"{n_allsky_bin1/n_deep_bin1:<18.1f}× {'50×':<15}")

    # S/N comparison for Halpha
    print("\n" + "="*70)
    print("HALPHA S/N COMPARISON")
    print("="*70)

    # Get deep-field S/N at alpha=1
    SNR_deep_z1 = results_deep_scan['Halpha'][0, np.argmin(np.abs(np.logspace(np.log10(0.1), np.log10(100), 20) - 1.0))]
    SNR_deep_z2 = results_deep_scan['Halpha'][1, np.argmin(np.abs(np.logspace(np.log10(0.1), np.log10(100), 20) - 1.0))]
    SNR_deep_z3 = results_deep_scan['Halpha'][2, np.argmin(np.abs(np.logspace(np.log10(0.1), np.log10(100), 20) - 1.0))]

    SNR_allsky_z1 = results_allsky['Halpha'][0]
    SNR_allsky_z2 = results_allsky['Halpha'][1]
    SNR_allsky_z3 = results_allsky['Halpha'][2]

    print(f"\n{'Config':<15} {'z=1.0':<12} {'z=2.0':<12} {'z=3.0':<12}")
    print("-"*70)
    print(f"{'Deep field':<15} {SNR_deep_z1:<12.1f} {SNR_deep_z2:<12.1f} {SNR_deep_z3:<12.1f}")
    print(f"{'All-sky':<15} {SNR_allsky_z1:<12.1f} {SNR_allsky_z2:<12.1f} {SNR_allsky_z3:<12.1f}")

    ratio_z1 = SNR_allsky_z1 / SNR_deep_z1
    ratio_z2 = SNR_allsky_z2 / SNR_deep_z2
    ratio_z3 = SNR_allsky_z3 / SNR_deep_z3

    print(f"{'Ratio (all/deep)':<15} {ratio_z1:<12.2f}× {ratio_z2:<12.2f}× {ratio_z3:<12.2f}×")

    # Cross-over analysis
    print("\n" + "="*70)
    print("CROSS-OVER REDSHIFT")
    print("="*70)

    if ratio_z1 > 1.0 and ratio_z3 < 1.0:
        print("\n✓ All-sky wins at z=1.0 (bright signals, extra modes help)")
        print("✓ Deep field wins at z=3.0 (faint signals, low noise helps)")

        if ratio_z2 > 1.0:
            print(f"\nCross-over redshift: z ~ 2.0-2.5")
        else:
            print(f"\nCross-over redshift: z ~ 1.5-2.0")
    else:
        print(f"\nNote: Unexpected cross-over behavior")
        print(f"  z=1: ratio = {ratio_z1:.2f}")
        print(f"  z=2: ratio = {ratio_z2:.2f}")
        print(f"  z=3: ratio = {ratio_z3:.2f}")

    print("\n" + "="*70)


if __name__ == '__main__':
    # Generate Figure 8
    results_deep_scan, results_allsky = generate_figure8()

    # Print summary table
    print_tradeoff_summary(results_deep_scan, results_allsky)

    print("\n" + "="*70)
    print("✓ Phase 3C complete: Figure 8 generation successful")
    print("="*70)
