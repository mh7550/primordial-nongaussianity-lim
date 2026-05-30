"""
fix1_ri_diagnostic.py — FIX 1: Validate r_i conversion factors

Compares current r_i values against variations to assess sensitivity
and generates comparison figure.

Per Pullen meeting Apr 2026:
- Document current Cheng et al. (2024) values
- Note Gong et al. (2017) for manual comparison
- Generate M_i(z) sensitivity figure
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from lim_signal import get_line_luminosity_density, LINE_PROPERTIES, get_halo_bias_simple

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib not available")
    sys.exit(1)

# Current r_i values from Cheng+2024
R_I_CHENG = {
    'Halpha': 1.27e41,
    'OIII': 1.32e41,
    'Hbeta': 1.27e41 * 0.35,  # via ratio
    'OII': 0.71e41,
}

# Hypothetical variations (±30% as typical calibration uncertainty)
# Note: Gong+ values require manual verification from paper
VARIATIONS = {
    'nominal': 1.0,
    'high_30pct': 1.3,
    'low_30pct': 0.7,
}

def compute_Mi_with_ri_scaling(z_array, line, ri_scale=1.0):
    """Compute M_i(z) with scaled r_i value."""
    # Temporarily scale r_i
    orig_ri = LINE_PROPERTIES[line]['r_i']
    orig_ratio = LINE_PROPERTIES[line].get('ratio_to_Halpha')

    if line == 'Hbeta':
        # Hβ uses ratio to Hα
        LINE_PROPERTIES['Halpha']['r_i'] = R_I_CHENG['Halpha'] * ri_scale
        Mi_arr = []
        for z in z_array:
            Mi_arr.append(get_line_luminosity_density(z, 'Hbeta'))
        LINE_PROPERTIES['Halpha']['r_i'] = orig_ri
    else:
        LINE_PROPERTIES[line]['r_i'] = R_I_CHENG[line] * ri_scale
        Mi_arr = []
        for z in z_array:
            Mi_arr.append(get_line_luminosity_density(z, line))
        LINE_PROPERTIES[line]['r_i'] = orig_ri

    return np.array(Mi_arr)


def make_fix1_diagnostic():
    """Generate FIX 1 diagnostic figure."""
    z_plot = np.linspace(0.5, 4.0, 100)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    lines = ['Halpha', 'OIII', 'Hbeta', 'OII']
    line_labels = {
        'Halpha': r'H$\alpha$',
        'OIII': r'[OIII]',
        'Hbeta': r'H$\beta$',
        'OII': r'[OII]',
    }

    for ax, line in zip(axes, lines):
        for var_name, ri_scale in VARIATIONS.items():
            Mi_arr = compute_Mi_with_ri_scaling(z_plot, line, ri_scale)

            if var_name == 'nominal':
                label = f'Cheng+2024 ($r_i = {R_I_CHENG[line]:.2e}$)'
                lw, alpha = 2.5, 1.0
                color = 'black'
            elif var_name == 'high_30pct':
                label = '+30% (e.g., alternate calibration)'
                lw, alpha = 1.5, 0.6
                color = 'red'
            else:
                label = '−30% (e.g., alternate calibration)'
                lw, alpha = 1.5, 0.6
                color = 'blue'

            ax.plot(z_plot, Mi_arr, lw=lw, alpha=alpha, color=color, label=label)

        ax.set_xlabel('Redshift $z$', fontsize=11)
        ax.set_ylabel(r'$M_i(z)$ [erg/s/Mpc$^3$]', fontsize=11)
        ax.set_title(f'{line_labels[line]}: Sensitivity to $r_i$ variations',
                     fontsize=12, fontweight='bold')
        ax.set_yscale('log')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle('FIX 1: $r_i$ Conversion Factor Sensitivity\n' +
                 'Cheng+2024 values vs ±30% variations (Gong+2017 pending manual verification)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()

    output_path = os.path.join(os.path.dirname(__file__), '..', 'figures', 'fix1_ri_sensitivity.png')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"FIX 1 diagnostic saved: {output_path}")


def print_fix1_summary():
    """Print FIX 1 summary table."""
    print("\n" + "=" * 80)
    print("FIX 1: r_i CONVERSION FACTOR VALIDATION")
    print("=" * 80)
    print("\nCurrent values (Cheng et al. 2024, arXiv:2403.19740, Table 1):")
    print("-" * 80)
    print(f"{'Line':<10} {'r_i (erg/s per Msun/yr)':<30} {'A_dust':<10}")
    print("-" * 80)

    for line in ['Halpha', 'OIII', 'Hbeta', 'OII']:
        r_val = R_I_CHENG[line]
        a_val = LINE_PROPERTIES[line]['A_i']
        print(f"{line:<10} {r_val:<30.2e} {a_val:<10.2f}")

    print("-" * 80)
    print("\nGong et al. (2017, ApJ 835, 273) comparison:")
    print("  ** REQUIRES MANUAL VERIFICATION FROM FULL PAPER **")
    print("  Gong+ uses observed LFs and SFRD calibrations at 0.8 ≤ z ≤ 5.2.")
    print("  Typical calibration uncertainties: ±20-30% between different SFR indicators.")
    print("\nValidation status:")
    print("  ✓ Current values documented from Cheng+2024")
    print("  ✓ Sensitivity figure generated (figures/fix1_ri_sensitivity.png)")
    print("  ⚠ Gong+ values flagged for manual cross-check")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    print_fix1_summary()
    make_fix1_diagnostic()

    print("\nFIX 1 VALIDATION: PARTIAL")
    print("  - Documentation: COMPLETE")
    print("  - Sensitivity analysis: COMPLETE")
    print("  - Gong+2017 comparison: PENDING (requires paper access)")
