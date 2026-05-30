"""
Figure 1: Joint forecast summary (2×2 panel, AAS journal style)
Publication quality matching Cheng et al. 2024
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

# Global rcParams for AAS journal style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'lines.linewidth': 1.2,
    'xtick.top': True,
    'ytick.right': True,
})

# Color palette
SLATE_GRAY = '#5d6d7e'
COLORS_LINES = {
    'Halpha': '#c0392b',
    'OIII': '#2980b9',
    'Hbeta': '#27ae60',
    'OII': '#8e44ad'
}

# Data
SIGMA_PLANCK = 5.1
SIGMA_SINGLE = 1.8
SIGMA_MULTI_DIAG = 0.89
SIGMA_MULTI_FULL = 0.71
SIGMA_LINES = {'Halpha': 1.8, 'OIII': 2.3, 'Hbeta': 3.1, 'OII': 2.7}

def main():
    fig = plt.figure(figsize=(7.0, 5.5))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.35,
                           left=0.09, right=0.97, top=0.96, bottom=0.08)

    # Panel (a): σ(f_NL) progression
    ax = fig.add_subplot(gs[0, 0])
    labels = ['Planck\nCMB', 'Single-\ntracer', 'Multi-\nauto', 'Multi-\nfull']
    sigmas = [SIGMA_PLANCK, SIGMA_SINGLE, SIGMA_MULTI_DIAG, SIGMA_MULTI_FULL]

    bars = ax.barh(labels, sigmas, color=SLATE_GRAY, alpha=0.85,
                   edgecolor='black', linewidth=0.4)

    # Threshold line
    ax.axvline(1.0, color='black', ls='--', lw=0.8, alpha=0.7)
    ax.text(1.05, 2.5, 'multi-field\nthreshold', fontsize=8, va='center')

    # Value labels
    for i, (bar, sigma) in enumerate(zip(bars, sigmas)):
        ax.text(sigma + 0.15, i, f'{sigma:.2f}', fontsize=8, va='center')

    ax.set_xlabel(r'$\sigma(f_{\mathrm{NL}}^{\mathrm{local}})$', fontsize=9)
    ax.set_xlim(0, 6)
    ax.text(0.05, 0.95, '(a)', transform=ax.transAxes, fontsize=9, va='top')

    # Panel (b): Hα S/N vs redshift
    ax = fig.add_subplot(gs[0, 1])
    z_plot = np.linspace(0.5, 4.0, 200)
    snr = 50 * (2.0 / (1 + z_plot))**1.5 / 4.0

    # Shaded 10σ region
    z_10sig = z_plot[snr > 10]
    if len(z_10sig) > 0:
        ax.axvspan(z_plot[0], z_10sig[-1], alpha=0.15, color=COLORS_LINES['Halpha'])

    ax.plot(z_plot, snr, '-', color=COLORS_LINES['Halpha'], lw=1.2)
    ax.axhline(10, color='gray', ls=':', lw=0.8, alpha=0.6)
    ax.text(1.4, 25, r'$10\sigma$ reach', fontsize=8, ha='left')

    ax.set_xlabel(r'Redshift $z$', fontsize=9)
    ax.set_ylabel(r'S/N (H$\alpha$)', fontsize=9)
    ax.set_xlim(0.5, 4)
    ax.set_ylim(0, 35)
    ax.text(0.05, 0.95, '(b)', transform=ax.transAxes, fontsize=9, va='top')

    # Panel (c): Deep vs all-sky
    ax = fig.add_subplot(gs[1, 0])
    configs = ['Deep\nfield', 'All-\nsky']
    sigma_vals = [10.0, SIGMA_MULTI_FULL]

    bars = ax.bar(configs, sigma_vals, color=SLATE_GRAY, alpha=0.85,
                  edgecolor='black', linewidth=0.4, width=0.5)
    ax.axhline(1.0, color='black', ls='--', lw=0.8, alpha=0.7)

    # Value labels
    for bar, sigma in zip(bars, sigma_vals):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.3,
                f'{sigma:.1f}', ha='center', fontsize=8)

    ax.set_ylabel(r'$\sigma(f_{\mathrm{NL}}^{\mathrm{local}})$', fontsize=9)
    ax.set_ylim(0, 12)
    ax.text(0.05, 0.95, '(c)', transform=ax.transAxes, fontsize=9, va='top')

    # Panel (d): Per-line contributions
    ax = fig.add_subplot(gs[1, 1])
    line_names = [r'H$\alpha$', '[O III]', r'H$\beta$', '[O II]', 'Multi-\ntracer']
    line_keys = ['Halpha', 'OIII', 'Hbeta', 'OII']
    sigmas_d = [SIGMA_LINES[k] for k in line_keys] + [SIGMA_MULTI_FULL]
    colors_d = [COLORS_LINES[k] for k in line_keys] + [SLATE_GRAY]

    bars = ax.bar(line_names, sigmas_d, color=colors_d, alpha=0.85,
                  edgecolor='black', linewidth=0.4, width=0.6)
    ax.axhline(1.0, color='black', ls='--', lw=0.8, alpha=0.7)

    # Value labels
    for bar, sigma in zip(bars, sigmas_d):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.1,
                f'{sigma:.1f}', ha='center', fontsize=8)

    ax.set_ylabel(r'$\sigma(f_{\mathrm{NL}}^{\mathrm{local}})$', fontsize=9)
    ax.set_ylim(0, 3.5)
    ax.text(0.05, 0.95, '(d)', transform=ax.transAxes, fontsize=9, va='top')

    # Save
    output_path = os.path.join(os.path.dirname(__file__), '..', 'figures',
                              'figure_1_joint_summary.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated: {output_path}")

if __name__ == '__main__':
    main()
