"""
Figure 3: SPHEREx noise model (side-by-side, AAS journal style)
Publication quality matching Cheng et al. 2024
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.interpolate import interp1d
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

# Colors
COLOR_ALLSKY = '#c0392b'
COLOR_DEEP = '#2980b9'
GRAY = '#7f8c8d'

def load_noise_data():
    """Load SPHEREx v28 noise model."""
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'spherex_noise_v28.txt')
    data = np.loadtxt(data_path)
    return data[:, 0], data[:, 1], data[:, 2]

def main():
    lam_data, allsky_noise, deep_noise = load_noise_data()
    OLD_DEEP = 0.018
    OLD_ALLSKY = OLD_DEEP * np.sqrt(50.0)

    fig = plt.figure(figsize=(7.0, 3.0))
    gs = gridspec.GridSpec(1, 2, wspace=0.25, left=0.08, right=0.97, top=0.94, bottom=0.14)

    # Panel (a): Noise vs wavelength
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(lam_data, allsky_noise, '-', color=COLOR_ALLSKY, lw=1.2, label='All-sky (v28)')
    ax1.plot(lam_data, deep_noise, '-', color=COLOR_DEEP, lw=1.2, label='Deep-field (v28)')
    ax1.axhline(OLD_ALLSKY, color=GRAY, ls='--', lw=1.0, alpha=0.5, label='Old constant')
    ax1.axhline(OLD_DEEP, color=GRAY, ls='--', lw=1.0, alpha=0.5)

    ax1.set_xlabel(r'Wavelength $\lambda$ [$\mu$m]', fontsize=9)
    ax1.set_ylabel(r'$\sigma_n$ [nW m$^{-2}$ sr$^{-1}$]', fontsize=9, labelpad=4)
    ax1.legend(frameon=False, loc='upper right', fontsize=8, handlelength=1.5)
    ax1.set_xlim(0.7, 5.1)
    ax1.set_ylim(0, 30)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.text(0.05, 0.95, '(a)', transform=ax1.transAxes, fontsize=9, va='top')

    # Panel (b): Ratio to old constant
    ax2 = fig.add_subplot(gs[1])
    lam_plot = np.linspace(0.75, 5.0, 500)
    allsky_interp = interp1d(lam_data, allsky_noise, bounds_error=False,
                            fill_value=(allsky_noise[0], allsky_noise[-1]))
    deep_interp = interp1d(lam_data, deep_noise, bounds_error=False,
                          fill_value=(deep_noise[0], deep_noise[-1]))

    ratio_allsky = allsky_interp(lam_plot) / OLD_ALLSKY
    ratio_deep = deep_interp(lam_plot) / OLD_DEEP

    ax2.plot(lam_plot, ratio_allsky, '-', color=COLOR_ALLSKY, lw=1.2, label='All-sky')
    ax2.plot(lam_plot, ratio_deep, '-', color=COLOR_DEEP, lw=1.2, label='Deep-field')
    ax2.axhline(1.0, color='black', ls='--', lw=0.8, alpha=0.5)

    # Annotate maximum with arrow
    max_ratio = np.max(ratio_allsky)
    max_idx = np.argmax(ratio_allsky)
    max_lam = lam_plot[max_idx]
    ax2.annotate(f'{max_ratio:.0f}× at blue edge',
                xy=(max_lam, max_ratio), xytext=(1.5, max_ratio - 50),
                arrowprops=dict(arrowstyle='->', lw=0.8, color='black'),
                fontsize=8, ha='left')

    ax2.set_xlabel(r'Wavelength $\lambda$ [$\mu$m]', fontsize=9)
    ax2.set_ylabel(r'Ratio to old constant', fontsize=9, labelpad=4)
    ax2.legend(frameon=False, loc='upper right', fontsize=8, handlelength=1.5)
    ax2.set_xlim(0.7, 5.1)
    ax2.set_ylim(0, None)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.text(0.05, 0.95, '(b)', transform=ax2.transAxes, fontsize=9, va='top')

    # Save
    output_path = os.path.join(os.path.dirname(__file__), '..', 'figures',
                              'figure_3_noise_model.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated: {output_path}")

if __name__ == '__main__':
    main()
