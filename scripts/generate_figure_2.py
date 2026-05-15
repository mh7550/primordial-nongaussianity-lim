"""
generate_figure_2.py — Publication-quality Figure 2 (LIM signal model)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rc

# Publication style settings
rc('font', **{'family': 'serif', 'serif': ['DejaVu Serif', 'Times']})
rc('text', usetex=False)
plt.rcParams['font.size'] = 9
plt.rcParams['axes.labelsize'] = 9
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 7
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['lines.linewidth'] = 1.5

# Colorblind-safe palette
LINE_COLORS = {
    'Halpha': '#E69F00',
    'OIII': '#56B4E9',
    'Hbeta': '#009E73',
    'OII': '#CC79A7'
}

# Emission line rest wavelengths (microns)
LINE_WAVELENGTHS = {
    'Halpha': 0.6563,
    'OIII': 0.5007,
    'Hbeta': 0.4861,
    'OII': 0.3727
}

def madau_sfrd(z):
    """Madau-Dickinson star formation rate density."""
    return 0.015 * (1 + z)**2.7 / (1 + ((1 + z)/2.9)**5.6)

def simple_bias(z):
    """Simple bias model."""
    return 1.0 + 0.84 * z

def figure_2_signal_model():
    """Generate Figure 2 with realistic LIM signal model."""
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.4))

    z_plot = np.linspace(0.5, 4.0, 100)

    # Conversion factors (from Cheng+2024)
    r_i = {'Halpha': 1.27e41, 'OIII': 1.32e41, 'Hbeta': 0.444e41, 'OII': 0.71e41}
    A_dust = {'Halpha': 1.0, 'OIII': 0.75, 'Hbeta': 1.25, 'OII': 2.30}

    # Panel (a): M_i(z) - comoving luminosity density
    ax = axes[0]
    for line in ['Halpha', 'OIII', 'Hbeta', 'OII']:
        sfrd = np.array([madau_sfrd(z) for z in z_plot])
        M_i = r_i[line] * sfrd * 10**(-A_dust[line]/2.5)
        label = line.replace('Halpha', r'H$\alpha$').replace('OIII', '[O III]').replace('Hbeta', r'H$\beta$').replace('OII', '[O II]')
        ax.plot(z_plot, M_i / 1e40, '-', lw=2, color=LINE_COLORS[line], label=label)

    ax.set_xlabel(r'Redshift $z$')
    ax.set_ylabel(r'$M_i(z)$ [$10^{40}$ erg s$^{-1}$ Mpc$^{-3}$]')
    ax.legend(frameon=False, loc='upper right')
    ax.text(0.02, 0.98, '(a)', transform=ax.transAxes, fontsize=10,
            fontweight='bold', va='top')
    ax.set_xlim(0.5, 4)
    ax.set_ylim(0, None)

    # Panel (b): bias-weighted intensity b_i * I_nu
    ax = axes[1]
    c = 3e5  # km/s
    for line in ['Halpha', 'OIII', 'Hbeta', 'OII']:
        sfrd = np.array([madau_sfrd(z) for z in z_plot])
        M_i = r_i[line] * sfrd * 10**(-A_dust[line]/2.5)
        # Geometric factor (simplified)
        chi = 3000 * (z_plot**2 / (1 + z_plot**3))  # Approximate comoving distance
        H_z = 70 * np.sqrt(0.3 * (1 + z_plot)**3 + 0.7)  # Hubble parameter
        A_0 = c / (4 * np.pi * (1 + z_plot) * H_z * chi**2)
        I_nu = M_i * A_0 * 1e9  # Convert to nW/m^2/sr
        b_i = simple_bias(z_plot)
        label = line.replace('Halpha', r'H$\alpha$').replace('OIII', '[O III]').replace('Hbeta', r'H$\beta$').replace('OII', '[O II]')
        ax.plot(z_plot, b_i * I_nu, '-', lw=2, color=LINE_COLORS[line])

    ax.set_xlabel(r'Redshift $z$')
    ax.set_ylabel(r'$b_i \, \bar{I}_\nu$ [nW m$^{-2}$ sr$^{-1}$]')
    ax.text(0.02, 0.98, '(b)', transform=ax.transAxes, fontsize=10,
            fontweight='bold', va='top')
    ax.set_xlim(0.5, 4)
    ax.set_ylim(0, None)

    # Panel (c): Intensity vs observed wavelength with noise floors
    ax = axes[2]

    # Load SPHEREx noise
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'spherex_noise_v28.txt')
    data = np.loadtxt(data_path)
    lam_data, allsky_noise, deep_noise = data[:, 0], data[:, 1], data[:, 2]

    # Plot noise floors
    ax.plot(lam_data, deep_noise, 'k:', lw=1.5, alpha=0.7, label='Deep-field noise', zorder=1)
    ax.plot(lam_data, allsky_noise, 'k--', lw=1.5, alpha=0.7, label='All-sky noise', zorder=1)

    # Plot emission lines at various redshifts
    for line in ['Halpha', 'OIII', 'Hbeta', 'OII']:
        z_sample = np.linspace(0.5, 3.5, 50)
        lam_obs = LINE_WAVELENGTHS[line] * (1 + z_sample)
        # Compute intensity at each z
        sfrd = np.array([madau_sfrd(z) for z in z_sample])
        M_i = r_i[line] * sfrd * 10**(-A_dust[line]/2.5)
        chi = 3000 * (z_sample**2 / (1 + z_sample**3))
        H_z = 70 * np.sqrt(0.3 * (1 + z_sample)**3 + 0.7)
        A_0 = c / (4 * np.pi * (1 + z_sample) * H_z * chi**2)
        I_nu = M_i * A_0 * 1e9
        b_i = simple_bias(z_sample)
        intensity = b_i * I_nu

        # Only plot within SPHEREx range
        mask = (lam_obs >= 0.75) & (lam_obs <= 5.0)
        ax.plot(lam_obs[mask], intensity[mask], '-', lw=2, color=LINE_COLORS[line],
                alpha=0.8, zorder=2)

    # Shade SPHEREx wavelength range
    ax.axvspan(0.75, 5.0, alpha=0.05, color='gray', zorder=0)

    ax.set_xlabel(r'Observed wavelength $\lambda$ [$\mu$m]')
    ax.set_ylabel(r'$b_i \, \bar{I}_\nu$ [nW m$^{-2}$ sr$^{-1}$]')
    ax.legend(frameon=False, loc='upper left', fontsize=7)
    ax.text(0.02, 0.98, '(c)', transform=ax.transAxes, fontsize=10,
            fontweight='bold', va='top')
    ax.set_xlim(0.7, 5.1)
    ax.set_yscale('log')
    ax.set_ylim(0.01, 30)

    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(__file__), '..', 'figures',
                              'figure_2_signal_model.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

if __name__ == '__main__':
    figure_2_signal_model()
