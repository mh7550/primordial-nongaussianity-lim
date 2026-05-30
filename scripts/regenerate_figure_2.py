"""
Figure 2: Bias-weighted intensity (3-panel stacked, AAS journal style)
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

# Muted 4-color palette
COLORS_LINES = {
    'Halpha': '#c0392b',
    'OIII': '#2980b9',
    'Hbeta': '#27ae60',
    'OII': '#8e44ad'
}

# Line wavelengths (microns)
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

def main():
    z_plot = np.linspace(0.5, 4.0, 100)

    # Conversion factors (from Cheng+2024)
    r_i = {'Halpha': 1.27e41, 'OIII': 1.32e41, 'Hbeta': 0.444e41, 'OII': 0.71e41}
    A_dust = {'Halpha': 1.0, 'OIII': 0.75, 'Hbeta': 1.25, 'OII': 2.30}

    fig = plt.figure(figsize=(3.375, 7.0))
    gs = gridspec.GridSpec(3, 1, hspace=0.08, left=0.18, right=0.94, top=0.98, bottom=0.06)

    # Panel (a): M_i(z)
    ax1 = fig.add_subplot(gs[0])
    for line in ['Halpha', 'OIII', 'Hbeta', 'OII']:
        sfrd = np.array([madau_sfrd(z) for z in z_plot])
        M_i = r_i[line] * sfrd * 10**(-A_dust[line]/2.5)
        label = line.replace('Halpha', r'H$\alpha$').replace('OIII', '[O III]')
        label = label.replace('Hbeta', r'H$\beta$').replace('OII', '[O II]')
        ax1.plot(z_plot, M_i / 1e40, '-', lw=1.2, color=COLORS_LINES[line], label=label)

    ax1.set_ylabel(r'$M_i(z)$ [$10^{40}$ erg s$^{-1}$ Mpc$^{-3}$]', fontsize=9, labelpad=4)
    ax1.legend(frameon=False, loc='upper right', fontsize=8, handlelength=1.5)
    ax1.set_xlim(0.5, 4)
    ax1.set_ylim(0, None)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.text(0.05, 0.95, '(a)', transform=ax1.transAxes, fontsize=9, va='top')
    ax1.set_xticklabels([])

    # Panel (b): b_i * I_nu vs z
    ax2 = fig.add_subplot(gs[1])
    c = 3e5  # km/s
    for line in ['Halpha', 'OIII', 'Hbeta', 'OII']:
        sfrd = np.array([madau_sfrd(z) for z in z_plot])
        M_i = r_i[line] * sfrd * 10**(-A_dust[line]/2.5)
        chi = 3000 * (z_plot**2 / (1 + z_plot**3))
        H_z = 70 * np.sqrt(0.3 * (1 + z_plot)**3 + 0.7)
        A_0 = c / (4 * np.pi * (1 + z_plot) * H_z * chi**2)
        I_nu = M_i * A_0 * 1e9
        b_i = simple_bias(z_plot)
        ax2.plot(z_plot, b_i * I_nu, '-', lw=1.2, color=COLORS_LINES[line])

    ax2.set_ylabel(r'$b_i \, \bar{I}_\nu$ [nW m$^{-2}$ sr$^{-1}$]', fontsize=9, labelpad=4)
    ax2.set_xlim(0.5, 4)
    ax2.set_ylim(0, None)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.text(0.05, 0.95, '(b)', transform=ax2.transAxes, fontsize=9, va='top')
    ax2.set_xticklabels([])

    # Panel (c): Intensity vs wavelength with noise
    ax3 = fig.add_subplot(gs[2])

    # Load noise data
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'spherex_noise_v28.txt')
    data = np.loadtxt(data_path)
    lam_data, allsky_noise, deep_noise = data[:, 0], data[:, 1], data[:, 2]

    # Plot noise floors as gray lines
    ax3.plot(lam_data, deep_noise, '--', lw=1.0, color='#7f8c8d', alpha=0.7, label='Deep-field noise')
    ax3.plot(lam_data, allsky_noise, ':', lw=1.0, color='#7f8c8d', alpha=0.7, label='All-sky noise')

    # Plot emission lines
    for line in ['Halpha', 'OIII', 'Hbeta', 'OII']:
        z_sample = np.linspace(0.5, 3.5, 50)
        lam_obs = LINE_WAVELENGTHS[line] * (1 + z_sample)
        sfrd = np.array([madau_sfrd(z) for z in z_sample])
        M_i = r_i[line] * sfrd * 10**(-A_dust[line]/2.5)
        chi = 3000 * (z_sample**2 / (1 + z_sample**3))
        H_z = 70 * np.sqrt(0.3 * (1 + z_sample)**3 + 0.7)
        A_0 = c / (4 * np.pi * (1 + z_sample) * H_z * chi**2)
        I_nu = M_i * A_0 * 1e9
        b_i = simple_bias(z_sample)
        intensity = b_i * I_nu

        mask = (lam_obs >= 0.75) & (lam_obs <= 5.0)
        ax3.plot(lam_obs[mask], intensity[mask], '-', lw=1.2, color=COLORS_LINES[line], alpha=0.8)

    ax3.set_xlabel(r'Observed wavelength $\lambda$ [$\mu$m]', fontsize=9)
    ax3.set_ylabel(r'$b_i \, \bar{I}_\nu$ [nW m$^{-2}$ sr$^{-1}$]', fontsize=9, labelpad=4)
    ax3.legend(frameon=False, loc='upper left', fontsize=7, handlelength=1.5)
    ax3.set_xlim(0.7, 5.1)
    ax3.set_yscale('log')
    ax3.set_ylim(0.01, 30)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.text(0.05, 0.95, '(c)', transform=ax3.transAxes, fontsize=9, va='top')

    # Save
    output_path = os.path.join(os.path.dirname(__file__), '..', 'figures',
                              'figure_2_signal_model.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated: {output_path}")

if __name__ == '__main__':
    main()
