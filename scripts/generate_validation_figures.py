"""
generate_validation_figures.py — Phase 3D validation figures.

Reproduces key figures from Cheng et al. (2024) arXiv:2403.19740 using our
pipeline, validating src/lim_signal.py, src/survey_configs.py.

Output figures:
  figures/validation_figure2.png — Reproduce Cheng+2024 Fig 2
  figures/validation_figure3.png — Reproduce Cheng+2024 Fig 3
  figures/validation_figure6.png — Reproduce Cheng+2024 Fig 6
  figures/validation_figure8.png — Updated Fig 8 with cross-over annotation

Usage:
    python scripts/generate_validation_figures.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from lim_signal import (
    get_line_luminosity_density,
    get_halo_bias_simple,
    get_line_intensity,
    LINE_PROPERTIES,
    load_spherex_noise,
)
from cosmology import get_hubble, get_comoving_distance, get_power_spectrum, h
from survey_configs import (
    SurveyConfig,
    N_CHANNELS,
    CHANNEL_CENTERS,
    compute_SNR_vs_redshift,
    compute_SNR_with_noise_scaling,
)

# ── Global style ──────────────────────────────────────────────────────────────
LINES   = ['Halpha', 'OIII', 'Hbeta', 'OII']
COLORS  = {'Halpha': 'red', 'OIII': 'blue', 'Hbeta': 'darkorange', 'OII': 'green'}
LABELS  = {'Halpha': r'H$\alpha$', 'OIII': r'[OIII]',
           'Hbeta':  r'H$\beta$',  'OII':  r'[OII]'}
L_SUN          = 3.826e33   # erg/s
ANALYSIS_ZMIN  = 0.7
ANALYSIS_ZMAX  = 6.0


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2 — Reproduce Cheng+2024 Figure 2
# ─────────────────────────────────────────────────────────────────────────────

def generate_figure2(save_path='figures/validation_figure2.png'):
    """
    Reproduce Cheng et al. (2024) Figure 2.

    Three panels:
      Top    — bias-weighted luminosity density M_i(z) vs z
      Middle — bias-weighted intensity b_i*nu*I_nu vs z
      Bottom — bias-weighted intensity vs observed wavelength (SPHEREx range)
               with deep-field noise sigma_n overlaid
    """
    print("\n" + "=" * 70)
    print("FIGURE 2: Bias-weighted luminosity density + intensity")
    print("=" * 70)

    # Redshift grid
    z_arr = np.linspace(0.05, 10.0, 80)

    print("  Computing b(z) on grid (colossus)...")
    b_arr = get_halo_bias_simple(z_arr)          # shape (80,), ~8 s

    print("  Computing line quantities...")
    M0   = {}   # luminosity density  [erg/s/Mpc³]
    I_nu = {}   # nu*I_nu (no bias)   [nW/m²/sr]
    I_bw = {}   # b*nu*I_nu           [nW/m²/sr]

    for line in LINES:
        print(f"    {line}...", end='', flush=True)
        M0[line]   = np.array([get_line_luminosity_density(z, line=line)
                               for z in z_arr])
        I_nu[line] = np.array([get_line_intensity(z, line=line,
                                                   return_bias_weighted=False)
                               for z in z_arr])
        I_bw[line] = b_arr * I_nu[line]
        print(" done")

    # ── Build figure ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(8, 12))
    fig.subplots_adjust(hspace=0.38)

    # ── Panel 1: M_i(z) = b(z)*M0_i in [10^7 L_sun h³ Mpc⁻³] ───────────────
    ax = axes[0]
    ax.axvspan(ANALYSIS_ZMIN, ANALYSIS_ZMAX, alpha=0.08, color='grey', zorder=0,
               label='Analysis range')

    for line in LINES:
        Mbar       = b_arr * M0[line]                      # erg/s/Mpc³
        Mbar_units = Mbar * h**3 / L_SUN / 1e7            # 10^7 L_sun h³ Mpc⁻³
        min_valid  = 1e-12   # floor for log axis

        in_range  = (z_arr >= ANALYSIS_ZMIN) & (z_arr <= ANALYSIS_ZMAX)
        out_range = ~in_range

        ax.semilogy(z_arr[in_range],  np.maximum(Mbar_units[in_range],  min_valid),
                    '-',  color=COLORS[line], lw=2.5, label=LABELS[line])
        ax.semilogy(z_arr[out_range], np.maximum(Mbar_units[out_range], min_valid),
                    ':',  color=COLORS[line], lw=1.5, alpha=0.55)

    ax.set_xlim(0, 10)
    ax.set_ylim(1e-4, 20)
    ax.set_xlabel('Redshift $z$', fontsize=11)
    ax.set_ylabel(r'$M(z)=b(z)\,dL/dV$' + '\n'
                  + r'$[10^7\,L_\odot\,h^3\,{\rm Mpc}^{-3}]$', fontsize=10)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9, ncol=2)
    ax.grid(True, alpha=0.2)
    ax.text(0.97, 0.95, '(a)', transform=ax.transAxes,
            fontsize=11, ha='right', va='top')

    # ── Panel 2: b(z)*nu*I_nu vs z ────────────────────────────────────────────
    ax = axes[1]
    ax.axvspan(ANALYSIS_ZMIN, ANALYSIS_ZMAX, alpha=0.08, color='grey', zorder=0)

    for line in LINES:
        in_range  = (z_arr >= ANALYSIS_ZMIN) & (z_arr <= ANALYSIS_ZMAX)
        out_range = ~in_range
        Ibw = np.maximum(I_bw[line], 1e-12)

        ax.semilogy(z_arr[in_range],  Ibw[in_range],  '-',
                    color=COLORS[line], lw=2.5, label=LABELS[line])
        ax.semilogy(z_arr[out_range], Ibw[out_range], ':',
                    color=COLORS[line], lw=1.5, alpha=0.55)

    ax.set_xlim(0, 10)
    ax.set_xlabel('Redshift $z$', fontsize=11)
    ax.set_ylabel(r'$b(z)\,\nu I_\nu$' + '\n'
                  + r'$[{\rm nW\,m^{-2}\,sr^{-1}}]$', fontsize=10)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9, ncol=2)
    ax.grid(True, alpha=0.2)
    ax.text(0.97, 0.95, '(b)', transform=ax.transAxes,
            fontsize=11, ha='right', va='top')

    # ── Panel 3: b(z)*nu*I_nu vs observed wavelength ─────────────────────────
    ax = axes[2]
    lam_min, lam_max = 0.75, 5.0

    for line in LINES:
        lam_rest = LINE_PROPERTIES[line]['lambda_rest']
        lam_obs  = lam_rest * (1.0 + z_arr)

        in_spherex  = (lam_obs >= lam_min) & (lam_obs <= lam_max)
        in_analysis = (z_arr >= ANALYSIS_ZMIN) & (z_arr <= ANALYSIS_ZMAX)
        mask = in_spherex & in_analysis

        if np.any(mask):
            Ibw = np.maximum(I_bw[line][mask], 1e-12)
            ax.semilogy(lam_obs[mask], Ibw, '-',
                        color=COLORS[line], lw=2.5, label=LABELS[line])

    # SPHEREx deep-field noise (nW/m²/sr)
    noise_data = load_spherex_noise(survey_mode='deep')
    wl  = noise_data['wavelength']
    sig = noise_data['noise']
    m   = (wl >= lam_min) & (wl <= lam_max)
    ax.semilogy(wl[m], sig[m], 'k-', lw=2.0,
                label=r'SPHEREx deep $\sigma_n$', zorder=5)

    ax.set_xlim(lam_min, lam_max)
    ax.set_xlabel(r'Observed wavelength $\lambda_{\rm obs}$ [$\mu$m]', fontsize=11)
    ax.set_ylabel(r'$b(z)\,\nu I_\nu$' + '\n'
                  + r'$[{\rm nW\,m^{-2}\,sr^{-1}}]$', fontsize=10)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9, ncol=2)
    ax.grid(True, alpha=0.2)
    ax.text(0.97, 0.95, '(c)', transform=ax.transAxes,
            fontsize=11, ha='right', va='top')

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: {save_path}")

    # Report key values
    for line in LINES:
        idx2 = np.argmin(np.abs(z_arr - 2.0))
        Mbar_at2 = b_arr[idx2] * M0[line][idx2] * h**3 / L_SUN / 1e7
        Ibw_at2  = I_bw[line][idx2]
        print(f"  {line:8s}: M_bar(z=2) = {Mbar_at2:.3f} × 10^7 L_sun h³/Mpc³,"
              f"  b*I(z=2) = {Ibw_at2:.4e} nW/m²/sr")

    return z_arr, b_arr, M0, I_bw


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    os.makedirs('figures', exist_ok=True)

    print("\n" + "=" * 70)
    print("PHASE 3D VALIDATION FIGURES")
    print("=" * 70)

    # Figure 2
    z_arr, b_arr, M0, I_bw = generate_figure2()

    print("\n" + "=" * 70)
    print("Phase 3D Figure 2 complete")
    print("=" * 70)
