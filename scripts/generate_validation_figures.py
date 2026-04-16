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
# FIGURE 3 — Reproduce Cheng+2024 Figure 3
# ─────────────────────────────────────────────────────────────────────────────

def _precompute_bias_grid(z_min=0.01, z_max=10.0, n=60):
    """
    Precompute b(z) on a coarse grid and return an interpolator.
    This avoids calling colossus inside tight loops.
    """
    z_grid = np.linspace(z_min, z_max, n)
    b_grid = get_halo_bias_simple(z_grid)
    from scipy.interpolate import interp1d
    return interp1d(z_grid, b_grid, kind='linear', bounds_error=False,
                    fill_value=(b_grid[0], b_grid[-1]))


def generate_figure3(save_path='figures/validation_figure3.png'):
    """
    Reproduce Cheng et al. (2024) Figure 3.

    Top panel    — log10 heatmap of the 92×92 C_ell signal matrix at ell~75
    Bottom panel — auto power spectrum (diagonal) of each line vs wavelength
                   with deep-field noise C_n overlaid
    """
    print("\n" + "=" * 70)
    print("FIGURE 3: C_ell matrix and auto-spectra")
    print("=" * 70)

    # ── Precompute bias interpolator ──────────────────────────────────────────
    print("  Precomputing bias grid for interpolation...")
    b_interp = _precompute_bias_grid()

    # Intensity interpolators for each line (bias-weighted, nW/m²/sr)
    from scipy.interpolate import interp1d
    z_grid = np.linspace(0.01, 10.0, 60)

    print("  Building intensity interpolators...")
    Ibw_interp = {}
    for line in LINES:
        Ibw_vals = np.array([
            get_line_intensity(z, line=line, return_bias_weighted=False)
            for z in z_grid
        ]) * b_interp(z_grid)
        Ibw_vals = np.maximum(Ibw_vals, 0.0)
        Ibw_interp[line] = interp1d(z_grid, Ibw_vals, kind='linear',
                                     bounds_error=False, fill_value=0.0)

    # ── Build signal amplitude array signal[i, l] = b*I_nu(z_il)  (nW/m²/sr) ─
    lambda_rests = np.array([LINE_PROPERTIES[l]['lambda_rest'] for l in LINES])
    z_channels = CHANNEL_CENTERS[:, None] / lambda_rests[None, :] - 1.0  # (N,4)
    valid = (z_channels > 0.05) & (z_channels < 9.0)

    print("  Computing signal amplitudes (b×I_nu at each channel redshift)...")
    signal = np.zeros((N_CHANNELS, 4))
    for il, line in enumerate(LINES):
        for i in range(N_CHANNELS):
            if not valid[i, il]:
                continue
            signal[i, il] = float(Ibw_interp[line](z_channels[i, il]))

    # ── Build C_ell matrix via Gaussian overlap in z-space ────────────────────
    print("  Building 92×92 C_ell matrix (Gaussian overlap)...")
    sigma_z = 0.12    # typical channel width in redshift for SPHEREx channels
    C = np.zeros((N_CHANNELS, N_CHANNELS))

    for il in range(4):
        z_i = z_channels[:, il]   # (N,)
        s_i = signal[:, il]       # (N,)
        for jl in range(4):
            z_j = z_channels[:, jl]   # (N,)
            s_j = signal[:, jl]       # (N,)
            # Gaussian weight matrix (N×N)
            dz = z_i[:, None] - z_j[None, :]          # (N, N)
            W  = np.exp(-0.5 * (dz / sigma_z)**2)
            # Zero out channels where the line is not in valid z range
            W *= valid[:, il, None].astype(float)
            W *= valid[None, :, jl].astype(float)
            C += np.outer(s_i, s_j) * W

    # Ensure symmetry
    C = 0.5 * (C + C.T)

    # ── Deep-field noise diagonal ─────────────────────────────────────────────
    deep = SurveyConfig.get_config('deep_field')
    C_n  = deep.C_n    # (92,) in (MJy/sr)² × sr

    print(f"  Matrix range: [{C[C > 0].min():.2e}, {C.max():.2e}]")
    print(f"  C_n range:    [{C_n.min():.2e}, {C_n.max():.2e}]")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(8, 10))
    fig.subplots_adjust(hspace=0.35)

    # ── Top: log10 heatmap ────────────────────────────────────────────────────
    ax = axes[0]
    C_pos  = C[C > 1e-30]
    c_floor = C_pos.min() if len(C_pos) > 0 else 1e-10
    C_plot = np.where(C > c_floor, C, c_floor)
    log_C  = np.log10(C_plot)
    vmin   = np.percentile(log_C, 5)   # clip very low outliers
    vmax   = log_C.max()

    im = ax.imshow(log_C, origin='lower', aspect='auto',
                   vmin=vmin, vmax=vmax, cmap='viridis')
    plt.colorbar(im, ax=ax, label=r'$\log_{10} C_\ell$ [arb. units]')

    ax.set_xlabel('Channel index $j$', fontsize=11)
    ax.set_ylabel('Channel index $i$', fontsize=11)
    ax.set_title(r'$C_\ell$ signal matrix at $\ell \approx 75$', fontsize=11)

    # Annotate key off-diagonal features
    # OIII channel index at z=1: lambda_obs = 0.5007*(1+1) = 1.0014 um
    # Find nearest channel
    oiii_chan_z1 = np.argmin(np.abs(CHANNEL_CENTERS - 1.002))
    hbeta_chan_z1 = np.argmin(np.abs(CHANNEL_CENTERS - 0.4861 * 2.0))

    # OIII-Hbeta cross-correlation band annotation
    ax.annotate('[OIII]×H$\\beta$', xy=(hbeta_chan_z1, oiii_chan_z1),
                xytext=(hbeta_chan_z1 + 8, oiii_chan_z1 - 8),
                color='white', fontsize=9,
                arrowprops=dict(arrowstyle='->', color='white', lw=1.2))

    ax.text(0.02, 0.97, r'$\ell \approx 75$', transform=ax.transAxes,
            color='white', fontsize=11, va='top',
            bbox=dict(boxstyle='round', fc='k', alpha=0.4))

    # ── Bottom: auto power spectra vs wavelength ──────────────────────────────
    ax = axes[1]
    diag_C = np.diag(C)    # auto-power (diagonal)

    for il, line in enumerate(LINES):
        wl = CHANNEL_CENTERS
        val = np.zeros(N_CHANNELS)
        for i in range(N_CHANNELS):
            if valid[i, il]:
                val[i] = signal[i, il]**2   # C_ell auto ~ signal²
        mask = (val > 0) & (wl >= 0.75) & (wl <= 5.0)
        if np.any(mask):
            ax.semilogy(wl[mask], val[mask], '-', color=COLORS[line],
                        lw=2.0, label=LABELS[line])

    # Noise power spectrum C_n (deep field)
    wl_mask = (CHANNEL_CENTERS >= 0.75) & (CHANNEL_CENTERS <= 5.0)
    ax.semilogy(CHANNEL_CENTERS[wl_mask], C_n[wl_mask], 'k--',
                lw=1.8, label=r'$C_n$ (deep field)', zorder=5)

    ax.set_xlabel(r'Observed wavelength $\lambda$ [$\mu$m]', fontsize=11)
    ax.set_ylabel(r'$C_\ell$ [arb. units]', fontsize=11)
    ax.set_xlim(0.75, 5.0)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9, ncol=2)
    ax.grid(True, alpha=0.2)
    ax.set_title(r'Auto power spectrum vs wavelength at $\ell \approx 75$',
                 fontsize=11)

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: {save_path}")

    # Report symmetry check
    sym_err = np.max(np.abs(C - C.T)) / (np.max(np.abs(C)) + 1e-30)
    print(f"  Matrix symmetry error: {sym_err:.2e}")

    # Check OIII-Hbeta dominance
    # Find strongest off-diagonal block by comparing max of each line-pair block
    oiii_idx = 1   # index in LINES
    hbeta_idx = 2  # index in LINES
    best_off_diag_val = -1.0
    best_off_diag_pair = None
    for il in range(4):
        for jl in range(il+1, 4):
            z_i = z_channels[:, il]
            z_j = z_channels[:, jl]
            dz  = z_i[:, None] - z_j[None, :]
            W   = np.exp(-0.5 * (dz / sigma_z)**2)
            W  *= valid[:, il, None].astype(float)
            W  *= valid[None, :, jl].astype(float)
            block_max = np.max(np.outer(signal[:, il], signal[:, jl]) * W)
            if block_max > best_off_diag_val:
                best_off_diag_val  = block_max
                best_off_diag_pair = (LINES[il], LINES[jl])
    print(f"  Strongest off-diagonal pair: {best_off_diag_pair}")

    return C, C_n, z_channels, valid, signal


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 6 — Reproduce Cheng+2024 Figure 6
# ─────────────────────────────────────────────────────────────────────────────

def generate_figure6(save_path='figures/validation_figure6.png'):
    """
    Reproduce Cheng et al. (2024) Figure 6.

    6-panel figure for the deep-field SPHEREx configuration:
      Panel 1 — bias-weighted intensity with ±1-sigma shaded region
      Panel 2 — S/N vs redshift for each line
      Panels 3-6 — relative 1-sigma constraint per line (1/S/N)
    """
    print("\n" + "=" * 70)
    print("FIGURE 6: S/N vs redshift (deep-field)")
    print("=" * 70)

    deep   = SurveyConfig.get_config('deep_field')
    z_bins = np.array([0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
    ell_bins = np.array([[50, 150], [150, 300]])

    print("  Computing S/N vs redshift (deep field)...")
    results = compute_SNR_vs_redshift(deep, z_bins=z_bins, ell_bins=ell_bins)

    # Precompute bias-weighted intensity on fine z grid
    b_interp = _precompute_bias_grid()
    from scipy.interpolate import interp1d
    z_fine = np.linspace(0.5, 4.5, 60)
    Ibw_fine = {}
    for line in LINES:
        Inu = np.array([get_line_intensity(z, line=line, return_bias_weighted=False)
                        for z in z_fine])
        Ibw_fine[line] = b_interp(z_fine) * Inu

    # Interpolate S/N onto fine grid for smooth curves
    from scipy.interpolate import interp1d as sci_interp1d
    SNR_interp = {}
    for line in LINES:
        snr = results[line]
        snr = np.maximum(snr, 1e-3)
        SNR_interp[line] = sci_interp1d(z_bins, snr, kind='linear',
                                         bounds_error=False, fill_value=1e-3)

    # Print summary
    for line in LINES:
        snr_15 = float(SNR_interp[line](1.5))
        snr_30 = float(SNR_interp[line](3.0))
        print(f"  {line:8s}: S/N(z=1.5)={snr_15:.1f},  S/N(z=3.0)={snr_30:.1f}")

    # ── Figure: 6 stacked panels ──────────────────────────────────────────────
    fig = plt.figure(figsize=(8, 14))
    heights = [3, 3, 1.8, 1.8, 1.8, 1.8]
    gs = fig.add_gridspec(6, 1, hspace=0.45, height_ratios=heights)
    axes = [fig.add_subplot(gs[k]) for k in range(6)]

    z_plot = np.linspace(0.7, 4.0, 80)

    # ── Panel 1: b*I_nu with ±1σ shaded ──────────────────────────────────────
    ax = axes[0]
    for line in LINES:
        Ibw = np.interp(z_plot, z_fine, Ibw_fine[line])
        snr = np.maximum(SNR_interp[line](z_plot), 1e-3)
        sigma = Ibw / snr

        ax.semilogy(z_plot, Ibw, '-', color=COLORS[line], lw=2.5,
                    label=LABELS[line])
        ax.fill_between(z_plot,
                        np.maximum(Ibw - sigma, 1e-6),
                        Ibw + sigma,
                        color=COLORS[line], alpha=0.18)

    ax.set_xlim(0.7, 4.0)
    ax.set_xlabel('Redshift $z$', fontsize=10)
    ax.set_ylabel(r'$b\,\nu I_\nu$ [nW m$^{-2}$ sr$^{-1}$]', fontsize=10)
    ax.legend(loc='upper right', fontsize=9, ncol=2, framealpha=0.9)
    ax.grid(True, alpha=0.2)
    ax.text(0.97, 0.95, '(a) Signal ± 1σ', transform=ax.transAxes,
            fontsize=9, ha='right', va='top')

    # ── Panel 2: S/N vs z ─────────────────────────────────────────────────────
    ax = axes[1]
    for line in LINES:
        snr = np.maximum(SNR_interp[line](z_plot), 1e-3)
        ax.semilogy(z_plot, snr, '-', color=COLORS[line], lw=2.5,
                    label=LABELS[line])

    for lvl, ls, lw in [(1.0, '-', 0.9), (3.0, '--', 0.9), (10.0, ':', 0.9)]:
        ax.axhline(lvl, color='k', ls=ls, lw=lw, alpha=0.5)
        ax.text(3.85, lvl * 1.05, f'S/N={lvl:.0f}', fontsize=8,
                ha='right', va='bottom', color='k', alpha=0.7)

    ax.set_xlim(0.7, 4.0)
    ax.set_xlabel('Redshift $z$', fontsize=10)
    ax.set_ylabel('S/N', fontsize=10)
    ax.legend(loc='upper right', fontsize=9, ncol=2, framealpha=0.9)
    ax.grid(True, alpha=0.2)
    ax.text(0.97, 0.95, '(b) S/N', transform=ax.transAxes,
            fontsize=9, ha='right', va='top')

    # ── Panels 3–6: relative 1σ constraint = 1/S/N per line ──────────────────
    for k, line in enumerate(LINES):
        ax = axes[2 + k]
        snr = np.maximum(SNR_interp[line](z_plot), 1e-3)
        rel = 1.0 / snr   # fractional 1-sigma constraint

        ax.fill_between(z_plot, -rel, rel, color=COLORS[line], alpha=0.4)
        ax.plot(z_plot,  rel, '-', color=COLORS[line], lw=1.5)
        ax.plot(z_plot, -rel, '-', color=COLORS[line], lw=1.5)
        ax.axhline(0, color='k', lw=0.7, alpha=0.5)
        for pm in [0.3, -0.3]:
            ax.axhline(pm, color='k', ls='--', lw=0.7, alpha=0.4)

        ax.set_xlim(0.7, 4.0)
        ax.set_ylim(-1.2, 1.2)
        ax.set_ylabel(r'$\Delta(bI)/bI$', fontsize=9)
        ax.set_xlabel('Redshift $z$', fontsize=9)
        ax.text(0.97, 0.93, f'({chr(99+k)}) {LABELS[line]}',
                transform=ax.transAxes, fontsize=9, ha='right', va='top',
                color=COLORS[line])
        ax.grid(True, alpha=0.2)

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: {save_path}")

    return results, z_bins


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 8 UPDATE — Add cross-over annotation
# ─────────────────────────────────────────────────────────────────────────────

def generate_figure8_update(save_path='figures/validation_figure8.png'):
    """
    Updated Figure 8 with cross-over annotation and [OII] note.

    Reproduces the deep-field noise scan + all-sky points from Phase 3C,
    adding:
      - Star marker and label at the Halpha cross-over redshift
      - '[OII] outside range' annotation in the z=1 panel
    """
    print("\n" + "=" * 70)
    print("FIGURE 8 UPDATE: Survey comparison with cross-over annotation")
    print("=" * 70)

    z_panels     = np.array([1.0, 2.0, 3.0])
    lines        = ['Halpha', 'OIII', 'Hbeta', 'OII']
    colors       = {'Halpha': 'red',  'OIII': 'blue',
                    'Hbeta': 'orange', 'OII': 'green'}
    labels_l     = {'Halpha': r'H$\alpha$', 'OIII': r'[OIII]',
                    'Hbeta':  r'H$\beta$',  'OII':  r'[OII]'}
    alpha_values = np.logspace(np.log10(0.1), np.log10(100), 20)
    ell_bins     = np.array([[50, 150], [150, 300]])

    # ── Deep-field S/N scan ───────────────────────────────────────────────────
    print("  Computing deep-field S/N scan...")
    results_deep = {}
    for line in lines:
        SNR_scan = np.zeros((len(z_panels), len(alpha_values)))
        for i_z, z in enumerate(z_panels):
            if line == 'OII' and z < 1.0:
                continue
            SNR_scan[i_z, :] = compute_SNR_with_noise_scaling(
                line, z, alpha_values, ell_bins=ell_bins)
        results_deep[line] = SNR_scan

    # ── All-sky points ────────────────────────────────────────────────────────
    print("  Computing all-sky S/N points...")
    allsky = SurveyConfig.get_config('all_sky')
    results_allsky = compute_SNR_vs_redshift(allsky, z_bins=z_panels,
                                              ell_bins=ell_bins)

    # ── Find Halpha cross-over redshift ───────────────────────────────────────
    # Cross-over is where S/N_deep(alpha=1) = S/N_allsky at the same z.
    # Scan over z, compute both S/N values, find where ratio crosses 1.
    deep = SurveyConfig.get_config('deep_field')
    z_scan = np.array([1.0, 1.5, 2.0, 2.5, 3.0])
    ratio_scan = []
    for z in z_scan:
        snr_d = compute_SNR_with_noise_scaling('Halpha', z, np.array([1.0]),
                                               ell_bins=ell_bins)[0]
        snr_a = compute_SNR_vs_redshift(allsky, z_bins=[z],
                                        ell_bins=ell_bins)['Halpha'][0]
        ratio_scan.append(snr_a / max(snr_d, 1e-10))
    ratio_scan = np.array(ratio_scan)

    z_cross = None
    for k in range(len(z_scan) - 1):
        if ratio_scan[k] >= 1.0 and ratio_scan[k+1] < 1.0:
            # linear interpolation
            frac = (ratio_scan[k] - 1.0) / (ratio_scan[k] - ratio_scan[k+1])
            z_cross = z_scan[k] + frac * (z_scan[k+1] - z_scan[k])
            break
        elif ratio_scan[k] < 1.0 and ratio_scan[k+1] >= 1.0:
            frac = (1.0 - ratio_scan[k]) / (ratio_scan[k+1] - ratio_scan[k])
            z_cross = z_scan[k] + frac * (z_scan[k+1] - z_scan[k])
            break

    if z_cross is not None:
        print(f"  Cross-over redshift: z_cross = {z_cross:.2f}")
    else:
        print(f"  Cross-over not found in z=[1,3]. Ratio scan: {ratio_scan}")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(8, 10))

    for i_panel, (ax, z) in enumerate(zip(axes, z_panels)):
        for line in lines:
            if line == 'OII' and z < 1.5:
                continue
            SNR_curve = results_deep[line][i_panel, :]
            ax.loglog(alpha_values, SNR_curve, '-', color=colors[line],
                      linewidth=2, label=labels_l[line] if i_panel == 0 else '')

        for line in lines:
            if line == 'OII' and z < 1.5:
                continue
            SNR_as = results_allsky[line][i_panel]
            ax.loglog(50.0, SNR_as, 'x', color=colors[line],
                      markersize=10, markeredgewidth=2.5)

        # Reference lines
        ax.axvline(1.0,  color='grey', ls='-',  lw=1,   alpha=0.5, zorder=0)
        ax.axvline(50.0, color='grey', ls='--', lw=1,   alpha=0.5, zorder=0)
        ax.axhline(1.0,  color='grey', ls='-',  lw=0.8, alpha=0.4, zorder=0)
        ax.axhline(3.0,  color='grey', ls='--', lw=0.8, alpha=0.4, zorder=0)
        ax.axhline(10.0, color='grey', ls=':',  lw=0.8, alpha=0.4, zorder=0)

        ax.set_xlabel(r'$\sigma_n^2 / (\sigma_n^{\rm deep})^2$', fontsize=11)
        if i_panel == 1:
            ax.set_ylabel(r'S/N', fontsize=12)
        ax.set_xlim(0.08, 120)
        ax.set_ylim(0.08, 400)

        ax.text(0.95, 0.95, f'$z = {z:.1f}$',
                transform=ax.transAxes, fontsize=12, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # [OII] absent annotation for z=1 panel
        if i_panel == 0:
            ax.legend(loc='lower left', fontsize=10, framealpha=0.9)
            ax.text(0.05, 0.15, '[OII] outside range', transform=ax.transAxes,
                    fontsize=9, color='green', alpha=0.7, va='bottom')

        ax.grid(True, alpha=0.2, which='both')

    # Cross-over star on the z=1.5 plot — mark on the panel whose z is
    # closest to z_cross (if found)
    if z_cross is not None:
        i_closest = int(np.argmin(np.abs(z_panels - z_cross)))
        ax_cross = axes[i_closest]
        # S/N of deep field at alpha=1 near z_cross
        snr_at_cross = compute_SNR_with_noise_scaling(
            'Halpha', z_panels[i_closest], np.array([1.0]), ell_bins=ell_bins)[0]
        ax_cross.plot(1.0, snr_at_cross, '*', color='k', markersize=14, zorder=6)
        ax_cross.annotate(f'cross-over ~ z = {z_cross:.1f}',
                          xy=(1.0, snr_at_cross),
                          xytext=(3.0, snr_at_cross * 0.4),
                          fontsize=9, color='k',
                          arrowprops=dict(arrowstyle='->', color='k', lw=1.2))

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: {save_path}")

    return results_deep, results_allsky, z_cross


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

    # Figure 3
    C_mat, C_n_deep, z_chans, valid_chans, sig_chans = generate_figure3()

    # Figure 6
    snr_results, z_bins_fig6 = generate_figure6()

    # Figure 8 update
    rd, ra, z_cross = generate_figure8_update()

    print("\n" + "=" * 70)
    print("Phase 3D: all four validation figures complete")
    print("=" * 70)
