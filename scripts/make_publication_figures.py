"""
make_publication_figures.py
===========================
Generate all publication-ready figures for the primordial non-Gaussianity
SPHEREx forecast paper.

Figures produced
----------------
1.  ``scale_dependent_bias.pdf/png``    — Δb(k) vs k for local PNG
2.  ``angular_power_spectrum.pdf/png``  — C_ell(fNL=0) and C_ell(fNL=10)
3.  ``multitracer_constraints.pdf/png`` — σ(f_NL) vs ell_max, multi-tracer
4.  ``sensitivity_fsky.pdf/png``        — σ(f_NL) vs f_sky
5.  ``sensitivity_ellmax.pdf/png``      — σ(f_NL) vs ell_max
6.  ``sensitivity_zmax.pdf/png``        — σ(f_NL) vs z_max
7.  ``sensitivity_bias.pdf/png``        — σ(f_NL) vs bias scaling
8.  ``sensitivity_density.pdf/png``     — σ(f_NL) vs number density scaling

Style conventions
-----------------
- matplotlib ``seaborn-v0_8-colorblind`` style, 12 pt fonts
- Figure size: 8×6 inches (single column), 12×5 inches (two-panel)
- 300 DPI, tight bounding box
- Both PDF and PNG saved for each figure

Run from the repository root:
    python scripts/make_publication_figures.py
"""

import sys, os, warnings
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Publication style
# ---------------------------------------------------------------------------
try:
    plt.style.use('seaborn-v0_8-colorblind')
except OSError:
    try:
        plt.style.use('seaborn-colorblind')
    except OSError:
        pass  # use default if neither is available

FONTSIZE   = 12
TITLESIZE  = 13
LEGENDSIZE = 10
LINEWIDTH  = 2.0
MARKERSIZE = 7

plt.rcParams.update({
    'font.size':         FONTSIZE,
    'axes.titlesize':    TITLESIZE,
    'axes.labelsize':    FONTSIZE,
    'xtick.labelsize':   FONTSIZE - 1,
    'ytick.labelsize':   FONTSIZE - 1,
    'legend.fontsize':   LEGENDSIZE,
    'lines.linewidth':   LINEWIDTH,
    'lines.markersize':  MARKERSIZE,
    'figure.dpi':        300,
    'savefig.dpi':       300,
    'savefig.bbox':      'tight',
})

FIGURES_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'figures'))
os.makedirs(FIGURES_DIR, exist_ok=True)

# Colorblind-friendly palette (Okabe & Ito 2008)
CB_COLORS = ['#0072B2', '#E69F00', '#009E73', '#D55E00', '#CC79A7',
             '#56B4E9', '#F0E442', '#000000']


def save_fig(fig, name):
    """Save figure as both PDF and PNG."""
    for ext in ('pdf', 'png'):
        path = os.path.join(FIGURES_DIR, f'{name}.{ext}')
        fig.savefig(path)
        print(f"  Saved: {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
from src.bias_functions import get_scale_dependent_bias, delta_b_local
from src.cosmology import get_power_spectrum, get_growth_factor
from src.limber import get_angular_power_spectrum, get_comoving_distance
from src.survey_specs import (
    SPHEREX_Z_BINS, N_SAMPLES, N_Z_BINS, F_SKY,
    get_bias, get_shot_noise_angular, get_number_density,
    SPHEREX_NUMBER_DENSITY, SPHEREX_BIAS,
)
from src.fisher import (
    compute_single_sample_forecast,
    compute_multitracer_full_forecast,
    compute_multitracer_fisher,
)

print("=" * 65)
print("PUBLICATION FIGURES")
print("=" * 65)

# ---------------------------------------------------------------------------
# Figure 1: Scale-dependent bias Δb(k) for local PNG
# ---------------------------------------------------------------------------
print("\nFigure 1: Scale-dependent bias")

k_arr  = np.logspace(-3, 0, 200)   # h/Mpc
z_vals = [0.5, 1.0, 2.0]
b1     = 2.0
fNL    = 10.0

fig, ax = plt.subplots(figsize=(8, 6))

for zi, col in zip(z_vals, CB_COLORS):
    db = np.array([get_scale_dependent_bias(k, b1, fNL=fNL, z=zi)
                   for k in k_arr])
    ax.loglog(k_arr, np.abs(db), color=col, lw=LINEWIDTH, label=f'$z = {zi}$')

# k^-2 reference line
k_ref = np.array([0.001, 0.1])
ax.loglog(k_ref, 0.08 * (k_ref / 0.001)**(-2), 'k--', lw=1.5, label=r'$k^{-2}$')

ax.set_xlabel(r'Wavenumber $k$ [$h$/Mpc]', fontsize=FONTSIZE)
ax.set_ylabel(r'$|\Delta b_{\rm local}(k, z)|$  ($f_{\rm NL} = 10$, $b_1 = 2$)',
              fontsize=FONTSIZE)
ax.set_title(r'Scale-Dependent Bias from Local PNG', fontsize=TITLESIZE)
ax.legend(fontsize=LEGENDSIZE)
ax.grid(True, alpha=0.3, which='both')
ax.set_xlim(k_arr[0], k_arr[-1])

save_fig(fig, 'scale_dependent_bias')


# ---------------------------------------------------------------------------
# Figure 2: Angular power spectrum C_ell with and without fNL
# ---------------------------------------------------------------------------
print("\nFigure 2: Angular power spectrum")

ell_arr = np.logspace(0.5, 2.5, 60)
z_min, z_max = SPHEREX_Z_BINS[4]   # z = [0.8, 1.0]
b1_s1 = get_bias(1, 4)             # sample 1

C0   = get_angular_power_spectrum(ell_arr, z_min, z_max, b1_s1, fNL=0)
C10  = get_angular_power_spectrum(ell_arr, z_min, z_max, b1_s1, fNL=10)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# Left: C_ell
ax1.loglog(ell_arr, C0,  color=CB_COLORS[0], lw=LINEWIDTH, label=r'$f_{\rm NL} = 0$')
ax1.loglog(ell_arr, C10, color=CB_COLORS[1], lw=LINEWIDTH, ls='--',
           label=r'$f_{\rm NL} = 10$')

z_mid = (z_min + z_max) / 2.0
chi_mid = get_comoving_distance(z_mid)
N_ell_s1 = get_shot_noise_angular(1, 4, z_mid, chi_mid)
ax1.axhline(N_ell_s1, color='grey', ls=':', lw=1.5,
            label=f'Shot noise (Sample 1)')

ax1.set_xlabel(r'Multipole $\ell$', fontsize=FONTSIZE)
ax1.set_ylabel(r'$C_\ell$  [dimensionless]', fontsize=FONTSIZE)
ax1.set_title(r'Angular Power Spectrum ($z \in [0.8, 1.0]$)', fontsize=TITLESIZE)
ax1.legend(fontsize=LEGENDSIZE)
ax1.grid(True, alpha=0.3, which='both')

# Right: fractional difference
frac = (C10 - C0) / C0
ax2.semilogx(ell_arr, 100 * frac, color=CB_COLORS[2], lw=LINEWIDTH)
ax2.axhline(0, color='k', lw=0.8, ls='-', alpha=0.4)
ax2.set_xlabel(r'Multipole $\ell$', fontsize=FONTSIZE)
ax2.set_ylabel(r'$[C_\ell(f_{\rm NL}=10) - C_\ell(0)] / C_\ell(0)$  [%]',
               fontsize=FONTSIZE)
ax2.set_title(r'Fractional PNG Enhancement ($f_{\rm NL}=10$)', fontsize=TITLESIZE)
ax2.grid(True, alpha=0.3, which='both')

fig.suptitle(r'Angular Power Spectrum with Local PNG  ($b_1 = 2.1$)',
             fontsize=TITLESIZE + 1, y=1.01)
fig.tight_layout()
save_fig(fig, 'angular_power_spectrum')


# ---------------------------------------------------------------------------
# Figure 3: Multi-tracer constraint vs ell_max
# ---------------------------------------------------------------------------
print("\nFigure 3: Multi-tracer sigma(f_NL) vs ell_max")

ell_max_arr = np.array([20, 30, 50, 80, 100, 150, 200])
sigma_multi = np.zeros(len(ell_max_arr))
sigma_s1    = np.zeros(len(ell_max_arr))

Z_ALL = list(range(N_Z_BINS))

for i, lmax in enumerate(ell_max_arr):
    ell_i = np.logspace(1, np.log10(lmax), max(10, int(20 * np.log10(lmax / 10))))
    sigma_multi[i], _ = compute_multitracer_full_forecast(
        ell_i, z_bin_indices=Z_ALL, shape='local', f_sky=F_SKY
    )
    sigma_s1[i] = compute_single_sample_forecast(ell_i, sample_num=1,
                                                  z_bin_indices=Z_ALL)
    print(f"  ell_max = {lmax:4d}: sigma_multi = {sigma_multi[i]:.3f}, "
          f"sigma_s1 = {sigma_s1[i]:.3f}")

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(ell_max_arr, sigma_multi, 'o-', color=CB_COLORS[0], lw=LINEWIDTH,
        ms=MARKERSIZE, label='Multi-tracer (5 samples, 11 z-bins)')
ax.plot(ell_max_arr, sigma_s1, 's--', color=CB_COLORS[1], lw=LINEWIDTH,
        ms=MARKERSIZE, label='Single-tracer (Sample 1)')
ax.axhline(1.0, color='grey', ls=':', lw=1.5, alpha=0.7, label=r'$\sigma = 1$ target')
ax.set_xlabel(r'Maximum multipole $\ell_{\rm max}$', fontsize=FONTSIZE)
ax.set_ylabel(r'$\sigma(f_{\rm NL}^{\rm local})$', fontsize=FONTSIZE)
ax.set_title(r'SPHEREx Constraint on Local PNG vs $\ell_{\rm max}$', fontsize=TITLESIZE)
ax.legend(fontsize=LEGENDSIZE)
ax.grid(True, alpha=0.3)
save_fig(fig, 'multitracer_constraints')


# ---------------------------------------------------------------------------
# Figure 4: Sensitivity — f_sky sweep
# ---------------------------------------------------------------------------
print("\nFigure 4: f_sky sensitivity")

ell_fast = np.logspace(1, np.log10(200), 20)

fsky_vals  = np.array([0.25, 0.50, 0.75, 1.0])
sigma_fsky = np.zeros(len(fsky_vals))
for i, fs in enumerate(fsky_vals):
    sigma_fsky[i], _ = compute_multitracer_full_forecast(
        ell_fast, z_bin_indices=Z_ALL, shape='local', f_sky=fs
    )

ref_idx    = 2  # f_sky = 0.75
sigma_ref  = sigma_fsky[ref_idx]
scale_fsky = sigma_ref * np.sqrt(fsky_vals[ref_idx] / fsky_vals)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fsky_vals, sigma_fsky, 'o-', color=CB_COLORS[0], lw=LINEWIDTH, ms=MARKERSIZE,
        label='Multi-tracer (numerical)')
ax.plot(fsky_vals, scale_fsky, 'k--', lw=1.5, label=r'$\propto 1/\sqrt{f_{\rm sky}}$')
ax.set_xlabel(r'Sky fraction $f_{\rm sky}$', fontsize=FONTSIZE)
ax.set_ylabel(r'$\sigma(f_{\rm NL}^{\rm local})$', fontsize=FONTSIZE)
ax.set_title(r'Sensitivity to Sky Coverage', fontsize=TITLESIZE)
ax.legend(fontsize=LEGENDSIZE)
ax.grid(True, alpha=0.3)
save_fig(fig, 'sensitivity_fsky')


# ---------------------------------------------------------------------------
# Figure 5: Sensitivity — ell_max sweep
# ---------------------------------------------------------------------------
print("\nFigure 5: ell_max sensitivity")

ellmax_vals  = np.array([50, 100, 200, 500, 1000])
sigma_ellmax = np.zeros(len(ellmax_vals))
for i, lmax in enumerate(ellmax_vals):
    ell_lo = np.arange(10, min(lmax, 50) + 1, dtype=float)
    if lmax > 50:
        ell_hi = np.logspace(np.log10(51), np.log10(lmax), 20)
        ell_i = np.unique(np.concatenate([ell_lo, ell_hi]))
    else:
        ell_i = ell_lo
    sigma_ellmax[i] = compute_single_sample_forecast(ell_i, sample_num=1)

fig, ax = plt.subplots(figsize=(8, 6))
ax.semilogx(ellmax_vals, sigma_ellmax, 'o-', color=CB_COLORS[1], lw=LINEWIDTH,
            ms=MARKERSIZE, label='Single-tracer Sample 1')
ax.set_xlabel(r'Maximum multipole $\ell_{\rm max}$', fontsize=FONTSIZE)
ax.set_ylabel(r'$\sigma(f_{\rm NL}^{\rm local})$', fontsize=FONTSIZE)
ax.set_title(r'Sensitivity to Maximum Multipole', fontsize=TITLESIZE)
ax.legend(fontsize=LEGENDSIZE)
ax.grid(True, alpha=0.3, which='both')
save_fig(fig, 'sensitivity_ellmax')


# ---------------------------------------------------------------------------
# Figure 6: Sensitivity — z_max sweep
# ---------------------------------------------------------------------------
print("\nFigure 6: z_max sensitivity")

z_max_vals = [1.0, 2.0, 3.0, 4.0, 4.6]
sigma_zmax = np.zeros(len(z_max_vals))
for i, zmax in enumerate(z_max_vals):
    bins_i = [idx for idx, (_, zhi) in enumerate(SPHEREX_Z_BINS)
              if zhi <= zmax + 0.01]
    if not bins_i:
        bins_i = [0]
    sigma_zmax[i], _ = compute_multitracer_full_forecast(
        ell_fast, z_bin_indices=bins_i, shape='local', f_sky=F_SKY
    )

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(z_max_vals, sigma_zmax, 'o-', color=CB_COLORS[2], lw=LINEWIDTH, ms=MARKERSIZE,
        label='Multi-tracer (numerical)')
ax.axhline(1.0, color='grey', ls=':', lw=1.5, alpha=0.7, label=r'$\sigma = 1$ target')
ax.set_xlabel(r'Maximum redshift $z_{\rm max}$', fontsize=FONTSIZE)
ax.set_ylabel(r'$\sigma(f_{\rm NL}^{\rm local})$', fontsize=FONTSIZE)
ax.set_title(r'Sensitivity to Redshift Coverage', fontsize=TITLESIZE)
ax.legend(fontsize=LEGENDSIZE)
ax.grid(True, alpha=0.3)
save_fig(fig, 'sensitivity_zmax')


# ---------------------------------------------------------------------------
# Figure 7: Sensitivity — bias scaling
# ---------------------------------------------------------------------------
print("\nFigure 7: bias scaling sensitivity")

original_bias = {s: list(v) for s, v in SPHEREX_BIAS.items()}
bias_factors  = np.array([0.8, 0.9, 1.0, 1.1, 1.2])
sigma_bias    = np.zeros(len(bias_factors))
for i, bf in enumerate(bias_factors):
    for s in range(1, N_SAMPLES + 1):
        SPHEREX_BIAS[s] = [v * bf for v in original_bias[s]]
    sigma_bias[i], _ = compute_multitracer_full_forecast(
        ell_fast, z_bin_indices=Z_ALL, shape='local', f_sky=F_SKY
    )
for s in range(1, N_SAMPLES + 1):
    SPHEREX_BIAS[s] = original_bias[s]

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(bias_factors, sigma_bias, 'o-', color=CB_COLORS[3], lw=LINEWIDTH, ms=MARKERSIZE,
        label='Multi-tracer (numerical)')
ax.axvline(1.0, color='grey', ls=':', lw=1.2, label='Fiducial bias')
ax.set_xlabel(r'Bias scaling factor $f_b$', fontsize=FONTSIZE)
ax.set_ylabel(r'$\sigma(f_{\rm NL}^{\rm local})$', fontsize=FONTSIZE)
ax.set_title(r'Sensitivity to Galaxy Bias', fontsize=TITLESIZE)
ax.legend(fontsize=LEGENDSIZE)
ax.grid(True, alpha=0.3)
save_fig(fig, 'sensitivity_bias')


# ---------------------------------------------------------------------------
# Figure 8: Sensitivity — number density scaling
# ---------------------------------------------------------------------------
print("\nFigure 8: number density scaling sensitivity")

from src.survey_specs import SPHEREX_NUMBER_DENSITY
original_density = {s: list(v) for s, v in SPHEREX_NUMBER_DENSITY.items()}
density_factors  = np.array([0.5, 0.75, 1.0, 1.25, 1.5])
sigma_density    = np.zeros(len(density_factors))
for i, nf in enumerate(density_factors):
    for s in range(1, N_SAMPLES + 1):
        SPHEREX_NUMBER_DENSITY[s] = [v * nf for v in original_density[s]]
    sigma_density[i], _ = compute_multitracer_full_forecast(
        ell_fast, z_bin_indices=Z_ALL, shape='local', f_sky=F_SKY
    )
for s in range(1, N_SAMPLES + 1):
    SPHEREX_NUMBER_DENSITY[s] = original_density[s]

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(density_factors, sigma_density, 'o-', color=CB_COLORS[4], lw=LINEWIDTH,
        ms=MARKERSIZE, label='Multi-tracer (numerical)')
ax.axvline(1.0, color='grey', ls=':', lw=1.2, label='Fiducial density')
ax.set_xlabel(r'Number density scaling factor $f_n$', fontsize=FONTSIZE)
ax.set_ylabel(r'$\sigma(f_{\rm NL}^{\rm local})$', fontsize=FONTSIZE)
ax.set_title(r'Sensitivity to Galaxy Number Density', fontsize=TITLESIZE)
ax.legend(fontsize=LEGENDSIZE)
ax.grid(True, alpha=0.3)
save_fig(fig, 'sensitivity_density')


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print()
print("=" * 65)
print("All publication figures saved to:", FIGURES_DIR)
print("Both PDF and PNG formats produced at 300 DPI.")
print("=" * 65)
