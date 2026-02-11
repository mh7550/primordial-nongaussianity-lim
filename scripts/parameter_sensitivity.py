"""
parameter_sensitivity.py
========================
Sensitivity analysis: how sigma(f_NL) varies with key survey/cosmology parameters.

Five sweeps:
  1. f_sky   in [0.25, 0.50, 0.75, 1.0]           — expected: sigma ∝ 1/sqrt(f_sky)
  2. ell_max in [50, 100, 200, 500, 1000]          — expected: improvement saturates
  3. z_max   in [1.0, 2.0, 3.0, 4.0, 4.6]         — expected: more volume → better
  4. bias × [0.8, 0.9, 1.0, 1.1, 1.2]             — expected: higher bias → better
  5. density × [0.5, 0.75, 1.0, 1.25, 1.5]        — expected: higher density → better

All results use the multi-tracer (5-sample) Fisher with the full 5×5 covariance matrix.
A coarse ell grid (20 points) is used for speed.

Run from the repository root:
    python scripts/parameter_sensitivity.py
"""

import sys, os, warnings
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.fisher import compute_multitracer_full_forecast, compute_single_sample_forecast
from src.survey_specs import (SPHEREX_Z_BINS, N_SAMPLES, N_Z_BINS, F_SKY,
                               get_bias, get_number_density, get_shot_noise_angular,
                               SPHEREX_NUMBER_DENSITY, SPHEREX_BIAS)
from src.limber import get_comoving_distance, get_hubble
from src.fisher import (compute_fisher_element, compute_multitracer_fisher,
                         compute_multitracer_full_forecast)

FIGURES_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'figures'))
os.makedirs(FIGURES_DIR, exist_ok=True)

# Coarse ell grid for speed (20 points, ell ∈ [10, 200])
ELL_ARRAY = np.logspace(1, np.log10(200), 20)
Z_ALL = list(range(N_Z_BINS))

# Helper: run multi-tracer forecast and return sigma(f_NL)
def run_forecast(ell_array=ELL_ARRAY, z_bin_indices=Z_ALL, f_sky=F_SKY):
    sigma, _ = compute_multitracer_full_forecast(
        ell_array, z_bin_indices=z_bin_indices, shape='local', f_sky=f_sky
    )
    return sigma

# ---------------------------------------------------------------------------
# 1. f_sky sweep
# ---------------------------------------------------------------------------
print("=" * 60)
print("SENSITIVITY 1: f_sky")
fsky_vals  = np.array([0.25, 0.50, 0.75, 1.0])
sigma_fsky = np.zeros(len(fsky_vals))
for i, fs in enumerate(fsky_vals):
    sigma_fsky[i] = run_forecast(f_sky=fs)
    print(f"  f_sky = {fs:.2f}  →  sigma(f_NL) = {sigma_fsky[i]:.3f}")

# Check sigma ∝ 1/sqrt(f_sky) scaling
ref_idx = 2  # f_sky = 0.75
sigma_ref_fsky = sigma_fsky[ref_idx]
scaling_fsky = sigma_ref_fsky * np.sqrt(fsky_vals[ref_idx] / fsky_vals)

# ---------------------------------------------------------------------------
# 2. ell_max sweep (single-tracer Sample 1 for speed)
# ---------------------------------------------------------------------------
print("\nSENSITIVITY 2: ell_max  (single-tracer Sample 1)")
ellmax_vals  = np.array([50, 100, 200, 500, 1000])
sigma_ellmax = np.zeros(len(ellmax_vals))
for i, lmax in enumerate(ellmax_vals):
    # Always sample the signal-rich low-ell regime (ell<50) densely, then
    # add log-spaced points up to ell_max so higher modes also contribute.
    ell_lo = np.arange(10, min(lmax, 50) + 1, dtype=float)
    if lmax > 50:
        ell_hi = np.logspace(np.log10(51), np.log10(lmax), 20)
        ell_i = np.unique(np.concatenate([ell_lo, ell_hi]))
    else:
        ell_i = ell_lo
    sigma_ellmax[i] = compute_single_sample_forecast(ell_i, sample_num=1)
    print(f"  ell_max = {lmax:5d}  →  sigma(f_NL) = {sigma_ellmax[i]:.3f}")

# ---------------------------------------------------------------------------
# 3. z_max sweep (multi-tracer, include increasing number of z-bins)
# ---------------------------------------------------------------------------
print("\nSENSITIVITY 3: z_max  (multi-tracer)")
z_max_vals  = [1.0, 2.0, 3.0, 4.0, 4.6]
sigma_zmax  = np.zeros(len(z_max_vals))
for i, zmax in enumerate(z_max_vals):
    bins_i = [idx for idx, (_, zhi) in enumerate(SPHEREX_Z_BINS) if zhi <= zmax + 0.01]
    if not bins_i:
        bins_i = [0]
    sigma_zmax[i] = run_forecast(z_bin_indices=bins_i)
    print(f"  z_max = {zmax:.1f}  ({len(bins_i)} bins)  →  sigma(f_NL) = {sigma_zmax[i]:.3f}")

# ---------------------------------------------------------------------------
# 4. Bias scaling (multiply all biases by a factor, use z_bin 4 only for speed)
# ---------------------------------------------------------------------------
print("\nSENSITIVITY 4: bias scaling  (multi-tracer, z_bin=4)")

import copy

bias_factors = np.array([0.8, 0.9, 1.0, 1.1, 1.2])
sigma_bias   = np.zeros(len(bias_factors))

# We patch SPHEREX_BIAS temporarily
original_bias = {s: list(v) for s, v in SPHEREX_BIAS.items()}

for i, bf in enumerate(bias_factors):
    for s in range(1, N_SAMPLES + 1):
        SPHEREX_BIAS[s] = [v * bf for v in original_bias[s]]
    sigma_bias[i] = run_forecast(z_bin_indices=Z_ALL)
    print(f"  bias × {bf:.2f}  →  sigma(f_NL) = {sigma_bias[i]:.3f}")

# Restore original biases
for s in range(1, N_SAMPLES + 1):
    SPHEREX_BIAS[s] = original_bias[s]

# ---------------------------------------------------------------------------
# 5. Number density scaling
# ---------------------------------------------------------------------------
print("\nSENSITIVITY 5: number density scaling  (multi-tracer)")

density_factors  = np.array([0.5, 0.75, 1.0, 1.25, 1.5])
sigma_density    = np.zeros(len(density_factors))

original_density = {s: list(v) for s, v in SPHEREX_NUMBER_DENSITY.items()}

for i, nf in enumerate(density_factors):
    for s in range(1, N_SAMPLES + 1):
        SPHEREX_NUMBER_DENSITY[s] = [v * nf for v in original_density[s]]
    sigma_density[i] = run_forecast(z_bin_indices=Z_ALL)
    print(f"  n × {nf:.2f}  →  sigma(f_NL) = {sigma_density[i]:.3f}")

# Restore
for s in range(1, N_SAMPLES + 1):
    SPHEREX_NUMBER_DENSITY[s] = original_density[s]

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
STYLE = dict(marker='o', linewidth=2.0, markersize=7)

def save_fig(fig, name):
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")

# --- f_sky ---
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fsky_vals, sigma_fsky, 'C0', **STYLE, label='Multi-tracer (numerical)')
ax.plot(fsky_vals, scaling_fsky, 'k--', lw=1.5, label=r'$\propto 1/\sqrt{f_{\rm sky}}$')
ax.set_xlabel(r'Sky fraction $f_{\rm sky}$', fontsize=12)
ax.set_ylabel(r'$\sigma(f_{\rm NL}^{\rm local})$', fontsize=12)
ax.set_title(r'Sensitivity to Sky Coverage', fontsize=13)
ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
save_fig(fig, 'sensitivity_fsky.png')

# --- ell_max ---
fig, ax = plt.subplots(figsize=(8, 6))
ax.semilogx(ellmax_vals, sigma_ellmax, 'C1', **STYLE, label='Single-tracer Sample 1')
ref_scale = sigma_ellmax[2] * np.sqrt(ellmax_vals[2] / ellmax_vals)
ax.semilogx(ellmax_vals, ref_scale, 'k--', lw=1.5, label=r'$\propto 1/\sqrt{\ell_{\rm max}}$')
ax.set_xlabel(r'Maximum multipole $\ell_{\rm max}$', fontsize=12)
ax.set_ylabel(r'$\sigma(f_{\rm NL}^{\rm local})$', fontsize=12)
ax.set_title(r'Sensitivity to Maximum Multipole', fontsize=13)
ax.legend(fontsize=11); ax.grid(True, alpha=0.3, which='both')
save_fig(fig, 'sensitivity_ellmax.png')

# --- z_max ---
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(z_max_vals, sigma_zmax, 'C2', **STYLE, label='Multi-tracer (numerical)')
ax.set_xlabel(r'Maximum redshift $z_{\rm max}$', fontsize=12)
ax.set_ylabel(r'$\sigma(f_{\rm NL}^{\rm local})$', fontsize=12)
ax.set_title(r'Sensitivity to Redshift Coverage', fontsize=13)
ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
save_fig(fig, 'sensitivity_zmax.png')

# --- bias ---
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(bias_factors, sigma_bias, 'C3', **STYLE, label='Multi-tracer (numerical)')
ax.axvline(1.0, color='grey', ls=':', lw=1.2, label='Fiducial bias')
ax.set_xlabel(r'Bias scaling factor', fontsize=12)
ax.set_ylabel(r'$\sigma(f_{\rm NL}^{\rm local})$', fontsize=12)
ax.set_title(r'Sensitivity to Galaxy Bias', fontsize=13)
ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
save_fig(fig, 'sensitivity_bias.png')

# --- density ---
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(density_factors, sigma_density, 'C4', **STYLE, label='Multi-tracer (numerical)')
ax.axvline(1.0, color='grey', ls=':', lw=1.2, label='Fiducial density')
ax.set_xlabel(r'Number density scaling factor', fontsize=12)
ax.set_ylabel(r'$\sigma(f_{\rm NL}^{\rm local})$', fontsize=12)
ax.set_title(r'Sensitivity to Galaxy Number Density', fontsize=13)
ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
save_fig(fig, 'sensitivity_density.png')

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("SENSITIVITY SUMMARY")
print("=" * 60)
print(f"{'Parameter':<30} {'sigma(fNL)':<12} {'Expected trend'}")
print("-" * 60)
print(f"{'f_sky scaling':<30} {'PASS' if abs(sigma_fsky[0]/sigma_fsky[2] - np.sqrt(0.75/0.25)) < 0.05 else 'FAIL'}")
print(f"  f_sky=0.25:  sigma = {sigma_fsky[0]:.3f}")
print(f"  f_sky=0.75:  sigma = {sigma_fsky[2]:.3f}  (ratio = {sigma_fsky[0]/sigma_fsky[2]:.3f}, expected {np.sqrt(0.75/0.25):.3f})")
print(f"\nell_max trend  (should decrease):  {'PASS' if sigma_ellmax[-1] < sigma_ellmax[0] else 'FAIL'}")
print(f"  ell_max= 50: sigma = {sigma_ellmax[0]:.3f}")
print(f"  ell_max=1000: sigma = {sigma_ellmax[-1]:.3f}")
print(f"\nz_max trend    (should decrease):  {'PASS' if sigma_zmax[-1] < sigma_zmax[0] else 'FAIL'}")
print(f"  z_max=1.0:   sigma = {sigma_zmax[0]:.3f}")
print(f"  z_max=4.6:   sigma = {sigma_zmax[-1]:.3f}")
print(f"\nBias trend     (higher → better):  {'PASS' if sigma_bias[-1] < sigma_bias[0] else 'FAIL'}")
print(f"  bias × 0.8:  sigma = {sigma_bias[0]:.3f}")
print(f"  bias × 1.2:  sigma = {sigma_bias[-1]:.3f}")
print(f"\nDensity trend  (higher → better):  {'PASS' if sigma_density[-1] < sigma_density[0] else 'FAIL'}")
print(f"  n × 0.5:     sigma = {sigma_density[0]:.3f}")
print(f"  n × 1.5:     sigma = {sigma_density[-1]:.3f}")
print("=" * 60)
