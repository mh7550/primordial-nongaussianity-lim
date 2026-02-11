"""
validate_pk_vs_class.py
=======================
Validates the project's get_power_spectrum() against a direct CLASS computation
using Planck 2018 cosmological parameters.

Run from the repository root:
    python scripts/validate_pk_vs_class.py

Unit convention:
    - CLASS pk(k, z) : k in Mpc^-1, P in Mpc^3
    - Our code        : k in h/Mpc,  P in (Mpc/h)^3
    - Conversion      : k_Mpc = k_hMpc * h
                        P_hMpc3 = P_Mpc3 * h^3
"""

import sys
import os

# Allow 'src' imports when run from repo root
sys.path.insert(0, '.')

import numpy as np
import matplotlib
matplotlib.use('Agg')          # non-interactive backend
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 1. Import our cosmology module
# ---------------------------------------------------------------------------
from src.cosmology import get_power_spectrum

# ---------------------------------------------------------------------------
# 2. Planck 2018 parameters for CLASS
#    These are the *exact* parameters requested; note that src/cosmology.py
#    uses Planck18 from astropy (h=0.6766, but ns=0.9649, not 0.9665) so
#    small differences are expected.
# ---------------------------------------------------------------------------
H_VALUE   = 0.6766
OMEGA_B   = 0.02242          # Ω_b h²
OMEGA_CDM = 0.11933          # Ω_cdm h²
NS        = 0.9665
AS        = 2.105e-9
K_PIVOT   = 0.05             # Mpc^-1
P_K_MAX   = 10.0             # h/Mpc  (CLASS param is in h/Mpc for P_k_max_h/Mpc)
Z_MAX_PK  = 4.0

# Wavenumbers for the comparison [h/Mpc]
K_TEST = np.array([0.001, 0.005, 0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 1.0])

# Redshifts for the comparison
Z_TEST = [0, 0.5, 1.0, 2.0]

# PASS band for ratio P_ours / P_CLASS
RATIO_LO = 0.7
RATIO_HI = 1.3

# ---------------------------------------------------------------------------
# 3. Run CLASS
# ---------------------------------------------------------------------------
print("=" * 68)
print("Running CLASS with Planck 2018 parameters ...")
print(f"  h           = {H_VALUE}")
print(f"  omega_b     = {OMEGA_B}")
print(f"  omega_cdm   = {OMEGA_CDM}")
print(f"  n_s         = {NS}")
print(f"  A_s         = {AS}")
print(f"  k_pivot     = {K_PIVOT} Mpc^-1")
print("=" * 68)

try:
    from classy import Class
except ImportError as exc:
    sys.exit(
        f"ERROR: classy not found ({exc}).\n"
        "Install CLASS and its Python wrapper: pip install classy"
    )

cosmo = Class()
cosmo.set({
    'h':             H_VALUE,
    'omega_b':       OMEGA_B,
    'omega_cdm':     OMEGA_CDM,
    'n_s':           NS,
    'A_s':           AS,
    'k_pivot':       K_PIVOT,      # Mpc^-1
    'output':        'mPk',
    'P_k_max_h/Mpc': P_K_MAX,
    'z_max_pk':      Z_MAX_PK,
})
cosmo.compute()
print("CLASS computation complete.\n")

h = H_VALUE   # shorthand

# ---------------------------------------------------------------------------
# Helper: CLASS P(k) in Mpc^3  →  (Mpc/h)^3
# ---------------------------------------------------------------------------
def class_pk_hMpc(k_hMpc, z):
    """Return P(k) in (Mpc/h)^3, given k in h/Mpc."""
    k_Mpc = k_hMpc * h           # convert to Mpc^-1
    P_Mpc3 = cosmo.pk(k_Mpc, z)  # CLASS returns Mpc^3
    return P_Mpc3 * h**3          # convert to (Mpc/h)^3

# ---------------------------------------------------------------------------
# 4. Comparison table
# ---------------------------------------------------------------------------
all_pass = True

for z in Z_TEST:
    print(f"--- z = {z} ---")
    print(f"{'k [h/Mpc]':>12}  {'P_ours [(Mpc/h)^3]':>20}  "
          f"{'P_CLASS [(Mpc/h)^3]':>20}  {'ratio':>8}  status")
    print("-" * 74)

    for k in K_TEST:
        P_ours  = get_power_spectrum(k, z)
        P_class = class_pk_hMpc(k, z)
        ratio   = P_ours / P_class
        ok      = RATIO_LO <= ratio <= RATIO_HI
        status  = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"{k:>12.4f}  {P_ours:>20.4e}  {P_class:>20.4e}  "
              f"{ratio:>8.4f}  {status}")
    print()

# ---------------------------------------------------------------------------
# 5. Generate validation figure
# ---------------------------------------------------------------------------
figures_dir = os.path.join(os.path.dirname(__file__), '..', 'figures')
figures_dir = os.path.normpath(figures_dir)
os.makedirs(figures_dir, exist_ok=True)
output_fig  = os.path.join(figures_dir, 'validation_pk_vs_class.png')

# Dense k grid for the left panel (smooth curves)
k_dense = np.logspace(np.log10(K_TEST[0]), np.log10(K_TEST[-1]), 200)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# ---- Left panel: P(k) at z = 0 ----
ax = axes[0]
z_left = 0.0
P_ours_dense  = np.array([get_power_spectrum(k, z_left) for k in k_dense])
P_class_dense = np.array([class_pk_hMpc(k, z_left)     for k in k_dense])

ax.loglog(k_dense, P_ours_dense,  color='C0', lw=2,   label='This work (E&H transfer fn)')
ax.loglog(k_dense, P_class_dense, color='C1', lw=2,
          ls='--', label='CLASS (Planck 2018)')

# Mark the discrete test points
P_ours_pts  = np.array([get_power_spectrum(k, z_left) for k in K_TEST])
P_class_pts = np.array([class_pk_hMpc(k, z_left)      for k in K_TEST])
ax.scatter(K_TEST, P_ours_pts,  color='C0', zorder=5, s=30)
ax.scatter(K_TEST, P_class_pts, color='C1', zorder=5, s=30)

ax.set_xlabel(r'$k$ [$h$/Mpc]', fontsize=12)
ax.set_ylabel(r'$P(k)$ [$({\rm Mpc}/h)^3$]', fontsize=12)
ax.set_title(r'Matter Power Spectrum at $z=0$', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, which='both', alpha=0.3)
ax.set_xlim(K_TEST[0] * 0.8, K_TEST[-1] * 1.2)

# ---- Right panel: ratio P_ours / P_CLASS at z = 0, 1, 2 ----
ax = axes[1]
colors_z = ['C0', 'C2', 'C3']
z_ratio_list = [0, 1.0, 2.0]

for zi, col in zip(z_ratio_list, colors_z):
    ratios = np.array([
        get_power_spectrum(k, zi) / class_pk_hMpc(k, zi)
        for k in K_TEST
    ])
    ax.semilogx(K_TEST, ratios, marker='o', color=col,
                lw=1.8, ms=6, label=f'$z={zi}$')

# ±10 % dashed reference lines
ax.axhline(1.10, color='grey', ls='--', lw=1.2, label='±10%')
ax.axhline(0.90, color='grey', ls='--', lw=1.2)
ax.axhline(1.00, color='black', ls='-',  lw=0.8, alpha=0.5)

ax.set_xlabel(r'$k$ [$h$/Mpc]', fontsize=12)
ax.set_ylabel(r'$P_{\rm ours}(k) / P_{\rm CLASS}(k)$', fontsize=12)
ax.set_title('Ratio: This Work / CLASS', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, which='both', alpha=0.3)
ax.set_xlim(K_TEST[0] * 0.8, K_TEST[-1] * 1.2)
ax.set_ylim(0.5, 1.5)

fig.suptitle(
    'Power Spectrum Validation: This Work vs CLASS (Planck 2018)',
    fontsize=14, y=1.02
)
fig.tight_layout()
fig.savefig(output_fig, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"Figure saved to: {output_fig}")

# ---------------------------------------------------------------------------
# 6. Cleanup CLASS and print overall result
# ---------------------------------------------------------------------------
cosmo.struct_cleanup()
cosmo.empty()

print()
print("=" * 68)
if all_pass:
    print("OVERALL RESULT: PASS")
    print(f"  All ratios P_ours/P_CLASS are within [{RATIO_LO}, {RATIO_HI}]")
else:
    print("OVERALL RESULT: FAIL")
    print(f"  Some ratios P_ours/P_CLASS fell outside [{RATIO_LO}, {RATIO_HI}]")
    print("  This may indicate a significant discrepancy in the power spectrum.")
print("=" * 68)
