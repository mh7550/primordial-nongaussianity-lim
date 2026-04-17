"""
run_phase4_inference.py — Main Phase 4 Bayesian inference script.

Runs the full Cheng et al. (2024) Section 5–6 inference pipeline:

  1. Generate fiducial data covariances for 8 redshift bins
  2. Run Newton-Raphson optimization to find MAP parameters
  3. Compute Fisher information matrix at MAP
  4. Report 1-sigma constraints and S/N per parameter
  5. Produce Figure 6 analog: recovered M_i(z) with 1-sigma bands

Usage
-----
    python scripts/run_phase4_inference.py [--no-plot]

Outputs
-------
    scripts/phase4_results.npz     — Fisher matrix, constraints, MAP estimates
    scripts/figure6_Mi_recovery.png — M_i(z) recovery plot (if matplotlib available)

References
----------
Cheng et al., Phys. Rev. D 109, 103011 (2024) — arXiv:2403.19740, Sections 5–6
"""

import numpy as np
import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from basis_functions import (
    EMISSION_LINES, N_M, Z_EVAL, Z_ANCHORS, FIDUCIAL_MIJ,
    get_fiducial_theta, mij_from_theta, cij_from_mij, evaluate_Mi,
    theta_from_mij,
)
from wishart_likelihood import (
    wishart_log_likelihood, make_fiducial_data_covariance,
)
from newton_raphson import newton_raphson_optimize
from fisher_posterior import (
    compute_fisher_matrix, compute_parameter_constraints,
    compute_snr_per_parameter, summarize_constraints, compute_line_constraints,
)
from survey_configs import SurveyConfig


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

Z_BINS = np.array([0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
ELL_BINS = np.array([[50, 150], [150, 300], [300, 500]])
FD_DELTA = 0.02
SURVEY = SurveyConfig.get_config('deep_field')

LINE_COLORS = {
    'Halpha': '#1f77b4',   # blue
    'OIII':   '#ff7f0e',   # orange
    'Hbeta':  '#2ca02c',   # green
    'OII':    '#d62728',   # red
}

LINE_LABELS = {
    'Halpha': r'H$\alpha$',
    'OIII':   r'[OIII]',
    'Hbeta':  r'H$\beta$',
    'OII':    r'[OII]',
}


def build_data_covariances():
    """Build fiducial data covariances at each z in Z_BINS."""
    print("Building fiducial data covariances...")
    C_data = {}
    for z in Z_BINS:
        C_data[z] = make_fiducial_data_covariance(z, SURVEY)
    return C_data


def run_optimization(theta_init, C_data, verbose=True):
    """Run Newton-Raphson optimization from theta_init."""
    print("\nRunning Newton-Raphson optimization...")
    result = newton_raphson_optimize(
        theta_init, C_data, Z_BINS, SURVEY,
        ell_bins=ELL_BINS,
        max_iter=30,
        tol_grad=1.0,
        tol_step=1e-4,
        verbose=verbose,
        fd_delta=FD_DELTA,
    )
    print(f"  Converged: {result['converged']}, iterations: {result['n_iter']}")
    print(f"  Final log L: {result['log_L']:.6e}")
    return result


def run_fisher_analysis(theta_map, verbose=True):
    """Compute Fisher matrix and posterior constraints at MAP."""
    print("\nComputing Fisher information matrix...")
    F = compute_fisher_matrix(
        theta_map, Z_BINS, SURVEY, ELL_BINS, fd_delta=FD_DELTA
    )
    print(f"  F shape: {F.shape}, rank: {np.linalg.matrix_rank(F)}")

    if verbose:
        summarize_constraints(theta_map, F, label="Phase 4 Posterior Constraints")

    line_res = compute_line_constraints(F, theta_map)
    print("\nPer-line constraints:")
    for line in EMISSION_LINES:
        res = line_res[line]
        print(f"  {line:<8}: sigma_mean = {res['sigma_mean']:.4f}, "
              f"snr_mean = {res['snr_mean']:.1f}")

    return F, line_res


def compute_recovered_Mi(theta_map, sigma):
    """
    Compute recovered M_i(z) ± 1-sigma band over a fine redshift grid.

    Returns dict: {line: (z_grid, M_central, M_upper, M_lower)}
    """
    z_grid = np.linspace(0.1, 6.0, 100)
    results = {}

    for i_line, line in enumerate(EMISSION_LINES):
        sl = slice(i_line * N_M, (i_line + 1) * N_M)
        m_map = mij_from_theta(theta_map[sl])
        sigma_line = sigma[sl]

        c_map = cij_from_mij(m_map)
        M_central = evaluate_Mi(c_map, z_grid)

        # Upper / lower: perturb m_ij by ±1 sigma
        m_up = np.exp(theta_map[sl] + sigma_line)
        m_dn = np.exp(theta_map[sl] - sigma_line)
        M_upper = evaluate_Mi(cij_from_mij(m_up), z_grid)
        M_lower = evaluate_Mi(cij_from_mij(m_dn), z_grid)
        M_lower = np.maximum(M_lower, 0)

        results[line] = (z_grid, M_central, M_upper, M_lower)

    return results


def compute_fiducial_Mi():
    """Compute fiducial M_i(z) over a fine redshift grid for comparison."""
    z_grid = np.linspace(0.1, 6.0, 100)
    results = {}
    for line in EMISSION_LINES:
        c_fid = cij_from_mij(FIDUCIAL_MIJ[line])
        M_fid = evaluate_Mi(c_fid, z_grid)
        results[line] = (z_grid, M_fid)
    return results


def make_figure6(recovered, fiducial, output_path):
    """
    Figure 6 analog: recovered M_i(z) with 1-sigma bands vs fiducial.

    Parameters
    ----------
    recovered : dict {line: (z_grid, M_central, M_upper, M_lower)}
    fiducial : dict {line: (z_grid, M_fid)}
    output_path : str
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping Figure 6.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    for ax, line in zip(axes, EMISSION_LINES):
        color = LINE_COLORS[line]
        label = LINE_LABELS[line]

        z_grid, M_c, M_up, M_dn = recovered[line]
        z_fid, M_fid = fiducial[line]

        ax.fill_between(z_grid, M_dn, M_up, alpha=0.3, color=color,
                        label=r'MAP $\pm 1\sigma$')
        ax.plot(z_grid, M_c, color=color, lw=2, label='MAP')
        ax.plot(z_fid, M_fid, color=color, lw=1.5, ls='--',
                alpha=0.7, label='Fiducial')
        ax.scatter(Z_EVAL, FIDUCIAL_MIJ[line], color=color, s=30, zorder=5)

        ax.set_xlabel('Redshift $z$')
        ax.set_ylabel(r'$M_i(z)$ [erg/s/Mpc$^3$]')
        ax.set_title(label, fontsize=13)
        ax.set_yscale('log')
        ax.set_xlim(0, 6.5)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)

    fig.suptitle('Phase 4: Recovered $M_i(z)$ with 1σ Posterior Bands\n'
                 '(Cheng+2024 Figure 6 analog)', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nFigure 6 saved: {output_path}")


def save_results(theta_map, F, sigma, result, output_path):
    """Save key outputs to .npz file."""
    np.savez(
        output_path,
        theta_map=theta_map,
        theta_fid=get_fiducial_theta(),
        F=F,
        sigma=sigma,
        z_bins=Z_BINS,
        z_eval=Z_EVAL,
        history_L=np.array(result['history_L']),
        fiducial_mij_Halpha=FIDUCIAL_MIJ['Halpha'],
        fiducial_mij_OIII=FIDUCIAL_MIJ['OIII'],
        fiducial_mij_Hbeta=FIDUCIAL_MIJ['Hbeta'],
        fiducial_mij_OII=FIDUCIAL_MIJ['OII'],
    )
    print(f"Results saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Phase 4 LIM Bayesian inference")
    parser.add_argument('--no-plot', action='store_true',
                        help="Skip Figure 6 generation")
    parser.add_argument('--perturb', type=float, default=0.0,
                        help="Perturb initial theta by this amount (log-space, default 0)")
    args = parser.parse_args()

    print("=" * 60)
    print("Phase 4: Bayesian Inference — Cheng+2024 Section 5–6")
    print("=" * 60)
    print(f"Survey: {SURVEY.name}, f_sky={SURVEY.f_sky}")
    print(f"Z bins: {Z_BINS}")
    print(f"Ell bins: {ELL_BINS.tolist()}")

    # Step 1: data covariances
    C_data = build_data_covariances()

    # Step 2: optimization
    theta_init = get_fiducial_theta()
    if args.perturb != 0.0:
        rng = np.random.default_rng(42)
        theta_init = theta_init + args.perturb * rng.standard_normal(len(theta_init))
        print(f"\nStarting from perturbed theta (perturbation={args.perturb})")

    result = run_optimization(theta_init, C_data, verbose=True)
    theta_map = result['theta_opt']

    # Step 3: Fisher analysis
    F, line_res = run_fisher_analysis(theta_map, verbose=True)
    sigma, rho, Sigma = compute_parameter_constraints(F)

    # Step 4: recovered M_i(z) and Figure 6
    recovered = compute_recovered_Mi(theta_map, sigma)
    fiducial = compute_fiducial_Mi()

    if not args.no_plot:
        fig_path = os.path.join(os.path.dirname(__file__), 'figure6_Mi_recovery.png')
        make_figure6(recovered, fiducial, fig_path)

    # Step 5: save results
    results_path = os.path.join(os.path.dirname(__file__), 'phase4_results.npz')
    save_results(theta_map, F, sigma, result, results_path)

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    snr = compute_snr_per_parameter(theta_map, F)
    well_constrained = ~np.isnan(snr) & (snr < 1e10)
    print(f"Well-constrained parameters: {np.sum(well_constrained)} / {len(theta_map)}")
    if np.any(well_constrained):
        print(f"Median S/N (well-constrained): {np.nanmedian(snr[well_constrained]):.1f}")
        print(f"Min S/N: {np.nanmin(snr[well_constrained]):.1f}")
        print(f"Max S/N: {np.nanmax(snr[well_constrained]):.1f}")

    return theta_map, F, sigma, result


if __name__ == '__main__':
    theta_map, F, sigma, result = main()
