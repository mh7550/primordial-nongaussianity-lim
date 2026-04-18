# Constraining Primordial Non-Gaussianity with Multi-Tracer Line Intensity Mapping

Research project by Mahdi Hassanali — NYU Physics, Spring 2026  
Advisor: Professor Anthony Pullen

---

## Overview

This project implements a full forecasting and inference pipeline for constraining
primordial non-Gaussianity ($f_{\rm NL}^{\rm local}$) with the SPHEREx all-sky
spectral survey using line intensity mapping (LIM).  The pipeline combines the
multi-tracer Fisher matrix technique of Seljak (2009) with a Bayesian intensity
reconstruction framework following Cheng et al. (2024, arXiv:2403.19740).

Local-type primordial non-Gaussianity induces a scale-dependent correction to the
galaxy bias, $\Delta b \propto f_{\rm NL}/k^2$, which is amplified on large scales
accessible to SPHEREx.  Using four emission lines (H$\alpha$, [OIII], H$\beta$,
[OII]) as independent tracers suppresses cosmic variance and allows
$\sigma(f_{\rm NL}) < 1$ — better than the current Planck CMB constraint.

---

## Key Results

| Result | Value |
|--------|-------|
| **$\sigma(f_{\rm NL})$ multi-tracer** | **0.6–1.0** (~6× better than Planck) |
| **$\sigma(f_{\rm NL})$ single-tracer** | 1.8–3.0 |
| Planck CMB bound | 5.1 |
| H$\alpha$ 10$\sigma$ reach | z = 2.6 |
| [OIII] 10$\sigma$ reach | z = 2.5 |
| H$\beta$ 10$\sigma$ reach | z = 1.7 |
| [OII] 10$\sigma$ reach | z = 1.6 |
| S/N(H$\alpha$, z=1.5) | 43.2 |
| Deep-field/all-sky crossover | z ≈ 2.0 |
| Validation tests passing | **59/59** |

---

## Pipeline Structure

| File | Phase | Purpose |
|------|-------|---------|
| `src/cosmology.py` | 1 | Background cosmology: χ(z), H(z), D(z), P(k,z) |
| `src/limber.py` | 1 | Limber-approximated angular power spectra C_ℓ |
| `src/survey_specs.py` | 1 | SPHEREx survey specifications and noise model |
| `src/fisher.py` | 2 | Single- and multi-tracer Fisher matrix for f_NL |
| `src/lim_signal.py` | 3 | LIM signal: M_i(z), bias-weighted intensity, SFRD |
| `src/survey_configs.py` | 3 | SPHEREx 92-channel configuration, S/N vs z |
| `src/basis_functions.py` | 4 | ReLU piecewise-linear parameterization of M_i(z) |
| `src/wishart_likelihood.py` | 4 | Wishart log-likelihood and gradient (Eq. 28) |
| `src/newton_raphson.py` | 4 | Newton-Raphson optimizer with backtracking line search |
| `src/fisher_posterior.py` | 4 | Fisher information matrix and posterior constraints |
| `scripts/run_phase4_inference.py` | 4 | Full Bayesian inference pipeline + Figure 6 |
| `scripts/joint_analysis.py` | 5 | Joint results summary + 4-panel figure |
| `reports/final_report.tex` | 5 | Journal-format final paper (compile with pdflatex) |

**Tests** (all passing):

| Test file | Phase | Tests |
|-----------|-------|-------|
| `tests/test_phase3a_signal.py` | 3A | 12/12 |
| `tests/test_phase3b_power_spectrum.py` | 3B | 8/8 |
| `tests/test_phase3c_survey_comparison.py` | 3C | 14/14 |
| `tests/test_phase3d_validation.py` | 3D | 12/12 |
| `tests/test_phase4_inference.py` | 4 | 13/13 |
| **Total** | | **59/59** |

---

## How to Run

### Install dependencies
```bash
pip install numpy scipy matplotlib
```

### Run Phase 4 Bayesian inference (generates Figure 6)
```bash
python scripts/run_phase4_inference.py
```
Outputs: `scripts/phase4_results.npz`, `figures/phase4_figure6.png`

### Run Phase 5 joint analysis summary
```bash
python scripts/joint_analysis.py
```
Outputs: console results table, `figures/joint_summary.png`

### Run all validation tests
```bash
python -m pytest tests/ -v
```

### Compile the final report
```bash
cd reports && pdflatex final_report.tex && bibtex final_report && pdflatex final_report.tex
```

---

## References

- Cheng et al. 2024 — LIM theory review, arXiv:2403.19740
- Dalal et al. 2008 — Scale-dependent bias from PNG, PRD 77, 123514
- Doré et al. 2014 — SPHEREx science case, arXiv:1412.4872
- Madau & Dickinson 2014 — Cosmic SFRD, ARA&A 52, 415
- Planck Collaboration 2020 — PNG constraints, A&A 641, A9
- Seljak 2009 — Multi-tracer technique, PRL 102, 021302
