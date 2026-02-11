"""
primordial_nongaussianity_lim
==============================
Fisher matrix forecast for primordial non-Gaussianity (PNG) with the
SPHEREx multi-tracer line-intensity mapping survey.

Modules
-------
cosmology
    Planck 2018 matter power spectrum, growth factor, and transfer function
    (Eisenstein & Hu 1998).

bias_functions
    Scale-dependent galaxy bias from local, equilateral, and orthogonal PNG
    (Dalal et al. 2008; Sefusatti & Komatsu 2007).

limber
    Angular power spectra C_ell via the Limber approximation
    (LoVerde & Afshordi 2008).

survey_specs
    SPHEREx survey geometry, galaxy samples, biases, and shot-noise model
    (Doré et al. 2014).

fisher
    Single- and multi-tracer Fisher matrix for f_NL (Seljak 2009;
    Hamaus et al. 2012).

Key Result
----------
For the full SPHEREx survey (f_sky = 0.75, z ∈ [0, 4.6], ell ∈ [2, 200]):
    sigma(f_NL^local) ~ 0.6-1.0  (5 galaxy samples, multi-tracer)

References
----------
Dalal et al., PRD 77, 123514 (2008)
Seljak, JCAP 0903, 007 (2009)
Doré et al., arXiv:1412.4872 (2014)
Planck Collaboration, A&A 641, A9 (2020)
"""
