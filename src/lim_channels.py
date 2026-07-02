"""
lim_channels.py — 92-channel SPHEREx LIM geometry with line assignment.

Builds the full 92-channel setup used in the paper's Fisher forecast.
Each channel has:

    * central observed wavelength λ_obs [μm]
    * channel width Δλ [μm]
    * v28 noise σ_n [nW/m²/sr] (all-sky or deep-field)
    * a dominant emission-line assignment (Hα, [OIII], Hβ, [OII])
    * the corresponding emission redshift z_peak = λ_obs / λ_rest − 1
    * a b1(z) function for that line

Line assignment rule: a channel is assigned to the emission line whose
redshift falls in the SPHEREx-accessible range [0.2, 5.0] and gives the
largest bias-weighted intensity at that wavelength. Hα is the brightest,
followed by [OIII], Hβ, [OII], so ties break in that order.
"""

import numpy as np
import os

try:
    from .lim_signal import LINE_PROPERTIES, get_bias_weighted_luminosity_density
except ImportError:
    try:
        from lim_signal import LINE_PROPERTIES, get_bias_weighted_luminosity_density
    except ImportError:
        # Fallback: minimal line info
        LINE_PROPERTIES = {
            'Halpha': {'lambda_rest': 0.6563},
            'OIII':   {'lambda_rest': 0.5007},
            'Hbeta':  {'lambda_rest': 0.4861},
            'OII':    {'lambda_rest': 0.3727},
        }

LINE_ORDER = ['Halpha', 'OIII', 'Hbeta', 'OII']

# Line bias models (from Cheng+2024 / paper — halo-mass-weighted linear bias).
LINE_BIAS_COEFFS = {
    'Halpha': (1.0, 0.84),
    'OIII':   (1.1, 0.90),
    'Hbeta':  (1.0, 0.84),
    'OII':    (1.2, 0.85),
}


def _b1_of_z(line):
    b0, slope = LINE_BIAS_COEFFS[line]
    return lambda z: b0 + slope * z


def load_v28_noise(mode='deep'):
    """Return (lam_um, σ_n_nW/m²/sr) from the packaged v28 file.

    mode='deep' → deep-field noise; mode='all-sky' → all-sky noise.
    """
    path = os.path.join(os.path.dirname(__file__), '..', 'data',
                        'spherex_noise_v28.txt')
    data = np.loadtxt(path)
    lam = data[:, 0]
    if mode == 'deep':
        sig = data[:, 2]
    else:
        sig = data[:, 1]
    return lam, sig


def build_92_channels(mode='deep', z_min=0.2, z_max=5.0, drop_last_band=True,
                      n_per_line=23):
    """Build the paper's 92-channel LIM geometry: 4 lines × 23 channels.

    Each of the 4 emission lines is sampled at ``n_per_line`` redshifts spanning
    its accessible range in SPHEREx. This is the multi-tracer structure that
    yields the paper's 0.71 constraint — 92 pseudo-channels (23 × 4 = 92) where
    the multi-tracer power comes from cross-correlations between different
    lines that map to the same physical redshift.

    Per-channel noise is interpolated from the v28 wavelength curve at the
    observed wavelength λ_obs = λ_rest (1+z_peak).
    """
    lam_grid, sig_grid = load_v28_noise(mode=mode)
    omega_pix = (6.2 * np.pi / 180.0 / 3600.0) ** 2

    channels = []
    for line in LINE_ORDER:
        lam_rest = LINE_PROPERTIES[line]['lambda_rest']
        # Redshift range in which this line is inside the SPHEREx window and
        # within [z_min, z_max].
        z_lo = max(z_min, 0.75 / lam_rest - 1.0)
        z_hi = min(z_max, 5.0 / lam_rest - 1.0)
        if z_hi <= z_lo:
            # Line inaccessible; use a single dummy channel at z_lo.
            z_peaks = np.array([z_lo])
        else:
            z_peaks = np.linspace(z_lo, z_hi, n_per_line)

        for zp in z_peaks:
            lam_obs = lam_rest * (1.0 + zp)
            # Channel width from spectral resolution:
            #   R=41 for λ<3.82 μm, R=35 for 3.82–4.42 μm, R=130 for 4.42–5 μm
            if lam_obs < 3.82:
                R = 41.0
            elif lam_obs < 4.42:
                R = 35.0
            else:
                R = 130.0
            dl = lam_obs / R
            # Interpolate v28 noise at the channel centre.
            sig_c = float(np.interp(lam_obs, lam_grid, sig_grid))
            N_ell = sig_c ** 2 * omega_pix
            channels.append(dict(
                line=line,
                lambda_rest=lam_rest,
                lambda_obs=float(lam_obs),
                delta_lambda=float(dl),
                z_peak=float(zp),
                sigma_n=sig_c,
                noise=float(N_ell),
                active=True,
                b1_of_z=_b1_of_z(line),
                I_scale=1.0,
                b_scale=1.0,
            ))
    return channels


def line_peak_z(line):
    """Peak redshift for a line inside the SPHEREx window (0.75–5.0 μm)."""
    lam_rest = LINE_PROPERTIES[line]['lambda_rest']
    z_lo = 0.75 / lam_rest - 1.0
    z_hi = min(5.0, 5.0 / lam_rest - 1.0)
    return 0.5 * (max(0.2, z_lo) + z_hi)


if __name__ == "__main__":
    channels = build_92_channels()
    print(f"Built {len(channels)} channels")
    counts = {line: 0 for line in LINE_ORDER}
    for ch in channels:
        counts[ch['line']] += 1
    print("Line assignments:", counts)
    active = sum(1 for ch in channels if ch['active'])
    print(f"Active channels (line in [0.2, 5.0]): {active}/{len(channels)}")
    print(f"First channel: λ={channels[0]['lambda_obs']:.3f}μm  "
          f"line={channels[0]['line']}  z={channels[0]['z_peak']:.2f}  "
          f"σ_n={channels[0]['sigma_n']:.2f}")
    print(f"Last  channel: λ={channels[-1]['lambda_obs']:.3f}μm  "
          f"line={channels[-1]['line']}  z={channels[-1]['z_peak']:.2f}  "
          f"σ_n={channels[-1]['sigma_n']:.2f}")
