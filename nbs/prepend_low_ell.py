"""
Hard-coded prepend of the two Planck low-ell TT bins (Prince & Dunkley 2019,
2018 release) into an existing lite_data dict.

Values come straight from planck-lite-py/data/:
    CTT_bin_low_ell_2018.dat, plmin_low_ell.dat, blmax_low_ell.dat.

Units match lite_data: raw C_ell in μK²; cov stores diagonal variance σ² in μK⁴.
TE and EE are untouched — planck-lite-py doesn't supply low-ell polarization.
"""

import numpy as np

# ---- the two low-ell TT bins (planck-lite-py, 2018) -----------------------
_LOW_ELL_TT = {
    'ell'  : np.array([ 8.5,   22.5]),                         # bin centers
    'dell' : np.array([14,     14   ], dtype=int),             # widths (ell ∈ [2,15] and [16,29])
    'c_ell': np.array([53.43946266415627,  9.695704299702811]),# mean C_ell (μK²)
    # cov stored as variance σ² to match lite_data convention; σ was 4.9229 and 0.58797 μK²
    'cov'  : np.array([24.235023922831867,  0.345706135681528]),
}


def prepend_planck_low_ell(lite_data: dict) -> dict:
    """Return a new dict with the two low-ell TT bins prepended to lite_data['TT'].
    TE/EE unchanged. Input is not mutated."""
    from copy import deepcopy
    out = deepcopy(lite_data)
    for key in ('c_ell', 'cov', 'ell', 'dell'):
        out['TT'][key] = np.concatenate([_LOW_ELL_TT[key], out['TT'][key]])
    return out