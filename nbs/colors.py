"""
Consistent color scheme for instrument families, built by sampling from
matplotlib colormaps.

Design:
- SAT     → sampled from 'summer'   (green → yellow-green → yellow)
- LAT     → sampled from 'cool'     (purple → magenta → blue → cyan)
- PK_lite → sampled from 'autumn'   (red → orange, trimmed before yellow)
- LB      → grey → yellow (hardcoded, LiteBIRD)

Defaults (index 0): SAT=green, LAT=purple, PK_lite=orange, LB=grey.
Higher indices diverge in hue (not just lightness) so same-family curves
stay visually distinct when overlaid.
"""
import matplotlib.cm as cm
import matplotlib.colors as mcolors


def _sample(cmap_name, positions):
    """Sample hex colors from a colormap at given [0,1] positions."""
    cmap = cm.get_cmap(cmap_name)
    return [mcolors.to_hex(cmap(p)) for p in positions]


# Default position for single-curve plots: first entry of each list.
FAMILY_PALETTES = {
    # summer: 0.0 teal-green → 1.0 yellow
    'SAT':     _sample('summer', [0.15, 0.45, 0.75, 0.95]),

    # cool: 0.0 cyan → 1.0 magenta
    # order: purple (default), magenta, blue, cyan
    'LAT':     _sample('cool',   [0.65, 0.95, 0.35, 0.05]),

    # autumn: 0.0 red → 1.0 yellow; trimmed before the yellow end
    # order: orange (default), red, deep red, burnt orange
    'PK_lite': _sample('autumn', [0.45, 0.0, 0.15, 0.3]),

    # LiteBIRD: hardcoded grey (default) + yellow
    'LB':      ['#7f7f7f', '#ffcc00'],
}


EXP_TO_FAMILY = {
    'SAT':            'SAT',
    'LAT':            'LAT',
    'LAT_pol':        'LAT',
    'LAT_cross':      'LAT',
    'PK':             'PK_lite',
    'PK_pol':         'PK_lite',
    'PK_cross':       'PK_lite',
    'PK_lite':        'PK_lite',
    'PK_lite_pol':    'PK_lite',
    'PK_lite_cross':  'PK_lite',
    'LB':             'LB',
}


def get_color(exp_key, index=0):
    """
    Return a hex color for a given experiment.

    Parameters
    ----------
    exp_key : str
        Experiment key, e.g. 'SAT', 'LAT_pol', 'PK_lite_cross'.
    index : int, default 0
        Which color within the family palette. 0 is the family default.
        Wraps around if index exceeds palette length.
    """
    family = EXP_TO_FAMILY[exp_key]
    palette = FAMILY_PALETTES[family]
    return palette[index % len(palette)]


def assign_colors(exp_keys):
    """
    Given a list of experiments to plot, hand each a color, cycling within
    family so same-family curves stay distinct.

    The first occurrence of a family gets palette[0] (the default), the
    second gets palette[1], and so on.
    """
    family_counter = {}
    colors = {}
    for exp in exp_keys:
        family = EXP_TO_FAMILY[exp]
        idx = family_counter.get(family, 0)
        colors[exp] = get_color(exp, idx)
        family_counter[family] = idx + 1
    return colors