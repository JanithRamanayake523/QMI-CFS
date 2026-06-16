"""Shared plotting style and color palette for publication figures."""
import matplotlib
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.style as mstyle


def set_publication_style():
    """Apply ggplot style and publication-quality font defaults."""
    mstyle.use('ggplot')
    matplotlib.rcParams.update({
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 12,
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.dpi': 150,
    })


# Consistent viridis palette across all paper figures.
_METHOD_ORDER = [
    'QMI-CFS',
    'QMI-CFS-Classic',
    'Boruta',
    'CMI',
    'LASSO',
    'MI',
    'All-Features',
]

_COLORS_ARRAY = cm.viridis([i / (len(_METHOD_ORDER) - 1)
                            for i in range(len(_METHOD_ORDER))])
METHOD_COLORS = {m: mcolors.to_hex(c) for m, c in zip(_METHOD_ORDER, _COLORS_ARRAY)}


def get_color(method: str) -> str:
    """Return the viridis color for a method name."""
    return METHOD_COLORS.get(method, '#7f8c8d')
