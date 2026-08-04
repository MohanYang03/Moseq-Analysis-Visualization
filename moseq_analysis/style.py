"""Journal plotting style defaults."""

from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns

JOURNAL_COLORS = [
    '#0072B2', '#E69F00', '#009E73', '#CC79A7',
    '#56B4E9', '#D55E00', '#F0E442', '#000000',
]
JOURNAL_CMAP = 'cividis'

JOURNAL_RC = {
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 8,
    'axes.titlesize': 9,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'legend.title_fontsize': 7,
    'legend.frameon': False,
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.25,
    'lines.markersize': 4.5,
    'errorbar.capsize': 3,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.prop_cycle': plt.cycler(color=JOURNAL_COLORS),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
}


def apply_journal_style(rc: dict = None) -> None:
    """Apply journal-ready matplotlib / seaborn defaults."""
    params = dict(JOURNAL_RC)
    if rc:
        params.update(rc)
    # Rebuild cycler in case JOURNAL_COLORS is referenced after import
    if 'axes.prop_cycle' not in (rc or {}):
        params['axes.prop_cycle'] = plt.cycler(color=JOURNAL_COLORS)
    plt.rcParams.update(params)
    sns.set_theme(style='ticks', rc=params)
