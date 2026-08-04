"""Group-name parsing, labels, colors, and small utilities."""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from .config import ExperimentConfig


def parse_group_meta(
    group_name: str,
    config: ExperimentConfig,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Return (treatment, sex, timepoint) from a group name like cort_b2_m."""
    g = str(group_name).lower()
    tokens = g.split('_')

    treatment = None
    for tx in config.treatments:
        if g.startswith(tx) or tx in tokens:
            treatment = tx
            break

    sex = None
    for code, label in config.sex_codes.items():
        if g.endswith(f'_{code}') or f'_{code}_' in g or tokens[-1] == code:
            sex = label
            break

    timepoint = next((key for key in config.time_order if key in tokens), None)
    return treatment, sex, timepoint


def color_for_group(group_name: str, config: ExperimentConfig) -> str:
    """Locked palette color for treatment x sex x timepoint."""
    treatment, sex, timepoint = parse_group_meta(group_name, config)
    default_key = config.default_palette_key
    palette = config.time_color_palettes.get(
        (treatment or default_key[0], sex or default_key[1]),
        config.time_color_palettes.get(default_key),
    )
    if palette is None:
        palette = ['#0072B2']
    if timepoint in config.time_order:
        idx = list(config.time_order).index(timepoint)
        return palette[min(idx, len(palette) - 1)]
    return palette[0]


def timepoint_label_from_group(group_name: str, config: ExperimentConfig) -> str:
    """Map group name to a display timepoint label."""
    _, _, timepoint = parse_group_meta(group_name, config)
    if timepoint in config.timepoint_labels:
        return config.timepoint_labels[timepoint]
    return str(group_name)


def pretty_group_label(group_name: str, config: ExperimentConfig) -> str:
    """e.g. cort_b2_m -> 'CORT Baseline Male'."""
    treatment, sex, timepoint = parse_group_meta(group_name, config)
    parts = []
    if treatment:
        parts.append(treatment.upper())
    if timepoint in config.timepoint_labels:
        parts.append(config.timepoint_labels[timepoint])
    if sex:
        parts.append(sex.capitalize())
    return ' '.join(parts) if parts else str(group_name)


def legend_label_for_groups(
    groups: Sequence[str],
    config: ExperimentConfig,
) -> List[str]:
    """Legend labels for time series or sex-contrast plots."""
    metas = [parse_group_meta(g, config) for g in groups]
    sexes = {m[1] for m in metas}
    timepoints = {m[2] for m in metas}
    sex_labels = set(config.sex_codes.values())

    if len(timepoints) == 1 and sexes == sex_labels:
        return [
            sex.capitalize() if sex else str(g)
            for g, (_, sex, _) in zip(groups, metas)
        ]

    if len(sexes) == 1 and len(timepoints) > 1:
        return [timepoint_label_from_group(g, config) for g in groups]

    labels = []
    for g, (_, sex, tp) in zip(groups, metas):
        parts = []
        if sex:
            parts.append(sex.capitalize())
        if tp in config.timepoint_labels:
            parts.append(config.timepoint_labels[tp])
        labels.append(' '.join(parts) if parts else str(g))
    return labels


def infer_sex_from_groups(
    groups: Sequence[str],
    config: ExperimentConfig,
) -> Optional[str]:
    tags = {parse_group_meta(g, config)[1] for g in groups}
    tags.discard(None)
    return tags.pop() if len(tags) == 1 else None


def infer_treatment_from_groups(
    groups: Sequence[str],
    config: ExperimentConfig,
) -> Optional[str]:
    tags = {parse_group_meta(g, config)[0] for g in groups}
    tags.discard(None)
    return tags.pop() if len(tags) == 1 else None


def time_scale_colors(
    n: int,
    config: ExperimentConfig,
    sex: Optional[str] = None,
    treatment: Optional[str] = None,
    cmap_name: Optional[str] = None,
) -> List:
    """Return n colors along a treatment x sex time palette."""
    if n < 1:
        return []
    if cmap_name is not None:
        cmap = plt.get_cmap(cmap_name)
        lo, hi = (0.5, 0.5) if n == 1 else (0.12, 0.88)
        return [cmap(x) for x in np.linspace(lo, hi, n)]

    default_key = config.default_palette_key
    base = config.time_color_palettes.get(
        (treatment or default_key[0], sex or default_key[1]),
        config.time_color_palettes.get(default_key, ['#0072B2']),
    )
    if n == len(base):
        return list(base)
    cmap = LinearSegmentedColormap.from_list('time_tx_sex', base)
    return [cmap(x) for x in np.linspace(0.0, 1.0, n)]


def time_scale_markers(n: int, config: ExperimentConfig) -> List[str]:
    markers = list(config.time_markers)
    return [markers[i % len(markers)] for i in range(n)]


def format_usage_plot_title(
    categories=None,
    sex: Optional[str] = None,
    treatment: Optional[str] = None,
) -> str:
    """Build title like: Locomotion Syllables (Female CORT)."""
    if categories is None:
        title = 'Syllables'
    else:
        if isinstance(categories, str):
            categories = [categories]
        title = f"{', '.join(c.capitalize() for c in categories)} Syllables"

    bits = []
    if sex:
        bits.append(sex.capitalize())
    if treatment:
        bits.append(treatment.upper())
    if bits:
        title += f" ({' '.join(bits)})"
    return title


def strip_sex_suffix(group_name: str, sex_codes: Optional[Iterable[str]] = None) -> str:
    """Remove trailing sex suffix (e.g. _m / _f) from a group name."""
    group_name = str(group_name)
    codes = tuple(sex_codes) if sex_codes is not None else ('m', 'f')
    for code in codes:
        suffix = f'_{code}'
        if group_name.endswith(suffix):
            return group_name[: -len(suffix)]
    return group_name


def shannon_entropy(probs) -> float:
    """Shannon entropy (bits) of a probability vector."""
    probs = np.asarray(probs, dtype=float)
    probs = probs / (probs.sum() + np.finfo(float).eps)
    return -(probs * np.log2(probs + np.finfo(float).eps)).sum()
