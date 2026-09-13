"""Fingerprint plot helpers (optional; requires moseq2_viz)."""

from __future__ import annotations

from os import makedirs
from os.path import join
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec
from sklearn.preprocessing import MinMaxScaler

from .config import ExperimentConfig
from .grouping import color_for_group, pretty_group_label

try:
    from moseq2_viz.model.fingerprint_classifier import create_fingerprint_dataframe
except ImportError:  # pragma: no cover - optional dependency
    create_fingerprint_dataframe = None


def load_moseq_filtered(csv_path, groups, chunksize=250000):
    """Stream-filter the large moseq_df.csv down to selected groups."""
    chunks = []
    n_kept = 0
    for i, chunk in enumerate(pd.read_csv(csv_path, chunksize=chunksize)):
        subset = chunk[chunk['group'].isin(groups)]
        if len(subset):
            chunks.append(subset)
            n_kept += len(subset)
        if (i + 1) % 20 == 0:
            print(f'  scanned {(i + 1) * chunksize:,} rows; kept {n_kept:,}')
    if not chunks:
        raise ValueError(f'No rows found for groups: {groups}')
    out = pd.concat(chunks, ignore_index=True)
    print(f'Loaded moseq_df subset: {out.shape[0]:,} rows x {out.shape[1]} columns')
    return out


def plotting_fingerprint_colored(
    summary,
    save_dir,
    range_dict,
    group_colors,
    figsize=(20, 18),
    preprocessor=None,
    level_names=None,
    vmin=None,
    vmax=None,
    plot_columns=None,
    col_names=None,
    filename_stem='moseq_fingerprint',
    eps_dir: Optional[str] = None,
    png_dir: Optional[str] = None,
    png_dpi: int = 300,
):
    """Fingerprint heatmap with a custom colored group strip.

    EPS and PNG are written to ``eps_dir`` / ``png_dir`` when provided;
    otherwise both formats (plus PDF) are written under ``save_dir``.
    """
    if level_names is None:
        level_names = ['Group']
    if plot_columns is None:
        plot_columns = [
            'dist_to_center_px', 'velocity_2d_mm', 'height_ave_mm', 'length_mm', 'MoSeq',
        ]
    if col_names is None:
        col_names = [
            ('Position', 'Dist. from center (px)'),
            ('Speed', 'Speed (mm/s)'),
            ('Height', 'Height (mm)'),
            ('Length', 'Length (mm)'),
            ('MoSeq', 'Syllable ID'),
        ]

    name_map = dict(zip(plot_columns, col_names))
    level = summary.index.get_level_values(0).astype(str)
    unique_groups = pd.unique(level)
    code_lookup = {g: i for i, g in enumerate(unique_groups)}
    level_codes = np.array([code_lookup[g] for g in level])
    boundaries = np.r_[0, np.argwhere(np.diff(level_codes)).ravel()]
    find_mid = (np.diff(np.r_[boundaries, len(level_codes)]) / 2).astype('int32')
    level_ticks = boundaries + find_mid
    cmap = ListedColormap([group_colors.get(g, '#cccccc') for g in unique_groups])

    fig = plt.figure(figsize=figsize, facecolor='white')
    gs = GridSpec(
        2, 1 + len(plot_columns),
        wspace=0.1, hspace=0.1,
        width_ratios=[1.4] + [8] * len(plot_columns),
        height_ratios=[10, 0.1],
        figure=fig,
    )

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.set_title(level_names[0], fontsize=20)
    ax0.imshow(
        level_codes[:, np.newaxis], aspect='auto', cmap=cmap,
        vmin=-0.5, vmax=len(unique_groups) - 0.5,
    )
    ax0.set_yticks(level_ticks)
    ax0.set_yticklabels(level[level_ticks], fontsize=16)
    ax0.get_xaxis().set_ticks([])

    plot_data = {}
    data_vmin, data_vmax = np.inf, -np.inf
    for col in plot_columns:
        data = summary[col].to_numpy()
        if preprocessor is not None:
            data = preprocessor.fit_transform(data.T).T
        data_vmin = min(data_vmin, np.min(data))
        data_vmax = max(data_vmax, np.max(data))
        plot_data[col] = data

    if vmin is None:
        vmin = data_vmin
    if vmax is None:
        vmax = data_vmax

    color_mappable = None
    for i, col in enumerate(plot_columns):
        title, xlabel = name_map[col]
        ax = fig.add_subplot(gs[0, i + 1])
        ax.set_title(title, fontsize=20)
        data = plot_data[col]
        if col == 'MoSeq':
            extent = [
                summary[col].columns[0], summary[col].columns[-1],
                len(summary) - 1, 0,
            ]
        else:
            extent = [
                range_dict[col].iloc[0], range_dict[col].iloc[1],
                len(summary) - 1, 0,
            ]
        color_mappable = ax.imshow(
            data, aspect='auto', interpolation='none',
            cmap='viridis', vmin=vmin, vmax=vmax, extent=extent,
        )
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_xticks(
            np.linspace(np.ceil(extent[0]), np.floor(extent[1]), 6).astype(int)
        )
        ax.set_yticks([])

    cax = fig.add_subplot(gs[1, -1])
    plt.colorbar(color_mappable, cax=cax, orientation='horizontal')
    cax.set_xlabel('Min Max' if preprocessor is not None else 'Percentage Usage')

    makedirs(save_dir, exist_ok=True)
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['pdf.fonttype'] = 42

    if eps_dir is None and png_dir is None:
        eps_dir = save_dir
        png_dir = save_dir
        fig.savefig(join(save_dir, f'{filename_stem}.pdf'), bbox_inches='tight')

    if eps_dir is not None:
        makedirs(eps_dir, exist_ok=True)
        eps_path = join(eps_dir, f'{filename_stem}.eps')
        fig.savefig(eps_path, format='eps', bbox_inches='tight')
        print(f'Saved: {eps_path}')
    if png_dir is not None:
        makedirs(png_dir, exist_ok=True)
        png_path = join(png_dir, f'{filename_stem}.png')
        fig.savefig(png_path, dpi=png_dpi, bbox_inches='tight')
        print(f'Saved: {png_path}')

    plt.show()
    plt.close(fig)


def run_fingerprint_for_treatment(
    config: ExperimentConfig,
    treatment: str = 'cort',
    sex_codes: Optional[Sequence[str]] = None,
    timepoints: Optional[Sequence[str]] = None,
    stat_type: str = 'mean',
    n_bins: int = 100,
    range_type: str = 'robust',
    preprocessor=None,
):
    """Build and plot fingerprint for one treatment (females then males by default)."""
    if create_fingerprint_dataframe is None:
        raise ImportError('moseq2_viz is required for fingerprint plots.')

    treatment = treatment.lower()
    if treatment not in config.treatments:
        raise ValueError(
            f"treatment must be one of {list(config.treatments)}; got {treatment!r}"
        )
    if preprocessor is None:
        preprocessor = MinMaxScaler()

    codes = list(sex_codes) if sex_codes is not None else list(config.sex_codes.keys())
    tps = list(timepoints) if timepoints is not None else list(config.time_order)
    selected_groups = [
        config.group_name(treatment, tp, sex) for sex in codes for tp in tps
    ]
    print(f'Fingerprint for {treatment.upper()}')
    print('Groups:', selected_groups)

    print('Loading stats_df...')
    fingerprint_stats = pd.read_csv(config.stats_csv)
    fingerprint_stats = fingerprint_stats[
        fingerprint_stats['group'].isin(selected_groups)
    ].copy()

    print('Loading moseq_df (chunked filter; file is large)...')
    fingerprint_moseq = load_moseq_filtered(config.moseq_csv, selected_groups)

    label_map = {g: pretty_group_label(g, config) for g in selected_groups}
    color_map = {label_map[g]: color_for_group(g, config) for g in selected_groups}
    pretty_order = [label_map[g] for g in selected_groups]

    fingerprint_moseq['group'] = fingerprint_moseq['group'].map(label_map)
    fingerprint_stats['group'] = fingerprint_stats['group'].map(label_map)
    fingerprint_moseq['group'] = pd.Categorical(
        fingerprint_moseq['group'], categories=pretty_order, ordered=True
    )
    fingerprint_stats['group'] = pd.Categorical(
        fingerprint_stats['group'], categories=pretty_order, ordered=True
    )
    fingerprint_moseq = fingerprint_moseq.sort_values(['group', 'uuid'])
    fingerprint_stats = fingerprint_stats.sort_values(['group', 'uuid'])

    summary, range_dict = create_fingerprint_dataframe(
        fingerprint_moseq,
        fingerprint_stats,
        stat_type=stat_type,
        n_bins=n_bins,
        range_type=range_type,
    )

    summary = summary.loc[
        summary.index.get_level_values(0).isin(pretty_order)
    ].copy()
    group_rank = {g: i for i, g in enumerate(pretty_order)}
    sort_keys = pd.DataFrame({
        'group_name': summary.index.get_level_values(0).astype(str),
        'session_id': summary.index.get_level_values(1).astype(str),
    })
    sort_keys['rank'] = sort_keys['group_name'].map(group_rank)
    summary = summary.iloc[
        sort_keys.sort_values(['rank', 'session_id']).index.to_numpy()
    ]

    plotting_fingerprint_colored(
        summary,
        config.fingerprint_plot_dir,
        range_dict,
        group_colors=color_map,
        preprocessor=preprocessor,
        filename_stem=f'moseq_fingerprint_{treatment}_pretty',
        eps_dir=join(config.figures_eps_dir, 'fingerprint'),
        png_dir=join(config.figures_png_dir, 'fingerprint'),
    )
    return summary, range_dict
