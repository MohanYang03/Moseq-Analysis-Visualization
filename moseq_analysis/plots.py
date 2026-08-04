"""Syllable usage and transition-matrix plotting."""

from __future__ import annotations

import os
import re
from os.path import join
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import seaborn as sns

from .config import ExperimentConfig
from .grouping import (
    color_for_group,
    format_usage_plot_title,
    infer_sex_from_groups,
    infer_treatment_from_groups,
    legend_label_for_groups,
    parse_group_meta,
    time_scale_markers,
)


def save_eps(path: str) -> None:
    """Save the current figure as EPS."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
    plt.savefig(path, format='eps', bbox_inches='tight')
    print(f'Saved: {path}')


def place_legend_right(title=None, ax=None, pad=1.02):
    """Place legend outside the axes on the right."""
    ax = ax or plt.gca()
    return ax.legend(
        title=title,
        loc='center left',
        bbox_to_anchor=(pad, 0.5),
        frameon=False,
        borderaxespad=0.0,
    )


def plot_syllable_usage_diff(
    pivot_mean,
    pivot_sem,
    groups,
    config: ExperimentConfig,
    y_max=None,
    sort_by=None,
    include_syllables=None,
    exclude_syllables=None,
    syllable_dict=None,
    categories=None,
    save_path=None,
    title=None,
    legend_title=None,
    width=7.2,
    height=3.6,
    show=True,
):
    """Plot syllable usage (mean +/- SEM), sorted by a group-pair difference."""
    if len(groups) < 2:
        raise ValueError('`groups` must contain at least two group names.')

    missing = [g for g in groups if g not in pivot_mean.columns]
    if missing:
        raise ValueError(f'Group names missing from pivot_mean.columns: {missing}')

    if categories is not None:
        if syllable_dict is None:
            raise ValueError('`syllable_dict` is required when `categories` is provided.')
        if isinstance(categories, str):
            categories = [categories]
        invalid = set(categories) - set(config.valid_syllable_categories)
        if invalid:
            raise ValueError(
                f'Invalid categories: {sorted(invalid)}. '
                f'Valid options: {sorted(config.valid_syllable_categories)}'
            )
        keep_ids = {sid for sid, cat in syllable_dict.items() if cat in categories}
        keep = pivot_mean.index.isin(keep_ids)
        pivot_mean = pivot_mean.loc[keep]
        pivot_sem = pivot_sem.loc[keep]
        if pivot_mean.empty:
            raise ValueError(f'No syllables found for categories: {categories}')

    if include_syllables is not None:
        keep = pivot_mean.index.isin(set(include_syllables))
        pivot_mean = pivot_mean.loc[keep]
        pivot_sem = pivot_sem.loc[keep]
        if pivot_mean.empty:
            raise ValueError('None of the included syllables were found in the data.')

    if exclude_syllables is not None:
        keep = ~pivot_mean.index.isin(set(exclude_syllables))
        pivot_mean = pivot_mean.loc[keep]
        pivot_sem = pivot_sem.loc[keep]
        if pivot_mean.empty:
            raise ValueError('All syllables were excluded; nothing left to plot.')

    if sort_by is None:
        group1, group2 = groups[0], groups[1]
    else:
        if len(sort_by) != 2:
            raise ValueError('`sort_by` must be a tuple/list of exactly two group names.')
        group1, group2 = sort_by
    for g in (group1, group2):
        if g not in pivot_mean.columns:
            raise ValueError(f'Sorting group {g} is not in pivot_mean.columns.')

    sorted_syllables = (
        pivot_mean[group1] - pivot_mean[group2]
    ).sort_values(ascending=False).index
    sorted_mean = pivot_mean.loc[sorted_syllables]
    sorted_sem = pivot_sem.loc[sorted_syllables]
    syllable_ids = sorted_mean.index.values
    x = range(len(syllable_ids))

    plt.figure(figsize=(width, height))
    sex = infer_sex_from_groups(groups, config)
    treatment = infer_treatment_from_groups(groups, config)
    colors = [color_for_group(g, config) for g in groups]
    markers = time_scale_markers(len(groups), config)
    legend_labels = legend_label_for_groups(groups, config)

    for group, color, marker, label in zip(groups, colors, markers, legend_labels):
        plt.errorbar(
            x,
            sorted_mean[group],
            yerr=sorted_sem[group],
            marker=marker,
            color=color,
            ecolor=color,
            elinewidth=0.8,
            markeredgecolor='white',
            markeredgewidth=0.4,
            label=label,
        )

    plt.xticks(ticks=x, labels=syllable_ids, rotation=90)
    plt.xlabel('Syllable ID')
    plt.ylabel('Average Usage Probability +/- SEM')
    if title is None:
        title = format_usage_plot_title(
            categories=categories, sex=sex, treatment=treatment
        )
    plt.title(title)

    if legend_title is None:
        metas = [parse_group_meta(g, config) for g in groups]
        sexes = {m[1] for m in metas}
        timepoints = {m[2] for m in metas}
        sex_labels = set(config.sex_codes.values())
        legend_title = (
            'Sex' if (len(timepoints) == 1 and sexes == sex_labels)
            else 'Timepoint'
        )
    place_legend_right(title=legend_title)

    if y_max is not None:
        plt.ylim(0, y_max)
    sns.despine()
    plt.tight_layout(rect=[0, 0, 0.82, 1])

    if save_path is not False:
        if save_path is None:
            stem = '_vs_'.join(groups)
            if categories is not None:
                stem += '_' + '_'.join(categories)
            stem = re.sub(r'[^A-Za-z0-9_.-]+', '_', stem)
            save_path = join(config.figures_dir, f'syllable_usage_diff_{stem}.eps')
        save_eps(save_path)

    if show:
        plt.show()
    plt.close()



def plot_transition_matrix(
    csv_path: str,
    config: ExperimentConfig,
    vmax: Optional[float] = None,
    cmap=None,
    save_path=None,
):
    """Plot a bigram transition matrix CSV and optionally save EPS."""
    import pandas as pd

    df = pd.read_csv(csv_path, header=None)
    group_name = os.path.basename(csv_path).replace(
        config.transition_matrix_suffix, ''
    )

    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(df, cmap=cmap, vmax=vmax, square=True, cbar_kws={'shrink': 0.8})
    plt.title(f'Bigram Transition Matrix - {group_name}')
    plt.xlabel('To Syllable')
    plt.ylabel('From Syllable')
    plt.tight_layout()

    if save_path is not False:
        if save_path is None:
            save_path = join(config.figures_dir, f'{group_name}_transition_matrix.eps')
        save_eps(save_path)

    plt.show()
    plt.close()


def list_transition_matrix_files(
    config: ExperimentConfig,
    included_groups: Optional[Sequence[str]] = None,
):
    """Return sorted transition-matrix CSV paths under the configured directory."""
    directory = config.transition_matrix_dir
    suffix = config.transition_matrix_suffix
    files = sorted(
        f for f in os.listdir(directory)
        if f.endswith(suffix)
        and (included_groups is None or f[: -len(suffix)] in set(included_groups))
    )
    return [join(directory, f) for f in files]
