"""Data loading and syllable usage / entropy tables."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .config import ExperimentConfig
from .grouping import shannon_entropy, strip_sex_suffix


def load_stats_and_labels(
    config: ExperimentConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load stats_df.csv and the syllable labeling spreadsheet."""
    stats_df = pd.read_csv(config.stats_csv)
    syllable_labels = pd.read_excel(
        config.labels_xlsx,
        engine='openpyxl',
        sheet_name=config.labels_sheet,
        header=1,
    )
    print(f'stats_df: {stats_df.shape[0]:,} rows x {stats_df.shape[1]} columns')
    print(f'Unique groups: {stats_df["group"].nunique()}')
    return stats_df, syllable_labels


def load_syllable_category_dict(syllable_labels: pd.DataFrame) -> Dict[int, str]:
    """Map syllable short id -> general category from the labeling sheet."""
    syllable_category_dict = (
        syllable_labels[['syllable short id', 'general category']]
        .dropna(subset=['syllable short id', 'general category'])
        .astype({'syllable short id': int})
        .set_index('syllable short id')['general category']
        .to_dict()
    )
    print(f'Mapped {len(syllable_category_dict)} syllables to categories')
    return syllable_category_dict


def maybe_combine_sex_groups(
    stats_df: pd.DataFrame,
    config: ExperimentConfig,
) -> pd.DataFrame:
    """Optionally strip sex suffixes from group names."""
    stats_df = stats_df.copy()
    if config.combine_sex_groups:
        stats_df['group'] = stats_df['group'].apply(
            lambda g: strip_sex_suffix(g, config.sex_codes.keys())
        )
        print(f'Combined sex groups: {stats_df["group"].nunique()} unique groups')
    else:
        print(f'Sex-separated groups: {stats_df["group"].nunique()} unique groups')
    print('Groups:', sorted(stats_df['group'].astype(str).unique()))
    return stats_df


def build_usage_by_mouse(
    stats_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Per-mouse usage pivot and group-level Shannon entropy summary."""
    usage_by_mouse = (
        stats_df.groupby(['uuid', 'group', 'syllable'], as_index=False)['usage']
        .mean()
        .pivot_table(
            index=['uuid', 'group'],
            columns='syllable',
            values='usage',
            fill_value=0.0,
        )
        .reset_index()
    )

    syllable_columns = [
        col for col in usage_by_mouse.columns
        if isinstance(col, (int, np.integer))
    ]
    usage_by_mouse['entropy'] = usage_by_mouse[syllable_columns].apply(
        shannon_entropy, axis=1
    )

    entropy_by_group = (
        usage_by_mouse.groupby('group')['entropy']
        .agg(['mean', 'std', 'count'])
        .reset_index()
        .rename(columns={'mean': 'entropy_mean', 'std': 'entropy_sd', 'count': 'n_mice'})
    )
    return usage_by_mouse, entropy_by_group


def build_group_usage_tables(
    stats_df: pd.DataFrame,
    config: ExperimentConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Group x syllable mean/SEM tables, optionally truncated by max syllable id."""
    group_usage_stats = (
        stats_df.groupby(['group', 'syllable'])['usage']
        .agg(['mean', 'std', 'count'])
        .reset_index()
    )
    group_usage_stats['sem'] = (
        group_usage_stats['std'] / np.sqrt(group_usage_stats['count'])
    )

    usage_mean_by_group = group_usage_stats.pivot(
        index='syllable', columns='group', values='mean'
    )
    usage_sem_by_group = group_usage_stats.pivot(
        index='syllable', columns='group', values='sem'
    )

    if config.max_syllable_id is not None:
        usage_mean_by_group = usage_mean_by_group[
            usage_mean_by_group.index <= config.max_syllable_id
        ]
        usage_sem_by_group = usage_sem_by_group[
            usage_sem_by_group.index <= config.max_syllable_id
        ]

    print(f'group_usage_stats: {group_usage_stats.shape}')
    print(f'usage_mean_by_group: {usage_mean_by_group.shape}')
    return group_usage_stats, usage_mean_by_group, usage_sem_by_group
