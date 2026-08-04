"""Experiment configuration for MoSeq syllable analysis."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from os.path import join
from typing import Dict, List, Optional, Sequence, Set, Tuple


def _default_timepoint_labels() -> Dict[str, str]:
    return {
        'b2': 'Baseline',
        'wk1': 'Week 1',
        'wk2': 'Week 2',
        'wk3': 'Week 3',
        'washout': 'Washout',
    }


def _default_time_color_palettes() -> Dict[Tuple[str, str], List[str]]:
    return {
        ('cort', 'male'): ['#472a7a', '#33628d', '#21918c', '#3fbc73', '#b0dd2f'],
        ('cort', 'female'): ['#6a00a8', '#9410a2', '#cc4778', '#f0804e', '#fdc627'],
        ('veh', 'male'): ['#000000', '#404040', '#737373', '#a6a6a6', '#d9d9d9'],
        ('veh', 'female'): ['#5c3317', '#8b5a2b', '#a67c52', '#c4a484', '#e6d5c3'],
    }


@dataclass
class ExperimentConfig:
    """User-facing settings for one MoSeq analysis experiment.

    Edit these in the notebook so the same analysis code works across
    different treatments, timepoints, and syllable label files.
    """

    # --- Paths ---
    data_dir: str
    labels_xlsx: str
    labels_sheet: str = 'Sheet1'
    stats_csv: Optional[str] = None
    moseq_csv: Optional[str] = None
    figures_dir: str = 'figures'
    fingerprint_plot_dir: Optional[str] = None
    transition_matrix_dir: Optional[str] = None
    transition_matrix_suffix: str = '_bigram_transition_matrix.csv'

    # --- Analysis options ---
    combine_sex_groups: bool = False
    max_syllable_id: Optional[int] = 54

    # --- Experiment design (edit for your study) ---
    treatments: Sequence[str] = ('cort', 'veh')
    time_order: Sequence[str] = ('b2', 'wk1', 'wk2', 'wk3', 'washout')
    timepoint_labels: Dict[str, str] = field(default_factory=_default_timepoint_labels)
    # Maps trailing group-name tokens to sex labels used in titles/legends
    sex_codes: Dict[str, str] = field(
        default_factory=lambda: {'m': 'male', 'f': 'female'}
    )
    valid_syllable_categories: Set[str] = field(
        default_factory=lambda: {
            'locomotion', 'exploratory', 'grooming', 'immobility', 'hesitation',
        }
    )

    # --- Plot appearance ---
    time_color_palettes: Dict[Tuple[str, str], List[str]] = field(
        default_factory=_default_time_color_palettes
    )
    time_markers: Sequence[str] = ('o', 's', '^', 'D', 'v')
    default_palette_key: Tuple[str, str] = ('cort', 'male')

    def __post_init__(self) -> None:
        if self.stats_csv is None:
            self.stats_csv = join(self.data_dir, 'stats_df.csv')
        if self.moseq_csv is None:
            self.moseq_csv = join(self.data_dir, 'moseq_df.csv')
        if self.fingerprint_plot_dir is None:
            self.fingerprint_plot_dir = join(self.data_dir, 'plots')
        if self.transition_matrix_dir is None:
            self.transition_matrix_dir = self.data_dir

        self.treatments = tuple(t.lower() for t in self.treatments)
        self.time_order = tuple(self.time_order)
        self.sex_codes = {str(k).lower(): str(v).lower() for k, v in self.sex_codes.items()}
        self.valid_syllable_categories = {
            str(c).lower() for c in self.valid_syllable_categories
        }

        os.makedirs(self.figures_dir, exist_ok=True)
        os.makedirs(self.fingerprint_plot_dir, exist_ok=True)

    def group_name(self, treatment: str, timepoint: str, sex_code: str) -> str:
        """Build a group name like cort_b2_m from parts."""
        return f'{treatment}_{timepoint}_{sex_code}'

    def groups_for(
        self,
        treatment: str,
        sex_code: str,
        timepoints: Optional[Sequence[str]] = None,
    ) -> List[str]:
        """List group names for one treatment x sex across timepoints."""
        tps = timepoints if timepoints is not None else self.time_order
        return [self.group_name(treatment, tp, sex_code) for tp in tps]

    def all_groups(self, sex_codes: Optional[Sequence[str]] = None) -> List[str]:
        """All treatment x timepoint x sex group names."""
        codes = sex_codes if sex_codes is not None else list(self.sex_codes.keys())
        return [
            self.group_name(tx, tp, sex)
            for tx in self.treatments
            for sex in codes
            for tp in self.time_order
        ]
