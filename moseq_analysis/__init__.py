"""MoSeq syllable analysis and visualization helpers."""

from .config import ExperimentConfig
from .data import (
    build_group_usage_tables,
    build_usage_by_mouse,
    load_stats_and_labels,
    load_syllable_category_dict,
    maybe_combine_sex_groups,
)
from .effects import (
    infer_sig_contrast,
    load_did_ddd_effects,
    q_to_stars,
)
from .fingerprint import (
    load_moseq_filtered,
    plotting_fingerprint_colored,
    run_fingerprint_for_treatment,
)
from .grouping import (
    color_for_group,
    format_usage_plot_title,
    infer_sex_from_groups,
    infer_treatment_from_groups,
    legend_label_for_groups,
    parse_group_meta,
    pretty_group_label,
    shannon_entropy,
    strip_sex_suffix,
    time_scale_colors,
    time_scale_markers,
    timepoint_label_from_group,
)
from .plots import (
    figure_stem,
    list_transition_matrix_files,
    plot_behavioral_state_map,
    plot_behavioral_state_map_pair,
    plot_syllable_usage_diff,
    plot_transition_matrix,
    save_eps,
    save_figure,
)
from .style import apply_journal_style, JOURNAL_COLORS, JOURNAL_CMAP, JOURNAL_RC

__all__ = [
    'ExperimentConfig',
    'JOURNAL_COLORS',
    'JOURNAL_CMAP',
    'JOURNAL_RC',
    'apply_journal_style',
    'build_group_usage_tables',
    'build_usage_by_mouse',
    'color_for_group',
    'format_usage_plot_title',
    'infer_sex_from_groups',
    'infer_sig_contrast',
    'infer_treatment_from_groups',
    'legend_label_for_groups',
    'figure_stem',
    'list_transition_matrix_files',
    'load_did_ddd_effects',
    'load_moseq_filtered',
    'load_stats_and_labels',
    'load_syllable_category_dict',
    'maybe_combine_sex_groups',
    'parse_group_meta',
    'plot_behavioral_state_map',
    'plot_behavioral_state_map_pair',
    'plot_syllable_usage_diff',
    'plot_transition_matrix',
    'plotting_fingerprint_colored',
    'pretty_group_label',
    'q_to_stars',
    'run_fingerprint_for_treatment',
    'save_eps',
    'save_figure',
    'shannon_entropy',
    'strip_sex_suffix',
    'time_scale_colors',
    'time_scale_markers',
    'timepoint_label_from_group',
]
