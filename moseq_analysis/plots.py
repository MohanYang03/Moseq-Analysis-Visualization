"""Syllable usage and transition-matrix plotting."""

from __future__ import annotations

import os
import re
import warnings
from os.path import join
from typing import Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D

from .config import ExperimentConfig
from .effects import (
    infer_sig_contrast,
    load_did_ddd_effects,
    normalize_sig_contrast,
    q_lookup,
    q_to_stars,
    sig_note_for_contrast,
    strip_timepoints,
)
from .grouping import (
    color_for_group,
    format_usage_plot_title,
    infer_sex_from_groups,
    infer_treatment_from_groups,
    legend_label_for_groups,
    parse_group_meta,
    pretty_group_label,
    time_scale_markers,
)


def figure_stem(path_or_stem: str) -> str:
    """Return a bare filename stem from a path or stem (strips image extensions)."""
    base = os.path.basename(str(path_or_stem))
    stem, ext = os.path.splitext(base)
    if ext.lower() in {".eps", ".png", ".pdf", ".svg"}:
        return stem
    return base


def save_figure(
    path_or_stem: str,
    config: ExperimentConfig,
    dpi: int = 300,
    subdir: Optional[str] = None,
) -> Tuple[str, str]:
    """Save the current figure as EPS and PNG into separate folders.

    ``path_or_stem`` may be a bare name (``cort_female``), a filename with
    extension (``cort_female.eps``), or a full path; only the basename stem is
    used. Files are written to ``config.figures_eps_dir`` and
    ``config.figures_png_dir``. If ``subdir`` is set (e.g. ``syllable_usage``),
    files go in that folder under both format directories.
    """
    stem = figure_stem(path_or_stem)
    eps_dir = config.figures_eps_dir
    png_dir = config.figures_png_dir
    if subdir:
        sub = str(subdir).strip("/\\")
        if sub:
            eps_dir = join(eps_dir, sub)
            png_dir = join(png_dir, sub)
    eps_path = join(eps_dir, f"{stem}.eps")
    png_path = join(png_dir, f"{stem}.png")
    os.makedirs(eps_dir, exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The PostScript backend does not support transparency",
        )
        plt.savefig(eps_path, format="eps", bbox_inches="tight")
    plt.savefig(png_path, format="png", dpi=dpi, bbox_inches="tight")
    print(f"Saved: {eps_path}")
    print(f"Saved: {png_path}")
    return eps_path, png_path


def save_eps(path: str, config: Optional[ExperimentConfig] = None) -> None:
    """Backward-compatible saver; prefers dual EPS/PNG when config is given."""
    if config is not None:
        save_figure(path, config)
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    plt.savefig(path, format="eps", bbox_inches="tight")
    print(f"Saved: {path}")


def place_legend_right(title=None, ax=None, pad=1.02):
    """Place legend outside the axes on the right."""
    ax = ax or plt.gca()
    return ax.legend(
        title=title,
        loc="center left",
        bbox_to_anchor=(pad, 0.5),
        frameon=False,
        borderaxespad=0.0,
    )


def _edge_legend_handle(color, prob, edge_width_scale, alpha, label):
    return Line2D(
        [0],
        [0],
        color=color,
        lw=max(abs(float(prob)) * float(edge_width_scale), 0.4),
        alpha=alpha,
        solid_capstyle="butt",
        label=label,
    )


def _add_bigram_edge_legends(
    ax_left,
    ax_right,
    edge_width_scale=300,
    edge_color="#3B6FA0",
    up_color="#3B6FA0",
    down_color="#C44E52",
    edge_alpha=0.45,
    diff_alpha=0.55,
):
    """Paper Fig. 1c-style edge-width legends on the left and right."""
    abs_handles = [
        _edge_legend_handle(edge_color, 0.025, edge_width_scale, edge_alpha, "2.5%"),
        _edge_legend_handle(edge_color, 0.010, edge_width_scale, edge_alpha, "1.0%"),
        _edge_legend_handle(edge_color, 0.001, edge_width_scale, edge_alpha, "0.1%"),
    ]
    ax_left.legend(
        handles=abs_handles,
        title="Bigram Probability",
        loc="center right",
        bbox_to_anchor=(-0.04, 0.5),
        frameon=False,
        handlelength=2.6,
        handletextpad=0.6,
        borderaxespad=0.0,
        labelspacing=0.9,
        fontsize=7,
        title_fontsize=8,
    )

    diff_handles = [
        _edge_legend_handle(up_color, 0.025, edge_width_scale, diff_alpha, "+2.5%"),
        _edge_legend_handle(up_color, 0.010, edge_width_scale, diff_alpha, "+1.0%"),
        _edge_legend_handle(down_color, 0.010, edge_width_scale, diff_alpha, "-1.0%"),
        _edge_legend_handle(down_color, 0.025, edge_width_scale, diff_alpha, "-2.5%"),
    ]
    ax_right.legend(
        handles=diff_handles,
        title="Bigram Probability Change",
        loc="center left",
        bbox_to_anchor=(1.04, 0.5),
        frameon=False,
        handlelength=2.6,
        handletextpad=0.6,
        borderaxespad=0.0,
        labelspacing=0.9,
        fontsize=7,
        title_fontsize=8,
    )


def _draw_significance_strip(
    ax_sig,
    syllable_ids,
    strip_tps,
    group_by_tp,
    lookup,
    config: ExperimentConfig,
):
    """Draw FDR stars under the usage plot, one row per post-baseline timepoint."""
    n_rows = len(strip_tps)
    ax_sig.set_xlim(-0.5, len(syllable_ids) - 0.5)
    ax_sig.set_ylim(-0.5, n_rows - 0.5)
    ax_sig.invert_yaxis()
    ax_sig.set_yticks(range(n_rows))
    ax_sig.set_yticklabels(
        [config.timepoint_labels.get(tp, tp) for tp in strip_tps],
        fontsize=6,
    )
    ax_sig.set_xticks(range(len(syllable_ids)))
    ax_sig.set_xticklabels(list(syllable_ids), rotation=90)
    ax_sig.tick_params(axis="y", length=0, pad=2)
    ax_sig.tick_params(axis="x", length=3)
    for spine in ("top", "right", "left"):
        ax_sig.spines[spine].set_visible(False)

    for row, timepoint in enumerate(strip_tps):
        color = color_for_group(group_by_tp[timepoint], config)
        ax_sig.get_yticklabels()[row].set_color(color)
        for x_pos, syllable_id in enumerate(syllable_ids):
            stars = q_to_stars(lookup.get((int(syllable_id), timepoint)))
            if not stars:
                continue
            ax_sig.text(
                x_pos,
                row,
                stars,
                ha="center",
                va="center",
                color=color,
                fontsize=7,
                fontweight="bold",
                clip_on=False,
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
    lines=True,
    effects=None,
    sig_contrast=None,
    sig_mode="strip",
    sig_alpha=0.05,
):
    """Plot syllable usage (mean +/- SEM), sorted by a group-pair difference.

    If ``lines`` is True (default), connect group means across syllables.
    If False, plot markers and SEM only.

    If ``effects`` is a DiD/DDD table or CSV path, CORT one-sex time series
    get an FDR significance strip under the x-axis. ``sig_contrast`` may be
    ``did_female``, ``did_male``, ``ddd``, or ``none``; the default infers DiD
    for CORT time series and skips VEH / sex-comparison plots. ``sig_alpha`` is
    reserved for a later binary mode and is unused by the star ladder.
    """
    if len(groups) < 2:
        raise ValueError("`groups` must contain at least two group names.")

    missing = [g for g in groups if g not in pivot_mean.columns]
    if missing:
        raise ValueError(f"Group names missing from pivot_mean.columns: {missing}")

    if categories is not None:
        if syllable_dict is None:
            raise ValueError(
                "`syllable_dict` is required when `categories` is provided."
            )
        if isinstance(categories, str):
            categories = [categories]
        invalid = set(categories) - set(config.valid_syllable_categories)
        if invalid:
            raise ValueError(
                f"Invalid categories: {sorted(invalid)}. "
                f"Valid options: {sorted(config.valid_syllable_categories)}"
            )
        keep_ids = {sid for sid, cat in syllable_dict.items() if cat in categories}
        keep = pivot_mean.index.isin(keep_ids)
        pivot_mean = pivot_mean.loc[keep]
        pivot_sem = pivot_sem.loc[keep]
        if pivot_mean.empty:
            raise ValueError(f"No syllables found for categories: {categories}")

    if include_syllables is not None:
        keep = pivot_mean.index.isin(set(include_syllables))
        pivot_mean = pivot_mean.loc[keep]
        pivot_sem = pivot_sem.loc[keep]
        if pivot_mean.empty:
            raise ValueError("None of the included syllables were found in the data.")

    if exclude_syllables is not None:
        keep = ~pivot_mean.index.isin(set(exclude_syllables))
        pivot_mean = pivot_mean.loc[keep]
        pivot_sem = pivot_sem.loc[keep]
        if pivot_mean.empty:
            raise ValueError("All syllables were excluded; nothing left to plot.")

    if sort_by is None:
        group1, group2 = groups[0], groups[1]
    else:
        if len(sort_by) != 2:
            raise ValueError(
                "`sort_by` must be a tuple/list of exactly two group names."
            )
        group1, group2 = sort_by
    for g in (group1, group2):
        if g not in pivot_mean.columns:
            raise ValueError(f"Sorting group {g} is not in pivot_mean.columns.")

    sorted_syllables = (
        (pivot_mean[group1] - pivot_mean[group2]).sort_values(ascending=False).index
    )
    sorted_mean = pivot_mean.loc[sorted_syllables]
    sorted_sem = pivot_sem.loc[sorted_syllables]
    syllable_ids = sorted_mean.index.values
    x = range(len(syllable_ids))

    contrast = normalize_sig_contrast(sig_contrast)
    strip_tps = []
    group_by_tp = {}
    lookup = {}
    if effects is not None:
        effects_df = load_did_ddd_effects(effects)
        if contrast is None:
            contrast = infer_sig_contrast(groups, config)
        if contrast != "none":
            if sig_mode != "strip":
                raise ValueError(
                    f"Unsupported sig_mode {sig_mode!r}; v1 only supports 'strip'."
                )
            strip_tps, group_by_tp = strip_timepoints(groups, config, effects_df)
            lookup = q_lookup(effects_df, contrast)
    use_strip = bool(strip_tps)
    _ = sig_alpha  # reserved for a later binary significance mode

    sex = infer_sex_from_groups(groups, config)
    treatment = infer_treatment_from_groups(groups, config)
    colors = [color_for_group(g, config) for g in groups]
    markers = time_scale_markers(len(groups), config)
    legend_labels = legend_label_for_groups(groups, config)

    if use_strip:
        n_rows = len(strip_tps)
        strip_h = max(0.32, 0.15 * n_rows)
        fig, (ax, ax_sig) = plt.subplots(
            2,
            1,
            figsize=(width, height + strip_h + 0.55),
            sharex=True,
            gridspec_kw={
                "height_ratios": [height, strip_h],
                "hspace": 0.04,
            },
        )
    else:
        fig, ax = plt.subplots(figsize=(width, height))
        ax_sig = None

    for group, color, marker, label in zip(groups, colors, markers, legend_labels):
        ax.errorbar(
            x,
            sorted_mean[group],
            yerr=sorted_sem[group],
            marker=marker,
            linestyle="-" if lines else "none",
            color=color,
            ecolor=color,
            elinewidth=0.8,
            markeredgecolor="white",
            markeredgewidth=0.4,
            label=label,
        )

    ax.set_xticks(list(x))
    ax.set_xlim(-0.5, len(syllable_ids) - 0.5)
    ax.set_ylabel("Average Usage Probability +/- SEM")
    if title is None:
        title = format_usage_plot_title(
            categories=categories, sex=sex, treatment=treatment
        )
    ax.set_title(title)

    if legend_title is None:
        metas = [parse_group_meta(g, config) for g in groups]
        sexes = {m[1] for m in metas}
        timepoints = {m[2] for m in metas}
        sex_labels = set(config.sex_codes.values())
        legend_title = (
            "Sex" if (len(timepoints) == 1 and sexes == sex_labels) else "Timepoint"
        )
    place_legend_right(title=legend_title, ax=ax)

    if y_max is not None:
        ax.set_ylim(0, y_max)
    sns.despine(ax=ax)

    if use_strip:
        ax.tick_params(axis="x", labelbottom=False, bottom=False)
        ax.set_xlabel("")
        _draw_significance_strip(
            ax_sig,
            syllable_ids,
            strip_tps,
            group_by_tp,
            lookup,
            config,
        )
        ax_sig.set_xlabel("Syllable ID")
        # tight_layout cannot place an external legend plus the strip on mpl 3.1
        fig.subplots_adjust(
            left=0.10,
            right=0.80,
            top=0.90,
            bottom=0.14,
            hspace=0.06,
        )
        fig.text(
            0.45,
            0.02,
            sig_note_for_contrast(contrast),
            ha="center",
            va="bottom",
            fontsize=6,
        )
    else:
        ax.set_xticklabels(list(syllable_ids), rotation=90)
        ax.set_xlabel("Syllable ID")
        fig.tight_layout(rect=[0, 0, 0.82, 1])

    if save_path is not False:
        if save_path is None:
            stem = "_vs_".join(groups)
            if categories is not None:
                stem += "_" + "_".join(categories)
            stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)
            save_path = f"syllable_usage_diff_{stem}"
        save_figure(save_path, config, subdir="syllable_usage")

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
    """Plot a bigram transition matrix CSV and save EPS + PNG."""
    import pandas as pd

    df = pd.read_csv(csv_path, header=None)
    group_name = os.path.basename(csv_path).replace(config.transition_matrix_suffix, "")

    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(df, cmap=cmap, vmax=vmax, square=True, cbar_kws={"shrink": 0.8})
    plt.title(f"Bigram Transition Matrix - {group_name}")
    plt.xlabel("To Syllable")
    plt.ylabel("From Syllable")
    plt.tight_layout()

    if save_path is not False:
        if save_path is None:
            save_path = f"{group_name}_transition_matrix"
        save_figure(save_path, config, subdir="transition_matrix")

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
        f
        for f in os.listdir(directory)
        if f.endswith(suffix)
        and (included_groups is None or f[: -len(suffix)] in set(included_groups))
    )
    return [join(directory, f) for f in files]


def _load_group_usage_dict(
    config: ExperimentConfig, group_name: str, max_syllables: int
):
    """Load syllable usage counts from ``{group}_syllable_counts.csv`` as a dict."""
    counts_path = join(
        config.transition_matrix_dir, f"{group_name}_syllable_counts.csv"
    )
    if not os.path.exists(counts_path):
        raise FileNotFoundError(f"Missing syllable counts file: {counts_path}")

    arr = np.loadtxt(counts_path, delimiter=",")
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    usage = {i: 0.0 for i in range(max_syllables)}
    for sid, count in zip(arr[:, 0].astype(int), arr[:, 1].astype(float)):
        if 0 <= sid < max_syllables:
            usage[int(sid)] = float(count)
    if sum(usage.values()) <= 0:
        raise ValueError(f"No syllable counts found for group {group_name}")
    return usage


def _normalize_usages(usage_dict):
    """MoSeq ``normalize_usages``: convert counts to frequencies."""
    total = sum(usage_dict.values()) or 1.0
    return {k: v / total for k, v in usage_dict.items()}


def _convert_ebunch_to_graph(ebunch):
    """MoSeq ``convert_ebunch_to_graph``."""
    graph = nx.DiGraph()
    graph.add_weighted_edges_from(ebunch)
    return graph


def _convert_transition_matrix_to_ebunch(
    weights,
    edge_threshold=0.0025,
    usages=None,
    usage_threshold=0.0,
    max_syllable=None,
):
    """MoSeq ``convert_transition_matrix_to_ebunch``, simplified for one group."""
    if max_syllable is not None:
        weights = weights[:max_syllable, :max_syllable]
    ebunch = []
    for (i, j), w in np.ndenumerate(weights):
        if i == j or w <= edge_threshold:
            continue
        if usages is not None:
            if (
                usages.get(i, 0.0) <= usage_threshold
                or usages.get(j, 0.0) <= usage_threshold
            ):
                continue
        ebunch.append((int(i), int(j), float(w)))
    return ebunch


def _connected_layout_graph(graph):
    """Undirected copy with components joined so spectral layout can run.

    Isolated syllables are linked with a moderate weight (10% of the median
    edge). Using a near-zero weight sends those nodes to the far periphery
    and, after rescaling, crushes the remaining core into a blob.
    """
    layout_graph = graph.to_undirected() if graph.is_directed() else graph.copy()
    if layout_graph.number_of_nodes() <= 1:
        return layout_graph
    comps = [list(c) for c in nx.connected_components(layout_graph)]
    if len(comps) <= 1:
        return layout_graph
    weights = [
        float(data.get("weight", 1.0)) for _, _, data in layout_graph.edges(data=True)
    ]
    join_weight = 0.1 * float(np.median(weights)) if weights else 1e-3
    join_weight = max(join_weight, 1e-4)
    reps = []
    for nodes in comps:
        sub = layout_graph.subgraph(nodes)
        reps.append(max(nodes, key=lambda n: sub.degree(n)))
    for a, b in zip(reps, reps[1:]):
        if not layout_graph.has_edge(a, b):
            layout_graph.add_edge(a, b, weight=join_weight)
    return layout_graph


def _relax_overlaps(pos, min_sep=0.16, n_iter=100, step=0.45):
    """Push overlapping nodes apart while keeping the global arrangement.

    ``min_sep`` is a fraction of the layout span (0.16 = 16% of the
    bounding-box width/height). Nearby syllables stay nearby; they are
    just not allowed to sit on top of each other.
    """
    if min_sep is None or min_sep <= 0 or len(pos) < 2:
        return pos
    nodes = list(pos)
    xy = np.array([pos[n] for n in nodes], dtype=float)
    span = float(np.max(xy.max(axis=0) - xy.min(axis=0)))
    sep = float(min_sep) * max(span, 1e-6)
    n = xy.shape[0]
    rng = np.random.RandomState(0)
    for _ in range(int(n_iter)):
        delta = np.zeros_like(xy)
        for i in range(n):
            for j in range(i + 1, n):
                dxy = xy[i] - xy[j]
                dist = float(np.sqrt((dxy**2).sum()))
                if dist < 1e-12:
                    dxy = rng.normal(0.0, 1e-3, size=2)
                    dist = float(np.sqrt((dxy**2).sum()))
                if dist >= sep:
                    continue
                push = 0.5 * (sep - dist) * (dxy / dist)
                delta[i] += push
                delta[j] -= push
        xy += step * delta
    return {node: (float(xy[i, 0]), float(xy[i, 1])) for i, node in enumerate(nodes)}


def _get_pos(
    graph,
    layout,
    nnodes,
    layout_spread=3.5,
    spring_weight=None,
    min_node_sep=0.16,
):
    """Node layout for one graph.

    ``spectral_spring`` (default for the paired maps):
    1. Build an undirected graph (for the pair, edges are max(P_A, P_B)).
    2. Spectral layout uses the Laplacian eigenvectors so syllables that
       share many / strong transitions start near each other.
    3. Spring layout then spreads nodes. ``layout_spread`` is the NetworkX
       ``k`` scale: larger values increase the preferred distance
       (~ ``layout_spread / sqrt(n)``). ``spring_weight=None`` gives every
       edge the same spring strength so high-probability transitions do not
       collapse the core. Pass ``'weight'`` to recover the tighter paper-style
       packing.
    4. ``min_node_sep`` then nudges overlapping nodes apart (fraction of
       the layout span; raise it if circles still overlap).
    """
    n = max(int(nnodes), 1)
    k = float(layout_spread) / np.sqrt(n)
    layout_graph = _connected_layout_graph(graph)
    if layout in ("spring", "spectral_spring"):
        pos0 = None
        if layout == "spectral_spring" and layout_graph.number_of_nodes() >= 3:
            try:
                pos0 = nx.spectral_layout(layout_graph, weight="weight")
                xy = np.array(list(pos0.values()), dtype=float)
                n_unique = len({tuple(np.round(p, 6)) for p in xy})
                if n_unique < max(3, xy.shape[0] // 2):
                    pos0 = None
            except Exception:
                pos0 = None
        pos = nx.spring_layout(
            layout_graph,
            pos=pos0,
            k=k,
            seed=0,
            weight=spring_weight,
            iterations=200,
        )
        return _relax_overlaps(pos, min_sep=min_node_sep)
    if layout == "circular":
        return nx.circular_layout(layout_graph)
    if layout == "spectral":
        return nx.spectral_layout(layout_graph, weight="weight")
    raise ValueError(
        "layout must be one of: 'spectral_spring', 'spring', 'circular', 'spectral'"
    )


def _node_sizes(nodes, usages, usage_scale, min_node_size):
    return [max(float(usages.get(n, 0.0)) * usage_scale, min_node_size) for n in nodes]


def _scaled_ring_areas(size_a, size_b, ring_width, inward=False):
    """Return (inner_area, outer_area) for a usage-change ring.

    Areas are matplotlib node sizes (∝ diameter²). ``ring_width`` multiplies
    the control-vs-treatment diameter gap (1 = the raw usage difference).
    """
    d_a = np.sqrt(np.maximum(np.asarray(size_a, dtype=float), 0.0))
    d_b = np.sqrt(np.maximum(np.asarray(size_b, dtype=float), 0.0))
    gap = float(ring_width) * (d_b - d_a)
    if inward:
        d_inner = np.clip(d_a + gap, 0.55 * d_a, d_a)
        d_color = 0.98 * d_a
        return d_color**2, d_inner**2
    d_outer = np.maximum(d_a + np.maximum(gap, 0.0), d_a)
    return np.asarray(size_a, dtype=float), d_outer**2


def _draw_graph(
    graph,
    pos,
    usages,
    ax,
    title,
    edge_width_scale=300,
    usage_scale=1e4,
    min_node_size=100,
    node_edge_color="#888888",
    edge_color="#3B6FA0",
    edge_alpha=0.45,
    arrows=True,
    font_size=8,
    arrow_rad=0.08,
):
    """MoSeq ``draw_graph`` with grey outlines, blue arrows, min size, and alpha.

    Reciprocal edges use a small arc so 1→2 and 2→1 stay visible.
    """
    for n in pos:
        if n not in graph:
            graph.add_node(n)
    nodes = list(pos.keys())
    node_size = _node_sizes(nodes, usages, usage_scale, min_node_size)
    edges = list(graph.edges)
    widths = [graph[u][v]["weight"] * edge_width_scale for u, v in edges]

    if edges:
        nx.draw_networkx_edges(
            graph,
            pos,
            ax=ax,
            edgelist=edges,
            width=widths,
            edge_color=edge_color,
            alpha=edge_alpha,
            arrows=arrows,
            arrowsize=12,
            arrowstyle="-|>",
            connectionstyle="arc3,rad={:.3f}".format(arrow_rad),
            node_size=node_size,
            nodelist=nodes,
        )
    nx.draw_networkx_nodes(
        graph,
        pos,
        ax=ax,
        nodelist=nodes,
        node_size=node_size,
        node_color="white",
        edgecolors=node_edge_color,
        linewidths=1.5,
    )
    nx.draw_networkx_labels(
        graph, pos, ax=ax, font_size=font_size, font_color="#333333"
    )
    ax.set_title(title)
    ax.axis("off")
    ax.set_aspect("equal")


def _draw_difference_graph(
    pos,
    usages_a,
    usages_b,
    diff_graph,
    ax,
    title,
    edge_width_scale=300,
    usage_scale=1e4,
    min_node_size=100,
    edge_alpha=0.55,
    arrows=True,
    font_size=8,
    arrow_rad=0.08,
    up_color="#3B6FA0",
    down_color="#C44E52",
    control_color="#111111",
    usage_ring_width=1.0,
):
    """Difference panel: black = control usage; colored ring = change.

    Upregulation: blue ring outside the black control circle.
    Downregulation: red ring inside the black control circle (white center).
    ``usage_ring_width`` scales those ring thicknesses (1 = raw usage Δ).
    """
    for n in pos:
        if n not in diff_graph:
            diff_graph.add_node(n)
    nodes = list(pos.keys())
    sizes_a = _node_sizes(nodes, usages_a, usage_scale, min_node_size)
    sizes_b = _node_sizes(nodes, usages_b, usage_scale, min_node_size)

    up_idx = [
        i for i, n in enumerate(nodes) if usages_b.get(n, 0.0) > usages_a.get(n, 0.0)
    ]
    down_idx = [
        i for i, n in enumerate(nodes) if usages_b.get(n, 0.0) < usages_a.get(n, 0.0)
    ]
    eq_idx = [
        i for i, n in enumerate(nodes) if usages_b.get(n, 0.0) == usages_a.get(n, 0.0)
    ]

    edges = list(diff_graph.edges)
    widths = [abs(diff_graph[u][v]["weight"]) * edge_width_scale for u, v in edges]
    edge_colors = [
        up_color if diff_graph[u][v]["weight"] > 0 else down_color for u, v in edges
    ]
    node_size_for_arrows = [max(a, b) for a, b in zip(sizes_a, sizes_b)]

    if edges:
        nx.draw_networkx_edges(
            diff_graph,
            pos,
            ax=ax,
            edgelist=edges,
            width=widths,
            edge_color=edge_colors,
            alpha=edge_alpha,
            arrows=arrows,
            arrowsize=12,
            arrowstyle="-|>",
            connectionstyle="arc3,rad={:.3f}".format(arrow_rad),
            node_size=node_size_for_arrows,
            nodelist=nodes,
        )

    def _subset(idxs):
        return (
            [nodes[i] for i in idxs],
            [sizes_a[i] for i in idxs],
            [sizes_b[i] for i in idxs],
        )

    if up_idx:
        n_up, sa_up, sb_up = _subset(up_idx)
        inner_up, outer_up = _scaled_ring_areas(
            sa_up, sb_up, usage_ring_width, inward=False
        )
        nx.draw_networkx_nodes(
            diff_graph,
            pos,
            ax=ax,
            nodelist=n_up,
            node_size=outer_up,
            node_color=up_color,
            edgecolors=up_color,
            linewidths=1.5,
        )
        nx.draw_networkx_nodes(
            diff_graph,
            pos,
            ax=ax,
            nodelist=n_up,
            node_size=inner_up,
            node_color="white",
            edgecolors=control_color,
            linewidths=1.5,
        )
    if down_idx:
        n_down, sa_down, sb_down = _subset(down_idx)
        outer_down, inner_down = _scaled_ring_areas(
            sa_down, sb_down, usage_ring_width, inward=True
        )
        nx.draw_networkx_nodes(
            diff_graph,
            pos,
            ax=ax,
            nodelist=n_down,
            node_size=sa_down,
            node_color="white",
            edgecolors=control_color,
            linewidths=1.5,
        )
        nx.draw_networkx_nodes(
            diff_graph,
            pos,
            ax=ax,
            nodelist=n_down,
            node_size=outer_down,
            node_color=down_color,
            edgecolors=down_color,
            linewidths=0.0,
        )
        nx.draw_networkx_nodes(
            diff_graph,
            pos,
            ax=ax,
            nodelist=n_down,
            node_size=inner_down,
            node_color="white",
            edgecolors="white",
            linewidths=0.0,
        )
    if eq_idx:
        n_eq, sa_eq, _ = _subset(eq_idx)
        nx.draw_networkx_nodes(
            diff_graph,
            pos,
            ax=ax,
            nodelist=n_eq,
            node_size=sa_eq,
            node_color="white",
            edgecolors=control_color,
            linewidths=1.5,
        )
    nx.draw_networkx_labels(
        diff_graph, pos, ax=ax, font_size=font_size, font_color="#333333"
    )
    ax.set_title(title)
    ax.axis("off")
    ax.set_aspect("equal")


def _load_group_tm(config: ExperimentConfig, group_name: str, max_syllables: int):
    suffix = config.transition_matrix_suffix
    tm_path = join(config.transition_matrix_dir, f"{group_name}{suffix}")
    if not os.path.exists(tm_path):
        raise FileNotFoundError(f"Missing transition matrix: {tm_path}")
    tm = np.loadtxt(tm_path, delimiter=",")
    return tm[:max_syllables, :max_syllables]


def _graph_from_tm(tm, usages, edge_threshold, usage_threshold, max_syllables):
    ebunch = _convert_transition_matrix_to_ebunch(
        tm,
        edge_threshold=edge_threshold,
        usages=usages,
        usage_threshold=usage_threshold,
        max_syllable=max_syllables,
    )
    graph = _convert_ebunch_to_graph(ebunch)
    for i, usage in usages.items():
        if usage > usage_threshold:
            graph.add_node(i)
    return graph


def plot_behavioral_state_map(
    config: ExperimentConfig,
    group_name: str,
    max_syllables: Optional[int] = None,
    edge_threshold: float = 0.0025,
    usage_threshold: float = 0.0,
    layout: str = "spectral_spring",
    layout_spread: float = 3.5,
    spring_weight=None,
    min_node_sep: float = 0.16,
    arrows: bool = True,
    figsize: float = 7.0,
    min_node_size: float = 100,
    node_edge_color: str = "#888888",
    edge_color: str = "#3B6FA0",
    edge_alpha: float = 0.45,
    arrow_rad: float = 0.08,
    edge_width_scale: float = 300,
    save_path=None,
    title=None,
    show: bool = True,
):
    """Draw a MoSeq-style transition graph for one group.

    Layout is spectral-seeded spring (local structure, then spread).
    Raise ``layout_spread`` to push nodes apart. Node size ∝ usage with a
    minimum diameter; edges are transparent blue arrows with a slight arc
    so reciprocal transitions do not overlap.
    """
    if max_syllables is None:
        max_syllables = (
            config.max_syllable_id + 1 if config.max_syllable_id is not None else 50
        )
    max_syllables = int(max_syllables)
    tm = _load_group_tm(config, group_name, max_syllables)
    usages = _normalize_usages(
        _load_group_usage_dict(config, group_name, max_syllables)
    )
    graph = _graph_from_tm(tm, usages, edge_threshold, usage_threshold, max_syllables)
    pos = _get_pos(
        graph,
        layout,
        nnodes=max(len(graph), 1),
        layout_spread=layout_spread,
        spring_weight=spring_weight,
        min_node_sep=min_node_sep,
    )
    if title is None:
        title = pretty_group_label(group_name, config)

    fig, ax = plt.subplots(figsize=(figsize, figsize))
    _draw_graph(
        graph,
        pos,
        usages,
        ax,
        title,
        arrows=arrows,
        min_node_size=min_node_size,
        node_edge_color=node_edge_color,
        edge_color=edge_color,
        edge_alpha=edge_alpha,
        arrow_rad=arrow_rad,
        edge_width_scale=edge_width_scale,
    )
    fig.tight_layout()

    if save_path is not False:
        if save_path is None:
            save_path = f"state_map_{group_name}"
        save_figure(save_path, config, subdir="statemaps")

    if show:
        plt.show()
    plt.close(fig)
    return fig


def plot_behavioral_state_map_pair(
    config: ExperimentConfig,
    group_a: str,
    group_b: str,
    max_syllables: Optional[int] = None,
    edge_threshold: float = 0.0025,
    usage_threshold: float = 0.0,
    difference_threshold: float = 0.005,
    layout: str = "spectral_spring",
    layout_spread: float = 3.5,
    spring_weight=None,
    min_node_sep: float = 0.16,
    arrows: bool = True,
    figsize: float = 5.6,
    min_node_size: float = 100,
    edge_alpha: float = 0.45,
    arrow_rad: float = 0.08,
    usage_ring_width: float = 1.0,
    edge_width_scale: float = 300,
    save_path=None,
    titles=None,
    show: bool = True,
):
    """Three-panel state maps: group A, group B, and B−A (paper Fig. 1c style).

    Shared layout is spectral-seeded spring on the union graph
    ``max(P_A, P_B)``. Raise ``layout_spread`` to reduce overlap.
    Absolute maps use blue arrows.
    The difference panel uses a blue ring outside the black control
    circle for increased usage, and a red ring inside the black control
    circle (white center) for decreased usage. ``usage_ring_width`` scales
    those ring thicknesses (1 = raw usage Δ; try 0.5–2.5). ``edge_width_scale``
    sets arrow thickness (line width = probability × scale; try 150–500).
    Difference edges are those with
    ``|P_B - P_A| >= difference_threshold`` (group-level matrices; a
    per-mouse p < 0.01 test is not available from the CSVs).
    """
    if max_syllables is None:
        max_syllables = (
            config.max_syllable_id + 1 if config.max_syllable_id is not None else 50
        )
    max_syllables = int(max_syllables)
    tm_a = _load_group_tm(config, group_a, max_syllables)
    tm_b = _load_group_tm(config, group_b, max_syllables)
    usages_a = _normalize_usages(_load_group_usage_dict(config, group_a, max_syllables))
    usages_b = _normalize_usages(_load_group_usage_dict(config, group_b, max_syllables))

    graph_a = _graph_from_tm(
        tm_a, usages_a, edge_threshold, usage_threshold, max_syllables
    )
    graph_b = _graph_from_tm(
        tm_b, usages_b, edge_threshold, usage_threshold, max_syllables
    )

    tm_union = np.maximum(tm_a, tm_b)
    usages_layout = {
        k: max(usages_a.get(k, 0.0), usages_b.get(k, 0.0))
        for k in set(usages_a) | set(usages_b)
    }
    layout_graph = _graph_from_tm(
        tm_union, usages_layout, edge_threshold, usage_threshold, max_syllables
    )
    for n in list(graph_a.nodes) + list(graph_b.nodes):
        layout_graph.add_node(n)
    pos = _get_pos(
        layout_graph,
        layout,
        nnodes=max(len(layout_graph), 1),
        layout_spread=layout_spread,
        spring_weight=spring_weight,
        min_node_sep=min_node_sep,
    )

    diff = tm_b - tm_a
    diff_ebunch = []
    for (i, j), w in np.ndenumerate(diff):
        if i == j or abs(w) < difference_threshold:
            continue
        diff_ebunch.append((int(i), int(j), float(w)))
    diff_graph = _convert_ebunch_to_graph(diff_ebunch)
    for n in pos:
        diff_graph.add_node(n)

    if titles is None:
        titles = (
            pretty_group_label(group_a, config),
            pretty_group_label(group_b, config),
            "{} − {}".format(
                pretty_group_label(group_b, config),
                pretty_group_label(group_a, config),
            ),
        )

    fig, axes = plt.subplots(1, 3, figsize=(3 * figsize + 2.4, figsize))
    _draw_graph(
        graph_a,
        pos,
        usages_a,
        axes[0],
        titles[0],
        arrows=arrows,
        min_node_size=min_node_size,
        edge_alpha=edge_alpha,
        arrow_rad=arrow_rad,
        edge_width_scale=edge_width_scale,
    )
    _draw_graph(
        graph_b,
        pos,
        usages_b,
        axes[1],
        titles[1],
        arrows=arrows,
        min_node_size=min_node_size,
        edge_alpha=edge_alpha,
        arrow_rad=arrow_rad,
        edge_width_scale=edge_width_scale,
    )
    diff_alpha = max(edge_alpha, 0.55)
    _draw_difference_graph(
        pos,
        usages_a,
        usages_b,
        diff_graph,
        axes[2],
        titles[2],
        arrows=arrows,
        min_node_size=min_node_size,
        edge_alpha=diff_alpha,
        arrow_rad=arrow_rad,
        usage_ring_width=usage_ring_width,
        edge_width_scale=edge_width_scale,
    )

    xy = np.array(list(pos.values()), dtype=float)
    span = float(np.max(xy.max(axis=0) - xy.min(axis=0)))
    pad = 0.18 * max(span, 1e-6)
    xlim = (xy[:, 0].min() - pad, xy[:, 0].max() + pad)
    ylim = (xy[:, 1].min() - pad, xy[:, 1].max() + pad)
    for ax in axes:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    fig.tight_layout()
    _add_bigram_edge_legends(
        axes[0],
        axes[2],
        edge_width_scale=edge_width_scale,
        edge_alpha=edge_alpha,
        diff_alpha=diff_alpha,
    )

    if save_path is not False:
        if save_path is None:
            save_path = "state_map_{}_vs_{}".format(group_a, group_b)
        save_figure(save_path, config, subdir="statemaps")

    if show:
        plt.show()
    plt.close(fig)
    return fig
