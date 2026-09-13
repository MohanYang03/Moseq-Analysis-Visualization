"""Load DiD / DDD FDR q-values and map them to significance stars."""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .config import ExperimentConfig
from .grouping import parse_group_meta

EffectsSource = Union[str, pd.DataFrame]

Q_COLUMNS = {
    "did_female": "DiD_Female_q",
    "did_male": "DiD_Male_q",
    "ddd": "DDD_q",
}
VALID_CONTRASTS = frozenset({"did_female", "did_male", "ddd", "none"})
REQUIRED_COLUMNS = (
    "syllable",
    "timepoint",
    "estimable",
    "DiD_Female_q",
    "DiD_Male_q",
    "DDD_q",
)
DEFAULT_STAR_LEVELS = (0.05, 0.01, 0.001)
SIG_NOTE_DID = (
    "* q<0.05, ** q<0.01, *** q<0.001; BH FDR. DiD: CORT vs VEH vs baseline."
)
SIG_NOTE_DDD = (
    "* q<0.05, ** q<0.01, *** q<0.001; BH FDR. DDD: male DiD minus female DiD."
)


def q_to_stars(q, levels: Sequence[float] = DEFAULT_STAR_LEVELS) -> str:
    """Map an FDR q-value to ``*`` / ``**`` / ``***``, or ``''`` if not significant."""
    if q is None:
        return ""
    try:
        q_val = float(q)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(q_val):
        return ""
    n_stars = sum(1 for thresh in levels if q_val < float(thresh))
    return "*" * n_stars


def _is_true(value) -> bool:
    if isinstance(value, str):
        return value.strip().upper() in {"TRUE", "T", "1", "YES"}
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    return bool(value)


def load_did_ddd_effects(source: EffectsSource) -> pd.DataFrame:
    """Return syllable × timepoint FDR q-values; non-estimable q is set to NA."""
    if isinstance(source, pd.DataFrame):
        df = source.copy()
    else:
        df = pd.read_csv(source)

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"DiD/DDD effects table is missing columns: {missing}. "
            f"Expected: {list(REQUIRED_COLUMNS)}"
        )

    out = df.loc[:, list(REQUIRED_COLUMNS)].copy()
    out["syllable"] = out["syllable"].astype(int)
    out["timepoint"] = out["timepoint"].astype(str)
    out["estimable"] = out["estimable"].map(_is_true)
    for col in ("DiD_Female_q", "DiD_Male_q", "DDD_q"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
        out.loc[~out["estimable"], col] = np.nan
    return out.drop_duplicates(subset=["syllable", "timepoint"], keep="first")


def normalize_sig_contrast(value: Optional[str]) -> Optional[str]:
    """Return a canonical contrast key, or None if ``value`` is None."""
    if value is None:
        return None
    key = str(value).strip().lower().replace("-", "_")
    if key not in VALID_CONTRASTS:
        raise ValueError(
            f"Invalid sig_contrast {value!r}. "
            f"Use one of: {sorted(VALID_CONTRASTS)}"
        )
    return key


def infer_sig_contrast(groups: Sequence[str], config: ExperimentConfig) -> str:
    """Choose a DiD contrast for a CORT one-sex time series; otherwise ``none``."""
    metas = [parse_group_meta(g, config) for g in groups]
    treatments = {m[0] for m in metas}
    sexes = {m[1] for m in metas}
    timepoints = {m[2] for m in metas}
    treatments.discard(None)
    sexes.discard(None)
    timepoints.discard(None)

    if len(treatments) != 1 or len(sexes) != 1 or len(timepoints) < 2:
        return "none"
    if next(iter(treatments)) != "cort":
        return "none"
    sex = next(iter(sexes))
    if sex == "female":
        return "did_female"
    if sex == "male":
        return "did_male"
    return "none"


def strip_timepoints(
    groups: Sequence[str],
    config: ExperimentConfig,
    effects: pd.DataFrame,
) -> Tuple[list, dict]:
    """Post-baseline timepoints in ``groups`` that exist in the effects table.

    Returns ``(timepoints, group_by_timepoint)`` ordered as ``config.time_order``.
    """
    group_by_tp = {}
    for group in groups:
        _, _, timepoint = parse_group_meta(group, config)
        if timepoint is not None:
            group_by_tp[timepoint] = group

    available = set(effects["timepoint"].astype(str))
    ordered = [
        tp
        for tp in config.time_order
        if tp in group_by_tp and tp in available
    ]
    return ordered, group_by_tp


def q_lookup(
    effects: pd.DataFrame,
    contrast: str,
) -> Dict[Tuple[int, str], float]:
    """Map ``(syllable, timepoint)`` to the FDR q-value for ``contrast``."""
    if contrast not in Q_COLUMNS:
        raise ValueError(f"No q-value column for contrast {contrast!r}")
    col = Q_COLUMNS[contrast]
    lookup = {}
    syllables = effects["syllable"].astype(int)
    timepoints = effects["timepoint"].astype(str)
    qvals = effects[col]
    for syllable, timepoint, q_val in zip(syllables, timepoints, qvals):
        lookup[(int(syllable), str(timepoint))] = q_val
    return lookup


def sig_note_for_contrast(contrast: str) -> str:
    if contrast == "ddd":
        return SIG_NOTE_DDD
    return SIG_NOTE_DID
