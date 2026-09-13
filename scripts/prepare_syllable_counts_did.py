#!/usr/bin/env python
"""Build mouse x timepoint syllable onset counts for longitudinal DiD GLMM analysis.

Reads frame-level ``moseq_df.csv``, keeps syllable onsets (usage-sort labels >= 0),
and writes one row per (session, syllable) with counts Y and session total N.

Mouse IDs come from ``SubjectName``. Batch-style SubjectNames (veh_sal, cort_sal,
veh_ket, cort_ket) are replaced by animal IDs parsed from ``SessionName``.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, Optional, Sequence, Set

import numpy as np
import pandas as pd

BATCH_SUBJECTS = {"veh_sal", "cort_sal", "veh_ket", "cort_ket"}
DEFAULT_TIMEPOINTS = ("b2", "wk1", "wk2", "wk3", "washout")
DEFAULT_TREATMENTS = ("cort", "veh")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data-dir",
        required=True,
        help="Directory containing moseq_df.csv (and optionally stats_df.csv).",
    )
    p.add_argument(
        "--out",
        required=True,
        help="Output CSV path for syllable_counts_did.csv.",
    )
    p.add_argument(
        "--timepoints",
        nargs="+",
        default=list(DEFAULT_TIMEPOINTS),
        help="Timepoint tokens to include (baseline first). Default: b2 wk1 wk2 wk3 washout",
    )
    p.add_argument(
        "--treatments",
        nargs="+",
        default=list(DEFAULT_TREATMENTS),
        help="Treatment tokens. Default: cort veh",
    )
    p.add_argument(
        "--exclude-group-tokens",
        nargs="+",
        default=["ket", "sal"],
        help="Drop groups whose names contain these tokens.",
    )
    p.add_argument("--chunksize", type=int, default=2_000_000)
    return p.parse_args()


def is_focus_group(
    group: str,
    treatments: Sequence[str],
    timepoints: Sequence[str],
    exclude_tokens: Sequence[str],
) -> bool:
    g = str(group).lower()
    if any(tok.lower() in g for tok in exclude_tokens):
        return False
    parts = g.split("_")
    if len(parts) < 3:
        return False
    treatment = parts[0]
    sex = parts[-1]
    timepoint = "_".join(parts[1:-1])
    return (
        treatment in treatments
        and timepoint in timepoints
        and sex in {"m", "f"}
    )


def parse_group(group: str) -> tuple[str, str, str]:
    parts = str(group).lower().split("_")
    return parts[0], "_".join(parts[1:-1]), parts[-1]


def recover_mouse_id(subject_name: str, session_name: str, uuid: str) -> str:
    subj = str(subject_name)
    if subj.lower() not in BATCH_SUBJECTS:
        return subj
    sess = str(session_name)
    m1 = re.search(r"_(\d{2,4})_(?:bb|wk\d|washout|bsln)", sess, re.I)
    if m1:
        return f"id_{m1.group(1)}"
    m2 = re.search(r"_(\d{2,4})$", sess)
    if m2:
        return f"id_{m2.group(1)}"
    m3 = re.search(r"(Veh|Cort)_(\d+)", sess, re.I)
    if m3:
        return f"{m3.group(1).lower()}_{m3.group(2)}"
    return f"uuid_{str(uuid)[:8]}"


def extract_session_meta(
    moseq_csv: Path,
    focus_groups: Set[str],
    chunksize: int,
) -> pd.DataFrame:
    parts = []
    for chunk in pd.read_csv(
        moseq_csv,
        usecols=["uuid", "SubjectName", "SessionName", "group"],
        chunksize=chunksize,
    ):
        chunk = chunk[chunk["group"].astype(str).isin(focus_groups)]
        if chunk.empty:
            continue
        parts.append(chunk.drop_duplicates())
    if not parts:
        raise RuntimeError("No focus sessions found in moseq_df.csv")
    meta = pd.concat(parts, ignore_index=True).drop_duplicates("uuid")
    parsed = meta["group"].map(parse_group)
    meta["treatment"] = parsed.map(lambda x: x[0])
    meta["timepoint"] = parsed.map(lambda x: x[1])
    meta["sex"] = parsed.map(lambda x: x[2])
    meta["mouse_id"] = [
        recover_mouse_id(s, sess, u)
        for s, sess, u in zip(meta["SubjectName"], meta["SessionName"], meta["uuid"])
    ]
    return meta


def aggregate_onset_counts(
    moseq_csv: Path,
    focus_uuids: Iterable[str],
    chunksize: int,
) -> pd.DataFrame:
    focus = set(map(str, focus_uuids))
    parts = []
    n_read = 0
    for chunk in pd.read_csv(
        moseq_csv,
        usecols=["uuid", "onset", "labels (usage sort)"],
        chunksize=chunksize,
    ):
        n_read += len(chunk)
        chunk = chunk[chunk["uuid"].astype(str).isin(focus)]
        if chunk.empty:
            continue
        chunk = chunk[chunk["onset"].astype(bool)]
        chunk = chunk[chunk["labels (usage sort)"] >= 0]
        if chunk.empty:
            continue
        g = chunk.groupby(["uuid", "labels (usage sort)"]).size().reset_index(name="y")
        g = g.rename(columns={"labels (usage sort)": "syllable"})
        parts.append(g)
        print(f"read {n_read:,} frames; kept count chunks={len(parts)}")
    if not parts:
        raise RuntimeError("No syllable onsets found for focus sessions")
    counts = pd.concat(parts, ignore_index=True)
    return counts.groupby(["uuid", "syllable"], as_index=False)["y"].sum()


def complete_count_grid(counts: pd.DataFrame, uuids: Sequence[str]) -> pd.DataFrame:
    syllables = sorted(counts["syllable"].unique())
    idx = pd.MultiIndex.from_product([list(uuids), syllables], names=["uuid", "syllable"])
    out = counts.set_index(["uuid", "syllable"]).reindex(idx, fill_value=0).reset_index()
    out["N"] = out.groupby("uuid")["y"].transform("sum")
    out["usage"] = np.where(out["N"] > 0, out["y"] / out["N"], np.nan)
    return out


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    moseq_csv = data_dir / "moseq_df.csv"
    if not moseq_csv.exists():
        raise FileNotFoundError(moseq_csv)

    # Discover focus groups from a light metadata pass
    group_set: Set[str] = set()
    for chunk in pd.read_csv(moseq_csv, usecols=["group"], chunksize=args.chunksize):
        group_set.update(chunk["group"].astype(str).unique())
    focus_groups = {
        g
        for g in group_set
        if is_focus_group(g, args.treatments, args.timepoints, args.exclude_group_tokens)
    }
    print(f"focus groups ({len(focus_groups)}): {sorted(focus_groups)}")

    meta = extract_session_meta(moseq_csv, focus_groups, args.chunksize)
    print(
        f"sessions={len(meta)} mice={meta['mouse_id'].nunique()} "
        f"subjects_raw={meta['SubjectName'].nunique()}"
    )
    dup = meta.groupby(["mouse_id", "timepoint"]).size()
    if (dup > 1).any():
        raise RuntimeError(
            "Duplicate mouse_id x timepoint sessions after ID recovery:\n"
            + dup[dup > 1].to_string()
        )

    counts = aggregate_onset_counts(moseq_csv, meta["uuid"], args.chunksize)
    grid = complete_count_grid(counts, meta["uuid"].astype(str).tolist())
    design_cols = [
        "uuid",
        "mouse_id",
        "SubjectName",
        "SessionName",
        "group",
        "treatment",
        "timepoint",
        "sex",
    ]
    df = grid.merge(meta[design_cols], on="uuid", how="left")
    df["T"] = (df["treatment"] == "cort").astype(int)
    df["S"] = (df["sex"] == "m").astype(int)
    df["timepoint"] = pd.Categorical(
        df["timepoint"], categories=list(args.timepoints), ordered=True
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote {out} shape={df.shape}")
    print(df.groupby(["treatment", "sex", "timepoint"])["mouse_id"].nunique().unstack())


if __name__ == "__main__":
    main()
