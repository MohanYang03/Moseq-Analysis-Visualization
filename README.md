# MoSeq Analysis Visualization

Analyze MoSeq syllable usage after you have fitted a model and labeled syllables by behavior category. Analysis and plotting code lives in `moseq_analysis/`; `syllable_stats.ipynb` is the main workflow.

## What you need

1. `stats_df.csv` from your MoSeq model output directory
2. `moseq_df.csv` (only required for fingerprint plots)
3. An Excel file of syllable behavior labels, with columns `syllable short id` and `general category`
4. Motion Sequencing conda enviorment (see the moseq wiki: https://github.com/dattalab/moseq2-app/wiki/Install-MoSeq-on-MacOS-using-Conda)

Group names in the CSVs should follow `{treatment}_{timepoint}_{sex}`, for example `cort_b2_m` or `veh_wk3_f`.

## Set up `syllable_stats.ipynb`

1. Clone this repo and open it as the working directory.
2. Select the moseq2-app conda env as the kernel.
3. Open `syllable_stats.ipynb` and edit **section 1 (Experiment settings)** first:

   - `DATA_DIR` folder containing `stats_df.csv` (and `moseq_df.csv` if used)
   - `LABELS_XLSX` / `LABELS_SHEET` path and sheet name for syllable labels
   - `TREATMENTS`, `TIME_ORDER`, `TIMEPOINT_LABELS`, `SEX_CODES` match the study design and group naming
   - `VALID_SYLLABLE_CATEGORIES` category names used in the label sheet
   - `TIME_COLOR_PALETTES` one color list per `(treatment, sex)`, same length/order as `TIME_ORDER`
   - `COMBINE_SEX_GROUPS`, `MAX_SYLLABLE_ID` optional analysis options

4. Run the notebook top to bottom:
   - load stats and labels
   - build per-mouse usage / entropy tables
   - build group mean / SEM tables
   - run the `plot_syllable_usage_diff` cells to save figures under `figures/eps/` and `figures/png/`

5. Optional sections:
   - transition matrices expects `{group}_bigram_transition_matrix.csv` under `DATA_DIR`
   - fingerprints requires `moseq2_viz` and streams the large `moseq_df.csv`
