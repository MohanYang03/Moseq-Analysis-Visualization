# Significance stars on syllable usage plots

Plan only. Do not implement until this mapping is accepted.

## Objective

Add academic-style significance marks (`*`, `**`, `***`) to `plot_syllable_usage_diff` figures, using `data/did_ddd_effects.csv` from `longitudinal_did_glmm.Rmd`.

The marks must show the **GLMM contrast that CSV actually estimates**, not a visual stand-in for “these two usage traces look different.”

## What the CSV tests

Each row is one syllable × post-baseline timepoint (`wk1`, `wk2`, `wk3`, `washout`). Baseline `b2` is the model reference and has **no row**.

| Column family | Meaning |
| --- | --- |
| `DiD_Female` | CORT vs VEH change from baseline, females (logit) |
| `DiD_Male` | Same contrast, males |
| `DDD` | Male DiD − female DiD (sex difference in the treatment effect) |
| `*_p` | Raw two-sided p from Wald z |
| `*_q` | BH FDR within each contrast family, across syllables × post timepoints |
| `estimable` | Finite SEs; `FALSE` rows were excluded from FDR |

Use **FDR q**, not raw p, to match the Rmd (`q < 0.05` is the reported significance rule). Do not star `estimable = FALSE` or missing q.

Star ladder (standard paper convention):

- `*` q < 0.05
- `**` q < 0.01
- `***` q < 0.001
- blank otherwise

Positive DiD means CORT increased usage relative to VEH more than at baseline. Do not encode sign in the star. Keep sign in the Rmd DiD/DDD figures.

## Why the current usage plots do not match the tests 1:1

`plot_syllable_usage_diff` draws **descriptive usage** (group mean ± SEM) with syllable ID on x.

The notebook currently makes three families:

1. **CORT time series** (all syllables or one category): several CORT timepoints, one sex.
2. **VEH time series**: same layout for vehicle.
3. **Male vs female at one timepoint**: two sexes, one treatment, one time.

None of these is “CORT vs VEH change from baseline.” In particular:

- DiD is **not** CORT week 3 vs CORT baseline.
- DiD is **not** contemporaneous CORT vs VEH usage.
- DDD is **not** male vs female usage at that timepoint.

Putting the wrong contrast on a plot is worse than omitting stars.

## Recommended mapping

| Plot family | Contrast to mark | How |
| --- | --- | --- |
| CORT female time series (all syllables or category) | `DiD_Female_q` at `wk1`/`wk2`/`wk3`/`washout` | Significance strip under x-axis |
| CORT male time series | `DiD_Male_q` | Same strip |
| VEH time series | none | VEH is the control arm of DiD; starring DiD here implies a VEH effect |
| Male vs female usage | none, by default | DDD is not a sex difference in usage |

Optional later (not in the first pass): a **new** CORT vs VEH usage plot at one post timepoint, still starred with DiD for that sex. Caption must say the star is DiD vs baseline, not a pairwise t-test of the two traces.

## Visual design: strip, not stars on the traces

A single “any-timepoint” star above each syllable is not useful: **44 / 49** estimable syllables have at least one female DiD with q < 0.05.

Timepoint-specific density (estimable, q < 0.05):

| Timepoint | Female DiD | Male DiD | DDD | n |
| --- | --- | --- | --- | --- |
| wk1 | 5 | 1 | 6 | 49 |
| wk2 | 22 | 5 | 14 | 49 |
| wk3 | 27 | 25 | 22 | 49 |
| washout | 33 | 23 | 21 | 49 |

Five overlaid usage series plus stacked `*` above the highest SEM will collide, especially on the full-syllable figures (`width=10`, `y_max=0.21`). Category panels are narrower (`PLOT_WIDTH=7`) and still too dense at wk3/washout for floating stars.

**Use a significance strip below the syllable axis** (common in genome-wide and some ethology figures):

```
usage mean ± SEM  (existing plot)
------------------------------------------------------------
wk1        *                    *
wk2           **     *      *
wk3     ***      **        ***
washout ***  **   *    ***   **
        0  1  2  3  ... syllable IDs in the plot order
```

Rules:

- One strip row per post timepoint, same order as `TIME_ORDER` after baseline.
- Star color matches that timepoint’s usage color (`TIME_COLOR_PALETTES`).
- Align stars to the same x as the usage points (including `exclude_syllables`, category filters, and sort order).
- No mark at syllables with no estimable q, or at IDs absent from the CSV.
- Leave a small gap between the x tick labels and the strip; expand `tight_layout` so the right-side legend still fits.
- Add a compact note on the figure or in the caption: `* q<0.05, ** q<0.01, *** q<0.001; BH FDR. DiD: CORT vs VEH vs baseline.`

Do not implement floating stars-on-traces in the first pass.

## Code changes (when implementing)

### 1. Load helper

Add a small loader, e.g. `moseq_analysis/effects.py` or a function in `plots.py`:

- Read `data/did_ddd_effects.csv` (path from notebook / config).
- Keep `syllable`, `timepoint`, `estimable`, `DiD_Female_q`, `DiD_Male_q`, `DDD_q`.
- Cast `syllable` to the same index type as `pivot_mean` (int).
- Treat `estimable != True` as missing q.

`q_to_stars(q, levels=(0.05, 0.01, 0.001))` → `*`, `**`, `***`, or `''`.

### 2. Extend `plot_syllable_usage_diff`

New arguments (defaults keep current figures unchanged):

- `effects=None`: DataFrame or CSV path.
- `sig_contrast=None`: `'did_female'`, `'did_male'`, `'ddd'`, or `'none'`.
- `sig_mode='strip'`: only `'strip'` in v1.
- `sig_alpha=0.05`: used only if we later add a binary filled/empty mode; star ladder can ignore this.

Auto-infer when `effects` is passed and `sig_contrast is None`:

- All groups CORT, one sex, ≥2 timepoints → that sex’s DiD.
- All groups VEH → `'none'` (no strip, no error).
- Mixed treatments or male-vs-female pair → `'none'` unless `sig_contrast` is set explicitly.

If `effects` is None, behavior is unchanged.

Implementation sketch:

1. Draw the existing errorbar plot.
2. If a contrast is active, look up q for each plotted syllable × each post timepoint present in `groups`.
3. Add N strip rows with `ax.annotate` / `ax.text` in data x and axes/figure y (below ticks).
4. Timepoints in `groups` that have no CSV row (`b2`) get no strip row.

Respect the same syllable subset as the plot: `categories`, `include_syllables`, `exclude_syllables`, and the sort order already computed from `sort_by`.

### 3. Notebook wiring (`syllable_stats.ipynb` usage cell)

Load once above the plot calls:

```python
effects = pd.read_csv(join(REPO_ROOT, 'data', 'did_ddd_effects.csv'))
```

Pass `effects=effects` only into **CORT** time-series calls (all-syllable and category, female and male). Leave VEH and male-vs-female calls unstarred.

Do not hard-code `lines=False` as part of this work; that flag stays independent.

### 4. Caption / title

Keep the plot title as usage. Put the statistical claim in `legend` subtitle or a one-line figtext, not in the y-axis (`Average Usage Probability +/- SEM` must remain usage, not DiD).

## What not to do

- Do not star raw p while the Rmd reports q.
- Do not star non-estimable rows (syllables 15 and 16 are the known failures).
- Do not collapse timepoints into one star per syllable.
- Do not put DDD stars on male-vs-female usage panels without a dedicated caption and a separate review.
- Do not change `longitudinal_did_glmm.Rmd` or the CSV schema for this.

## Validation

After implementation (not now):

- CORT female full plot: strip rows only for `wk1`–`washout`; wk1 sparse (~5 stars), washout dense (~33).
- Excluded syllable 29 has no star and no usage point.
- Category plots only star IDs that remain after `categories=...`.
- VEH plots identical to today (no strip).
- Re-save EPS/PNG; confirm stars are not clipped and are readable in EPS (use text `*` characters, not unicode glyphs that fail in some PostScript fonts).

## Open choice (needed before coding)

Default recommendation: **CORT time-series only, FDR star strip, no VEH/sex-comparison stars.**

If you want a different first pass, say which:

1. Keep the recommendation above.
2. Also star male-vs-female plots with DDD for that timepoint (floating `*` above the pair).
3. Add new CORT vs VEH usage panels at wk1/wk2/wk3/washout, starred with DiD.
4. Binary `*` only (q < 0.05), no `**` / `***`.
