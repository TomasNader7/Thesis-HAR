# Thesis-HAR: HAPT -> WISDM Cross-Dataset Human Activity Recognition

Cross-dataset generalization study for Human Activity Recognition (HAR): a
stacking-ensemble model is trained on the **UCI HAPT** dataset (source domain)
and evaluated on the **WISDM** dataset (target domain), with several domain
adaptation strategies applied to recover performance lost to distribution
shift between the two sensor datasets. This is the working codebase behind
Tomas Nader's M.S. thesis on cross-dataset generalization in HAR using
ensemble learning.

## Repo layout

```
src/        pipeline scripts (run in the order below)
data/       NOT checked in -- raw + regenerated intermediate data (see "Data")
results/    small result artifacts (metrics, confusion matrices, reports) for phase1/phase2/phase3
tables/     master_table.csv -- consolidated phase3 results table
figures/    KS-statistic distribution-shift plots/tables (phase2)
reference/  CATA 2025 single-dataset paper code/figures -- background, not part of the pipeline
```

## Data

Raw datasets are **not included** in this repo (they're large and licensed
for research use, not redistribution). Source them yourself:

- **UCI HAR / HAPT** ("Smartphone-Based Recognition of Human Activities and
  Postural Transitions") -- UCI Machine Learning Repository.
- **WISDM** ("WISDM Smartphone and Smartwatch Activity and Biometrics
  Dataset") -- WISDM Lab, Fordham University.

Place the raw downloads under:

```
data/raw/hapt/       <- HAPT dataset root
data/raw/uci_har/    <- original UCI HAR dataset root (used by analysis_feature_dataset.py)
data/raw/wisdm/      <- WISDM raw/ folder
```

Running the pipeline (below) generates intermediate feature matrices under
`data/interim/`. Both `data/raw/` and `data/interim/` are gitignored --
some of these intermediate files are hundreds of MB and are meant to be
regenerated locally, not committed.

## Pipeline stages

Run from the repo root so the relative `results/`, `tables/`, and `figures/`
output paths resolve correctly. Each script's input paths are anchored to
the repo root internally (via `Path(__file__)`), so they don't depend on
your current working directory either way.

1. **Feature extraction** -- convert raw sensor readings into a UCI-HAR-style
   feature table for each dataset.
   ```
   python src/feature_extraction_WISDM.py
   python src/feature_extraction_HAPT.py   # reuses feature code from feature_extraction_WISDM.py
   ```
2. **Filtering + initial distribution-shift check** -- filter both datasets
   to the shared activity classes (3-class: WALKING/SITTING/STANDING, or
   6-class) and run an initial Kolmogorov-Smirnov test on raw features.
   ```
   python src/analysis_feature_dataset.py
   ```
3. **Naive baseline (Phase 1/2 driver)** -- train the stacking ensemble on
   HAPT, evaluate directly on WISDM with no adaptation. Set `PHASE_TAG` in
   the script to `"phase1"` or `"phase2"` depending on which feature-set
   run you're reproducing.
   ```
   python src/cross_dataset_hapt_to_wisdm.py
   ```
4. **Z-score normalization / distribution alignment, meta-learner
   adaptation, and fine-tuning** -- the domain-adaptation experiments
   (naive -> z-score -> meta-only adaptation -> fine-tuning at several
   target-data fractions).
   ```
   python src/phase3_domain_adaptation.py
   ```
5. **Base-learner diagnostics** -- per-base-learner breakdown (confidence,
   per-class F1, prediction distribution) across the phase3 conditions.
   ```
   python src/base_learner_diagnostics.py
   ```
6. **Fine-tuning learning curve** -- accuracy/recall vs. fraction of target
   data used for fine-tuning.
   ```
   python src/learning_curve.py
   ```
7. **KS-statistic distribution-shift analysis (phase2)** -- quantifies
   feature-distribution shift between HAPT and WISDM, raw vs. z-scored.
   ```
   python src/ks_analysis_phase2.py
   ```
8. **Consolidated results table** -- run after step 4, scans all
   `results/phase3/**/metrics.json` and builds one comparison table.
   ```
   python src/build_master_table.py
   ```

`results/phase1/`, `results/phase2/`, and `results/phase3/` in this repo
already contain the output of a completed run (confusion matrices,
classification reports, metrics, plots) for reference -- you don't need to
re-run everything to see the numbers, only to reproduce or extend them.

## Known open questions

- **Stack-of-a-stack meta-learner in `cross_dataset_hapt_to_wisdm.py`.**
  `get_advanced_stacking_model()` (around the `level0`/`level1` block) builds
  a `StackingClassifier` as the meta-learner, then wraps *that* in a second,
  outer `StackingClassifier` using the same base learners -- i.e. a
  stack-of-a-stack, not the "5 base learners + 1 logistic-regression
  meta-learner" architecture described in the thesis text. This is flagged
  with a `# TODO(verify):` comment at the exact spot in the file. It has
  **not** been resolved -- it needs a decision (intentional deeper ensemble,
  or a copy/paste bug to fix) before the phase1 results produced by this
  script are treated as final.

## Related work (reference only, not part of this pipeline)

`reference/cata2025/` holds `Human_Activity_Recognition.py` (a stacking +
CNN model on the single-dataset UCI HAR problem, no cross-dataset transfer)
plus its result figures, behind the separate, already-published undergrad
paper *"Human Activity Recognition using an Ensemble Learning Approach"*
(CATA 2025, Nader/Murad/Rahimi). Same stacking architecture as this thesis,
applied to an easier single-dataset setting -- useful background/methodology
context, not something this pipeline runs or depends on. Note: the figures
under `reference/cata2025/cross_val_results/` and `figures/` weren't
regenerable from any script found in the old repo during migration, so
treat them as archival images rather than reproducible artifacts.

## Excluded from this repo

- **`cross_dataset_eval.py`** -- confirmed dead code. It repeats an old,
  already-fixed bug (a 561-feature-to-61-feature `SelectKBest` mismatch) and
  is not part of the current working pipeline. Deliberately not migrated.
- Large regenerated feature-matrix dumps (`Filtered_datasets_and_KS_results/`,
  `hapt_3class_output_phase2/`, `wisdm_phase2/`, loose `X_wisdm.txt` /
  `wisdm_features_normalized.csv`, etc. -- several hundred MB combined) are
  gitignored. They're intermediate outputs of the feature-extraction and
  filtering scripts above, not source material.
