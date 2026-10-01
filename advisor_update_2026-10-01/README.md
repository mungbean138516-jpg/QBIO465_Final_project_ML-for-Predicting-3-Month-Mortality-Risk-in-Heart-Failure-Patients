# October 1 advisor update: aggregate result bundle

This directory contains **aggregate-only** tables, plots, run settings and validation records. It does not contain `dat.csv`, fitted estimators, row identifiers, row-level scores, or the executed notebook. Those local artifacts stay in the Git-ignored `../comparison_results/`.

The full Chinese [teacher report](../TEACHER_PROGRESS_REPORT_2026-10-01.md) explains the source, sample, methods, results and limits. [Machine-readable handoff](../HEART_FAILURE_ORGLAB_HANDOFF_2026-10-01.json) is intended for combining with the separate OrgLab project.

The main comparison is a separate rerun of the September workflow, with the same input hash and split but different package versions. `timing_sensitivity_*` files use a more conservative admission/day-one feature list; `feature_set_overlap.csv` directly compares each model's main and restricted high-risk lists at the same capacity. Both are exploratory analyses on the previously inspected test split.

To regenerate with an authorized **local** `../dat.csv` and the environment in `environment.yml`, run `python reproduce.py` in this directory. Nested baseline tuning can be computationally expensive. No script pushes files to GitHub.
