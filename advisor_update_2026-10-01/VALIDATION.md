# Validation status

Completed locally with the authorized `dat.csv` whose SHA-256 is recorded in `data_audit.json`:

1. `python -m unittest -q test_risk_comparison.py`: 3 synthetic tests passed. They check equal capacity, ties, outcome accounting, row-order invariance, invalid inputs and train-only feature filtering.
2. `python audit_source_data.py dat.csv`: input has 2,008 rows, 167 columns, 42 positive 3-month labels; no missing mortality labels or nonmonotone cumulative death labels. Two discharge death fields disagree on seven rows, including five `Died` destinations with 3-month label zero. This is unresolved and is reported, not repaired.
3. `python run_comparison.py`: completed all notebook code and exported private fitted models and test predictions. `summarize_advisor_update.py` independently reconstructs each confusion matrix from private row decisions, verifies fixed budgets and matching data hash, then exports aggregates only.
4. `python verify_saved_comparison.py dat.csv`: all 402 scores per model match reloaded fitted estimators within absolute tolerance `1e-6`; measured maxima are in `model_roundtrip_validation.json`.
5. `python timing_sensitivity.py` and `python compare_feature_sets.py`: completed on the same split; feature-set overlap checked against aligned keys and unchanged labels.
6. Visual review completed for `recall_uncertainty.png` and `overlap_by_budget.png`; axes and denominators are explicit. Patient-level row keys were not found in the aggregate bundle or teacher report.

The test set was previously inspected; these are exploratory results. No external validity, admission-instant feature availability, patient independence, follow-up completeness, score calibration, or clinical utility is established.
