# Model choice and high-risk identification

Status: implementation prepared; no new patient-level analysis has been run.

The Oxford meeting emphasizes the technical workflow and how choosing models and evaluation criteria changes judgments about people. The meeting did not prescribe top-k percentages. The 5%, 10%, and 20% budgets below are our concrete exploratory design to investigate that question.

## Run

Put the original project dataset `dat.csv` beside the notebooks, install the repository requirements, and run `Heart_Failure_Model_Disagreement.ipynb` from top to bottom. It imports `risk_comparison.py`; keep the two files together. Nested baseline tuning is computationally expensive. TensorFlow is required for the neural network. The original notebook and saved historical results remain intact.

The revised notebook separates train2, validation, and test before hyperparameter selection. Missingness and constant-column filters, imputation, scaling and encoding are fitted within each fitting partition. LR and RF use nested CV on train2; the NN uses a validation split, not cross-validation. Validation selects F1 thresholds. NN validation also selects the early-stopping epoch; a further independent assessment would be needed to measure selection instability. The exact models evaluated on the test data are saved without a later refit. All three models use the same partitions.

## Questions and outputs

| Question | Output in `comparison_results/` |
| --- | --- |
| At equal screening budgets, how many deaths does each model identify or miss? | `screening_summary.csv`, `screening_tradeoffs.png` |
| Which model pairs flag the same rows? | `model_overlap.csv`: overlap, Jaccard, disagreements, deaths found by only one model |
| Which observed deaths are missed by each model? | `death_case_review.csv` |
| What are the three scores and decisions for each test row? | `patient_comparison.csv`, `test_predictions.csv` |
| Does the preferred model change with the evaluation metric? | `ranking_summary.csv` and threshold-policy rows in `screening_summary.csv` |
| What exactly was run? | `run_manifest.json`, `comparison_settings.json`, saved fitted models |

Top-k selects exactly ceil(n × fraction) rows, independent of true outcomes. Ties use a shared seeded random row-key ordering; lower/upper recall attainable within the tied boundary are reported. Those bounds are not confidence intervals. Test labels are used only to evaluate the pre-specified policies. These budgets are not clinical thresholds. The F1-threshold policy is reported separately because it need not select the same number of people.

## Interpretation boundaries

- Historical test results have already been inspected. This extension is exploratory, not a fresh independent test or external validation.
- Only eight deaths were in the historical test split. Differences of one death change recall by 12.5 percentage points. No reliable superiority or significance claim follows automatically.
- Do not interpret uncalibrated prediction scores as verified absolute clinical risks.
- Different flags establish model disagreement. They do not demonstrate actual treatment consequences, causal effects, improved survival, or effects on user trust.
- Data provenance, feature measurement timing, repeated patients, follow-up completeness, and cumulative mortality label consistency still require source-data verification.
- The revised results will differ from historical results because tuning no longer uses the later validation partition. Do not combine the two runs.
- Summary evaluation in code uses average precision; historical `PR_AUC` column names denote AP, not trapezoidal integration.
- Patient-level outputs and the row-position manifest are ignored by git. Analysis keys are only within-run identifiers. No raw dataset or patient-level predictions are committed.

## Verification completed without patient data

Synthetic tests verify matched budgets, death/missed-death accounting, pairwise disagreement, tied-boundary bounds, invariance to row order, invalid-input rejection, and training-only feature filtering. These are software checks, not study results. The full notebook has not been executed because `dat.csv` is unavailable.
