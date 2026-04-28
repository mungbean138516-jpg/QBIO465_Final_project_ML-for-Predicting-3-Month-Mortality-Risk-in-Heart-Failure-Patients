# Saved Result Artifacts

This folder stores exported artifacts generated from `QBIO465_Final_Project_TunedBaselines.ipynb`.

## Files

- `cv_model_performance_summary.csv`
  - Cross-validation summary for tuned baseline models.

- `final_model_performance_summary.csv`
  - Held-out test-set metrics for Logistic Regression, Random Forest, and the multi-task neural network.

- `logistic_regression_coefficients.csv`
  - Logistic Regression coefficients after preprocessing.

- `logistic_regression_pipeline.joblib`
  - Serialized preprocessing + Logistic Regression pipeline.

- `multitask_heart_failure_model.keras`
  - Saved Keras multi-task neural network.

- `nn_preprocessor.joblib`
  - Serialized preprocessing object used for neural network inputs.

- `random_forest_feature_importance.csv`
  - Random Forest feature importance table.

- `random_forest_pipeline.joblib`
  - Serialized preprocessing + Random Forest pipeline.

- `risk_group_summary.csv`
  - Observed 3-month mortality rate across model-ranked risk quartiles.

## Note

These artifacts are included to make the course project easier to inspect without rerunning the full notebook.
