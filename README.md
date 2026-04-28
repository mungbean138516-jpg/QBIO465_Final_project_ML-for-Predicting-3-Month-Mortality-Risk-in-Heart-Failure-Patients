# QBIO465 Final Project: Machine Learning for Predicting 3-Month Mortality Risk in Heart Failure Patients

## Overview

This repository contains a QBIO 465 final project focused on predicting **3-month mortality risk** in hospitalized heart failure patients using structured clinical data.

The project compares three modeling approaches:

- Logistic Regression
- Random Forest
- Multi-task Neural Network with shared layers and three outputs:
  - death within 28 days
  - death within 3 months
  - death within 6 months

The main prediction target is `death.within.3.months`.

## Research Question

Can clinical variables collected during hospitalization be used to estimate which heart failure patients are at elevated risk of death within 3 months?

## Dataset

- Main dataset path expected by the notebook: `dat.csv`
- Data dictionary: `dataDictionary.csv`
- Number of patients/rows: 2008
- Raw CSV columns: 167
- Outcome prevalence:
  - 28-day mortality: 37 / 2008 (1.84%)
  - 3-month mortality: 42 / 2008 (2.09%)
  - 6-month mortality: 57 / 2008 (2.84%)

The dataset is highly imbalanced, so **PR-AUC, recall, and F1** are more informative than accuracy alone.

By default, the repository is configured to **not track `dat.csv` in git** unless you explicitly decide that public redistribution is allowed.

## Project Workflow

The notebook performs the following steps:

1. Load the dataset and variable dictionary.
2. Explore outcome imbalance, missingness, age/gender patterns, and feature correlations.
3. Remove obvious leakage variables and non-predictive identifiers before modeling.
4. Build a preprocessing pipeline with imputation, scaling, and one-hot encoding.
5. Train and tune baseline models with 5-fold cross-validation.
6. Train a multi-task neural network that jointly predicts 28-day, 3-month, and 6-month mortality.
7. Tune classification thresholds on a validation split to improve F1 under class imbalance.
8. Export trained models and summary tables.

## Modeling Summary

### Baselines

- **Logistic Regression**
  - Class-weighted
  - Hyperparameters tuned with `RandomizedSearchCV`
  - Stronger interpretability through coefficients

- **Random Forest**
  - Hyperparameters tuned with `RandomizedSearchCV`
  - Best overall held-out ranking performance in this project

### Multi-task Neural Network

- Shared hidden layers with three binary outputs
- Auxiliary tasks:
  - 28-day mortality
  - 6-month mortality
- Main task:
  - 3-month mortality
- Uses class weighting on the 3-month task to stabilize learning under severe imbalance

## Evaluation Design

- Main split: 80% train / 20% test
- Stratified by 3-month mortality label
- 5-fold cross-validation performed on the training portion
- Logistic Regression and Random Forest tuned for **average precision (PR-AUC)**
- Probability thresholds selected on a separate validation split to improve F1

## Key Results

### Held-Out Test Set Performance

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Threshold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.978 | 0.400 | 0.250 | 0.308 | 0.804 | 0.210 | 0.77 |
| Random Forest | 0.965 | 0.200 | 0.250 | 0.222 | 0.876 | 0.331 | 0.10 |
| Multi-task Neural Network | 0.953 | 0.176 | 0.375 | 0.240 | 0.762 | 0.278 | 0.39 |

### 5-Fold Cross-Validation Summary

| Model | ROC-AUC Mean | PR-AUC Mean | Recall Mean | F1 Mean | Precision Mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.833 | 0.333 | 0.524 | 0.199 | 0.123 |
| Random Forest | 0.899 | 0.420 | 0.000 | 0.000 | 0.000 |

## Interpretation

- The **Random Forest** achieved the strongest held-out **PR-AUC** and **ROC-AUC**, making it the best ranking model in this repository.
- The **Multi-task Neural Network** reached the highest **recall**, which may be useful when missing high-risk patients is more costly than generating false positives.
- The **Logistic Regression** remains the easiest model to interpret and still performed competitively despite the rare-event setting.
- Because the positive class rate is only about **2.1%**, high accuracy should not be over-interpreted.

## Example Model Insights

Top variables highlighted by the trained models include:

- `hydroxybutyrate.dehydrogenase`
- `eye.opening`
- `GCS`
- `high.sensitivity.troponin`
- `D.dimer`
- `cystatin`
- `Inorganic.Phosphorus`
- `urea`

These are **model-derived importance signals**, not causal clinical conclusions.

## Risk Stratification Output

Using the best model by held-out PR-AUC, the test set was divided into four ranked risk groups:

| Risk Group | Observed 3-Month Mortality Rate | Patients |
| --- | ---: | ---: |
| Low | 0.000 | 101 |
| Medium-low | 0.000 | 100 |
| Medium-high | 0.020 | 100 |
| High | 0.059 | 101 |

## Repository Structure

```text
.
|-- QBIO465_Final_Project_TunedBaselines.ipynb
|-- dataDictionary.csv
|-- requirements.txt
|-- environment.yml
|-- saved_results_data/
|   |-- cv_model_performance_summary.csv
|   |-- final_model_performance_summary.csv
|   |-- logistic_regression_coefficients.csv
|   |-- logistic_regression_pipeline.joblib
|   |-- multitask_heart_failure_model.keras
|   |-- nn_preprocessor.joblib
|   |-- random_forest_feature_importance.csv
|   |-- random_forest_pipeline.joblib
|   |-- risk_group_summary.csv
|   `-- README.md
`-- README.md
```

If you keep the raw dataset private, place `dat.csv` at the repository root before running the notebook.

## Reproducibility

### Option 1: Conda

```bash
conda env create -f environment.yml
conda activate qbio465-heart-failure
jupyter notebook QBIO465_Final_Project_TunedBaselines.ipynb
```

### Option 2: pip + virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
jupyter notebook QBIO465_Final_Project_TunedBaselines.ipynb
```

Recommended Python version: **3.10**

## Saved Outputs

The repository already includes exported result artifacts in `saved_results_data/`, including:

- trained Logistic Regression pipeline
- trained Random Forest pipeline
- trained multi-task neural network
- neural network preprocessor
- coefficient and feature-importance tables
- cross-validation and final test-set summaries
- risk-group calibration summary

## Limitations

- Small number of positive events relative to feature count
- No external validation cohort
- Threshold tuning is sensitive in rare-event settings
- Notebook-based workflow rather than a fully packaged training pipeline
- This is a course research project and **not** a clinical decision support tool

## Data Sharing Note

This project uses patient-level clinical data in `dat.csv`. Before making the repository public, confirm that:

- the dataset is permitted for redistribution
- all protected or identifiable information has been removed
- public sharing complies with course, institutional, and data-use requirements

The current `.gitignore` excludes `dat.csv` by default. If redistribution is not allowed, keep it excluded and provide data access instructions instead of uploading the raw file.

## Suggested Next Improvements

- Move notebook logic into a reproducible training script
- Add calibration plots and confidence intervals
- Evaluate external validation or temporal validation
- Add SHAP or permutation importance for more robust interpretation
- Add a lightweight inference example for new patients
