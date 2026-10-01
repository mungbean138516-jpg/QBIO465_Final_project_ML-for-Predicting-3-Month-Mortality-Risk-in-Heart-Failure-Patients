# Analysis contract and interpretation

- **Question:** At an identical number of records flagged, how do LR, RF and NN change detected deaths, missed deaths and selected individuals? How sensitive are these patterns to feature timing?
- **Unit/denominator:** one row per index hospitalization as supplied by the local Zigong table; 402 held-out rows, eight recorded 3-month deaths. Repeated-person linkage is not available.
- **Endpoint:** supplied `death.within.3.months` binary label. No relabeling of discordant discharge fields.
- **Primary contrast:** equal-capacity top 21, 41 and 81 for each model (exploratory 5%, 10%, 20% of 402); pairwise overlap and detected/missed deaths. Validation-F1 thresholds are a separate policy with unequal capacities.
- **Data use:** same stratified seed-42 train/validation/test split for all models. Baseline 5-fold inner AP tuning and 5-fold outer CV on training only. NN uses single validation early stopping; thresholds selected on validation. Fitted filters and imputers see training partitions only.
- **Uncertainty:** exact binomial intervals for fixed-policy recall and 2,000 paired test-row bootstrap draws (1,999 valid) for rankings and equal-capacity recall. Conditional on fixed fitted models and a previously seen split. No model retraining, new cohort, or clinical effect claim.
- **Timing sensitivity:** 134 raw fields from source-documented admission demographics, admission-day status, admission-note comorbidities and day-one labs; same split. Baseline hyperparameters inherited from main training-only search; all models refitted; validation thresholds reselected. This is not fully retuned or prospective.
- **Invalid run:** abort on changed input hash, misaligned test rows, wrong event counts, mismatched confusion-matrix counts, or unequal top-k capacity.
- **Reproducibility:** seeds, package versions, input hash and commands in `reanalysis_settings.json`; selected feature list and NN epoch count in `timing_sensitivity_settings.json`.
