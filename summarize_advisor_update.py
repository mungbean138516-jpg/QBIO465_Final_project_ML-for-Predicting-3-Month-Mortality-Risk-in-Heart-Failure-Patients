"""Validate a local rerun and export aggregates only to a separate dated bundle.

Patient-level tables and fitted estimators stay in ignored comparison_results/.
Intervals are exploratory, conditional on the fixed fitted models and test split.
"""
from itertools import combinations
from pathlib import Path
import hashlib
import json
import platform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.metrics import confusion_matrix
from importlib.metadata import version

ROOT = Path(__file__).resolve().parent
PRIVATE = ROOT / "comparison_results"
PUBLIC = ROOT / "advisor_update_2026-10-01"
PUBLIC.mkdir(exist_ok=True)
PRED = pd.read_csv(PRIVATE / "test_predictions.csv")
PATIENT = pd.read_csv(PRIVATE / "patient_comparison.csv")
SCREEN = pd.read_csv(PRIVATE / "screening_summary.csv")
OVERLAP = pd.read_csv(PRIVATE / "model_overlap.csv")
MANIFEST = json.loads((PRIVATE / "run_manifest.json").read_text())
DATA_AUDIT = json.loads((PUBLIC / "data_audit.json").read_text())
assert MANIFEST["data_sha256"] == DATA_AUDIT["input_sha256"]
assert len(PRED) == len(MANIFEST["test_rows"]) == 402
assert PRED.row_key.is_unique and int(PRED.y_true.sum()) == 8
assert set(PRED.row_key) == set(PATIENT.row_key)
assert not PATIENT[["row_key", "policy"]].duplicated().any()
for row in SCREEN.itertuples():
    part = PATIENT[PATIENT.policy == row.policy]
    cm = confusion_matrix(part.y_true, part[f"{row.model}_selected"], labels=[0, 1]).ravel()
    np.testing.assert_array_equal(cm, [row.tn, row.fp, row.fn, row.tp])
for policy, group in SCREEN.groupby("policy"):
    if policy != "validation_f1":
        assert group.selected.nunique() == 1

for name in ["screening_summary", "model_overlap", "ranking_summary",
             "final_model_performance_summary", "cv_model_performance_summary", "comparison_settings"]:
    suffix = ".json" if name == "comparison_settings" else ".csv"
    (PUBLIC / f"{name}{suffix}").write_bytes((PRIVATE / f"{name}{suffix}").read_bytes())

# Exact binomial recall interval, conditional on a fixed model and this test set.
fixed = SCREEN[SCREEN.policy != "validation_f1"].copy()
fixed["recall_95_low"] = fixed.apply(
    lambda r: 0.0 if r.tp == 0 else beta.ppf(.025, r.tp, r.fn + 1), axis=1)
fixed["recall_95_high"] = fixed.apply(
    lambda r: 1.0 if r.fn == 0 else beta.ppf(.975, r.tp + 1, r.fn), axis=1)
fixed[["policy", "model", "selected", "deaths", "tp", "fn", "recall",
       "recall_95_low", "recall_95_high"]].to_csv(PUBLIC / "screening_uncertainty.csv", index=False)

# Paired nonparametric test-row bootstrap, recomputing equal-capacity rankings.
# This quantifies finite test-set variation, not refitting, temporal, or transport uncertainty.
models = ["LR", "RF", "NN"]
fractions = [.05, .10, .20]
y = PRED.y_true.to_numpy(dtype=int)
scores = PRED[models].to_numpy(dtype=float)
rng = np.random.default_rng(20261001)
B = 2000
records = []
for _ in range(B):
    ix = rng.integers(0, len(y), len(y))
    yy, ss = y[ix], scores[ix]
    positives = int(yy.sum())
    if positives == 0 or positives == len(yy):
        continue
    result = {"average_precision": {}, "roc_auc": {}}
    for j, model in enumerate(models):
        result["average_precision"][model] = average_precision_score(yy, ss[:, j])
        result["roc_auc"][model] = roc_auc_score(yy, ss[:, j])
    for fraction in fractions:
        k = int(np.ceil(len(yy) * fraction))
        metric = f"top_{100*fraction:g}_recall"
        result[metric] = {}
        for j, model in enumerate(models):
            order = np.argsort(-ss[:, j], kind="stable")
            result[metric][model] = float(yy[order[:k]].sum() / positives)
    records.append(result)

interval_rows = []
for metric in records[0]:
    for model in models:
        vals = np.array([r[metric][model] for r in records])
        interval_rows.append({"metric": metric, "contrast": model,
                              "bootstrap_mean": float(vals.mean()),
                              "percentile_2_5": float(np.quantile(vals, .025)),
                              "percentile_97_5": float(np.quantile(vals, .975))})
    for a, b in combinations(models, 2):
        vals = np.array([r[metric][a] - r[metric][b] for r in records])
        interval_rows.append({"metric": metric, "contrast": f"{a}-{b}",
                              "bootstrap_mean": float(vals.mean()),
                              "percentile_2_5": float(np.quantile(vals, .025)),
                              "percentile_97_5": float(np.quantile(vals, .975))})
pd.DataFrame(interval_rows).to_csv(PUBLIC / "paired_bootstrap_intervals.csv", index=False)

fig, ax = plt.subplots(figsize=(8, 4.5))
colors = {"LR": "#34679a", "RF": "#d1812e", "NN": "#558c6a"}
offset = {"LR": -.22, "RF": 0, "NN": .22}
for model, group in fixed.groupby("model", sort=False):
    xx = group.selected.to_numpy() + offset[model] * 12
    yy = group.recall.to_numpy()
    ax.errorbar(xx, yy, yerr=[yy - group.recall_95_low.to_numpy(),
                             group.recall_95_high.to_numpy() - yy],
                fmt="o-", capsize=3, color=colors[model], label=model)
ax.set_xticks([21, 41, 81])
ax.set_xlabel("Records flagged of 402 (equal capacity)")
ax.set_ylabel("Recall among 8 observed deaths")
ax.set_ylim(0, 1.02)
ax.grid(axis="y", alpha=.2)
ax.legend(frameon=False, ncol=3)
ax.set_title("Observed recall; conditional 95% binomial intervals")
fig.tight_layout()
fig.savefig(PUBLIC / "recall_uncertainty.png", dpi=180)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
pair_data = OVERLAP[OVERLAP.policy != "validation_f1"].copy()
pair_data["selected_each"] = pair_data.policy.map(
    {"top_5_percent": 21, "top_10_percent": 41, "top_20_percent": 81})
pair_data["pair"] = pair_data.model_a + " / " + pair_data.model_b
pair_colors = {"LR / RF": "#34679a", "LR / NN": "#d1812e", "RF / NN": "#558c6a"}
for pair, group in pair_data.groupby("pair", sort=False):
    group = group.sort_values("selected_each")
    axes[0].plot(group.selected_each, group.jaccard, "o-", label=pair, color=pair_colors[pair])
    axes[1].plot(group.selected_each, group.disagreement, "o-", label=pair, color=pair_colors[pair])
axes[0].set_ylabel("Jaccard overlap")
axes[0].set_ylim(0, 1)
axes[1].set_ylabel("Different flag decisions (of 402)")
for ax in axes:
    ax.set_xticks([21, 41, 81])
    ax.set_xlabel("Flagged by each model")
    ax.grid(axis="y", alpha=.2)
axes[1].legend(frameon=False, fontsize=8)
fig.suptitle("Model pairs disagree despite matched capacity")
fig.tight_layout()
fig.savefig(PUBLIC / "overlap_by_budget.png", dpi=180)
plt.close(fig)

env = {name: version(name) for name in ["numpy", "pandas", "scikit-learn", "tensorflow", "keras", "joblib", "matplotlib", "scipy"]}
(PUBLIC / "reanalysis_settings.json").write_text(json.dumps({
    "base_commit": "18cbbfb0614535e278ac36a754b39fd7ec13dda9",
    "data_sha256": MANIFEST["data_sha256"],
    "python": platform.python_version(), "packages": env,
    "training_command": "/Users/moyingli/miniconda3/envs/qbio465/bin/python run_comparison.py",
    "summary_command": "/Users/moyingli/miniconda3/envs/qbio465/bin/python summarize_advisor_update.py",
    "training_seed": MANIFEST["random_state"],
    "bootstrap_seed": 20261001, "bootstrap_replicates_requested": B,
    "bootstrap_replicates_valid": len(records),
    "bootstrap_scope": "Paired resampling of fixed test rows, score models held fixed; equal-capacity top-k reranked per sample; exploratory only.",
    "test_predictions_sha256": hashlib.sha256((PRIVATE / "test_predictions.csv").read_bytes()).hexdigest(),
    "patient_level_outputs_public": False
}, indent=2) + "\n")
print("Exported aggregate-only bundle", PUBLIC)
