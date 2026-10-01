"""Exploratory sensitivity using only feature groups documented at admission/day one.

The source paper describes admission demographics, admission-day clinical status,
comorbidities from admission notes, and day-one laboratory measurements. This
restricted set excludes echocardiography and other fields without a stated
measurement time. It is not a prospective admission-time validation.
"""
import hashlib
import json
import os
from pathlib import Path

for key, value in {"MPLBACKEND": "Agg", "TF_NUM_INTEROP_THREADS": "1",
                   "TF_NUM_INTRAOP_THREADS": "1", "OMP_NUM_THREADS": "1",
                   "OPENBLAS_NUM_THREADS": "1", "TF_CPP_MIN_LOG_LEVEL": "2"}.items():
    os.environ.setdefault(key, value)

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

from risk_comparison import TrainingFeatureFilter, compare_predictions

ROOT = Path(__file__).resolve().parent
PRIVATE = ROOT / "comparison_results/timing_sensitivity"
PUBLIC = ROOT / "advisor_update_2026-10-01"
PRIVATE.mkdir(parents=True, exist_ok=True)
PUBLIC.mkdir(exist_ok=True)
DATA = ROOT / "dat.csv"
manifest = json.loads((ROOT / "comparison_results/run_manifest.json").read_text())
assert hashlib.sha256(DATA.read_bytes()).hexdigest() == manifest["data_sha256"]
df = pd.read_csv(DATA)

demographic = ["admission.ward", "admission.way", "occupation", "gender", "ageCat"]
admission_clinical = ["body.temperature", "pulse", "respiration",
    "systolic.blood.pressure", "diastolic.blood.pressure", "map", "weight",
    "height", "BMI", "type.of.heart.failure",
    "NYHA.cardiac.function.classification", "Killip.grade", "GCS"]
comorbidity = ["myocardial.infarction", "congestive.heart.failure",
    "peripheral.vascular.disease", "cerebrovascular.disease", "dementia",
    "Chronic.obstructive.pulmonary.disease", "connective.tissue.disease",
    "peptic.ulcer.disease", "diabetes",
    "moderate.to.severe.chronic.kidney.disease", "hemiplegia", "leukemia",
    "malignant.lymphoma", "solid.tumor", "liver.disease", "AIDS", "CCI.score"]
columns = list(df.columns)
start, end = columns.index("creatinine.enzymatic.method"), columns.index("total.hemoglobin")
labs = columns[start:end + 1]
assert len(labs) > 80 and not any("death" in c or "admission.time" in c for c in labs)
features = demographic + admission_clinical + comorbidity + labs
assert len(set(features)) == len(features) and set(features).issubset(df.columns)
assert not set(features).intersection({"No.", "outcome.during.hospitalization", "dischargeDay"})

train_ix, val_ix, test_ix = [manifest[name] for name in
    ["train_rows", "validation_rows", "test_rows"]]
xtrain, xval, xtest = [df.loc[ix, features] for ix in [train_ix, val_ix, test_ix]]
target = "death.within.3.months"
ytrain, yval, ytest = [df.loc[ix, target].astype(int) for ix in [train_ix, val_ix, test_ix]]
assert (len(ytrain), len(yval), len(ytest)) == (1284, 322, 402)
assert tuple(int(x.sum()) for x in [ytrain, yval, ytest]) == (27, 7, 8)

numeric = Pipeline([("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())])
categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])

def preprocess():
    return Pipeline([("filter", TrainingFeatureFilter()),
        ("columns", ColumnTransformer([
            ("num", numeric, make_column_selector(dtype_include=np.number)),
            ("cat", categorical, make_column_selector(dtype_exclude=np.number))]))])

lr = Pipeline([("preprocess", preprocess()), ("model", LogisticRegression(
    C=.01, penalty="l1", solver="liblinear", class_weight="balanced",
    max_iter=5000, random_state=42))])
rf = Pipeline([("preprocess", preprocess()), ("model", RandomForestClassifier(
    n_estimators=500, min_samples_split=10, min_samples_leaf=2,
    max_features="log2", max_depth=None, class_weight="balanced",
    random_state=42, n_jobs=1))])
lr.fit(xtrain, ytrain)
rf.fit(xtrain, ytrain)

np.random.seed(42)
tf.keras.utils.set_random_seed(42)
tf.config.experimental.enable_op_determinism()
nn_pre = preprocess()
xxtrain = nn_pre.fit_transform(xtrain).astype("float32")
xxval = nn_pre.transform(xval).astype("float32")
xxtest = nn_pre.transform(xtest).astype("float32")
inputs = keras.Input(shape=(xxtrain.shape[1],), name="clinical_features")
z = layers.Dense(32, activation="relu", kernel_regularizer=regularizers.l2(.001))(inputs)
z = layers.Dropout(.5)(z)
z = layers.Dense(16, activation="relu", kernel_regularizer=regularizers.l2(.001))(z)
z = layers.Dropout(.3)(z)
outputs = [layers.Dense(1, activation="sigmoid", name=name)(z)
           for name in ["death_28", "death_3m", "death_6m"]]
nn = keras.Model(inputs=inputs, outputs=outputs)
nn.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4),
           loss=["binary_crossentropy"] * 3, loss_weights=[.2, 1., .2])
targets = [df["death.within.28.days"], df[target], df["death.within.6.months"]]
train_targets = [t.loc[train_ix].to_numpy(dtype="float32").reshape(-1, 1) for t in targets]
val_targets = [t.loc[val_ix].to_numpy(dtype="float32").reshape(-1, 1) for t in targets]
weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=ytrain.to_numpy())
main_weight = np.clip(np.where(ytrain.to_numpy() == 1, weights[1], weights[0]), .5, 8.).astype("float32")
sample_weights = [np.ones(len(ytrain), dtype="float32"), main_weight,
                  np.ones(len(ytrain), dtype="float32")]
history = nn.fit(xxtrain, train_targets, sample_weight=sample_weights,
                 validation_data=(xxval, val_targets), epochs=80, batch_size=32,
                 callbacks=[keras.callbacks.EarlyStopping(
                     monitor="val_loss", patience=8, restore_best_weights=True)], verbose=0)

val_scores = {"LR": lr.predict_proba(xval)[:, 1],
              "RF": rf.predict_proba(xval)[:, 1],
              "NN": nn.predict(xxval, verbose=0)[1].ravel()}
test_scores = {"LR": lr.predict_proba(xtest)[:, 1],
               "RF": rf.predict_proba(xtest)[:, 1],
               "NN": nn.predict(xxtest, verbose=0)[1].ravel()}
grid = np.linspace(.01, .99, 99)
thresholds = {name: float(max(grid, key=lambda th: f1_score(yval, score >= th, zero_division=0)))
              for name, score in val_scores.items()}
pred = pd.DataFrame({"row_key": [f"test_{i:04d}" for i in range(len(ytest))],
                     "y_true": ytest.to_numpy(), **test_scores})
screen, overlap, _, rank = compare_predictions(pred, thresholds, PRIVATE, seed=42)
for name, table in [("timing_sensitivity_screening", screen),
                    ("timing_sensitivity_overlap", overlap),
                    ("timing_sensitivity_ranking", rank)]:
    table.to_csv(PUBLIC / f"{name}.csv", index=False)
(PUBLIC / "timing_sensitivity_tradeoffs.png").write_bytes((PRIVATE / "screening_tradeoffs.png").read_bytes())
(PUBLIC / "timing_sensitivity_settings.json").write_text(json.dumps({
    "purpose": "Exploratory conservative admission/day-one feature sensitivity; no claim of prospective validation.",
    "source_for_feature_timing": "https://www.nature.com/articles/s41597-021-00835-9",
    "input_data_sha256": manifest["data_sha256"],
    "same_split_as_main_run": True,
    "feature_count_before_fitted_filter": len(features),
    "features": features,
    "selection": "Admission demographics, admission-day clinical characteristics, comorbidities from admission notes, day-one laboratory block as documented by source; exclude echo and uncertain-time interventions/diagnoses.",
    "baseline_hyperparameters": "Reused train-only selected settings from main comparison; not retuned on reduced features.",
    "nn_epochs_run": len(history.history["loss"]),
    "thresholds_chosen_on_validation": thresholds,
    "random_seed": 42,
    "patient_level_outputs_public": False,
}, indent=2) + "\n")
print("Admission/day-one sensitivity aggregate results:")
print(screen[["policy", "model", "selected", "tp", "fp", "fn", "recall"]].to_string(index=False))
print(rank.to_string(index=False))
