"""Check private saved estimators against private test scores, without exporting rows."""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from tensorflow import keras

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "comparison_results"
EXCLUDE = [
    "Unnamed: 0", "No.", "Way.of.leaving.hospital", "discharge.department",
    "outcome.during.hospitalization", "death.within.28.days",
    "death.within.3.months", "death.within.6.months",
    "re.admission.within.28.days", "re.admission.within.3.months",
    "re.admission.within.6.months", "time.of.death..days.from.admission.",
    "re.admission.time..days.from.admission.",
    "return.to.emergency.department.within.6.months",
    "time.to.emergency.department.within.6.months", "dischargeDay",
]


def verify(data_path):
    manifest = json.loads((OUT / "run_manifest.json").read_text())
    assert hashlib.sha256(data_path.read_bytes()).hexdigest() == manifest["data_sha256"]
    df = pd.read_csv(data_path)
    x = df.drop(columns=[c for c in EXCLUDE if c in df]).loc[manifest["test_rows"]]
    y = df.loc[manifest["test_rows"], "death.within.3.months"].to_numpy()
    pred = pd.read_csv(OUT / "test_predictions.csv")
    np.testing.assert_array_equal(y, pred.y_true.to_numpy())
    models = {
        "LR": joblib.load(OUT / "logistic_regression_pipeline.joblib"),
        "RF": joblib.load(OUT / "random_forest_pipeline.joblib"),
    }
    max_diffs = {}
    for name, model in models.items():
        actual = model.predict_proba(x)[:, 1]
        max_diffs[name] = float(np.max(np.abs(actual - pred[name].to_numpy())))
        np.testing.assert_allclose(actual, pred[name].to_numpy(), rtol=1e-6, atol=1e-6)
    pre = joblib.load(OUT / "nn_preprocessor.joblib")
    model = keras.models.load_model(OUT / "multitask_heart_failure_model.keras")
    actual = model.predict(pre.transform(x).astype("float32"), verbose=0)[1].ravel()
    max_diffs["NN"] = float(np.max(np.abs(actual - pred.NN.to_numpy())))
    np.testing.assert_allclose(actual, pred.NN.to_numpy(), rtol=1e-6, atol=1e-6)
    return max_diffs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    args = parser.parse_args()
    differences = verify(args.csv)
    target = ROOT / "advisor_update_2026-10-01/model_roundtrip_validation.json"
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps({
        "validated_rows": 402,
        "all_models_match_exported_scores_with_atol": 1e-6,
        "maximum_absolute_score_difference": differences,
    }, indent=2) + "\n")
    print("Maximum absolute score differences after reload:", differences)
