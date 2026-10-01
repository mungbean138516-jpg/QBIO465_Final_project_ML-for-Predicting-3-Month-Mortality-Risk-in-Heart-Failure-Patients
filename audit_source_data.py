"""Write aggregate-only quality checks for the locally held Zigong CSV.

This does not establish a patient-level longitudinal audit or verify a
particular PhysioNet release; the CSV does not contain those metadata.
"""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def audit(path):
    df = pd.read_csv(path)
    d28 = df["death.within.28.days"]
    d3 = df["death.within.3.months"]
    d6 = df["death.within.6.months"]
    death_day = df["time.of.death..days.from.admission."]
    recorded_dead = df["outcome.during.hospitalization"] == "Dead"
    destination_died = df["Way.of.leaving.hospital"] == "Died"
    labels = [d28, d3, d6]
    if any(not x.dropna().isin([0, 1]).all() for x in labels):
        raise ValueError("Mortality labels must be binary")
    excluded = ["No.", "Unnamed: 0"]
    return {
        "input_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        "rows": len(df),
        "raw_columns": len(df.columns),
        "three_month_deaths": int(d3.sum()),
        "death_counts": {"28_days": int(d28.sum()), "3_months": int(d3.sum()), "6_months": int(d6.sum())},
        "missing_mortality_labels": {"28_days": int(d28.isna().sum()), "3_months": int(d3.isna().sum()), "6_months": int(d6.isna().sum())},
        "duplicate_no": int(df["No."].duplicated().sum()),
        "duplicate_rows_without_ids": int(df.drop(columns=excluded).duplicated().sum()),
        "nonmonotone_mortality_labels": {"28_to_3_months": int((d28 > d3).sum()), "3_to_6_months": int((d3 > d6).sum())},
        "death_day_available": int(death_day.notna().sum()),
        "three_month_positive_without_death_day": int(((d3 == 1) & death_day.isna()).sum()),
        "death_day_nonmissing_three_month_negative": int(((d3 == 0) & death_day.notna()).sum()),
        "three_month_negative_death_day_at_most_90": int(((d3 == 0) & (death_day <= 90)).sum()),
        "three_month_negative_death_day_after_90": int(((d3 == 0) & (death_day > 90)).sum()),
        "six_month_positive_without_death_day": int(((d6 == 1) & death_day.isna()).sum()),
        "six_month_negative_death_day_at_most_180": int(((d6 == 0) & (death_day <= 180)).sum()),
        "outcome_during_hospitalization_dead": int(recorded_dead.sum()),
        "discharge_destination_died": int(destination_died.sum()),
        "dead_field_disagreement": int((recorded_dead != destination_died).sum()),
        "outcome_dead_not_three_month_positive": int((recorded_dead & (d3 == 0)).sum()),
        "destination_died_not_three_month_positive": int((destination_died & (d3 == 0)).sum()),
        "columns_more_than_80pct_missing": int((df.isna().mean() > .8).sum()),
        "columns_entirely_missing": int(df.isna().all().sum()),
        "columns_constant_nonmissing": int((df.nunique(dropna=True) <= 1).sum()),
        "limits": [
            "A unique No. is not proof of unique patients across hospitalizations.",
            "Nonmissing binary labels do not prove complete follow-up or exclude unobserved deaths.",
            "No per-feature timestamps are present, so admission-time availability is not verified for every retained feature.",
            "A raw-file hash alone does not identify the exact PhysioNet release or confirm license conditions.",
            "Two discharge-related death fields disagree; the mortality label cannot be adjudicated from this CSV alone."
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path, help="Local, authorized dat.csv")
    parser.add_argument("--out", type=Path, default=Path("advisor_update_2026-10-01/data_audit.json"))
    args = parser.parse_args()
    result = audit(args.csv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote aggregate audit: {args.out}")
