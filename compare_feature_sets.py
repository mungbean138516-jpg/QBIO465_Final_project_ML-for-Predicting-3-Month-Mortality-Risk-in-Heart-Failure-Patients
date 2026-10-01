"""Aggregate how restricted feature timing changes equal-capacity test flags."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
OLD = pd.read_csv(ROOT / "comparison_results/patient_comparison.csv")
NEW = pd.read_csv(ROOT / "comparison_results/timing_sensitivity/patient_comparison.csv")
OUT = ROOT / "advisor_update_2026-10-01"
assert OLD[["policy", "row_key"]].equals(NEW[["policy", "row_key"]])
assert OLD.y_true.equals(NEW.y_true)
rows = []
for policy in ["top_5_percent", "top_10_percent", "top_20_percent"]:
    old = OLD[OLD.policy == policy]
    new = NEW[NEW.policy == policy]
    for model in ["LR", "RF", "NN"]:
        a = old[f"{model}_selected"].to_numpy(dtype=bool)
        b = new[f"{model}_selected"].to_numpy(dtype=bool)
        y = old.y_true.to_numpy(dtype=bool)
        assert int(a.sum()) == int(b.sum())
        rows.append({"policy": policy, "model": model,
                     "selected_each": int(a.sum()),
                     "overlap": int((a & b).sum()),
                     "changed_flag_decisions": int((a != b).sum()),
                     "jaccard": float((a & b).sum() / (a | b).sum()),
                     "deaths_both": int((y & a & b).sum()),
                     "deaths_full_only": int((y & a & ~b).sum()),
                     "deaths_restricted_only": int((y & ~a & b).sum()),
                     "deaths_neither": int((y & ~a & ~b).sum())})
result = pd.DataFrame(rows)
result.to_csv(OUT / "feature_set_overlap.csv", index=False)
fig, ax = plt.subplots(figsize=(7.2, 4.2))
colors = {"LR": "#34679a", "RF": "#d1812e", "NN": "#558c6a"}
for model, part in result.groupby("model", sort=False):
    ax.plot(part.selected_each, part.changed_flag_decisions, "o-", label=model, color=colors[model])
ax.set_xticks([21, 41, 81])
ax.set_xlabel("Flagged by each version of a model (equal capacity)")
ax.set_ylabel("Test records with changed flag decision (of 402)")
ax.grid(axis="y", alpha=.2)
ax.legend(frameon=False)
ax.set_title("Changing eligible feature timing changes who is flagged")
fig.tight_layout()
fig.savefig(OUT / "feature_set_disagreement.png", dpi=180)
plt.close(fig)
print(result.to_string(index=False))
