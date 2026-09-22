"""Compare aligned, held-out model predictions. No clinical action is implied."""
from itertools import combinations
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def compare_predictions(predictions, thresholds, output_dir, fractions=(0.05, 0.10, 0.20), seed=42):
    """Input: row_key, y_true, then one score column per model.

    row_key is a unique analysis identifier, not a clinical patient identifier.
    Thresholds must have been selected without using these held-out outcomes.
    Exact top-k ties use a common seeded, outcome-independent row-key ordering.
    Boundary-tie recall bounds describe all possible selections within that tie.
    """
    p = predictions.copy()
    models = list(thresholds)
    if not models or len(models) != len(set(models)):
        raise ValueError('Provide at least one uniquely named model.')
    required = ['row_key', 'y_true', *models]
    if not set(required).issubset(p.columns) or p.empty:
        raise ValueError('Missing required columns or empty predictions.')
    if p[required].isna().any().any() or p.row_key.duplicated().any():
        raise ValueError('Missing values or duplicate row keys.')
    p['row_key'] = p.row_key.astype(str)
    if p.row_key.duplicated().any():
        raise ValueError('Row keys must also be unique as strings.')
    if not p.y_true.isin([0, 1]).all() or p.y_true.nunique() != 2:
        raise ValueError('Held-out outcomes must contain both binary classes.')
    for m in models:
        if not np.isfinite(p[m]).all() or not p[m].between(0, 1).all():
            raise ValueError(f'Invalid probability scores: {m}')
        if not np.isfinite(thresholds[m]) or not 0 <= thresholds[m] <= 1:
            raise ValueError(f'Invalid validation threshold: {m}')
    if any(not 0 < f < 1 for f in fractions):
        raise ValueError('Fractions must be between 0 and 1.')
    # Sorting before the seeded permutation makes tie breaking invariant to CSV row order.
    p = p.sort_values('row_key').reset_index(drop=True)
    tie_order = np.random.default_rng(seed).permutation(len(p))
    y = p.y_true.to_numpy(dtype=int)
    positive = y == 1
    n_deaths = int(positive.sum())
    rows, pair_rows, patient_rows = [], [], []
    policies = [('validation_f1', None)] + [(f'top_{100*f:g}_percent', int(np.ceil(len(p)*f))) for f in fractions]
    for policy, k in policies:
        decisions = {}
        details = p[required].copy()
        details['policy'] = policy
        for model in models:
            score = p[model].to_numpy()
            if k is None:
                selected = score >= thresholds[model]
                cutoff = float(thresholds[model])
                tie_n, tie_slots, low_tp, high_tp = 0, 0, None, None
            else:
                order = np.lexsort((tie_order, -score))
                selected = np.zeros(len(p), dtype=bool)
                selected[order[:k]] = True
                cutoff = float(score[order[k-1]])
                above, tied = score > cutoff, score == cutoff
                tie_n = int(tied.sum())
                tie_slots = k - int(above.sum())
                tie_deaths = int((tied & positive).sum())
                above_deaths = int((above & positive).sum())
                low_tp = above_deaths + max(0, tie_slots - (tie_n-tie_deaths))
                high_tp = above_deaths + min(tie_slots, tie_deaths)
            decisions[model] = selected
            tp = int((selected & positive).sum())
            fp = int((selected & ~positive).sum())
            fn, tn = n_deaths-tp, int((~selected & ~positive).sum())
            rows.append(dict(policy=policy, model=model, n=len(p), deaths=n_deaths,
                selected=int(selected.sum()), tp=tp, fp=fp, fn=fn, tn=tn,
                precision=tp/(tp+fp) if tp+fp else 0., recall=tp/n_deaths,
                f1=2*tp/(2*tp+fp+fn), cutoff=cutoff,
                boundary_tie_size=tie_n, boundary_tie_selected=tie_slots,
                tie_recall_min=low_tp/n_deaths if low_tp is not None else np.nan,
                tie_recall_max=high_tp/n_deaths if high_tp is not None else np.nan))
            details[f'{model}_selected'] = selected
            details[f'{model}_missed_death'] = positive & ~selected
        for a,b in combinations(models, 2):
            sa,sb = decisions[a],decisions[b]
            both, union = sa & sb, sa | sb
            pair_rows.append(dict(policy=policy,model_a=a,model_b=b,
                overlap=int(both.sum()),union=int(union.sum()),
                jaccard=float(both.sum()/union.sum()) if union.any() else np.nan,
                disagreement=int((sa != sb).sum()),
                deaths_both=int((positive & both).sum()),
                deaths_only_a=int((positive & sa & ~sb).sum()),
                deaths_only_b=int((positive & sb & ~sa).sum()),
                deaths_neither=int((positive & ~union).sum())))
        details['models_selecting'] = np.stack(list(decisions.values())).sum(axis=0)
        details['model_disagreement'] = details.models_selecting.between(1,len(models)-1)
        patient_rows.append(details)
    summary = pd.DataFrame(rows)
    overlap = pd.DataFrame(pair_rows)
    patients = pd.concat(patient_rows,ignore_index=True)
    ranking = pd.DataFrame([dict(model=m, average_precision=average_precision_score(y,p[m]),
        roc_auc=roc_auc_score(y,p[m])) for m in models])
    out = Path(output_dir)
    out.mkdir(parents=True,exist_ok=True)
    for name,table in [('screening_summary',summary),('model_overlap',overlap),
        ('patient_comparison',patients),('death_case_review',patients[patients.y_true == 1]),
        ('ranking_summary',ranking)]:
        table.to_csv(out / f'{name}.csv',index=False)
    (out/'comparison_settings.json').write_text(json.dumps(dict(seed=seed,
        fractions=list(fractions),thresholds={m:float(v) for m,v in thresholds.items()},
        interpretation='Retrospective model flags; not observed treatment or clinical benefit.',
        ties='Common seeded row-key ordering; tie recall bounds are not confidence intervals.'),indent=2))
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(1,2,figsize=(11,4))
    top = summary[summary.policy != 'validation_f1']
    for m in models:
        sub = top[top.model == m]
        axes[0].plot(sub.selected,sub.recall,marker='o',label=m)
        axes[1].plot(sub.selected,sub.fp,marker='o',label=m)
    for ax in axes:
        ax.set_xlabel('Number flagged (same budget per model)')
        ax.legend()
    axes[0].set_ylabel('Recall of observed deaths')
    axes[0].set_ylim(0,1.05)
    axes[1].set_ylabel('False positives')
    fig.tight_layout()
    fig.savefig(out/'screening_tradeoffs.png',dpi=160)
    plt.close(fig)
    return summary,overlap,patients,ranking

# This estimator is importable so the fitted pipelines can be loaded with joblib.
from sklearn.base import BaseEstimator, TransformerMixin

class TrainingFeatureFilter(BaseEstimator, TransformerMixin):
    """Learn missingness/constant filters only from each fitting partition."""
    def __init__(self, missing_cutoff=0.8):
        self.missing_cutoff = missing_cutoff

    def fit(self, X, y=None):
        self.columns_ = X.columns[(X.isna().mean() <= self.missing_cutoff)
                                  & (X.nunique(dropna=True) > 1)].tolist()
        if not self.columns_:
            raise ValueError('No usable features in training partition.')
        return self

    def transform(self, X):
        return X.loc[:, self.columns_].copy()
