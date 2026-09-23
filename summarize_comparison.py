"""Validate real run outputs and prepare aggregate figures for review."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

root=Path(__file__).resolve().parent
out=root/'comparison_results'
s=pd.read_csv(out/'screening_summary.csv')
p=pd.read_csv(out/'patient_comparison.csv')
y=pd.read_csv(out/'test_predictions.csv')
assert len(y)==402 and y.y_true.sum()==8
for r in s.itertuples():
 rows=p[p.policy==r.policy]
 cm=confusion_matrix(rows.y_true,rows[f'{r.model}_selected'],labels=[0,1]).ravel()
 np.testing.assert_array_equal(cm,[r.tn,r.fp,r.fn,r.tp])
assert s.groupby('policy').apply(lambda g:g.selected.nunique(),include_groups=False).drop('validation_f1').eq(1).all()
# Publish aggregate plots; patient-level tables remain local/private.
pub=root/'comparison_summary'
pub.mkdir(exist_ok=True)
for name in ['screening_summary.csv','model_overlap.csv','ranking_summary.csv','final_model_performance_summary.csv','cv_model_performance_summary.csv','environment_versions.json','comparison_settings.json']:
 (pub/name).write_bytes((out/name).read_bytes())
(pub/'screening_tradeoffs.png').write_bytes((out/'screening_tradeoffs.png').read_bytes())
# A compact plot of detected versus missed deaths at each fixed budget.
policies=['top_5_percent','top_10_percent','top_20_percent']
fig,axes=plt.subplots(1,3,figsize=(10,3.6),sharey=True)
for ax,policy in zip(axes,policies):
 sub=s[s.policy==policy]
 ax.bar(sub.model,sub.tp,color='#387f83',label='Detected deaths')
 ax.bar(sub.model,sub.fn,bottom=sub.tp,color='#ddd8d2',label='Missed deaths')
 for i,r in enumerate(sub.itertuples()):
  ax.text(i,r.tp/2,str(r.tp),ha='center',va='center',color='white',fontweight='bold')
  if r.fn:ax.text(i,r.tp+r.fn/2,str(r.fn),ha='center',va='center',color='#444')
 ax.set_title(f'Flag {int(sub.selected.iloc[0])} of 402 rows')
 ax.set_ylim(0,8.7)
 ax.spines[['top','right']].set_visible(False)
axes[0].set_ylabel('Observed deaths in test set (n=8)')
axes[-1].legend(loc='upper center',bbox_to_anchor=(.5,-.12),frameon=False)
fig.suptitle('Same screening budget, different death-case detection')
fig.tight_layout()
fig.savefig(pub/'detected_and_missed.png',dpi=180,bbox_inches='tight')
plt.close(fig)
print('All exported decision counts verified against patient-level rows.')
print(s[['policy','model','selected','tp','fn','fp','precision','recall']].to_string(index=False))
print(pd.read_csv(out/'ranking_summary.csv').to_string(index=False))
print(pd.read_csv(out/'model_overlap.csv').to_string(index=False))
