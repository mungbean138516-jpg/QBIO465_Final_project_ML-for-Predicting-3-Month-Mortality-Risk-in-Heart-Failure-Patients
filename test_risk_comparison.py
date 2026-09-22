"""Small synthetic checks; these are not heart-failure study results."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from risk_comparison import compare_predictions, TrainingFeatureFilter

class ComparisonTests(unittest.TestCase):
    def test_matched_budgets_disagreement_and_ties(self):
        p=pd.DataFrame({'row_key':['a','b','c','d'], 'y_true':[1,0,1,0],
                        'LR':[.9,.8,.2,.1], 'RF':[.9,.2,.8,.1], 'NN':[.5,.5,.5,.5]})
        with tempfile.TemporaryDirectory() as d:
            s,o,patients,_=compare_predictions(p,dict(LR=.5,RF=.5,NN=.5),d,fractions=(.5,))
            top=s[s.policy=='top_50_percent'].set_index('model')
            self.assertEqual(top.selected.tolist(),[2,2,2])
            self.assertEqual(top.loc['LR','tp'],1)
            self.assertEqual(top.loc['RF','tp'],2)
            self.assertEqual(top.loc['NN','tie_recall_min'],0)
            self.assertEqual(top.loc['NN','tie_recall_max'],1)
            pair=o[(o.policy=='top_50_percent') & (o.model_a=='LR') & (o.model_b=='RF')].iloc[0]
            self.assertEqual(pair.overlap,1)
            self.assertEqual(pair.deaths_only_b,1)
            self.assertEqual(pair.deaths_neither,0)
            self.assertTrue((s.tp+s.fn==2).all())
            self.assertTrue((s.tp+s.fp+s.fn+s.tn==4).all())
            _,_,shuffled,_=compare_predictions(p.sample(frac=1,random_state=8),dict(LR=.5,RF=.5,NN=.5),d,fractions=(.5,))
            pd.testing.assert_frame_equal(patients,shuffled)
            self.assertTrue((Path(d)/'screening_tradeoffs.png').exists())

    def test_invalid_alignment_and_nonfinite_score(self):
        p=pd.DataFrame({'row_key':['a','a'], 'y_true':[0,1], 'LR':[.1,.9]})
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError): compare_predictions(p,{'LR':.5},d)
            p['row_key']=['a','b'];p.loc[1,'LR']=np.inf
            with self.assertRaises(ValueError): compare_predictions(p,{'LR':.5},d)

    def test_filter_uses_training_only(self):
        train=pd.DataFrame({'vary':[1,2,3], 'constant':[1,1,1], 'missing':[np.nan]*3})
        f=TrainingFeatureFilter().fit(train)
        test=pd.DataFrame({'vary':[4], 'constant':[2], 'missing':[3]})
        self.assertEqual(f.transform(test).columns.tolist(),['vary'])

if __name__=='__main__': unittest.main()
