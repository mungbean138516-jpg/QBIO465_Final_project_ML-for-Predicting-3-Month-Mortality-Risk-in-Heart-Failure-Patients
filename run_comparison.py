"""Execute notebook code in order without a Jupyter network kernel.

Checkpoints baseline state locally; --resume-baselines resumes after cell 29.
"""
import os
for key,value in {'MPLBACKEND':'Agg','TF_NUM_INTEROP_THREADS':'1','TF_NUM_INTRAOP_THREADS':'1','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','LOKY_MAX_CPU_COUNT':'4','TF_CPP_MIN_LOG_LEVEL':'2'}.items():
    os.environ.setdefault(key,value)
from pathlib import Path
import json,time,sys,contextlib,io,traceback
import joblib
from IPython.display import display
root=Path(__file__).resolve().parent
os.chdir(root)
out=root/'comparison_results';out.mkdir(exist_ok=True)
nb=json.loads((root/'Heart_Failure_Model_Disagreement.ipynb').read_text())
ns={'__name__':'__main__','display':display}
resume='--resume-baselines' in sys.argv
for i,cell in enumerate(nb['cells']):
    if cell['cell_type']!='code':continue
    if resume and i<30 and i not in [2,19,20,22]:continue
    print(time.strftime('%H:%M:%S'),f'cell {i}:', ''.join(cell['source']).splitlines()[0][:100],flush=True)
    buffer=io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer),contextlib.redirect_stderr(buffer):
            exec(compile(''.join(cell['source']),f'notebook_cell_{i}','exec'),ns)
        cell['execution_count']=i+1
        cell['outputs']=[{'output_type':'stream','name':'stdout','text':buffer.getvalue().splitlines(keepends=True)}]
        print(buffer.getvalue()[-1500:],flush=True)
        if resume and i==22:ns.update(joblib.load(out/'baseline_checkpoint.joblib'))
        if i==29:
            keys=['DATA_PATH','RANDOM_STATE','preprocessor','logreg_pipe','rf_pipe','cv_summary','X_train2_raw','X_val_raw','X_test_raw','y_train2','y_val','y_test','y28_train2','y28_val','y28_test','y6_train2','y6_val','y6_test','logreg_val_prob','rf_val_prob','logreg_test_prob','rf_test_prob','logreg_best_th','rf_best_th','baseline_test_summary']
            joblib.dump({k:ns[k] for k in keys},out/'baseline_checkpoint.joblib')
        if 'plt' in ns:ns['plt'].close('all')
    except Exception:
        print(buffer.getvalue()[-1500:],flush=True)
        traceback.print_exc()
        raise
    finally:
        (out/'executed_comparison.ipynb').write_text(json.dumps(nb,indent=1))
print('COMPLETE',flush=True)
