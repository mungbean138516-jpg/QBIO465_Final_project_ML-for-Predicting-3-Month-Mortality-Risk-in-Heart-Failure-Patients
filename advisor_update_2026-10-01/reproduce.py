"""Run this aggregate-only bundle's workflow with a local authorized dat.csv."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent.parent
if not (ROOT / "dat.csv").is_file():
    raise SystemExit("Provide an authorized local dat.csv at the repository root; it is Git-ignored.")
env = os.environ.copy()
env.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "hf-mpl"))
for args in [
    ["run_comparison.py"],
    ["audit_source_data.py", "dat.csv"],
    ["summarize_advisor_update.py"],
    ["verify_saved_comparison.py", "dat.csv"],
    ["timing_sensitivity.py"],
    ["compare_feature_sets.py"],
]:
    subprocess.run([sys.executable, *args], cwd=ROOT, env=env, check=True)
