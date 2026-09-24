"""Run the Stage 3 study.  --quick is a smoke check; full mode retains 30 trials."""
from __future__ import annotations
import subprocess, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
SCRIPTS=("e1_power_convergence.py","e2_error_vs_samples.py","e3_variance_validation.py","e4_parameters.py","e5_runtime_and_forests.py")
if __name__ == "__main__":
    for script in SCRIPTS:
        print(f"Running {script}",flush=True)
        subprocess.run([sys.executable,str(HERE/script),*sys.argv[1:]],check=True)
