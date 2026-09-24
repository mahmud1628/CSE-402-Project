"""Run every stage-1 experiment in sequence.

    python run_all.py            # full run (all datasets incl. com-youtube; about 1-2 h)
    python run_all.py --quick    # small datasets, fewer trials (about 10 min)
    python run_all.py --no-large # everything except com-youtube

Extra arguments are passed on to every experiment. Output goes to results/stage1/.
"""

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPERIMENTS = [
    "e1_power_convergence.py",
    "e2_variance_validation.py",
    "e3_error_vs_samples.py",
    "e4_error_vs_time.py",
    "e5_parameters.py",
    "e6_walk_vs_forest.py",
    "e7_relative_error.py",
]

if __name__ == "__main__":
    extra = sys.argv[1:]
    for exp in EXPERIMENTS:
        t0 = time.time()
        print(f"\n===== {exp} {' '.join(extra)} =====", flush=True)
        subprocess.run([sys.executable, str(HERE / exp), *extra], check=True, cwd=HERE)
        print(f"===== {exp} finished in {time.time() - t0:.0f}s =====", flush=True)
