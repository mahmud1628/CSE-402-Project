"""Run the Stage 3 study.  --quick is a smoke check; full mode retains 30 trials.

Progress lines are printed while each step runs, and the latest status is kept
in ``<output>/progress.json``.  Re-running the same command resumes.
"""
from __future__ import annotations
import subprocess, sys, time
from pathlib import Path
HERE=Path(__file__).resolve().parent
SCRIPTS=("e1_power_convergence.py","e2_error_vs_samples.py","e3_variance_validation.py","e4_parameters.py","e5_runtime_and_forests.py")


def _clock(seconds):
    seconds=int(round(seconds)); h,rem=divmod(seconds,3600); m,s=divmod(rem,60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"


if __name__ == "__main__":
    start=time.monotonic()
    for i,script in enumerate(SCRIPTS,1):
        step=time.monotonic()
        print(f"\n=== Stage 3 step {i}/{len(SCRIPTS)}: {script} (started {time.strftime('%H:%M:%S')}) ===",flush=True)
        subprocess.run([sys.executable,str(HERE/script),*sys.argv[1:]],check=True)
        print(f"=== step {i}/{len(SCRIPTS)} done in {_clock(time.monotonic()-step)} · total {_clock(time.monotonic()-start)} ===",flush=True)
    print(f"\nStage 3 finished in {_clock(time.monotonic()-start)}.",flush=True)
