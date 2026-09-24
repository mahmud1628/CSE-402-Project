"""Run Stage 2 locally: build networks, solve PPR, draw figures and smoke-test MC."""

from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
EXPERIMENTS = ["e1_build_networks.py", "e2_rankings.py", "e3_figures.py", "e4_mc_smoke.py"]


if __name__ == "__main__":
    for script in EXPERIMENTS:
        print(f"Running {script}", flush=True)
        subprocess.run([sys.executable, str(HERE / script), *sys.argv[1:]], check=True)
