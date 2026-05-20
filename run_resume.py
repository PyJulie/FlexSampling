"""Wrapper to run FlexSampling - discards stdout to avoid pipe blocking."""
import subprocess, sys, os
os.chdir(r"F:\research\projects\FlexSampling")
# Redirect subprocess stdout/stderr to DEVNULL to prevent pipe blocking
# All output goes to the log file inside the script
proc = subprocess.run(
    [sys.executable, "-u", "examples/run_flexsampling.py"],
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
)
sys.exit(proc.returncode)
