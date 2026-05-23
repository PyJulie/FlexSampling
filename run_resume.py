"""Wrapper to run FlexSampling — discards stdout to avoid pipe blocking.

Usage:
    python run_resume.py            # uses FLEX_SEED env var (default 42)
    FLEX_SEED=123 python run_resume.py
"""
import subprocess, sys, os
os.chdir(r"F:\research\projects\FlexSampling")
proc = subprocess.run(
    [sys.executable, "-u", "examples/run_flexsampling.py"],
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    env=os.environ.copy(),
)
sys.exit(proc.returncode)
