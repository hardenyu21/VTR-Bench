#!/usr/bin/env python3
"""Run all CPU-only checks; no vLLM, GPU or videos required."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
commands = [
    [sys.executable, 'evaluate.py', '--validate-only'],
    [sys.executable, 'engine/test_evaluate_checklist_retry_v2.py'],
    [sys.executable, 'engine/test_evaluate_vlm_wer_retry.py'],
    [sys.executable, '-m', 'unittest', 'discover', '-s', 'engine', '-p', 'test_vtextbench*.py', '-v'],
    [sys.executable, '-m', 'unittest', 'test_portable', '-v'],
]
for command in commands:
    subprocess.run(command, cwd=ROOT, check=True)
print('All offline checks passed. This is not a GPU inference test.')
