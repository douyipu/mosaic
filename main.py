#!/usr/bin/env python3
"""
Main entry point for the MOSAIC closed-loop search.

This is a thin wrapper around `scripts/run_loop.py`.
All implementation lives in the local packages under `scripts/`, `score/`,
and `packages/finetune/`.

Usage:
    python main.py --max-iters 5 --T-max 1000000
    python main.py --mock  # Local test mode
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Delegate to run_loop
from scripts.run_loop import main as run_loop_main

if __name__ == "__main__":
    run_loop_main()
