#!/usr/bin/env python3
"""
Single-source XGuard fine-tuning script with a 2M-token budget.

Workflow:
  1. Ensure the 4096-token XGuard sliding-window cache exists
  2. Build a 2M-token training set from XGuard only
  3. Run LoRA fine-tuning

Dataset: dataset/train/xguard-train.parquet
Model: defaults to /model/llm/Meta-Llama-3.1-8B-Instruct

Usage:
    uv run python scripts/finetune_xguard_only.py
    uv run python scripts/finetune_xguard_only.py --model-path /path/to/model --output-dir ./output
"""

import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Reuse the step helpers from run_loop.
from scripts.data_constructor import construct_training_set, ensure_slided_for_sources
from scripts.run_loop import (
    DEFAULT_MODEL_PATH,
    _format_duration,
    _get_gpu_count,
    step_finetune,
)

T_MAX = 2_000_000
MAX_LENGTH = 4096
NUM_EPOCHS = 2
BATCH_SIZE = 4
GRAD_ACCUM = 1
QUALITY_THRESHOLD = 0.0
SEED = 42


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Full-XGuard 2M-token fine-tuning run")
    parser.add_argument(
        "--model-path",
        type=str,
        default=os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH),
        help="Path to the base model",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory; defaults to results/finetune_xguard_<timestamp>",
    )
    parser.add_argument(
        "--mock", action="store_true", help="Skip the actual training step"
    )
    parser.add_argument("--num-epochs", type=int, default=NUM_EPOCHS)
    args = parser.parse_args()

    model_path = args.model_path
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else (
            PROJECT_ROOT
            / "results"
            / f"finetune_xguard_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    iter_dir = output_dir / "iter_00"
    iter_dir.mkdir(parents=True, exist_ok=True)

    w = {"xguard": 1.0, "orbench": 0.0, "ifeval": 0.0}
    dims = {}

    print("=" * 60)
    print("Full-XGuard 2M-token fine-tuning")
    print("=" * 60)
    print(f"  Model: {model_path}")
    print(f"  T_max: {T_MAX:,} tokens")
    print(f"  Max length: {MAX_LENGTH}")
    print(f"  Epochs: {args.num_epochs}")
    print(f"  Output: {output_dir}")
    print("=" * 60)

    # Step 0: ensure the sliding-window cache exists
    print("\n[Step 0] Ensuring the XGuard 4096-token sliding-window cache...")
    ensure_slided_for_sources(model_path, MAX_LENGTH, sources=["xguard"])

    # Step 1: build the training set
    print("\n[Step 1] Building training data (100% XGuard, 2M tokens)...")
    data_path = construct_training_set(
        w=w,
        T_max=T_MAX,
        max_length=MAX_LENGTH,
        quality_threshold=QUALITY_THRESHOLD,
        dims=dims,
        seed=SEED,
        output_path=iter_dir / "train_data.parquet",
        model_path=model_path,
    )

    # Step 2: fine-tuning
    print("\n[Step 2] Running LoRA fine-tuning...")
    t0 = time.time()
    lora_path = step_finetune(
        data_path=data_path,
        iteration=0,
        iter_dir=iter_dir,
        model_path=model_path,
        max_length=MAX_LENGTH,
        mock=args.mock,
        num_epochs=args.num_epochs,
        batch_size=BATCH_SIZE,
        grad_accum=GRAD_ACCUM,
    )
    elapsed = time.time() - t0

    print("\n" + "=" * 60)
    print("Done")
    print("=" * 60)
    print(f"  LoRA output: {lora_path}")
    print(f"  Elapsed: {_format_duration(elapsed)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
