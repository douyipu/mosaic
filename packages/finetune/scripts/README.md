# finetune scripts

LoRA fine-tuning utilities for the MOSAIC experiments.

## Supported workflow

- XGuard multi-turn jailbreak-defense data
- OrBench benign boundary queries
- Optional turn-aware sliding windows for long XGuard conversations
- Token-budgeted sampling
- Window-level reweighting so a long attack chain does not dominate training after slicing

## Directory convention

```text
packages/finetune/
├── scripts/
│   ├── launch.sh
│   ├── train.py
│   ├── analyze_lengths.py
│   └── preprocess_slide_by_turn.py
├── data/
├── cache/
└── outputs/
```

## Script overview

### `launch.sh`

Entry point for LoRA training. It reads environment variables, prints the run
configuration, and calls `train.py`.

Common variables:

- `CUDA_VISIBLE_DEVICES`
- `MODEL_PATH`
- `XGUARD_PARQUET`, `ORBENCH_PARQUET`
- `MAX_LENGTH`
- `TARGET_TOTAL_TOKENS`
- `BATCH_SIZE`, `GRAD_ACCUM`, `NUM_EPOCHS`
- `ENABLE_WINDOW_REWEIGHT`

### `train.py`

Core training script built on TRL SFTTrainer and PEFT LoRA.

Key features:

- Cached tokenization in `cache/`
- Approximate token-budget sampling via `TARGET_TOTAL_TOKENS`
- Window-level reweighting for sliced XGuard conversations

### `analyze_lengths.py`

Reports token-length and turn-count statistics to help decide the slicing and
training budget settings.

### `preprocess_slide_by_turn.py`

Turn-aware sliding-window preprocessing for long multi-turn conversations.

## Typical workflow

1. Put the training parquet files into `packages/finetune/data/`.
2. Optionally run `uv run python packages/finetune/scripts/analyze_lengths.py`.
3. Slice long XGuard conversations with `preprocess_slide_by_turn.py`.
4. Launch training with `./packages/finetune/scripts/launch.sh`.
