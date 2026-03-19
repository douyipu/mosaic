# MOSAIC

Public code release for the paper **"MOSAIC: Multi-Objective Slice-Aware Iterative Curation for Alignment."**

`MOSAIC` is short for **Multi-Objective Slice-Aware Iterative Curation for Alignment**.

This repository contains the core evaluation, data-mixture search, and fine-tuning utilities used to study closed-loop supervised fine-tuning under a fixed token budget. The public release is intentionally trimmed down for GitHub: large datasets, generated results, virtual environments, and internal-only sync tooling are not included.

## What is in this repository

- A unified L1-L3 evaluation interface for XGuard, OrBench, and IFEval
- Slice-aware closed-loop data mixture search with an agent-based proposal step
- LoRA fine-tuning utilities and vLLM inference helpers
- Minimal configs needed to reproduce the MOSAIC workflow once the datasets are prepared

## What is intentionally excluded

- Raw or processed datasets
- Experiment outputs under `results/`
- Local virtual environments
- Archived exploratory scripts and notebooks
- Internal cloud storage / synchronization tooling

## Quickstart

### 1. Create the environment

```bash
uv sync
```

To run local fine-tuning or managed vLLM inference, install the extra training dependencies:

```bash
uv sync --extra finetune
```

### 2. Configure environment variables

Copy `.env.example` to `.env` and fill in the keys you actually use.

At minimum, the evaluation and proposal-agent pipeline expects:

- `DEEPSEEK_API_KEY` for L3 extraction and the proposal agent
- `MODEL_PATH` when using local model inference or local fine-tuning

### 3. Prepare the dataset layout

Create the following directories locally:

```text
dataset/
  train/
    xguard-train.parquet
    orbench.parquet
    tulu-3.parquet
  eval/
    xteaming.jsonl
    orbench.jsonl
    tulu-3.jsonl
```

The repository expects preprocessed training pools and evaluation sets to follow these names unless you modify the scripts.

### 4. Pre-score the training pools

```bash
uv run python -m score.xguard_eval
uv run python -m score.orbench_eval
uv run python -m score.ifeval_eval
```

### 5. Merge the evaluation signals back into parquet files

```bash
uv run python scripts/merge_eval_to_parquet.py
```

### 6. Run the MOSAIC search loop

```bash
uv run python main.py --max-iters 5 --T-max 1000000 --quality-threshold 4.0
```

`main.py` is a thin wrapper around [`scripts/run_loop.py`](/C:/Users/14685/Desktop/thesis_project/seu_paper_220235229/AJAR_arxiv/mosaic/scripts/run_loop.py).

## Repository layout

```text
mosaic/
|- configs/                     # Small benchmark-specific configs
|- instruction_following_eval/  # IFEval support code
|- packages/finetune/           # LoRA fine-tuning utilities
|- score/                       # Benchmark scoring and aggregation
|- scripts/                     # Closed-loop orchestration scripts
|- .env.example                 # Minimal environment variable template
|- main.py                      # Entry point for the MOSAIC loop
|- pyproject.toml               # uv project metadata
`- schema.py                    # Shared record schema
```

## Recommended workflow

1. Prepare the three training pools and three evaluation sets.
2. Run the benchmark-specific evaluators to extract L3 annotations and aggregate scores.
3. Merge the resulting slice/score/need columns back into the training parquet files.
4. Launch the closed-loop search with a fixed token budget.
5. Inspect the per-iteration `results/run_*/` outputs, Pareto archive, and agent traces.

## Notes on reproducibility

- The public repository does not ship the datasets used in the paper.
- The release keeps the original experimental code structure, but trims large artifacts and outdated internal utilities.
- Some scripts still assume the benchmark file names listed above. If your local files differ, update the path constants in the relevant script before running.

## Citation

If you use this repository, cite the MOSAIC paper:

```bibtex
@article{dou2026mosaic,
  title   = {MOSAIC: Multi-Objective Slice-Aware Iterative Curation for Alignment},
  author  = {Dou, Yipu and Yang, Wang},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2026}
}
```
