# packages/finetune

This subpackage contains the LoRA fine-tuning utilities used by MOSAIC.

## Usage

Install the training dependencies first:

```bash
uv sync --extra finetune
```

Then launch a fine-tuning run:

```bash
./packages/finetune/scripts/launch.sh
```

## Path configuration

All paths can be overridden with environment variables. Otherwise they are
derived from the script location:

- `PROJECT_ROOT`: finetune package root
- `DATA_DIR`, `CACHE_DIR`, `OUTPUT_DIR`: data, cache, and output directories
- `MODEL_PATH`: local HuggingFace model directory
- `HF_HOME`: HuggingFace cache location

See [`scripts/README.md`](/C:/Users/14685/Desktop/thesis_project/seu_paper_220235229/AJAR_arxiv/mosaic/packages/finetune/scripts/README.md) for the script-level workflow.
