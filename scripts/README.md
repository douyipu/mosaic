# Scripts

The `scripts/` directory contains the orchestration layer for the MOSAIC loop.

## Core entry points

- `run_loop.py`: main closed-loop controller
- `distribution_agent.py`: proposal agent that maps failure profiles to the next data mixture
- `data_constructor.py`: token-budgeted subset construction from pre-scored training pools
- `eval_model.py`: unified evaluation wrapper over XGuard, OrBench, and IFEval
- `vllm_inference.py`: local or API-based generation for evaluation sets
- `merge_eval_to_parquet.py`: writes slice/score/need signals back into parquet files
- `pareto.py`: Pareto archive utilities for multi-objective tracking

## Optional utilities

- `finetune_xguard_only.py`: single-source baseline fine-tuning helper
- `test_l3_reliability.py`: reliability checks for the L3 annotation pipeline

## Typical usage

```bash
uv run python main.py --max-iters 5 --T-max 1000000 --quality-threshold 4.0
```
