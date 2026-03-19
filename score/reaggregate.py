"""从 L3 缓存重新聚合所有任务（不调 LLM）。

用法:
    uv run python -m score.reaggregate           # 重聚合全部
    uv run python -m score.reaggregate --task xguard   # 只重聚合 xguard
    uv run python -m score.reaggregate --task orbench  # 只重聚合 orbench
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="从 L3 缓存重新聚合评测结果")
    parser.add_argument("--task", choices=["xguard", "orbench", "all"], default="all",
                        help="要重聚合的任务 (default: all)")
    args = parser.parse_args()

    root = Path(__file__).parent.parent
    tasks_to_run = []

    if args.task in ("all", "xguard"):
        tasks_to_run.append("xguard")
    if args.task in ("all", "orbench"):
        tasks_to_run.append("orbench")

    for task in tasks_to_run:
        cache_path = root / f"results/{task}/l3_cache.jsonl"
        out_path = root / f"results/{task}/eval_results.jsonl"

        if not cache_path.exists():
            print(f"[{task}] L3 缓存不存在: {cache_path}，跳过")
            print(f"  → 先运行 uv run python -m score.{task}_eval 生成缓存")
            continue

        print(f"\n{'='*60}")
        print(f"重聚合: {task}")
        print(f"{'='*60}")

        if task == "xguard":
            from score.xguard_eval import reaggregate
        elif task == "orbench":
            from score.orbench_eval import reaggregate

        reaggregate(cache_path, out_path)


if __name__ == "__main__":
    main()
