"""统一报告：分桶统计、Need 热力、预算分配建议。"""

import json
import random
from collections import defaultdict
from pathlib import Path

from score.need import compute_bucket_need, allocate_budget


def _load_jsonl(path: Path) -> list[dict]:
    samples = []
    if not path.exists():
        return samples
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
    return samples


def _bootstrap_stability(samples: list[dict], n_iter: int = 100, frac: float = 0.8) -> dict:
    """子集 bootstrap：随机抽 frac 比例 × n_iter 次，计算 score 均值方差。"""
    if len(samples) < 10:
        return {"mean": 0, "std": 0, "n": len(samples)}
    scores = [s.get("response_channel", {}).get("score", 0) for s in samples]
    means = []
    for _ in range(n_iter):
        subset = random.sample(scores, max(1, int(len(scores) * frac)))
        means.append(sum(subset) / len(subset))
    return {
        "mean": sum(means) / len(means),
        "std": (sum((m - sum(means) / len(means)) ** 2 for m in means) / len(means)) ** 0.5,
        "n": len(samples),
    }


def main() -> None:
    root = Path(__file__).parent.parent
    results_dir = root / "results"

    tasks = ["ifeval", "xguard", "orbench"]
    all_buckets = {}
    all_samples = {}

    for task in tasks:
        path = results_dir / task / "eval_results.jsonl"
        samples = _load_jsonl(path)
        all_samples[task] = samples
        if not samples:
            print(f"[{task}] 无数据")
            continue
        bucket_need = compute_bucket_need(samples)
        all_buckets[task] = bucket_need

        score_dist = defaultdict(int)
        for s in samples:
            score_dist[s.get("response_channel", {}).get("score", 0)] += 1

        print(f"\n=== {task.upper()} 分桶统计 ===")
        print(f"样本数: {len(samples)}")
        print("Score 分布:", dict(sorted(score_dist.items())))
        print("Need 按桶:")
        for k in sorted(bucket_need.keys(), key=lambda x: (x.split("_")[-1] if "_" in x else x)):
            print(f"  {k}: {bucket_need[k]:.1f}")

        if len(samples) >= 20:
            stab = _bootstrap_stability(samples, n_iter=50)
            print(f"Bootstrap 稳定性 (50次×80%): mean={stab['mean']:.3f}, std={stab['std']:.3f}")

    total_need = defaultdict(float)
    for task, buckets in all_buckets.items():
        for k, v in buckets.items():
            total_need[f"{task}:{k}"] = v

    print("\n=== 跨任务 Need 汇总 ===")
    for k in sorted(total_need.keys()):
        print(f"  {k}: {total_need[k]:.1f}")

    total = sum(total_need.values())
    if total > 0:
        budget = 1000
        alloc = allocate_budget(dict(total_need), budget)
        print(f"\n预算分配建议 (总预算 B={budget}):")
        for k in sorted(alloc.keys()):
            if alloc[k] > 0:
                print(f"  {k}: {alloc[k]}")


if __name__ == "__main__":
    main()
