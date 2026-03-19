"""统一 need 计算与预算分配，三个评测任务共用。"""

from collections import defaultdict


def compute_sample_need(valid: int, score: int, weight: int, s_max: int = 5) -> float:
    """样本级 need = valid × gap(score) × weight。"""
    gap = (s_max + 1) - score
    return valid * gap * weight


def compute_bucket_need(samples: list[dict], slice_key: str = "slice") -> dict[str, float]:
    """按切片聚合 Need(s) = sum(need_i for i in slice s)。"""
    buckets = defaultdict(float)
    for s in samples:
        slice_val = s.get("prompt_channel", {}).get(slice_key)
        if slice_val is not None:
            buckets[slice_val] += s.get("need", 0)
    return dict(buckets)


def allocate_budget(bucket_need: dict[str, float], total_budget: int) -> dict[str, int]:
    """按比例分配预算 B(s) = round(B * Need(s) / sum(Need))。"""
    total = sum(bucket_need.values())
    if total == 0:
        return {k: 0 for k in bucket_need}
    return {k: round(total_budget * v / total) for k, v in bucket_need.items()}


def compute_two_level_budget(
    task_bucket_needs: dict[str, dict[str, float]],
    total_budget: int,
    task_weights: dict[str, float] | None = None,
) -> dict[str, dict[str, int]]:
    """两级预算分配：先按任务分配，再按桶内分配。

    Args:
        task_bucket_needs: {task_name: {bucket_label: need_value}}
            例如 {"ifeval": {"complexity_1": 8129, ...},
                   "xguard": {"complexity_3": 375, ...},
                   "orbench": {"boundary_1": 45, ...}}
        total_budget: 总预算（样本条数或 token 数）
        task_weights: 可选，{task_name: weight}。
            若为 None，则按任务间 Need 总量比例分配。
            若给定，则先按 task_weights 归一化分配，再在桶内按 Need 分配。
            典型用法: 等权 {"ifeval": 1, "xguard": 1, "orbench": 1}。

    Returns:
        {task_name: {bucket_label: allocated_budget}}
    """
    if not task_bucket_needs:
        return {}

    # --- 第一级：任务间分配 ---
    if task_weights is not None:
        # 按给定权重分配
        total_w = sum(task_weights.get(t, 1.0) for t in task_bucket_needs)
        if total_w == 0:
            total_w = len(task_bucket_needs)
        task_budgets = {
            t: round(total_budget * task_weights.get(t, 1.0) / total_w)
            for t in task_bucket_needs
        }
    else:
        # 按各任务 Need 总值比例分配
        task_totals = {t: sum(bns.values()) for t, bns in task_bucket_needs.items()}
        grand_total = sum(task_totals.values())
        if grand_total == 0:
            task_budgets = {t: 0 for t in task_bucket_needs}
        else:
            task_budgets = {
                t: round(total_budget * v / grand_total) for t, v in task_totals.items()
            }

    # --- 第二级：桶内分配 ---
    result = {}
    for task, bucket_needs in task_bucket_needs.items():
        result[task] = allocate_budget(bucket_needs, task_budgets.get(task, 0))

    return result
