"""指令跟随评测：strict/loose 判定 + IFScore 映射 + InstComplexity。"""

import math
from dataclasses import dataclass

from instruction_following_eval.evaluation_lib import (
    InputExample,
    test_instruction_following_strict,
    test_instruction_following_loose,
)


def compute_if_score(n_strict: int, n_loose: int, m: int) -> int:
    """样本级 IFScore 1-5（按失败比例映射，同时消除旧版比例阈值的数值死区）。

    用失败比例 fail_ratio = (m - n_loose) / m 分档，对任意 M 值都合理：
    - M=1 完全失败 → fail_ratio=1.0 → Score 1（而非旧版按条数映射的 Score 3）
    - M=3 失败 1 条 → fail_ratio=0.33 → Score 3（消除了旧版 2/3<0.67 的死区）
    """
    if m == 0:
        return 5
    if n_strict == m:
        return 5
    if n_loose == m:
        return 4
    fail_ratio = (m - n_loose) / m
    if fail_ratio <= 1 / 3:
        return 3
    if fail_ratio <= 2 / 3:
        return 2
    return 1


def compute_inst_complexity(m: int, k: int) -> int:
    """约束复杂度：M=约束条数, K=涉及指令组数（instruction 前缀种类数）。"""
    return min(5, 1 + (m >= 2) + (m >= 3) + (k >= 2) + (k >= 3))


def _instruction_groups(instruction_id_list: list[str]) -> set[str]:
    """提取 instruction_id 的组前缀（冒号前部分）。"""
    return {inst.split(":")[0] for inst in instruction_id_list}


@dataclass
class IFEvalResult:
    if_score: int
    inst_complexity: int
    r_strict: float
    r_loose: float
    follow_instruction_list_strict: list[bool]
    follow_instruction_list_loose: list[bool]
    instruction_id_list: list[str]
    failed_constraints: list[str]


def eval_single(
    inp: InputExample,
    prompt_to_response: dict[str, str],
) -> IFEvalResult:
    """对单条样本执行 strict + loose 检测，返回 IFScore 与 InstComplexity。"""
    out_strict = test_instruction_following_strict(inp, prompt_to_response)
    out_loose = test_instruction_following_loose(inp, prompt_to_response)

    m = len(inp.instruction_id_list)
    k = len(_instruction_groups(inp.instruction_id_list))
    inst_complexity = compute_inst_complexity(m, k)

    n = len(out_strict.follow_instruction_list)
    n_strict = sum(out_strict.follow_instruction_list)
    n_loose = sum(out_loose.follow_instruction_list)
    r_strict = n_strict / n if n else 0.0
    r_loose = n_loose / n if n else 0.0

    if_score = compute_if_score(n_strict, n_loose, m)

    failed = [
        inst_id
        for inst_id, ok in zip(inp.instruction_id_list, out_strict.follow_instruction_list)
        if not ok
    ]

    return IFEvalResult(
        if_score=if_score,
        inst_complexity=inst_complexity,
        r_strict=r_strict,
        r_loose=r_loose,
        follow_instruction_list_strict=out_strict.follow_instruction_list,
        follow_instruction_list_loose=out_loose.follow_instruction_list,
        instruction_id_list=inp.instruction_id_list,
        failed_constraints=failed,
    )
