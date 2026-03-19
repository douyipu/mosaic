"""指令跟随评测端到端流程：读约束配置 → strict/loose 检测 → IFScore → need → 输出。"""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

EVAL_SAMPLE_LIMIT = int(os.environ.get("EVAL_SAMPLE_LIMIT", 0))

from instruction_following_eval.evaluation_lib import InputExample

from score.ifeval_score import eval_single
from score.need import compute_sample_need, compute_bucket_need


def _load_constraints(path: Path) -> list[dict]:
    entries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    return entries


def _load_responses(path: Path) -> dict[str, str]:
    prompt_to_response = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            prompt_to_response[obj["prompt"]] = obj["response"]
    return prompt_to_response


def _build_input_example(entry: dict) -> InputExample:
    return InputExample(
        key=entry.get("id", entry.get("key", 0)),
        instruction_id_list=entry["instruction_id_list"],
        prompt=entry["prompt"],
        kwargs=entry["kwargs"],
    )


def main() -> None:
    root = Path(__file__).parent.parent
    constraints_path = root / "configs/ifeval_constraints/tulu3_constraints.jsonl"
    if not constraints_path.exists():
        print("请先运行: python -m score.ifeval_extract")
        sys.exit(1)

    response_path = root / "configs/ifeval_constraints/tulu3_responses.jsonl"
    if not response_path.exists():
        print(f"未找到 response 文件: {response_path}")
        sys.exit(1)

    constraints = _load_constraints(constraints_path)
    prompt_to_response = _load_responses(response_path)

    inputs = [
        _build_input_example(e)
        for e in constraints
        if e["prompt"] in prompt_to_response
    ]
    if EVAL_SAMPLE_LIMIT > 0:
        inputs = inputs[:EVAL_SAMPLE_LIMIT]
        print(f"限制样本数: {EVAL_SAMPLE_LIMIT}")
    skipped = len(constraints) - len(inputs)
    if skipped:
        print(f"跳过 {skipped} 条无匹配 response 的样本")

    if not inputs:
        print("无有效样本")
        sys.exit(1)

    results_dir = root / "results/ifeval"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "eval_results.jsonl"

    samples = []
    for inp in inputs:
        result = eval_single(inp, prompt_to_response)
        valid = 1
        slice_label = f"complexity_{result.inst_complexity}"
        weight = result.inst_complexity
        need = compute_sample_need(valid, result.if_score, weight)

        sample = {
            "id": str(inp.key),
            "task": "ifeval",
            "prompt_channel": {
                "valid": valid,
                "slice": slice_label,
                "weight": weight,
                "l3": {"instruction_id_list": inp.instruction_id_list, "m": len(inp.instruction_id_list)},
                "l2": {"inst_complexity": result.inst_complexity},
            },
            "response_channel": {
                "score": result.if_score,
                "l3": {
                    "r_strict": result.r_strict,
                    "r_loose": result.r_loose,
                    "follow_strict": result.follow_instruction_list_strict,
                    "follow_loose": result.follow_instruction_list_loose,
                },
                "l2": {"r_strict": result.r_strict, "r_loose": result.r_loose},
                "diag": {"failed_constraints": result.failed_constraints},
            },
            "need": need,
        }
        samples.append(sample)

    with open(out_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    bucket_need = compute_bucket_need(samples)
    score_dist = defaultdict(int)
    for s in samples:
        score_dist[s["response_channel"]["score"]] += 1

    print("\n=== IFEval 评测报告 ===")
    print(f"样本数: {len(samples)}")
    print("\nIFScore 分布:")
    for score in sorted(score_dist.keys()):
        print(f"  分数 {score}: {score_dist[score]} 条 ({100*score_dist[score]/len(samples):.1f}%)")
    print("\n分桶统计 (按 InstComplexity):")
    for slice_name in sorted(bucket_need.keys(), key=lambda x: int(x.split("_")[1])):
        print(f"  {slice_name}: Need={bucket_need[slice_name]:.1f}")
    print(f"\n结果已写入: {out_path}")


if __name__ == "__main__":
    main()
