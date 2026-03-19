#!/usr/bin/env python3
"""
Evaluate model responses against three benchmarks.

Takes inference output (response jsonl files) and computes scores using
the existing score/ evaluation framework (L1-L3 extraction and deterministic aggregation).
"""

import asyncio
import concurrent.futures
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from collections import defaultdict


def _run_async_from_sync(coro):
    """Run async coroutine from sync context. Safe when caller is already inside an event loop."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    # 已在事件循环中：用 nest_asyncio 在同一循环内运行，避免 ThreadPoolExecutor 新建循环
    # 导致 httpx AsyncClient 在循环关闭后 aclose 报 "Event loop is closed"
    try:
        import nest_asyncio
        nest_asyncio.apply()
    except ImportError:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    return loop.run_until_complete(coro)

import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _distribution(scores: List[float], key_type: type = str) -> Dict[str, int]:
    """Count distribution of integer scores."""
    dist = defaultdict(int)
    for s in scores:
        if key_type == str:
            dist[str(int(s))] += 1
        else:
            dist[int(s)] += 1
    return dict(sorted(dist.items()))


# ---------------------------------------------------------------------------
# XGuard evaluation — safety scoring of model responses to jailbreak attacks
# ---------------------------------------------------------------------------

async def _evaluate_xguard_async(responses_path: Path, api_key: Optional[str] = None) -> Dict[str, Any]:
    from score.xguard_eval import _extract_all_l3, reaggregate

    key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")

    # Load generation outputs
    with open(responses_path, "r", encoding="utf-8") as f:
        items = [json.loads(line) for line in f]

    if not items:
        return {"score": 0.0, "count": 0, "details": []}

    # Load original xteaming data to reconstruct the conversation (metadata + messages)
    eval_dir = PROJECT_ROOT / "dataset" / "eval"
    xteaming_path = eval_dir / "xteaming.jsonl"
    originals = {}
    if xteaming_path.exists():
        # 使用 utf-8-sig 自动处理 BOM 头
        with open(xteaming_path, "r", encoding="utf-8-sig") as f:
            for i, line in enumerate(f):
                data = json.loads(line)
                originals[str(data.get("id", f"xteaming_{i}"))] = data

    # Prepare rows for LLM L3 extraction
    # 逐轮推理输出：每条已有完整对话 messages（含模型每轮回复），直接用于打分
    rows_for_l3 = []
    for idx, item in enumerate(items):
        item_id = str(item.get("id", idx))
        orig = originals.get(item_id, {})

        if item.get("messages"):
            messages_with_resp = item["messages"]
            metadata = item.get("metadata", orig.get("metadata", {}))
        else:
            # 无 messages 时从 originals 重建（单轮推理输出格式）
            messages = orig.get("messages", [])
            if messages and messages[-1].get("role") == "assistant":
                messages = messages[:-1]
            messages_with_resp = messages + [{"role": "assistant", "content": item.get("response", "")}]
            metadata = orig.get("metadata", {})

        rows_for_l3.append({
            "id": item_id,
            "messages": messages_with_resp,
            "metadata": metadata,
        })

    # Output paths for this run's temporary L3 cache
    run_dir = responses_path.parent
    cache_path = run_dir / "xguard_l3_cache.jsonl"
    results_path = run_dir / "xguard_eval_results.jsonl"

    # Extract L3
    sys.argv = [] # ensure no conflict with CLI args parsed in score module if any
    await _extract_all_l3(rows_for_l3, cache_path)

    # Reaggregate to produce scores
    samples = reaggregate(cache_path, results_path)

    scores = [s["response_channel"]["score"] for s in samples]
    avg_score = sum(scores) / len(scores) if scores else 0.0

    # Clean up temp cache
    if cache_path.exists():
        cache_path.unlink()

    return {
        "score": avg_score,
        "count": len(scores),
        "score_distribution": _distribution(scores),
        "details": samples,  # Return the full atomic details for the advisor
    }


# ---------------------------------------------------------------------------
# OrBench evaluation — false refusal / over-sensitivity scoring
# ---------------------------------------------------------------------------

async def _evaluate_orbench_async(responses_path: Path, api_key: Optional[str] = None) -> Dict[str, Any]:
    from score.orbench_eval import _extract_all_l3, reaggregate

    # Load generation outputs
    with open(responses_path, "r", encoding="utf-8") as f:
        items = [json.loads(line) for line in f]

    if not items:
        return {"score": 0.0, "count": 0, "details": []}

    # Prepare rows for LLM L3 extraction (OrBench is usually single-turn)
    rows_for_l3 = []
    for idx, item in enumerate(items):
        item_id = str(item.get("id", idx))
        prompt = item.get("prompt", "")
        # Build messages format
        msgs = [{"role": "user", "content": prompt}]
        msgs.append({"role": "assistant", "content": item.get("response", "")})
        
        rows_for_l3.append({
            "id": item_id,
            "messages": msgs,
        })

    # Output paths for this run's temporary L3 cache
    run_dir = responses_path.parent
    cache_path = run_dir / "orbench_l3_cache.jsonl"
    results_path = run_dir / "orbench_eval_results.jsonl"

    # Extract L3
    await _extract_all_l3(rows_for_l3, cache_path)

    # Reaggregate to produce scores
    samples = reaggregate(cache_path, results_path)

    scores = [s["response_channel"]["score"] for s in samples]
    avg_score = sum(scores) / len(scores) if scores else 0.0

    # Clean up temp cache
    if cache_path.exists():
        cache_path.unlink()

    return {
        "score": avg_score,
        "count": len(scores),
        "score_distribution": _distribution(scores),
        "details": samples,
    }

async def _evaluate_xguard_and_orbench_async(
    xteaming_path: Path,
    orbench_path: Path,
    api_key: Optional[str] = None,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Run xguard and orbench in a single event loop.
    Avoids 'Event loop is closed' when httpx clients from the first run
    try to cleanup after asyncio.run() has already closed the loop.
    """
    xguard_result: Dict[str, Any] = {"score": 0.0, "count": 0}
    orbench_result: Dict[str, Any] = {"score": 0.0, "count": 0}

    if xteaming_path.exists():
        print("\n[eval] Evaluating xteaming (safety via LLM L3 extraction)...")
        xguard_result = await _evaluate_xguard_async(xteaming_path, api_key)
    if orbench_path.exists():
        print("\n[eval] Evaluating orbench (anti-overrefusal via LLM L3 extraction)...")
        orbench_result = await _evaluate_orbench_async(orbench_path, api_key)

    return xguard_result, orbench_result


def evaluate_xguard_responses(responses_path: Path, api_key: Optional[str] = None) -> Dict[str, Any]:
    """Standalone entry for xguard-only evaluation (e.g. CLI)."""
    return asyncio.run(_evaluate_xguard_async(responses_path, api_key))


def evaluate_orbench_responses(responses_path: Path, api_key: Optional[str] = None) -> Dict[str, Any]:
    """Standalone entry for orbench-only evaluation (e.g. CLI)."""
    return asyncio.run(_evaluate_orbench_async(responses_path, api_key))


# ---------------------------------------------------------------------------
# IFEval evaluation — instruction following with verifiable constraints
# ---------------------------------------------------------------------------

def _load_ifeval_prompts_by_id() -> Dict[str, str]:
    """从 dataset/eval/tulu-3.jsonl 加载 id->prompt 映射。"""
    path = PROJECT_ROOT / "dataset" / "eval" / "tulu-3.jsonl"
    if not path.exists():
        return {}
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                key = str(data.get("key", ""))
                prompt = data.get("prompt", "")
                if key and prompt:
                    out[key] = prompt
            except json.JSONDecodeError:
                pass
    return out


def evaluate_ifeval_responses(responses_path: Path) -> Dict[str, Any]:
    """
    Score model responses against verifiable instruction constraints.
    Uses the existing evaluation_lib for strict/loose checking.
    """
    with open(responses_path, "r", encoding="utf-8") as f:
        items = [json.loads(line) for line in f]

    if not items:
        return {"score": 0.0, "count": 0, "details": []}

    # Try to use the real IFEval evaluation library
    try:
        sys.path.insert(0, str(PROJECT_ROOT / "instruction_following_eval"))
        return _evaluate_ifeval_real(items)
    except ImportError:
        print("[eval] WARNING: evaluation_lib not available, defaulting to heuristic")
        return {"score": 0.0, "count": 0, "details": []}


def _evaluate_ifeval_real(items) -> Dict[str, Any]:
    """使用 eval_single 正确调用 evaluation_lib（需 InputExample + prompt_to_response）。"""
    from instruction_following_eval.evaluation_lib import InputExample
    from score.ifeval_score import eval_single
    from score.need import compute_sample_need

    prompt_by_id = _load_ifeval_prompts_by_id()

    scores = []
    samples = []

    for item in tqdm(items, desc="Evaluating IFEval", unit="resp"):
        prompt = item.get("prompt") or prompt_by_id.get(str(item.get("id", "")), "")
        if not prompt:
            scores.append(0.0)
            samples.append({
                "id": item.get("id", ""),
                "task": "ifeval",
                "prompt_channel": {"valid": 0, "slice": "complexity_1"},
                "response_channel": {"score": 0.0},
            })
            continue

        # 修复kwargs中的兼容性问题，并展开不支持的relation为语义等价的约束
        instruction_id_list = item.get("instruction_id_list", [])
        kwargs = item.get("kwargs", [])
        fixed_instruction_ids = []
        fixed_kwargs = []
        prompt_lower = prompt.lower()
        prompt_at_most = any(x in prompt_lower for x in ["at most", "no longer than", "no more than"])

        for idx, kw in enumerate(kwargs):
            inst_id = instruction_id_list[idx] if idx < len(instruction_id_list) else ""

            if not isinstance(kw, dict):
                fixed_instruction_ids.append(inst_id)
                fixed_kwargs.append(kw)
                continue

            fixed_kw = dict(kw)

            # 修复1: "exactly N" 展开为 "at least N" AND "less than N+1"（语义等价）
            is_exactly = fixed_kw.get("relation") == "exactly" or fixed_kw.get("let_relation") == "exactly"
            if is_exactly:
                if "num_sentences" in fixed_kw:
                    n = fixed_kw["num_sentences"]
                    fixed_instruction_ids.append(inst_id)
                    fixed_instruction_ids.append(inst_id)
                    fixed_kwargs.append({"num_sentences": n, "relation": "at least"})
                    fixed_kwargs.append({"num_sentences": n + 1, "relation": "less than"})
                elif "num_words" in fixed_kw:
                    n = fixed_kw["num_words"]
                    fixed_instruction_ids.append(inst_id)
                    fixed_instruction_ids.append(inst_id)
                    fixed_kwargs.append({"num_words": n, "relation": "at least"})
                    fixed_kwargs.append({"num_words": n + 1, "relation": "less than"})
                elif "frequency" in fixed_kw and "keyword" in fixed_kw:
                    n = fixed_kw["frequency"]
                    fixed_instruction_ids.append(inst_id)
                    fixed_instruction_ids.append(inst_id)
                    fixed_kwargs.append({"keyword": fixed_kw["keyword"], "frequency": n, "relation": "at least"})
                    fixed_kwargs.append({"keyword": fixed_kw["keyword"], "frequency": n + 1, "relation": "less than"})
                elif "let_frequency" in fixed_kw and "letter" in fixed_kw:
                    n = fixed_kw["let_frequency"]
                    fixed_instruction_ids.append(inst_id)
                    fixed_instruction_ids.append(inst_id)
                    fixed_kwargs.append({"letter": fixed_kw["letter"], "let_frequency": n, "let_relation": "at least"})
                    fixed_kwargs.append({"letter": fixed_kw["letter"], "let_frequency": n + 1, "let_relation": "less than"})
                else:
                    fixed_kw["relation"] = "at least"
                    fixed_instruction_ids.append(inst_id)
                    fixed_kwargs.append(fixed_kw)
                continue

            # 修复2: "at most N" -> "less than N+1"（语义等价）；兼容历史错误转换的 "less than N"
            if fixed_kw.get("relation") == "at most":
                if "num_sentences" in fixed_kw:
                    fixed_kw = {"num_sentences": fixed_kw["num_sentences"] + 1, "relation": "less than"}
                elif "num_words" in fixed_kw:
                    fixed_kw = {"num_words": fixed_kw["num_words"] + 1, "relation": "less than"}
                elif "frequency" in fixed_kw and "keyword" in fixed_kw:
                    fixed_kw = {"keyword": fixed_kw["keyword"], "frequency": fixed_kw["frequency"] + 1, "relation": "less than"}
            elif fixed_kw.get("let_relation") == "at most":
                fixed_kw = {"letter": fixed_kw["letter"], "let_frequency": fixed_kw["let_frequency"] + 1, "let_relation": "less than"}
            elif fixed_kw.get("capital_relation") == "at most":
                fixed_kw = {"capital_frequency": fixed_kw["capital_frequency"] + 1, "capital_relation": "less than"}
            elif fixed_kw.get("relation") == "less than" and prompt_at_most:
                # 历史错误：将 "at most" 误转为 "less than" 且未改 num，需补 +1
                if "num_sentences" in fixed_kw:
                    fixed_kw = {"num_sentences": fixed_kw["num_sentences"] + 1, "relation": "less than"}
                elif "num_words" in fixed_kw:
                    fixed_kw = {"num_words": fixed_kw["num_words"] + 1, "relation": "less than"}

            # 修复3: 为combination:repeat_prompt添加prompt_to_repeat
            if inst_id == "combination:repeat_prompt" and "prompt_to_repeat" not in fixed_kw:
                fixed_kw["prompt_to_repeat"] = prompt

            fixed_instruction_ids.append(inst_id)
            fixed_kwargs.append(fixed_kw)

        inp = InputExample(
            key=item.get("id", 0),
            instruction_id_list=fixed_instruction_ids,
            prompt=prompt,
            kwargs=fixed_kwargs,
        )
        prompt_to_response = {prompt: item.get("response", "")}

        try:
            result = eval_single(inp, prompt_to_response)
        except Exception:
            scores.append(0.0)
            samples.append({
                "id": item.get("id", ""),
                "task": "ifeval",
                "prompt_channel": {"valid": 0, "slice": "complexity_1"},
                "response_channel": {"score": 0.0},
            })
            continue

        score = float(result.if_score)
        scores.append(score)

        # 优先使用推理时传递的预计算 complexity（如果存在），否则使用 eval_single 计算的
        inst_complexity = item.get("_inst_complexity")
        if inst_complexity is not None:
            inst_complexity = int(inst_complexity)
        else:
            inst_complexity = result.inst_complexity

        slice_label = f"complexity_{inst_complexity}"
        weight = inst_complexity
        need = compute_sample_need(1, int(score), weight)

        samples.append({
            "id": item.get("id", ""),
            "task": "ifeval",
            "prompt_channel": {
                "valid": 1,
                "slice": slice_label,
                "weight": weight,
                "l3": {"instruction_id_list": inp.instruction_id_list, "m": len(inp.instruction_id_list)},
                "l2": {"inst_complexity": inst_complexity},
            },
            "response_channel": {
                "score": score,
                "l2": {"r_strict": result.r_strict, "r_loose": result.r_loose},
                "l3": {
                    "r_strict": result.r_strict,
                    "r_loose": result.r_loose,
                    "follow_strict": result.follow_instruction_list_strict,
                    "follow_loose": result.follow_instruction_list_loose,
                },
                "diag": {"failed_constraints": result.failed_constraints},
            },
            "need": need,
        })

    avg_score = sum(scores) / len(scores) if scores else 0.0

    return {
        "score": avg_score,
        "count": len(scores),
        "score_distribution": _distribution(scores),
        "details": samples,
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def _compute_bucket_stats(details: List[Dict]) -> Dict[str, float]:
    """Per-slice average scores for the advisor overview."""
    bucket_scores = defaultdict(list)
    for sample in details:
        if "prompt_channel" not in sample or "response_channel" not in sample:
            continue
        slice_val = sample["prompt_channel"].get("slice", "unknown")
        bucket_scores[slice_val].append(sample["response_channel"].get("score", 0))

    stats = {}
    for bucket, b_scores in bucket_scores.items():
        if b_scores:
            stats[bucket] = round(sum(b_scores) / len(b_scores), 3)
    return stats

def evaluate_all(
    inference_dir: Path,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluate all three benchmarks from inference output directory.
    Output compatible with the Pareto loop and LLM advisor.
    """
    results = {}

    # XGuard + OrBench: run in a single event loop to avoid "Event loop is closed"
    # (httpx clients from first run would try to aclose() after loop already closed)
    xteaming_path = inference_dir / "xteaming_responses.jsonl"
    orbench_path = inference_dir / "orbench_responses.jsonl"
    if xteaming_path.exists() or orbench_path.exists():
        xguard_result, orbench_result = _run_async_from_sync(
            _evaluate_xguard_and_orbench_async(xteaming_path, orbench_path, api_key)
        )
        results["xguard"] = xguard_result
        results["orbench"] = orbench_result
        if results["xguard"].get("details"):
            results["xguard"]["bucket_score_avgs"] = _compute_bucket_stats(results["xguard"]["details"])
        if results["orbench"].get("details"):
            results["orbench"]["bucket_score_avgs"] = _compute_bucket_stats(results["orbench"]["details"])
    else:
        results["xguard"] = {"score": 0.0, "count": 0}
        results["orbench"] = {"score": 0.0, "count": 0}

    # IFEval
    ifeval_path = inference_dir / "ifeval_responses.jsonl"
    if ifeval_path.exists():
        print("\n[eval] Evaluating ifeval (instruction following via true program check)...")
        results["ifeval"] = evaluate_ifeval_responses(ifeval_path)
        results["ifeval"]["bucket_score_avgs"] = _compute_bucket_stats(results["ifeval"].get("details", []))
    else:
        results["ifeval"] = {"score": 0.0, "count": 0}

    # Compose f vector
    f = (
        results["xguard"].get("score", 0.0),
        results["orbench"].get("score", 0.0),
        results["ifeval"].get("score", 0.0),
    )
    results["f"] = f

    print(f"\n[eval] Final scores: xguard={f[0]:.4f}, orbench={f[1]:.4f}, ifeval={f[2]:.4f}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate model responses")
    parser.add_argument("--inference-dir", type=str, required=True,
                        help="Directory containing *_responses.jsonl files")
    args = parser.parse_args()

    results = evaluate_all(Path(args.inference_dir))
    print(json.dumps({k: v for k, v in results.items() if k != "f"}, indent=2, default=str))
