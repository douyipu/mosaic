#!/usr/bin/env python3
"""
将 eval_results.jsonl 合并回 raw parquet，一次执行后 data_constructor 可直接从 parquet 读 slice/score/need。

用法:
    uv run python -m scripts.merge_eval_to_parquet
    uv run python -m scripts.merge_eval_to_parquet --task xguard
"""

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"

# 与 data_constructor 一致
RAW_PATHS = {
    "xguard": DATASET_DIR / "train" / "xguard-train.parquet",
    "orbench": DATASET_DIR / "train" / "orbench.parquet",
    "ifeval": DATASET_DIR / "train" / "tulu-3.parquet",
}
EVAL_PATHS = {
    "xguard": PROJECT_ROOT / "results" / "xguard" / "eval_results.jsonl",
    "orbench": PROJECT_ROOT / "results" / "orbench" / "eval_results.jsonl",
    "ifeval": PROJECT_ROOT / "results" / "ifeval" / "eval_results.jsonl",
}


def _record_from_eval(d: dict, source_name: str) -> dict:
    """从 eval 行提取合并用字段，列名与 raw parquet 兼容。"""
    pc = d.get("prompt_channel", {}) or {}
    rc = d.get("response_channel", {}) or {}
    l2_prompt = pc.get("l2", {}) or {}
    l2_resp = rc.get("l2", {}) or {}
    l3 = rc.get("l3", {}) or {}
    diag = rc.get("diag", {}) or {}

    rec = {
        "score": float(rc.get("score", 0)),
        "slice": pc.get("slice", "unknown"),
        "need": float(d.get("need", 0)),
    }
    if "_row_idx" in d:
        rec["_row_idx"] = int(d["_row_idx"])
    rec["id"] = str(d.get("id", ""))

    if source_name == "xguard":
        rec["_pressure"] = l2_prompt.get("pressure", 0)
        rec["_concealment"] = l2_prompt.get("concealment", 0)
        rec["_safety"] = l2_resp.get("safety", "unknown")
        leaks = l3.get("safety_leaks", {}) or {}
        for leak_type, val in leaks.items():
            if val:
                rec[f"_leak_{leak_type}"] = 1
    elif source_name == "orbench":
        rec["_refusal_type"] = l2_resp.get("refusal_type", "unknown")
        rec["_help_level"] = l2_resp.get("help_level", "unknown")
        rec["_friction"] = l2_resp.get("friction", "unknown")
    elif source_name == "ifeval":
        ic = l2_prompt.get("inst_complexity")
        if ic is None:
            slice_val = pc.get("slice", "")
            if isinstance(slice_val, str) and slice_val.startswith("complexity_"):
                try:
                    ic = int(slice_val.split("_")[1])
                except (IndexError, ValueError):
                    ic = 0
            else:
                ic = 0
        rec["_inst_complexity"] = ic
        rec["_failed_constraints"] = diag.get("failed_constraints", [])

    return rec


def merge_one(source_name: str) -> bool:
    raw_path = RAW_PATHS[source_name]
    eval_path = EVAL_PATHS[source_name]

    if not raw_path.exists():
        print(f"  [{source_name}] raw 不存在: {raw_path}")
        return False
    if not eval_path.exists():
        print(f"  [{source_name}] eval_results 不存在: {eval_path}")
        return False

    records = []
    with open(eval_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            records.append(_record_from_eval(d, source_name))

    if not records:
        print(f"  [{source_name}] eval_results 为空")
        return False

    eval_df = pd.DataFrame(records)
    raw_df = pd.read_parquet(raw_path)

    # 确定 merge key，优先级：id > _row_idx（与 parquet 行一一对应）> 行索引
    if "id" in raw_df.columns:
        merge_key_raw = "id"
        merge_key_eval = "id"
    elif "_row_idx" in eval_df.columns:
        merge_key_raw = "_merge_idx"
        merge_key_eval = "_row_idx"
        raw_df = raw_df.copy()
        raw_df["_merge_idx"] = range(len(raw_df))
    else:
        merge_key_raw = "_merge_idx"
        merge_key_eval = "_merge_idx"
        raw_df = raw_df.copy()
        raw_df["_merge_idx"] = range(len(raw_df))
        eval_df = eval_df.copy()
        eval_df["_merge_idx"] = range(len(eval_df))

    score_cols = [c for c in eval_df.columns if c not in (merge_key_eval, "id", "_row_idx")]
    scores = eval_df[[merge_key_eval] + score_cols].drop_duplicates(subset=[merge_key_eval])

    merged = raw_df.merge(
        scores,
        left_on=merge_key_raw,
        right_on=merge_key_eval,
        how="left",
        suffixes=("_old", "_new"),
    )
    merged = merged.drop(columns=[merge_key_eval, "_merge_idx", "_row_idx"], errors="ignore")

    for c in score_cols:
        src = f"{c}_new" if f"{c}_new" in merged.columns else c
        if src in merged.columns:
            merged[c] = merged[src]
        merged = merged.drop(columns=[f"{c}_old", f"{c}_new"], errors="ignore")

    merged.to_parquet(raw_path, index=False)
    print(f"  [{source_name}] 已合并 {len(records)} 条打分 -> {raw_path.name}")
    return True


def main():
    import argparse
    parser = argparse.ArgumentParser(description="将 eval_results.jsonl 合并回 raw parquet")
    parser.add_argument("--task", choices=["xguard", "orbench", "ifeval", "all"], default="all")
    args = parser.parse_args()

    tasks = ["xguard", "orbench", "ifeval"] if args.task == "all" else [args.task]
    print("[merge_eval_to_parquet] 合并预打分到 parquet\n")

    ok = 0
    for t in tasks:
        if merge_one(t):
            ok += 1

    print(f"\n完成: {ok}/{len(tasks)}")
    return 0 if ok == len(tasks) else 1


if __name__ == "__main__":
    sys.exit(main())
