#!/usr/bin/env python3
"""
Data constructor — build training subsets from a distribution vector.

Given a data distribution w = {"xguard": ..., "orbench": ..., "ifeval": ...}
and total token budget T_max, construct a training parquet by:
  1. Allocating token budget per source
  2. Loading slided parquet data
  3. Using training set's own pre-scored columns (score, slice, L2/L3) — no merge with eval
  4. (Optional) Filtering out low-quality rows based on response_score
  5. Per-dimension DimPlan: bucket_weights + focus_criteria (include/exclude)
  6. Token-constrained sampling (focus first, then random fill)
"""

import json
import math
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FINETUNE_DATA = PROJECT_ROOT / "packages" / "finetune" / "data"
DATASET_DIR = PROJECT_ROOT / "dataset"

RAW_TO_SLIDED_BASE = {
    "xguard": (DATASET_DIR / "train" / "xguard-train.parquet", FINETUNE_DATA / "xguard-train-slided"),
    "orbench": (DATASET_DIR / "train" / "orbench.parquet", FINETUNE_DATA / "orbench-slided"),
    "ifeval": (DATASET_DIR / "train" / "tulu-3.parquet", FINETUNE_DATA / "tulu-3-slided"),
}


def get_slided_path(name: str, max_length: int) -> Path:
    if name not in RAW_TO_SLIDED_BASE:
        raise ValueError(f"Unknown source: {name}")
    _, base = RAW_TO_SLIDED_BASE[name]
    return Path(str(base) + f"-{max_length}.parquet")


def get_default_source_paths(max_length: int) -> Dict[str, Path]:
    return {name: get_slided_path(name, max_length) for name in RAW_TO_SLIDED_BASE}


def ensure_slided_for_sources(
    model_path: str,
    max_length: int,
    sources: Optional[List[str]] = None,
) -> None:
    preprocess_script = PROJECT_ROOT / "packages" / "finetune" / "scripts" / "preprocess_slide_by_turn.py"
    names = sources or list(RAW_TO_SLIDED_BASE.keys())
    for name in names:
        if name not in RAW_TO_SLIDED_BASE:
            continue
        raw, _ = RAW_TO_SLIDED_BASE[name]
        slided = get_slided_path(name, max_length)
        if not raw.exists():
            print(f"[preprocess] {name}: raw file missing, skipping: {raw}")
            continue
        if slided.exists():
            print(f"[preprocess] {name}: slided cache hit (max_length={max_length}): {slided.name}")
            continue
        print(f"[preprocess] {name}: cache miss for max_length={max_length}, generating: {slided.name}")
        env = os.environ.copy()
        env["INPUT_PARQUET"] = str(raw)
        env["OUTPUT_PARQUET"] = str(slided)
        env["MAX_LENGTH"] = str(max_length)
        env["MODEL_PATH"] = model_path
        env["PROJECT_ROOT"] = str(PROJECT_ROOT)
        result = subprocess.run(
            [sys.executable, str(preprocess_script)],
            env=env, cwd=str(PROJECT_ROOT), stdout=sys.stdout, stderr=sys.stderr,
        )
        if result.returncode != 0:
            raise RuntimeError(f"preprocess failed for {name}, cannot continue without _win_tokens")


def _resolve_source_path(name: str, paths_cfg: Dict[str, Any]) -> Optional[Path]:
    val = paths_cfg.get(name)
    if val is None:
        return None
    candidates = [val] if isinstance(val, (str, Path)) else val
    for p in candidates:
        path = Path(p) if isinstance(p, str) else p
        if path.exists():
            return path
    return None


def _load_source(name: str, path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["_source"] = name
    return df


def _get_token_estimate(df: pd.DataFrame, source_name: str) -> None:
    if df.empty:
        df["_token_estimate"] = pd.Series(dtype=int)
        return
    for col in ("_win_tokens", "_orig_tokens", "token_count", "tokens", "orig_tokens"):
        if col in df.columns and df[col].notna().any():
            df["_token_estimate"] = df[col].fillna(0).astype(int)
            if col == "_win_tokens":
                print(f"  [{source_name}] token estimate: using _win_tokens (per-window actual length)")
            elif col == "_orig_tokens":
                print(f"  [{source_name}] WARNING: _win_tokens not found, falling back to _orig_tokens "
                      "(original conv length, may overestimate). Re-run preprocess to fix.")
            return
    raise ValueError(
        f"[{source_name}] No precomputed token column (_win_tokens, _orig_tokens, token_count, ...). "
        "Run ensure_slided_for_sources(model_path) first."
    )


def _estimate_tokens(row: pd.Series) -> int:
    for col in ("_token_estimate", "_win_tokens", "_orig_tokens", "token_count", "tokens", "orig_tokens"):
        if col in row.index and pd.notna(row.get(col)):
            return int(row[col])
    raise ValueError("No token column; ensure _get_token_estimate was called first.")


def allocate_budget(w: Dict[str, float], T_max: int) -> Dict[str, int]:
    total_w = sum(w.values())
    if total_w <= 0:
        return {k: 0 for k in w}
    budget = {}
    allocated = 0
    keys = list(w.keys())
    for k in keys[:-1]:
        b = int(T_max * w[k] / total_w)
        budget[k] = b
        allocated += b
    budget[keys[-1]] = max(0, T_max - allocated)
    return budget


def _ensure_training_scores_from_raw(
    df: pd.DataFrame,
    source_name: str,
    raw_path: Path,
) -> pd.DataFrame:
    """从 raw parquet 加载训练集预打分列，按 source_id 合并到 slided df。"""
    has_raw_fields = (
        "slice" in df.columns or "score" in df.columns
    ) and "_slice" not in df.columns and "_score" not in df.columns
    if has_raw_fields:
        df = df.copy()
        rename = {"score": "_score", "response_score": "_score", "slice": "_slice"}
        for old, new in rename.items():
            if old in df.columns and new not in df.columns:
                df = df.rename(columns={old: new})
        return df

    if not raw_path.exists():
        return df

    raw = pd.read_parquet(raw_path)
    id_col = "source_id" if "source_id" in df.columns else "id"
    raw_id = "id" if "id" in raw.columns else None

    if raw_id is None:
        raw = raw.copy()
        if "metadata" in raw.columns:
            raw["id"] = raw["metadata"].apply(
                lambda m: m.get("original_id") if isinstance(m, dict) else None
            )
        else:
            raw["id"] = list(range(len(raw)))
        raw_id = "id"

    score_cols = []
    if "score" in raw.columns or "response_score" in raw.columns:
        score_cols.append("score" if "score" in raw.columns else "response_score")
    if "slice" in raw.columns:
        score_cols.append("slice")
    for c in ("_l2", "_pressure", "_concealment", "_safety", "_refusal_type", "_help_level",
              "_friction", "_inst_complexity", "_failed_constraints"):
        if c in raw.columns:
            score_cols.append(c)
    for leak_col in raw.columns:
        if leak_col.startswith("_leak_"):
            score_cols.append(leak_col)

    if not score_cols:
        return df

    merge_cols = [raw_id] + score_cols
    merge_cols = [c for c in merge_cols if c in raw.columns]
    scores_df = raw[merge_cols].drop_duplicates(subset=[raw_id]).copy()
    scores_df[raw_id] = scores_df[raw_id].astype(str)

    left_on = "source_id" if id_col == "source_id" else "id"
    if left_on not in df.columns:
        return df

    df = df.copy()
    df[left_on] = df[left_on].astype(str)

    df = df.merge(
        scores_df,
        left_on=left_on,
        right_on=raw_id,
        how="left",
        suffixes=("", "_raw"),
    )
    if raw_id in df.columns and raw_id != left_on:
        df = df.drop(columns=[raw_id], errors="ignore")

    rename = {"score": "_score", "response_score": "_score", "slice": "_slice"}
    for old, new in rename.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})

    return df


def _detail_to_record(d: dict, source_name: str) -> dict:
    """从 eval detail 提取通用 + source 专用字段。"""
    pc = d.get("prompt_channel", {})
    rc = d.get("response_channel", {})
    l2_prompt = pc.get("l2", {}) or {}
    l2_resp = rc.get("l2", {}) or {}
    l3 = rc.get("l3", {}) or {}
    diag = rc.get("diag", {}) or {}

    rec = {
        "id": str(d.get("id", "")),
        "_slice": pc.get("slice", "unknown"),
        "_score": float(rc.get("score", 0)),
    }

    if source_name == "xguard":
        rec["_safety"] = l2_resp.get("safety", "unknown")
        rec["_pressure"] = l2_prompt.get("pressure", 0)
        rec["_concealment"] = l2_prompt.get("concealment", 0)
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


def _load_eval_results_from_json(eval_results_path: Path, source_name: str) -> Optional[pd.DataFrame]:
    if not eval_results_path.exists():
        return None
    try:
        with open(eval_results_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None

    bench = data.get(source_name, {})
    details = bench.get("details", [])
    if not details:
        return None

    records = [_detail_to_record(d, source_name) for d in details]
    eval_df = pd.DataFrame(records)

    if source_name == "ifeval" and source_name in RAW_TO_SLIDED_BASE:
        raw_path = RAW_TO_SLIDED_BASE[source_name][0]
        if raw_path.exists():
            raw_df = pd.read_parquet(raw_path)
            if "id" in raw_df.columns and len(raw_df) >= len(eval_df):
                def _remap_id(eval_id: str) -> str:
                    try:
                        idx = int(eval_id)
                        if 0 <= idx < len(raw_df):
                            return str(raw_df.iloc[idx]["id"])
                    except (ValueError, TypeError):
                        pass
                    return eval_id

                numeric_count = sum(1 for x in eval_df["id"] if str(x).isdigit())
                if numeric_count == len(eval_df):
                    eval_df = eval_df.copy()
                    eval_df["id"] = eval_df["id"].apply(_remap_id)

    return eval_df


def _load_eval_results_legacy(source_name: str) -> Optional[pd.DataFrame]:
    res_path = PROJECT_ROOT / "results" / source_name / "eval_results.jsonl"
    if not res_path.exists():
        return None

    records = []
    with open(res_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            records.append(_detail_to_record(d, source_name))
    if not records:
        return None

    eval_df = pd.DataFrame(records)

    if source_name == "ifeval" and source_name in RAW_TO_SLIDED_BASE:
        raw_path = RAW_TO_SLIDED_BASE[source_name][0]
        if raw_path.exists():
            raw_df = pd.read_parquet(raw_path)
            if "id" in raw_df.columns and len(raw_df) >= len(eval_df):
                def _remap_id(eval_id: str) -> str:
                    try:
                        idx = int(eval_id)
                        if 0 <= idx < len(raw_df):
                            return str(raw_df.iloc[idx]["id"])
                    except (ValueError, TypeError):
                        pass
                    return eval_id

                numeric_count = sum(1 for x in eval_df["id"] if str(x).isdigit())
                if numeric_count == len(eval_df):
                    eval_df = eval_df.copy()
                    eval_df["id"] = eval_df["id"].apply(_remap_id)

    return eval_df


def _load_eval_results(
    source_name: str,
    eval_results_path: Optional[Path] = None,
) -> Optional[pd.DataFrame]:
    if eval_results_path is not None:
        df = _load_eval_results_from_json(eval_results_path, source_name)
        if df is not None:
            return df
    return _load_eval_results_legacy(source_name)


# ---------------------------------------------------------------------------
# Focus criteria matching (include/exclude format)
# ---------------------------------------------------------------------------


def _parse_focus_patterns(patterns: list[str]) -> list[tuple[int, int]]:
    """解析 P/C 模式，如 ["P4_C3", "P5_C3"] -> [(4,3), (5,3)]"""
    result = []
    for pattern in patterns:
        try:
            p_part, c_part = pattern.split("_")
            result.append((int(p_part[1:]), int(c_part[1:])))
        except Exception:
            continue
    return result


def _matches_focus_include(
    row: pd.Series,
    include: Dict[str, Any],
    source_name: str,
) -> bool:
    """检查行是否符合 include 条件中的任一项。"""
    if not include:
        return False

    if source_name == "xguard":
        patterns = include.get("pc_patterns", [])
        if patterns:
            tuples = _parse_focus_patterns(patterns)
            p = row.get("_pressure", 0)
            c = row.get("_concealment", 0)
            if (p, c) in tuples:
                return True

        safety_vals = include.get("safety", [])
        if safety_vals and row.get("_safety", "unknown") in safety_vals:
            return True

        leak_types = include.get("leak_types", [])
        for lt in leak_types:
            if row.get(f"_leak_{lt}") == 1:
                return True

        pr = include.get("pressure_range")
        if pr and len(pr) >= 2:
            if pr[0] <= row.get("_pressure", 0) <= pr[1]:
                return True

        cr = include.get("concealment_range")
        if cr and len(cr) >= 2:
            if cr[0] <= row.get("_concealment", 0) <= cr[1]:
                return True

    elif source_name == "orbench":
        for key, col in [("refusal_type", "_refusal_type"), ("help_level", "_help_level"), ("friction", "_friction")]:
            vals = include.get(key, [])
            if vals and row.get(col, "unknown") in vals:
                return True

    elif source_name == "ifeval":
        ic_vals = include.get("inst_complexity", [])
        if ic_vals and row.get("_inst_complexity", 0) in ic_vals:
            return True

        fc_vals = include.get("failed_constraint", [])
        if fc_vals:
            failed = row.get("_failed_constraints", [])
            if isinstance(failed, list) and any(fc in failed for fc in fc_vals):
                return True

    return False


def _matches_focus_criteria(
    row: pd.Series,
    focus_criteria: Dict[str, Any],
    source_name: str,
) -> bool:
    """检查行是否符合 focus_criteria。支持新格式 (include/exclude) 和旧格式 (focus_on 等)。"""
    if not focus_criteria:
        return False

    # 新格式：include/exclude
    include = focus_criteria.get("include")
    if include is not None:
        return _matches_focus_include(row, include, source_name)

    # 旧格式兼容：直接把 focus_criteria 当 include 用
    return _matches_focus_include(row, _convert_legacy_focus(focus_criteria), source_name)


def _convert_legacy_focus(legacy: Dict[str, Any]) -> Dict[str, Any]:
    """将旧格式 focus_criteria (focus_on, focus_safety, ...) 转为新 include 格式。"""
    include: Dict[str, Any] = {}
    if legacy.get("focus_on"):
        include["pc_patterns"] = legacy["focus_on"]
    if legacy.get("focus_safety"):
        include["safety"] = legacy["focus_safety"]
    if legacy.get("focus_leak_types"):
        include["leak_types"] = legacy["focus_leak_types"]
    if legacy.get("focus_pressure_range"):
        include["pressure_range"] = legacy["focus_pressure_range"]
    if legacy.get("focus_concealment_range"):
        include["concealment_range"] = legacy["focus_concealment_range"]
    if legacy.get("focus_refusal_type"):
        include["refusal_type"] = legacy["focus_refusal_type"]
    if legacy.get("focus_help_level"):
        include["help_level"] = legacy["focus_help_level"]
    if legacy.get("focus_friction"):
        include["friction"] = legacy["focus_friction"]
    if legacy.get("focus_inst_complexity"):
        include["inst_complexity"] = legacy["focus_inst_complexity"]
    if legacy.get("focus_failed_constraint"):
        include["failed_constraint"] = legacy["focus_failed_constraint"]
    return include


# ---------------------------------------------------------------------------
# Token-constrained sampling
# ---------------------------------------------------------------------------


def sample_by_tokens(
    df: pd.DataFrame,
    token_budget: int,
    seed: int = 42,
    focus_criteria: Optional[Dict[str, Any]] = None,
    focus_ratio: float = 0.5,
    source_name: str = "xguard",
) -> pd.DataFrame:
    """Sample rows from df without exceeding token_budget.

    focus 样本优先选取（占 focus_ratio 的预算），剩余预算随机填充。
    """
    if df.empty or token_budget <= 0:
        return df.iloc[:0]

    has_focus = bool(focus_criteria)

    if "_token_estimate" not in df.columns:
        tokens = df.apply(_estimate_tokens, axis=1).values
        df = df.copy()
        df["_token_estimate"] = tokens
    else:
        df = df.copy()

    # 标记 focus 样本
    if has_focus:
        df["_is_focus"] = df.apply(
            lambda row: _matches_focus_criteria(row, focus_criteria, source_name), axis=1
        )
    else:
        df["_is_focus"] = False

    focus_df = df[df["_is_focus"]]
    non_focus_df = df[~df["_is_focus"]]

    focus_budget = int(token_budget * focus_ratio) if has_focus else 0
    rng = random.Random(seed)

    selected_indices = []
    used_tokens = 0

    # 阶段 1：优先选择 focus 样本
    if focus_budget > 0 and len(focus_df) > 0:
        focus_indices = focus_df.index.tolist()
        rng.shuffle(focus_indices)

        for idx in focus_indices:
            t = focus_df.loc[idx, "_token_estimate"]
            if used_tokens + t > focus_budget:
                continue
            selected_indices.append(idx)
            used_tokens += t

        if selected_indices:
            print(f"    [FOCUS] 已选择 {len(selected_indices)} 个 focus 样本 (~{used_tokens:,} tokens)")

    # 阶段 2：用非 focus 样本填充剩余预算
    remaining_budget = token_budget - used_tokens
    if remaining_budget > 0 and len(non_focus_df) > 0:
        non_focus_indices = non_focus_df.index.tolist()
        rng.shuffle(non_focus_indices)

        for idx in non_focus_indices:
            t = non_focus_df.loc[idx, "_token_estimate"]
            if used_tokens + t > token_budget:
                continue
            selected_indices.append(idx)
            used_tokens += t

    # 阶段 3：如果还有余量，从所有剩余样本补
    remaining_budget = token_budget - used_tokens
    if remaining_budget > 50 and len(selected_indices) < len(df):
        all_remaining = df[~df.index.isin(selected_indices)]
        if len(all_remaining) > 0:
            remaining_indices = all_remaining.index.tolist()
            rng.shuffle(remaining_indices)
            for idx in remaining_indices:
                t = df.loc[idx, "_token_estimate"]
                if used_tokens + t > token_budget:
                    continue
                selected_indices.append(idx)
                used_tokens += t

    result = df.loc[selected_indices].copy()
    result = result.drop(columns=["_is_focus"], errors="ignore")
    return result


# ---------------------------------------------------------------------------
# Main: construct_training_set
# ---------------------------------------------------------------------------


def construct_training_set(
    w: Dict[str, float],
    T_max: int = 1_000_000,
    max_length: int = 4096,
    source_paths: Optional[Dict[str, Path]] = None,
    quality_threshold: float = 0.0,
    # --- 新接口：per-dimension plan ---
    dims: Optional[Dict[str, dict]] = None,
    # --- 旧接口（自动转为 dims） ---
    sampling_mode: str = "uniform",
    bucket_weights: Optional[Dict[str, Dict[str, float]]] = None,
    focus_criteria: Optional[Dict[str, Any]] = None,
    # --- 通用 ---
    seed: int = 42,
    output_path: Optional[Path] = None,
    dry_run: bool = False,
    model_path: Optional[str] = None,
    eval_results_path: Optional[Path] = None,
) -> Path:
    """
    Construct a training dataset from the distribution vector.

    新接口使用 dims (per-dimension DimPlan)。旧接口 bucket_weights/focus_criteria 自动转换。
    """
    # 旧接口 → 新接口兼容
    if dims is None and (bucket_weights or focus_criteria):
        dims = {}
        legacy_focus = focus_criteria or {}
        for source_name in w:
            dim_plan: dict = {}
            if bucket_weights and source_name in bucket_weights:
                dim_plan["bucket_weights"] = bucket_weights[source_name]
            if legacy_focus:
                dim_plan["focus_criteria"] = {"include": _convert_legacy_focus(legacy_focus), "exclude": {}}
            if dim_plan:
                dims[source_name] = dim_plan
    dims = dims or {}

    paths_cfg = source_paths if source_paths is not None else get_default_source_paths(max_length)
    budget = allocate_budget(w, T_max)
    model_path = model_path or os.environ.get("MODEL_PATH", "/model/llm/Meta-Llama-3.1-8B-Instruct")
    ensure_slided_for_sources(model_path, max_length, sources=list(budget.keys()))

    print(f"[data_constructor] Q-Thresh={quality_threshold}")
    print(f"[data_constructor] T_max={T_max:,}")
    print(f"[data_constructor] distribution: {w}")
    print(f"[data_constructor] budget allocation: {budget}")

    all_samples = []
    stats = {}

    for source_name, token_budget in budget.items():
        path = _resolve_source_path(source_name, paths_cfg)
        if path is None:
            print(f"  [WARN] {source_name} data not found, skipping")
            continue

        df = _load_source(source_name, path)
        is_slided = "_win_tokens" in df.columns or "_orig_tokens" in df.columns
        if is_slided:
            has_win = "_win_tokens" in df.columns
            print(f"  [{source_name}] using slided data (has _win_tokens={has_win}): {path.name}")

        raw_path = RAW_TO_SLIDED_BASE[source_name][0]
        df = _ensure_training_scores_from_raw(df, source_name, raw_path)

        has_slice_from_raw = "_slice" in df.columns and (df["_slice"].fillna("") != "").any()
        if not has_slice_from_raw:
            eval_df = _load_eval_results(source_name, eval_results_path)
            if eval_df is not None:
                score_cols = [c for c in eval_df.columns if c != "id"]
                scores_sub = eval_df[["id"] + score_cols].drop_duplicates(subset=["id"])
                left_on = "source_id" if "source_id" in df.columns else "id"
                df = df.merge(scores_sub, left_on=left_on, right_on="id", how="left")
                df = df.drop(columns=["id"], errors="ignore")
                print(f"  [{source_name}] merged scores from eval_results (fallback)")

        if "_score" not in df.columns:
            df["_score"] = 5.0
        df["_score"] = df["_score"].fillna(5.0)
        if "_slice" not in df.columns:
            df["_slice"] = "unknown"
        df["_slice"] = df["_slice"].fillna("unknown")
        if "_pressure" in df.columns:
            df["_pressure"] = df["_pressure"].fillna(0)
        if "_concealment" in df.columns:
            df["_concealment"] = df["_concealment"].fillna(0)
        if "_safety" in df.columns:
            df["_safety"] = df["_safety"].fillna("unknown")
        if "_refusal_type" in df.columns:
            df["_refusal_type"] = df["_refusal_type"].fillna("unknown")
        if "_help_level" in df.columns:
            df["_help_level"] = df["_help_level"].fillna("unknown")
        if "_friction" in df.columns:
            df["_friction"] = df["_friction"].fillna("unknown")
        if "_inst_complexity" in df.columns:
            df["_inst_complexity"] = df["_inst_complexity"].fillna(0)
        if "_failed_constraints" in df.columns:
            df["_failed_constraints"] = df["_failed_constraints"].apply(
                lambda x: x if isinstance(x, list) else []
            )

        if quality_threshold > 0:
            num_before = len(df)
            df = df[df["_score"] >= quality_threshold]
            print(f"  [{source_name}] score >= {quality_threshold}: {num_before} -> {len(df)} rows")

        _get_token_estimate(df, source_name)

        # 从 parquet _l2 补充 pressure/concealment
        if "_pressure" not in df.columns and "_l2" in df.columns:
            def parse_l2(l2_val):
                try:
                    if isinstance(l2_val, str):
                        l2 = json.loads(l2_val)
                    else:
                        l2 = l2_val
                    return l2.get("pressure", 0), l2.get("concealment", 0)
                except:
                    return 0, 0
            df[["_pressure", "_concealment"]] = df["_l2"].apply(lambda x: pd.Series(parse_l2(x)))

        # 读取当前维度的 DimPlan
        dim_plan = dims.get(source_name, {})
        bw = dim_plan.get("bucket_weights", {})
        source_focus = dim_plan.get("focus_criteria", {})

        if source_focus:
            include = source_focus.get("include", source_focus)
            if include:
                print(f"  [{source_name}] Focus include: {include}")
            focus_mask = df.apply(
                lambda row: _matches_focus_criteria(row, source_focus, source_name), axis=1
            )
            focus_count = focus_mask.sum()
            print(f"  [{source_name}] Focus criteria: {focus_count} matching samples")

        # 按桶采样
        if bw:
            buckets_with_data = {b: w_val for b, w_val in bw.items() if len(df[df["_slice"] == b]) > 0}
            total_b_weight = sum(buckets_with_data.values()) if buckets_with_data else 0

            if buckets_with_data:
                sampled_dfs = []
                for b_name, b_w in buckets_with_data.items():
                    b_budget = int(token_budget * (b_w / total_b_weight)) if total_b_weight > 0 else 0
                    b_df = df[df["_slice"] == b_name]
                    sampled_b = sample_by_tokens(
                        b_df, b_budget, seed=seed,
                        focus_criteria=source_focus, source_name=source_name,
                    )
                    sampled_dfs.append(sampled_b)
                    print(f"    - bucket '{b_name}': budget {b_budget:,}, selected {len(sampled_b)} rows")
                sampled = pd.concat(sampled_dfs, ignore_index=True)
            else:
                for b_name in bw:
                    print(f"    - bucket '{b_name}': skipped (no data)")
                slice_vals = df["_slice"].dropna().unique().tolist()
                if slice_vals:
                    print(f"    [诊断] 实际 _slice 取值: {slice_vals[:10]}{'...' if len(slice_vals) > 10 else ''}")
                sampled = sample_by_tokens(
                    df, token_budget, seed=seed,
                    focus_criteria=source_focus, source_name=source_name,
                )
        else:
            sampled = sample_by_tokens(
                df, token_budget, seed=seed,
                focus_criteria=source_focus, source_name=source_name,
            )

        if source_focus:
            focus_mask = sampled.apply(
                lambda row: _matches_focus_criteria(row, source_focus, source_name), axis=1
            )
            focus_count = int(focus_mask.sum())
            print(f"    [FOCUS] 最终选中 {focus_count} 个符合 L2/L3 条件的样本")

        total_tokens = int(sampled["_token_estimate"].sum()) if len(sampled) > 0 else 0
        print(f"  [{source_name}] FINISHED: selected {len(sampled)} rows, ~{total_tokens:,} tokens")

        stats[source_name] = {
            "selected_rows": len(sampled),
            "budget_tokens": token_budget,
            "estimated_tokens": total_tokens,
        }
        all_samples.append(sampled)

    if not all_samples:
        print("[data_constructor] ERROR: no samples collected")
        return Path("/dev/null")

    combined = pd.concat(all_samples, ignore_index=True)
    combined = combined.sample(frac=1, random_state=seed).reset_index(drop=True)
    total_tokens = int(combined["_token_estimate"].sum())
    print(f"\n[data_constructor] TOTAL: {len(combined)} rows, ~{total_tokens:,} tokens")

    if dry_run:
        print("[data_constructor] dry_run=True, not saving")
        return Path("/dev/null")

    if output_path is None:
        output_path = PROJECT_ROOT / "packages" / "finetune" / "data" / "train_constructed.parquet"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if "_n_windows" in combined.columns:
        combined["n_windows_total"] = combined["_n_windows"].astype(int).clip(lower=1)
    elif "source_id" in combined.columns and "window_id" in combined.columns:
        g = combined.groupby("source_id")["window_id"].transform("count")
        combined["n_windows_total"] = g.astype(int).clip(lower=1)
    save_cols = [c for c in combined.columns if not c.startswith("_") and c != "id_str"]
    combined[save_cols].to_parquet(output_path, index=False)

    stats_path = output_path.with_suffix(".stats.json")
    with open(stats_path, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2, ensure_ascii=False)

    print(f"[data_constructor] saved: {output_path}")
    return output_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Construct training dataset")
    parser.add_argument("--w", type=str, default="1.0,0.0,0.0", help="Distribution weights (xguard,orbench,ifeval)")
    parser.add_argument("--T_max", type=int, default=1_000_000, help="Total token budget")
    parser.add_argument("--max-length", type=int, default=4096, help="Slided window max_length (cache key)")
    parser.add_argument("--quality-threshold", type=float, default=4.0)
    parser.add_argument("--dims", type=str, default=None,
                        help="JSON string of per-dimension plans: {\"xguard\": {\"bucket_weights\": {...}, \"focus_criteria\": {...}}}")
    # Legacy args (converted to dims internally)
    parser.add_argument("--sampling-mode", type=str, choices=["uniform", "need-driven"], default="uniform",
                        help="(legacy, ignored) Sampling strategy")
    parser.add_argument("--bucket-weights", type=str, default=None, help="(legacy) JSON string of bucket weights")
    parser.add_argument("--focus-criteria", type=str, default=None, help="(legacy) JSON dict of focus criteria")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--eval-results-path", type=str, default=None)
    args = parser.parse_args()

    weights = [float(x) for x in args.w.split(",")]
    w = {"xguard": weights[0], "orbench": weights[1], "ifeval": weights[2]}

    dims_arg = json.loads(args.dims) if args.dims else None
    bw = json.loads(args.bucket_weights) if args.bucket_weights else None
    fc = json.loads(args.focus_criteria) if args.focus_criteria else None
    eval_rp = Path(args.eval_results_path) if args.eval_results_path else None
    out_pth = Path(args.output) if args.output else None

    construct_training_set(
        w=w,
        T_max=args.T_max,
        max_length=args.max_length,
        quality_threshold=args.quality_threshold,
        dims=dims_arg,
        bucket_weights=bw,
        focus_criteria=fc,
        seed=args.seed,
        output_path=out_pth,
        dry_run=args.dry_run,
        model_path=args.model_path,
        eval_results_path=eval_rp,
    )
