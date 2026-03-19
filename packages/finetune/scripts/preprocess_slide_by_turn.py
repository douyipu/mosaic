#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Turn-aware sliding window slicing for long multi-turn chats.
- Only slice samples whose full length > MAX_LENGTH
- Slide by USER TURN stride (TURN_STRIDE, default=2)
- Window length constrained by tokens (<= MAX_LENGTH)
- Slice at message boundaries, keep all system messages in every window
- Parallel + progress bar via datasets.map(num_proc=...)

Input parquet schema:
  - id (optional)
  - messages (required): list[{"role":..., "content":...}, ...]

Output parquet schema:
  - source_id
  - window_id
  - messages
"""

import os
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer


# -------------------------
# Env config（未设 PROJECT_ROOT 时从脚本位置推导）
# -------------------------
PROJECT_ROOT = Path(
    os.environ.get("PROJECT_ROOT", str(Path(__file__).resolve().parent.parent))
).resolve()
DATA_DIR = Path(os.environ.get("DATA_DIR", str(PROJECT_ROOT / "data"))).resolve()
OUT_DIR = Path(os.environ.get("OUT_DIR", str(DATA_DIR))).resolve()

MODEL_PATH = os.environ.get("MODEL_PATH", "/model/llm/Meta-Llama-3.1-8B-Instruct")

INPUT_PARQUET = Path(os.environ.get("INPUT_PARQUET", str(DATA_DIR / "xguard-train.parquet"))).resolve()
OUTPUT_PARQUET = Path(os.environ.get("OUTPUT_PARQUET", str(DATA_DIR / "xguard-train-slided.parquet"))).resolve()

MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "8192"))
TURN_STRIDE = int(os.environ.get("TURN_STRIDE", "2"))
NUM_PROC = int(os.environ.get("NUM_PROC", "8"))  # parallelism
BATCH_SIZE = int(os.environ.get("MAP_BATCH_SIZE", "64"))  # datasets.map batch size
LIMIT_ROWS = int(os.environ.get("LIMIT_ROWS", "0"))  # 0 = all

# -------------------------
# Per-process tokenizer cache (important!)
# -------------------------
_TOKENIZER = None


def get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        _TOKENIZER = tok
    return _TOKENIZER


def ensure_list(messages: Any) -> List[Dict[str, Any]]:
    if hasattr(messages, "tolist"):
        messages = messages.tolist()
    return messages if isinstance(messages, list) else []


def split_by_user_turn(messages: List[Dict[str, Any]]):
    """
    Split messages into:
      system_msgs: list[system msg]
      turns: list[list[msg]] where each turn starts with a user msg, followed by assistant/tool/etc until next user.
    """
    system_msgs = [m for m in messages if m.get("role") == "system"]

    turns = []
    current = []
    for m in messages:
        role = m.get("role")
        if role == "system":
            continue
        if role == "user":
            if current:
                turns.append(current)
            current = [m]
        else:
            current.append(m)
    if current:
        turns.append(current)
    return system_msgs, turns


def count_tokens(tok, messages: List[Dict[str, Any]]) -> int:
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    eos = tok.eos_token
    if eos and not text.endswith(eos):
        text += eos
    ids = tok(text, truncation=False, add_special_tokens=False)["input_ids"]
    return len(ids)


def slice_one(messages: List[Dict[str, Any]], source_id: Any):
    """
    Return list of windows: [{"source_id":..., "window_id":k, "messages":..., "win_tokens":int}, ...]
    Each window carries its own token count (win_tokens) for accurate budget estimation downstream.
    """
    tok = get_tokenizer()
    messages = ensure_list(messages)

    total_tokens = count_tokens(tok, messages)
    if total_tokens <= MAX_LENGTH:
        return [{
            "source_id": source_id,
            "window_id": 0,
            "messages": messages,
            "win_tokens": total_tokens,
        }], {
            "was_sliced": 0,
            "orig_tokens": total_tokens,
            "n_windows": 1
        }

    system_msgs, turns = split_by_user_turn(messages)
    n_turns = len(turns)

    windows = []
    start = 0
    win_id = 0

    while start < n_turns:
        window_msgs = list(system_msgs)
        for t in turns[start:]:
            window_msgs.extend(t)
            if count_tokens(tok, window_msgs) > MAX_LENGTH:
                window_msgs = window_msgs[:-len(t)]
                break

        if len(window_msgs) <= len(system_msgs):
            window_msgs = list(system_msgs) + list(turns[start])

        windows.append({
            "source_id": source_id,
            "window_id": win_id,
            "messages": window_msgs,
            "win_tokens": count_tokens(tok, window_msgs),
        })
        win_id += 1
        start += TURN_STRIDE

    return windows, {
        "was_sliced": 1,
        "orig_tokens": total_tokens,
        "n_windows": len(windows)
    }


def map_fn(batch):
    """
    batched map: returns expanded rows (list-of-lists -> flattened)
    We also emit per-window metadata for later aggregation.
    """
    out_source_id = []
    out_window_id = []
    out_messages = []
    out_win_tokens = []

    out_was_sliced = []
    out_orig_tokens = []
    out_n_windows = []

    ids = batch.get("id", None)
    msgs_list = batch["messages"]

    for i, msgs in enumerate(msgs_list):
        source_id = ids[i] if ids is not None else batch["_row_id"][i]
        windows, meta = slice_one(msgs, source_id)

        for w in windows:
            out_source_id.append(w["source_id"])
            out_window_id.append(w["window_id"])
            out_messages.append(w["messages"])
            out_win_tokens.append(w["win_tokens"])

            out_was_sliced.append(meta["was_sliced"])
            out_orig_tokens.append(meta["orig_tokens"])
            out_n_windows.append(meta["n_windows"])

    return {
        "source_id": out_source_id,
        "window_id": out_window_id,
        "messages": out_messages,
        "_win_tokens": out_win_tokens,
        "_was_sliced": out_was_sliced,
        "_orig_tokens": out_orig_tokens,
        "_n_windows": out_n_windows,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[cfg] INPUT={INPUT_PARQUET}")
    print(f"[cfg] OUTPUT={OUTPUT_PARQUET}")
    print(f"[cfg] MAX_LENGTH={MAX_LENGTH} TURN_STRIDE={TURN_STRIDE} NUM_PROC={NUM_PROC} BATCH={BATCH_SIZE}")

    df = pd.read_parquet(INPUT_PARQUET)
    if LIMIT_ROWS and LIMIT_ROWS > 0:
        df = df.head(LIMIT_ROWS).copy()

    if "messages" not in df.columns:
        raise ValueError("Input parquet must contain column 'messages'")

    # ensure an id column exists (helps traceability)
    # Prefer metadata.original_id (e.g., xguard) over row index for stable IDs
    if "id" not in df.columns:
        if "metadata" in df.columns:
            df["id"] = df["metadata"].apply(
                lambda m: m.get("original_id") if isinstance(m, dict) else None
            )
        else:
            df["id"] = list(range(len(df)))
    # Fallback: if id is still null/empty, use row index
    if df["id"].isna().any() or (df["id"] == "").any():
        df["id"] = df["id"].fillna(pd.Series(range(len(df)), index=df.index).astype(str))

    # Keep original id-to-metadata mapping for later merge
    # Identify columns to preserve (all except messages which gets transformed)
    preserve_cols = [c for c in df.columns if c != "messages"]
    id_to_meta = df[preserve_cols].copy()
    id_to_meta["id"] = id_to_meta["id"].astype(str)

    ds = Dataset.from_pandas(df[["id", "messages"]], preserve_index=False)
    ds = ds.add_column("_row_id", list(range(len(ds))))  # stable fallback id

    # Map with multiprocessing (shows progress bar)
    out = ds.map(
        map_fn,
        batched=True,
        batch_size=BATCH_SIZE,
        num_proc=NUM_PROC,
        remove_columns=ds.column_names,  # keep only returned fields
        desc="Sliding windows (turn-stride)",
    )

    # Merge preserved columns from original data (using source_id = original id)
    odf = out.to_pandas()
    odf["source_id"] = odf["source_id"].astype(str)
    id_to_meta_renamed = id_to_meta.rename(columns={"id": "source_id"})
    # Avoid duplicate columns that map_fn already outputs
    cols_to_merge = [c for c in id_to_meta_renamed.columns
                     if c not in odf.columns or c == "source_id"]
    if len(cols_to_merge) > 1:  # more than just source_id
        odf = odf.merge(id_to_meta_renamed[cols_to_merge], on="source_id", how="left")

    # 丢弃 slice 为空的行（训练不需要无标签样本）
    before_drop = len(odf)
    if "slice" in odf.columns:
        odf = odf[odf["slice"].notna() & (odf["slice"] != "")]
    dropped = before_drop - len(odf)
    if dropped > 0:
        print(f"[filter] 丢弃 {dropped} 条 slice 为空的样本")

    # Save parquet
    odf.to_parquet(str(OUTPUT_PARQUET), index=False)
    print(f"[save] wrote: {OUTPUT_PARQUET} (rows={len(out)})")

    # Summarize stats from emitted meta columns
    meta = odf.groupby("source_id", as_index=False).first()[["_was_sliced", "_orig_tokens", "_n_windows"]]

    stats = {
        "input_rows": int(len(ds)),
        "output_rows": int(len(out)),
        "expansion_factor": float(len(out) / max(1, len(ds))),
        "sliced_fraction": float(meta["_was_sliced"].mean()),
        "orig_tokens_mean": float(meta["_orig_tokens"].mean()),
        "orig_tokens_p90": float(meta["_orig_tokens"].quantile(0.90)),
        "orig_tokens_p95": float(meta["_orig_tokens"].quantile(0.95)),
        "orig_tokens_max": int(meta["_orig_tokens"].max()),
        "win_tokens_mean": float(odf["_win_tokens"].mean()),
        "win_tokens_p90": float(odf["_win_tokens"].quantile(0.90)),
        "win_tokens_p95": float(odf["_win_tokens"].quantile(0.95)),
        "win_tokens_max": int(odf["_win_tokens"].max()),
        "windows_per_sample_mean": float(meta["_n_windows"].mean()),
        "windows_per_sliced_mean": float(meta[meta["_was_sliced"] == 1]["_n_windows"].mean()) if (meta["_was_sliced"] == 1).any() else 1.0,
        "windows_per_sample_max": int(meta["_n_windows"].max()),
    }
    print("\n" + "=" * 80)
    print("[stats]")
    print("=" * 80)
    print(json.dumps(stats, ensure_ascii=False, indent=2))

    stats_path = OUTPUT_PARQUET.with_suffix(".stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"[save] stats: {stats_path}")


if __name__ == "__main__":
    main()
