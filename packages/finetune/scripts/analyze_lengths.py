#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyze multi-turn chat length stats:
- turn counts (user/assistant turns)
- token lengths (using tokenizer.apply_chat_template + tokenization)
- joint stats: turns x tokens, long-tail rates, percentiles

Inputs:
  - XGUARD_PARQUET: parquet with columns ["id", "messages"]
  - ORBENCH_PARQUET: same format
Model/tokenizer:
  - MODEL_PATH: local model dir (hf format), used only for tokenizer

Outputs:
  - prints summary to stdout
  - writes CSV: outputs/length_stats_<timestamp>.csv
  - writes JSON: outputs/length_summary_<timestamp>.json
  - optional histogram png if matplotlib installed
"""

import os
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

# Optional plots
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None

from transformers import AutoTokenizer


# 未设 PROJECT_ROOT 时从脚本位置推导
PROJECT_ROOT = Path(
    os.environ.get("PROJECT_ROOT", str(Path(__file__).resolve().parent.parent))
).resolve()
DATA_DIR = Path(os.environ.get("DATA_DIR", str(PROJECT_ROOT / "data"))).resolve()
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", str(PROJECT_ROOT / "outputs"))).resolve()

MODEL_PATH = os.environ.get("MODEL_PATH", "/model/llm/Meta-Llama-3.1-8B-Instruct")
XGUARD_PARQUET = Path(os.environ.get("XGUARD_PARQUET", str(DATA_DIR / "xguard-train.parquet"))).resolve()
ORBENCH_PARQUET = Path(os.environ.get("ORBENCH_PARQUET", str(DATA_DIR / "orbench.parquet"))).resolve()

# Performance knobs
MAX_ROWS_PER_SOURCE = int(os.environ.get("MAX_ROWS_PER_SOURCE", "0"))  # 0 = all
TOKENIZE_BATCH_SIZE = int(os.environ.get("TOKENIZE_BATCH_SIZE", "256"))
NUM_SAMPLES_FOR_TEXT_CHECK = int(os.environ.get("NUM_SAMPLES_FOR_TEXT_CHECK", "3"))


def _count_turns(messages: List[Dict[str, Any]]) -> Tuple[int, int, int]:
    """Return (num_messages, num_user, num_assistant)."""
    n = len(messages)
    u = sum(1 for m in messages if (m.get("role") == "user"))
    a = sum(1 for m in messages if (m.get("role") == "assistant"))
    return n, u, a


def _ensure_list(messages: Any) -> List[Dict[str, Any]]:
    if hasattr(messages, "tolist"):
        messages = messages.tolist()
    if not isinstance(messages, list):
        return []
    return messages


def _quantiles(series: pd.Series, qs=(0.5, 0.9, 0.95, 0.99)) -> Dict[str, float]:
    out = {}
    for q in qs:
        out[f"p{int(q*100)}"] = float(series.quantile(q))
    return out


def _bucket_turns(u_turns: int) -> str:
    if u_turns <= 1:
        return "1"
    if u_turns == 2:
        return "2"
    if u_turns == 3:
        return "3"
    if 4 <= u_turns <= 5:
        return "4-5"
    if 6 <= u_turns <= 8:
        return "6-8"
    return "9+"


def _bucket_tokens(tok: int) -> str:
    # buckets aligned to common context sizes
    if tok <= 512:
        return "<=512"
    if tok <= 1024:
        return "513-1024"
    if tok <= 2048:
        return "1025-2048"
    if tok <= 4096:
        return "2049-4096"
    if tok <= 6144:
        return "4097-6144"
    if tok <= 8192:
        return "6145-8192"
    return ">8192"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load data
    frames = []
    for name, path in [("xguard", XGUARD_PARQUET), ("orbench", ORBENCH_PARQUET)]:
        if not path.exists():
            raise FileNotFoundError(f"Missing parquet: {path}")
        df = pd.read_parquet(path)
        if "messages" not in df.columns:
            raise ValueError(f"{path} missing column 'messages'")
        if MAX_ROWS_PER_SOURCE and MAX_ROWS_PER_SOURCE > 0:
            df = df.head(MAX_ROWS_PER_SOURCE)
        df = df.copy()
        df["source"] = name
        frames.append(df[["source", "messages"]])

    df = pd.concat(frames, ignore_index=True)
    print(f"[load] total rows = {len(df)} (xguard+orbench)")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Basic turn stats first
    stats_rows = []
    texts_for_sanity = []
    for idx, raw in enumerate(df["messages"].tolist()):
        msgs = _ensure_list(raw)
        n_msg, n_user, n_asst = _count_turns(msgs)
        # "turns" here: number of user turns (common for multi-turn attack trajectories)
        stats_rows.append((n_msg, n_user, n_asst, _bucket_turns(n_user)))

        if len(texts_for_sanity) < NUM_SAMPLES_FOR_TEXT_CHECK:
            try:
                t = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
                texts_for_sanity.append(t[:400].replace("\n", "\\n"))
            except Exception:
                pass

    df["n_messages"] = [x[0] for x in stats_rows]
    df["n_user"] = [x[1] for x in stats_rows]
    df["n_assistant"] = [x[2] for x in stats_rows]
    df["turn_bucket"] = [x[3] for x in stats_rows]

    if texts_for_sanity:
        print("\n[sanity] chat_template preview (first 400 chars):")
        for i, t in enumerate(texts_for_sanity, 1):
            print(f"  sample#{i}: {t}")

    # Token lengths (batched)
    # We compute full token length WITHOUT truncation to see the true distribution.
    token_lens = []
    token_buckets = []
    total = len(df)

    for start in range(0, total, TOKENIZE_BATCH_SIZE):
        end = min(total, start + TOKENIZE_BATCH_SIZE)
        batch_msgs = df["messages"].iloc[start:end].tolist()

        texts = []
        for raw in batch_msgs:
            msgs = _ensure_list(raw)
            text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            # add eos if needed
            eos = tokenizer.eos_token
            if eos and not text.endswith(eos):
                text = text + eos
            texts.append(text)

        enc = tokenizer(
            texts,
            truncation=False,  # IMPORTANT: no truncation for analysis
            add_special_tokens=False,
            padding=False,
            return_attention_mask=False,
        )
        lens = [len(ids) for ids in enc["input_ids"]]
        token_lens.extend(lens)
        token_buckets.extend([_bucket_tokens(x) for x in lens])

    df["tokens"] = token_lens
    df["token_bucket"] = token_buckets

    # Summaries
    def summarize(label: str, sub: pd.DataFrame) -> Dict[str, Any]:
        s = sub["tokens"]
        out = {
            "label": label,
            "rows": int(len(sub)),
            "tokens_mean": float(s.mean()),
            "tokens_min": int(s.min()),
            "tokens_max": int(s.max()),
            **_quantiles(s, qs=(0.5, 0.9, 0.95, 0.99)),
            "over_2048": float((s > 2048).mean()),
            "over_4096": float((s > 4096).mean()),
            "over_6144": float((s > 6144).mean()),
            "over_8192": float((s > 8192).mean()),
            "turns_user_mean": float(sub["n_user"].mean()),
            "turns_user_max": int(sub["n_user"].max()),
        }
        return out

    overall = summarize("overall", df)
    by_source = [summarize(f"source={src}", df[df["source"] == src]) for src in sorted(df["source"].unique())]

    print("\n" + "=" * 80)
    print("[summary] token length distribution")
    print("=" * 80)
    print(json.dumps(overall, ensure_ascii=False, indent=2))
    for item in by_source:
        print(json.dumps(item, ensure_ascii=False, indent=2))

    # Cross-tab: turn_bucket x token_bucket
    ctab = pd.crosstab(df["turn_bucket"], df["token_bucket"], normalize="index") * 100.0
    ctab = ctab.round(2)

    print("\n" + "=" * 80)
    print("[summary] turn_bucket x token_bucket (row %)")
    print("=" * 80)
    print(ctab.to_string())

    # Save per-row stats (for your later sampling/need analysis)
    out_csv = OUTPUT_DIR / f"length_stats_{ts}.csv"
    out_json = OUTPUT_DIR / f"length_summary_{ts}.json"

    df_out = df.drop(columns=["messages"])  # keep it small; messages can be huge
    df_out.to_csv(out_csv, index=False)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "overall": overall,
                "by_source": by_source,
                "turn_token_crosstab_rowpct": ctab.to_dict(),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\n[save] per-row stats: {out_csv}")
    print(f"[save] summary json : {out_json}")

    # Optional histograms
    if plt is not None:
        # tokens histogram
        plt.figure(figsize=(8, 4))
        plt.hist(df["tokens"].values, bins=60)
        plt.title("Token length histogram")
        plt.xlabel("tokens")
        plt.ylabel("count")
        tok_png = OUTPUT_DIR / f"token_hist_{ts}.png"
        plt.tight_layout()
        plt.savefig(tok_png, dpi=140)
        plt.close()

        # turns histogram (user turns)
        plt.figure(figsize=(8, 4))
        plt.hist(df["n_user"].values, bins=range(1, int(df["n_user"].max()) + 2))
        plt.title("User-turn histogram")
        plt.xlabel("user turns")
        plt.ylabel("count")
        turn_png = OUTPUT_DIR / f"turn_hist_{ts}.png"
        plt.tight_layout()
        plt.savefig(turn_png, dpi=140)
        plt.close()

        print(f"[plot] token hist: {tok_png}")
        print(f"[plot] turn  hist: {turn_png}")


if __name__ == "__main__":
    main()
