#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import hashlib
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer

# Optional plotting
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None


# =========================
# Paths / env（未设 PROJECT_ROOT 时从脚本位置推导，适配任意 checkout 路径）
# =========================
PROJECT_ROOT = Path(
    os.environ.get("PROJECT_ROOT", str(Path(__file__).resolve().parent.parent))
).resolve()
DATA_DIR = Path(os.environ.get("DATA_DIR", str(PROJECT_ROOT / "data"))).resolve()
CACHE_DIR = Path(os.environ.get("CACHE_DIR", str(PROJECT_ROOT / "cache"))).resolve()
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", str(PROJECT_ROOT / "outputs"))).resolve()

MODEL_PATH = os.environ.get("MODEL_PATH", "/model/llm/Meta-Llama-3.1-8B-Instruct")

XGUARD_PARQUET = Path(os.environ.get("XGUARD_PARQUET", str(DATA_DIR / "xguard-train.parquet"))).resolve()
ORBENCH_PARQUET = Path(os.environ.get("ORBENCH_PARQUET", str(DATA_DIR / "orbench.parquet"))).resolve()
TRAIN_PARQUET = Path(os.environ.get("TRAIN_PARQUET", "")).resolve() if os.environ.get("TRAIN_PARQUET") else None

MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "8192"))
TARGET_TOTAL_TOKENS = int(os.environ.get("TARGET_TOTAL_TOKENS", "2000000"))  # 0/neg => no sampling

TOKENIZE_BATCH_SIZE = int(os.environ.get("TOKENIZE_BATCH_SIZE", "512"))
TOKENIZE_NUM_PROC = int(os.environ.get("TOKENIZE_NUM_PROC", "4"))

SEED = int(os.environ.get("SEED", "42"))

# train
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "1"))
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", "16"))
NUM_EPOCHS = int(os.environ.get("NUM_EPOCHS", "2"))
LR = float(os.environ.get("LR", "2e-4"))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", "0.01"))
WARMUP_RATIO = float(os.environ.get("WARMUP_RATIO", "0.1"))
LOGGING_STEPS = int(os.environ.get("LOGGING_STEPS", "10"))

BF16 = os.environ.get("BF16", "1") == "1"
GRADIENT_CHECKPOINTING = os.environ.get("GRADIENT_CHECKPOINTING", "1") == "1"
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "-1"))
IS_DDP = LOCAL_RANK != -1

if IS_DDP:
    torch.cuda.set_device(LOCAL_RANK)
    device = torch.device("cuda", LOCAL_RANK)
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# LoRA
LORA_R = int(os.environ.get("LORA_R", "16"))
LORA_ALPHA = int(os.environ.get("LORA_ALPHA", "32"))
LORA_DROPOUT = float(os.environ.get("LORA_DROPOUT", "0.05"))
TARGET_MODULES = os.environ.get(
    "LORA_TARGET_MODULES",
    "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
).split(",")

# window-level reweight
ENABLE_WINDOW_REWEIGHT = os.environ.get("ENABLE_WINDOW_REWEIGHT", "1") == "1"
# weight = 1 / n_windows_total (clamped to >=1)
WINDOW_WEIGHT_CLAMP_MIN = float(os.environ.get("WINDOW_WEIGHT_CLAMP_MIN", "1.0"))


# =========================
# Utils
# =========================
def _tokenized_cache_path(max_length: int) -> Path:
    if TRAIN_PARQUET:
        # Include full path hash so different iter data files don't share cache
        path_str = str(TRAIN_PARQUET.resolve())
        path_hash = hashlib.md5(path_str.encode()).hexdigest()[:12]
        return CACHE_DIR / f"tokenized_train_max{max_length}_{path_hash}.parquet"
    return CACHE_DIR / f"tokenized_xguard_orbench_max{max_length}.parquet"


def _ensure_list(messages: Any) -> List[Dict[str, Any]]:
    if hasattr(messages, "tolist"):
        messages = messages.tolist()
    return messages


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds/60:.1f}min"
    return f"{seconds/3600:.1f}h"


def _plot_training_curves(log_history: list, output_dir: Path) -> None:
    if plt is None:
        return
    entries = [x for x in log_history if "loss" in x]
    if not entries:
        return
    steps = [e.get("step", i) for i, e in enumerate(entries)]
    losses = [float(e["loss"]) for e in entries]
    lrs = [e.get("learning_rate") for e in entries]
    has_lr = any(lr is not None for lr in lrs)

    if has_lr:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    else:
        fig, ax1 = plt.subplots(figsize=(8, 4))
        ax2 = None

    ax1.plot(steps, losses, linewidth=1.5)
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss")
    ax1.grid(True, alpha=0.3)

    if ax2 is not None and has_lr:
        ax2.plot(steps, [float(lr or 0.0) for lr in lrs], linewidth=1.5)
        ax2.set_xlabel("Step")
        ax2.set_ylabel("Learning Rate")
        ax2.set_title("Learning Rate Schedule")
        ax2.grid(True, alpha=0.3)
    else:
        ax1.set_xlabel("Step")

    fig.tight_layout()
    fig.savefig(output_dir / "loss_curve.png", dpi=120)
    plt.close()


def _sample_by_tokens(dataset: Dataset, target_tokens: int, seed: int = 42) -> Tuple[Dataset, int]:
    if target_tokens <= 0:
        return dataset, 0
    if "input_ids" not in dataset.column_names:
        raise ValueError("Token sampling requires 'input_ids' in dataset.")

    n = len(dataset)
    lengths = [len(dataset[i]["input_ids"]) for i in range(n)]
    idxs = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idxs)

    selected, total = [], 0
    for i in idxs:
        selected.append(i)
        total += lengths[i]
        if total >= target_tokens:
            break
    return dataset.select(selected), total


# =========================
# Tokenization cache builder (preserve source_id / n_windows_total)
# =========================
def _load_and_normalize_parquet(path: Path, source_name: str) -> pd.DataFrame:
    """
    Normalize to columns:
      - source (xguard/orbench)
      - source_id (original conversation id)
      - n_windows_total (>=1)  # for window-level reweight
      - messages
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing data file: {path}")
    df = pd.read_parquet(path)
    if "messages" not in df.columns:
        raise ValueError(f"{path} missing 'messages' column")

    df = df.copy()
    df["source"] = source_name

    # Determine source_id
    if "source_id" in df.columns:
        # slided xguard
        df["source_id"] = df["source_id"]
    elif "id" in df.columns:
        df["source_id"] = df["id"]
    else:
        # fallback
        df["source_id"] = list(range(len(df)))

    # Determine n_windows_total (best effort)
    if "_n_windows" in df.columns:
        # produced by our fast slicer
        df["n_windows_total"] = df["_n_windows"].astype(int)
    elif "window_id" in df.columns:
        # can derive by group count
        g = df.groupby("source_id")["window_id"].transform("count")
        df["n_windows_total"] = g.astype(int)
    else:
        df["n_windows_total"] = 1

    # clamp >=1
    df["n_windows_total"] = df["n_windows_total"].clip(lower=1)

    return df[["source", "source_id", "n_windows_total", "messages"]]


def _ensure_tokenized_cache(tokenizer, max_length: int) -> Path:
    cache_path = _tokenized_cache_path(max_length)
    # In DDP, only rank 0 creates cache; others wait to avoid race (0-byte read)
    is_cache_creator = not IS_DDP or LOCAL_RANK == 0

    if cache_path.exists() and cache_path.stat().st_size > 0:
        return cache_path

    if not is_cache_creator:
        # Non-creator ranks: wait for rank 0 to finish writing
        for _ in range(300):
            if cache_path.exists() and cache_path.stat().st_size > 0:
                return cache_path
            time.sleep(1)
        raise RuntimeError(f"Timeout waiting for tokenized cache: {cache_path}")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[cache] Missing max_length={max_length} cache, creating: {cache_path}")

    if TRAIN_PARQUET:
        print(f"[cache] Loading single training parquet: {TRAIN_PARQUET}")
        dfs = [_load_and_normalize_parquet(TRAIN_PARQUET, "combined")]
    else:
        dfs = [
            _load_and_normalize_parquet(XGUARD_PARQUET, "xguard"),
            _load_and_normalize_parquet(ORBENCH_PARQUET, "orbench"),
        ]
    ds = Dataset.from_pandas(pd.concat(dfs, ignore_index=True), preserve_index=False)

    eos_token = tokenizer.eos_token

    def tokenize_batch(examples):
        texts = []
        for msgs in examples["messages"]:
            msgs = _ensure_list(msgs)
            text = tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=False,
            )
            if eos_token and not text.endswith(eos_token):
                text = text + eos_token
            texts.append(text)

        out = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding=False,
            add_special_tokens=False,
        )
        return {
            "input_ids": out["input_ids"],
            "attention_mask": out["attention_mask"],
            # carry-through meta
            "source": examples["source"],
            "source_id": examples["source_id"],
            "n_windows_total": examples["n_windows_total"],
        }

    ds = ds.map(
        tokenize_batch,
        batched=True,
        batch_size=TOKENIZE_BATCH_SIZE,
        remove_columns=["messages"],  # keep source/source_id/n_windows_total
        desc="Tokenizing",
        num_proc=TOKENIZE_NUM_PROC,
    )

    # 先写临时文件再原子重命名，避免 DDP 下其他 rank 读到未写完的 parquet
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    ds.to_parquet(str(tmp_path))
    tmp_path.replace(cache_path)
    print(f"[cache] Saved {len(ds)} rows -> {cache_path}")
    return cache_path


def _load_tokenized_dataset(cache_path: Path) -> Dataset:
    df = pd.read_parquet(cache_path)
    return Dataset.from_pandas(df, preserve_index=False)


# =========================
# Data collator: only compute loss on assistant responses
# =========================
class AssistantOnlyDataCollator:
    """
    Pads batch and creates labels that only train on assistant response tokens.
    For each sequence, finds the response_template pattern (e.g. the assistant
    header tokens), then marks tokens from there until <|eot_id|> as trainable.
    All other positions get label=-100 (ignored by loss).
    """

    def __init__(self, response_template_ids, tokenizer):
        self.response_template_ids = response_template_ids
        self.tokenizer = tokenizer
        self.eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")

    def _find_template_positions(self, ids):
        tpl = self.response_template_ids
        tpl_len = len(tpl)
        positions = []
        for i in range(len(ids) - tpl_len + 1):
            if ids[i : i + tpl_len] == tpl:
                positions.append(i)
        return positions

    def __call__(self, examples):
        pad_id = self.tokenizer.pad_token_id or 0
        tpl_len = len(self.response_template_ids)

        id_lists = []
        extra = {}

        for ex in examples:
            ids = ex["input_ids"]
            if hasattr(ids, "tolist"):
                ids = ids.tolist()
            id_lists.append(ids)
            for k, v in ex.items():
                if k in ("input_ids", "attention_mask"):
                    continue
                extra.setdefault(k, []).append(v)

        max_len = max(len(ids) for ids in id_lists)
        batch_ids, batch_mask, batch_labels = [], [], []

        for ids in id_lists:
            length = len(ids)
            pad_len = max_len - length

            padded = ids + [pad_id] * pad_len
            mask = [1] * length + [0] * pad_len
            labels = [-100] * max_len

            for pos in self._find_template_positions(ids):
                start = pos + tpl_len
                for j in range(start, length):
                    labels[j] = ids[j]
                    if ids[j] == self.eot_id:
                        break

            batch_ids.append(padded)
            batch_mask.append(mask)
            batch_labels.append(labels)

        result = {
            "input_ids": torch.tensor(batch_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }
        for k, vals in extra.items():
            if vals and isinstance(vals[0], (int, float)):
                result[k] = torch.tensor(vals)
        return result


# =========================
# Weighted trainer
# =========================
class WeightedSFTTrainer(SFTTrainer):
    """
    Apply per-sample weights for causal LM loss.
    Uses `labels` from DataCollatorForCompletionOnlyLM when available,
    so loss is only computed on assistant response tokens.
    """
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        sample_weight = inputs.pop("sample_weight", None)
        labels = inputs.pop("labels", None)

        outputs = model(**inputs)
        logits = outputs.logits  # (B, T, V)

        shift_logits = logits[:, :-1, :]
        has_labels = labels is not None
        shift_labels = labels[:, 1:] if has_labels else inputs["input_ids"][:, 1:]

        vocab_size = shift_logits.size(-1)
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
        token_loss = loss_fct(
            shift_logits.reshape(-1, vocab_size),
            shift_labels.reshape(-1),
        ).reshape(shift_labels.size())

        if has_labels:
            valid_mask = (shift_labels != -100).to(token_loss.dtype)
            denom = valid_mask.sum(dim=1).clamp_min(1.0)
        else:
            attention_mask = inputs.get("attention_mask", None)
            if attention_mask is not None:
                valid_mask = attention_mask[:, 1:].contiguous().to(token_loss.dtype)
                denom = valid_mask.sum(dim=1).clamp_min(1.0)
            else:
                denom = torch.full(
                    (token_loss.size(0),), token_loss.size(1),
                    device=token_loss.device, dtype=token_loss.dtype,
                )

        per_example_loss = token_loss.sum(dim=1) / denom

        if sample_weight is None:
            loss = per_example_loss.mean()
        else:
            w = sample_weight.to(per_example_loss.device, dtype=per_example_loss.dtype)
            w = torch.clamp(w, min=1.0 / WINDOW_WEIGHT_CLAMP_MIN)
            loss = (per_example_loss * w).sum() / w.sum().clamp_min(1e-8)

        return (loss, outputs) if return_outputs else loss


def _add_sample_weight(ds: Dataset) -> Dataset:
    """
    Add `sample_weight` column:
      - if ENABLE_WINDOW_REWEIGHT: weight = 1 / n_windows_total
      - else: weight = 1
    """
    if not ENABLE_WINDOW_REWEIGHT:
        return ds.add_column("sample_weight", [1.0] * len(ds))

    if "n_windows_total" not in ds.column_names:
        # fallback: derive from source_id counts if possible
        if "source_id" in ds.column_names:
            # convert to pandas for grouping (ok for ~40k)
            pdf = ds.to_pandas()
            counts = pdf.groupby("source_id")["source_id"].transform("count").astype(int)
            pdf["n_windows_total"] = counts.clip(lower=1)
            ds = Dataset.from_pandas(pdf, preserve_index=False)
        else:
            return ds.add_column("sample_weight", [1.0] * len(ds))

    def map_weight(examples):
        n = examples["n_windows_total"]
        # n might be list of ints
        w = [1.0 / max(1, int(x)) for x in n]
        return {"sample_weight": w}

    ds = ds.map(map_weight, batched=True, desc="Add sample_weight")
    return ds


# =========================
# Main
# =========================
def main():
    try:
        _main_impl()
    finally:
        # 必须在退出前调用，否则 NCCL 会报资源泄漏警告
        if IS_DDP and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def _main_impl():
    t_start = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    torch.manual_seed(SEED)

    # ===== CUDA 性能优化（A800/AMP 推荐）=====
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_DIR / f"lora_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"[run] output_dir = {run_dir}")
    print(f"[run] model_path = {MODEL_PATH}")
    print(f"[run] max_length = {MAX_LENGTH}, target_tokens = {TARGET_TOTAL_TOKENS:,}")
    print(f"[run] window_reweight = {ENABLE_WINDOW_REWEIGHT}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    cache_path = _ensure_tokenized_cache(tokenizer, MAX_LENGTH)
    dataset = _load_tokenized_dataset(cache_path)
    print(f"[data] tokenized cache: {cache_path} (rows={len(dataset)})")

    # add sample_weight BEFORE sampling (weights based on original n_windows_total)
    dataset = _add_sample_weight(dataset)

    if TARGET_TOTAL_TOKENS and TARGET_TOTAL_TOKENS > 0:
        dataset, total_tokens = _sample_by_tokens(dataset, TARGET_TOTAL_TOKENS, seed=SEED)
        print(f"[data] sampled: rows={len(dataset)}, tokens={total_tokens:,}")

    # model
    dtype = torch.bfloat16 if BF16 else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model.to(device)
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id

    if GRADIENT_CHECKPOINTING and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    n_samples = len(dataset)
    steps_per_epoch = max(1, (n_samples // (BATCH_SIZE * GRAD_ACCUM)))
    total_steps = steps_per_epoch * NUM_EPOCHS
    warmup_steps = max(1, int(total_steps * WARMUP_RATIO))
    print(f"[train] steps/epoch={steps_per_epoch}, total_steps={total_steps}, warmup_steps={warmup_steps}")

    training_args = SFTConfig(
        output_dir=str(run_dir),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=warmup_steps,
        logging_steps=LOGGING_STEPS,
        save_strategy="no",  # 只保留最终模型，不保存中间 checkpoint
        bf16=BF16,
        fp16=(not BF16),
        report_to="none",
        remove_unused_columns=False,
        disable_tqdm=False,
        max_length=MAX_LENGTH,
        packing=False,
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
        ddp_find_unused_parameters=False,
    )

    TrainerCls = WeightedSFTTrainer if ENABLE_WINDOW_REWEIGHT else SFTTrainer

    response_template_ids = (
        [tokenizer.convert_tokens_to_ids("<|start_header_id|>")]
        + tokenizer.encode("assistant", add_special_tokens=False)
        + [tokenizer.convert_tokens_to_ids("<|end_header_id|>")]
        + tokenizer.encode("\n\n", add_special_tokens=False)
    )
    collator = AssistantOnlyDataCollator(response_template_ids, tokenizer)

    trainer = TrainerCls(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        data_collator=collator,
    )

    t_before = time.perf_counter()
    trainer.train()
    t_after = time.perf_counter()

    trainer.save_model(str(run_dir))
    tokenizer.save_pretrained(str(run_dir))
    print(f"[save] saved to: {run_dir}")

    if trainer.state.log_history:
        _plot_training_curves(trainer.state.log_history, run_dir)
        metrics_path = run_dir / "train_metrics.jsonl"

        def _to_json(v):
            if hasattr(v, "item"):
                return v.item()
            return v

        with open(metrics_path, "w", encoding="utf-8") as f:
            for entry in trainer.state.log_history:
                f.write(json.dumps(entry, ensure_ascii=False, default=_to_json) + "\n")
        print(f"[save] metrics: {metrics_path}")

    # run_loop 模式下每次 iteration 数据不同，缓存不会复用，训练完即删
    if TRAIN_PARQUET and cache_path.exists() and (not IS_DDP or LOCAL_RANK == 0):
        cache_path.unlink()
        print(f"[cache] removed after training: {cache_path}")

    t_end = time.perf_counter()
    print("\n" + "=" * 60)
    print("Time breakdown")
    print("=" * 60)
    print(f"  setup: {_format_duration(t_before - t_start)}")
    print(f"  train: {_format_duration(t_after - t_before)}")
    print(f"  tail : {_format_duration(t_end - t_after)}")
    print(f"  total: {_format_duration(t_end - t_start)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
