#!/usr/bin/env python3
"""
run_loop.py — Main closed-loop controller for iterative fine-tuning.

Orchestrates the full cycle:
  1. Construct training data from distribution vector w and dims
  2. Fine-tune model (LoRA on GPU server)
  3. Run inference on evaluation sets (vLLM)
  4. Evaluate responses (three benchmarks)
  5. Update Pareto frontier
  6. Ask LLM advisor for next distribution
  7. Repeat

Usage (GPU server):
    uv run python scripts/run_loop.py --max-iters 5 --T-max 1000000

Local mock test:
    uv run python scripts/run_loop.py --max-iters 2 --T-max 100000 --mock
"""

import asyncio
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.pareto import ParetoArchive
from scripts.data_constructor import construct_training_set, ensure_slided_for_sources
from scripts.distribution_agent import get_next_distribution
from scripts.vllm_inference import (
    load_eval_prompts,
    run_api_inference,
    run_inference_with_managed_server,
    run_mock_inference,
)
from scripts.eval_model import evaluate_all


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_MODEL_PATH = os.environ.get("MODEL_PATH", "/model/llm/Meta-Llama-3.1-8B-Instruct")
DEFAULT_T_MAX = 1_000_000
DEFAULT_MAX_ITERS = 5
DEFAULT_MAX_LENGTH = 4096
DEFAULT_SEED = 42

RESULTS_DIR = PROJECT_ROOT / "results"
EVAL_DIR = PROJECT_ROOT / "dataset" / "eval"
BASELINE_CACHE_DIR = RESULTS_DIR / "baseline_cache"


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}min"
    else:
        return f"{seconds/3600:.1f}h"


def _get_gpu_count(env: dict) -> int:
    """Detect number of visible GPUs in a faster, more stable way."""
    # 1. First respect CUDA_VISIBLE_DEVICES if explicitly set
    cvd = env.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None:
        cvd = cvd.strip()
        if not cvd:
            return 1  # 环境变量为空白，回退到 1
        return len(cvd.split(","))

    # 2. Use nvidia-smi which is extremely fast and doesn't load fat libraries
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            lines = [line for line in result.stdout.splitlines() if line.strip().startswith("GPU")]
            n = len(lines)
            return max(1, n)
    except Exception as e:
        print(f"[warning] Failed to query nvidia-smi: {e}", file=sys.stderr)
        
    # 3. Fallback
    return 1


class _Tee:
    """Writes to multiple streams (e.g. console + log file)."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, s: str) -> None:
        for st in self._streams:
            st.write(s)
            st.flush()

    def flush(self) -> None:
        for st in self._streams:
            st.flush()

    def close(self) -> None:
        # absl logging handler calls stream.close() at shutdown; we must not close
        # the real stdout/stderr, only flush so buffered output is written
        self.flush()

    def fileno(self) -> int:
        # subprocess.run inherits stdout and needs fileno(); delegate to real stdout
        return self._streams[0].fileno()

    def isatty(self) -> bool:
        return hasattr(self._streams[0], 'isatty') and self._streams[0].isatty()


def _run_subprocess_tee(
    cmd: List[str],
    env: Dict[str, str],
    cwd: str,
    timeout_seconds: Optional[float] = None,
) -> int:
    """Run subprocess, stream stdout/stderr to current sys.stdout (goes to Tee -> console + log)."""
    proc = subprocess.Popen(
        cmd,
        env=env,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    try:
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
        return proc.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        print(f"[ERROR] Process timed out after {timeout_seconds}s, killing...")
        proc.kill()
        proc.wait()
        return -1


# ---------------------------------------------------------------------------
# Distribution validation
# ---------------------------------------------------------------------------

def _validate_distribution(w: Dict[str, float]) -> Dict[str, float]:
    """Ensure weights are non-negative and sum to 1. Falls back to xguard-only if all zero."""
    w = {k: max(0.0, v) for k, v in w.items()}
    total = sum(w.values())
    if total <= 0:
        print("[WARN] Agent returned all-zero distribution, falling back to xguard-only")
        return {"xguard": 1.0, "orbench": 0.0, "ifeval": 0.0}
    return {k: round(v / total, 6) for k, v in w.items()}


# ---------------------------------------------------------------------------
# Iteration checkpoint (resume within an iteration after partial failure)
# ---------------------------------------------------------------------------

def _load_checkpoint(iter_dir: Path) -> Dict[str, Any]:
    cp_path = iter_dir / "checkpoint.json"
    if cp_path.exists():
        with open(cp_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_checkpoint(iter_dir: Path, step: str, **kwargs) -> None:
    cp = _load_checkpoint(iter_dir)
    cp["last_completed_step"] = step
    cp.update(kwargs)
    with open(iter_dir / "checkpoint.json", "w", encoding="utf-8") as f:
        json.dump(cp, f, indent=2, ensure_ascii=False, default=str)


def _clear_checkpoint(iter_dir: Path) -> None:
    cp_path = iter_dir / "checkpoint.json"
    if cp_path.exists():
        cp_path.unlink()


# ---------------------------------------------------------------------------
# Step executors
# ---------------------------------------------------------------------------

def step_ensure_slided(model_path: str, max_length: int, mock: bool = False) -> None:
    """
    Step 0: Ensure slided cache exists for all sources at given max_length.
    Cache is keyed by max_length; changing it triggers automatic regeneration.
    """
    if mock:
        return
    ensure_slided_for_sources(model_path, max_length)


def step_construct_data(
    w: Dict[str, float],
    dims: Dict[str, dict],
    quality_threshold: float,
    T_max: int,
    max_length: int,
    iteration: int,
    iter_dir: Path,
    run_dir: Path,
    seed: int,
    model_path: str,
) -> Path:
    """Step 1: Construct training dataset from distribution vector.

    训练集使用 parquet 预打分列（score/slice），不 merge 评测集。
    使用 packages/finetune/data 下已有的 slided 缓存。
    """
    print(f"\n{'='*60}")
    print(f"[Step 1] Constructing training data (iter={iteration})")
    print(f"{'='*60}")

    output_path = iter_dir / "train_data.parquet"
    return construct_training_set(
        w=w,
        T_max=T_max,
        max_length=max_length,
        quality_threshold=quality_threshold,
        dims=dims,
        seed=seed + iteration,
        output_path=output_path,
        model_path=model_path,
    )


def step_finetune(
    data_path: Path,
    iteration: int,
    iter_dir: Path,
    model_path: str,
    max_length: int = 4096,
    mock: bool = False,
    num_epochs: int = 1,
    batch_size: int = 4,
    grad_accum: int = 4,
) -> Path:
    """Step 2: Fine-tune model using LoRA."""
    print(f"\n{'='*60}")
    print(f"[Step 2] Fine-tuning model (iter={iteration})")
    print(f"{'='*60}")

    output_dir = iter_dir / "lora_output"

    if mock:
        print("[finetune] MOCK MODE — skipping actual training")
        output_dir.mkdir(parents=True, exist_ok=True)
        # Create a dummy marker file
        (output_dir / "mock_adapter_config.json").write_text(
            json.dumps({"mock": True, "iteration": iteration}),
            encoding="utf-8"
        )
        return output_dir

    # Real training: call launch.sh with custom env vars
    finetune_root = PROJECT_ROOT / "packages" / "finetune"
    env = os.environ.copy()
    env["XGUARD_PARQUET"] = str(data_path)
    env["ORBENCH_PARQUET"] = str(data_path)  # combined parquet
    env["OUTPUT_DIR"] = str(output_dir)
    env["MODEL_PATH"] = model_path
    env["TARGET_TOTAL_TOKENS"] = "0"  # Disable sampling, use all rows
    env["NUM_EPOCHS"] = str(num_epochs)
    env["BATCH_SIZE"] = str(batch_size)
    env["GRAD_ACCUM"] = str(grad_accum)
    env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    env["MAX_LENGTH"] = str(max_length)

    # For single-parquet training, we need to adapt the script
    # The train.py can accept a single TRAIN_PARQUET env var
    env["TRAIN_PARQUET"] = str(data_path)

    n_gpus = _get_gpu_count(env)
    if n_gpus > 1:
        # 强制：用 IPv4 本机地址 + 让 accelerate 自动选端口
        env["MASTER_ADDR"] = "127.0.0.1"
        env["MASTER_PORT"] = "0"
        env["GLOO_SOCKET_IFNAME"] = "lo"
        env["NCCL_SOCKET_IFNAME"] = "lo"
        env["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"  # 可选，便于定位
        
        cmd = [
            sys.executable, "-m", "accelerate.commands.launch",
            "--multi_gpu",
            "--num_processes", str(n_gpus),
            "--num_machines", "1",
            "--mixed_precision", "bf16",
            "--dynamo_backend", "no",
            "--main_process_port", "0",          # 关键：自动端口
            str(finetune_root / "scripts" / "train.py"),
        ]
        print(f"[finetune] Using {n_gpus} GPUs (accelerate launch)")
    else:
        cmd = [sys.executable, str(finetune_root / "scripts" / "train.py")]

    print(f"[finetune] Running: {' '.join(cmd)}")
    print(f"[finetune] Data: {data_path}")
    print(f"[finetune] Output: {output_dir}")

    t0 = time.time()
    returncode = _run_subprocess_tee(cmd, env, str(PROJECT_ROOT))
    elapsed = time.time() - t0

    if returncode != 0:
        print(f"[finetune] ERROR: training failed (exit code {returncode})")
        raise RuntimeError(f"Fine-tuning failed at iteration {iteration}")

    print(f"[finetune] Done in {_format_duration(elapsed)}")
    return output_dir


def step_inference(
    lora_path: Optional[Path],
    iteration: int,
    iter_dir: Path,
    model_path: str,
    mock: bool = False,
) -> Path:
    """Step 3: Run inference on evaluation sets."""
    print(f"\n{'='*60}")
    print(f"[Step 3] Running inference (iter={iteration})")
    print(f"{'='*60}")

    inference_dir = iter_dir / "inference"
    prompts = load_eval_prompts(EVAL_DIR)

    if mock:
        run_mock_inference(prompts, inference_dir)
    elif api_base := os.environ.get("INFERENCE_API_BASE_URL"):
        # 外部 API：直接连接，不启停
        api_model = os.environ.get("INFERENCE_API_MODEL", model_path)
        xteaming_concurrency = int(os.environ.get("XTEAMING_API_CONCURRENCY", "50"))
        run_api_inference(
            api_base_url=api_base,
            api_model=api_model,
            prompts=prompts,
            output_dir=inference_dir,
            xteaming_concurrency=xteaming_concurrency,
        )
    else:
        # 默认：自动启动 vLLM serve → API 推理 → 关闭服务
        port = int(os.environ.get("VLLM_SERVE_PORT", "8000"))
        xteaming_concurrency = int(os.environ.get("XTEAMING_API_CONCURRENCY", "50"))
        run_inference_with_managed_server(
            model_path=model_path,
            lora_path=lora_path,
            prompts=prompts,
            output_dir=inference_dir,
            port=port,
            xteaming_concurrency=xteaming_concurrency,
        )

    return inference_dir


def step_evaluate(
    inference_dir: Path,
    iteration: int,
) -> Dict[str, Any]:
    """Step 4: Evaluate model responses."""
    print(f"\n{'='*60}")
    print(f"[Step 4] Evaluating responses (iter={iteration})")
    print(f"{'='*60}")

    results = evaluate_all(inference_dir)

    # Save results
    results_path = inference_dir.parent / "eval_results.json"
    save_data = {k: v for k, v in results.items() if k != "f"}
    save_data["f"] = list(results["f"])
    with open(results_path, "w", encoding="utf-8") as fh:
        json.dump(save_data, fh, indent=2, ensure_ascii=False, default=str)
    print(f"[eval] Results saved to {results_path}")

    return results


async def step_pareto_and_advise(
    archive: ParetoArchive,
    history: List[Dict[str, Any]],
    w: Dict[str, float],
    dims: Dict[str, dict],
    eval_results: Dict[str, Any],
    iteration: int,
    T_max: int,
    iter_dir: Path,
) -> tuple[dict, dict, dict, bool]:
    """Step 5-6: Update Pareto frontier + get LLM advice (Async).

    Returns:
        (w_next, dims_next, record, accepted)
    """
    print(f"\n{'='*60}")
    print(f"[Step 5-6] Pareto update + Agent advisor (iter={iteration})")
    print(f"{'='*60}")

    f = eval_results["f"]
    record = {
        "iter": iteration,
        "w": w,
        "dims": dims,
        "f": list(f),
        "timestamp": _timestamp(),
    }

    accepted = archive.add(record)
    record["accepted"] = accepted

    history.append(record)

    print(f"  Scores: xguard={f[0]:.4f}, orbench={f[1]:.4f}, ifeval={f[2]:.4f}")
    print(f"  Accepted into frontier: {'[OK]' if accepted else '[NO]'}")
    print(f"  Frontier size: {archive.size()}")
    print(f"\n{archive.to_markdown_table()}")

    print(f"\n[advisor] Asking DeepSeek for next distribution...")

    eval_detail_parts = []
    for bench_name in ["xguard", "orbench", "ifeval"]:
        bench = eval_results.get(bench_name, {})
        dist = bench.get("score_distribution", {})
        if dist:
            eval_detail_parts.append(f"### {bench_name} score distribution: {dist}")
        bucket_avgs = bench.get("bucket_score_avgs", {})
        if bucket_avgs:
            eval_detail_parts.append(f"### {bench_name} per-bucket avg scores:")
            for bucket, avg in sorted(bucket_avgs.items()):
                eval_detail_parts.append(f"  - {bucket}: avg_score={avg:.2f}")

    eval_results_file = iter_dir / "eval_results.json"
    agent_trace_file = iter_dir / "agent_trace.jsonl"

    prev_record = history[-2] if len(history) >= 2 else None

    w_next, dims_next, extra_info = await get_next_distribution(
        frontier_summary=archive.summary_for_llm(),
        history_table=ParetoArchive.history_to_markdown(history),
        current_record=record,
        T_max=T_max,
        eval_details="\n".join(eval_detail_parts),
        source_stats={
            "_eval_results_path": str(eval_results_file),
            "_trace_log_path": str(agent_trace_file),
        },
        prev_record=prev_record,
        n_samples=1,
        temperature=0.0,
        aggregate_method="median",
    )

    if dims_next:
        for dim_name, dim_plan in dims_next.items():
            fc = dim_plan.get("focus_criteria", {}).get("include", {})
            if fc:
                print(f"  [advisor] [{dim_name}] focus include: {fc}")

    return w_next, dims_next, record, accepted


async def _advise_from_baseline(
    archive: ParetoArchive,
    history: List[Dict[str, Any]],
    baseline_record: Dict[str, Any],
    baseline_eval: Dict[str, Any],
    baseline_iter_dir: Path,
    T_max: int,
) -> tuple[dict, dict]:
    """用 baseline 评测结果让 Agent 建议 iter 0 的分布。返回 (w_next, dims_next)。"""
    print(f"\n{'='*60}")
    print("[advisor] Asking Agent for initial distribution (from baseline eval)")
    print(f"{'='*60}")

    eval_detail_parts = []
    for bench_name in ["xguard", "orbench", "ifeval"]:
        bench = baseline_eval.get(bench_name, {})
        dist = bench.get("score_distribution", {})
        if dist:
            eval_detail_parts.append(f"### {bench_name} score distribution: {dist}")
        bucket_avgs = bench.get("bucket_score_avgs", {})
        if bucket_avgs:
            eval_detail_parts.append(f"### {bench_name} per-bucket avg scores:")
            for bucket, avg in sorted(bucket_avgs.items()):
                eval_detail_parts.append(f"  - {bucket}: avg_score={avg:.2f}")

    w_next, dims_next, _extra = await get_next_distribution(
        frontier_summary=archive.summary_for_llm(),
        history_table=ParetoArchive.history_to_markdown(history),
        current_record=baseline_record,
        T_max=T_max,
        eval_details="\n".join(eval_detail_parts),
        source_stats={
            "_eval_results_path": str(baseline_iter_dir / "eval_results.json"),
            "_trace_log_path": str(baseline_iter_dir / "agent_trace.jsonl"),
        },
        n_samples=1,
        temperature=0.0,
        aggregate_method="median",
    )

    return w_next, dims_next


# ---------------------------------------------------------------------------
# Baseline cache (首次生成，后续复用)
# ---------------------------------------------------------------------------

def _baseline_cache_path(model_path: str, max_length: int) -> Path:
    """Cache 路径：按 model_path + max_length 区分。"""
    key = f"{model_path}_{max_length}"
    h = hashlib.md5(key.encode()).hexdigest()[:12]
    return BASELINE_CACHE_DIR / f"baseline_{h}"


def _get_or_create_baseline(
    run_dir: Path,
    model_path: str,
    max_length: int,
    mock: bool,
) -> Tuple[Dict[str, Any], Dict[str, Any], Path]:
    """
    获取或创建 baseline（未微调基座评测）。
    首次运行生成并缓存，后续直接读取。
    Returns:
        (baseline_record, eval_results, iter_dir)
    """
    cache_dir = _baseline_cache_path(model_path, max_length)
    record_path = cache_dir / "baseline_record.json"
    eval_path = cache_dir / "eval_results.json"
    iter_dir = cache_dir / "iter_baseline"

    if record_path.exists() and eval_path.exists():
        print(f"\n[baseline] Cache hit: {cache_dir}")
        with open(record_path, "r", encoding="utf-8") as f:
            record = json.load(f)
        with open(eval_path, "r", encoding="utf-8") as f:
            eval_results = json.load(f)
        iter_dir.mkdir(parents=True, exist_ok=True)
        with open(iter_dir / "eval_results.json", "w", encoding="utf-8") as fh:
            json.dump(eval_results, fh, indent=2, ensure_ascii=False, default=str)
        return record, eval_results, iter_dir

    print(f"\n[baseline] Cache miss, generating: {cache_dir}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    iter_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("[baseline] Running inference on base model (no LoRA)")
    print(f"{'='*60}")

    inference_dir = step_inference(None, 0, iter_dir, model_path, mock)
    eval_results = step_evaluate(inference_dir, 0)

    f = eval_results["f"]
    print(f"\n[baseline] Scores: xguard={f[0]:.4f}, orbench={f[1]:.4f}, ifeval={f[2]:.4f}")

    record = {
        "iter": -1,  # baseline 排在最前，与 iter 0,1,2... 区分
        "iter_label": "baseline",
        "w": {"xguard": 0.0, "orbench": 0.0, "ifeval": 0.0},
        "bucket_weights": {},
        "f": list(f),
        "timestamp": _timestamp(),
        "accepted": True,
    }

    with open(record_path, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2, ensure_ascii=False, default=str)
    save_data = {k: v for k, v in eval_results.items() if k != "f"}
    save_data["f"] = list(eval_results["f"])
    with open(eval_path, "w", encoding="utf-8") as fh:
        json.dump(save_data, fh, indent=2, ensure_ascii=False, default=str)
    with open(iter_dir / "eval_results.json", "w", encoding="utf-8") as fh:
        json.dump(save_data, fh, indent=2, ensure_ascii=False, default=str)

    return record, eval_results, iter_dir


# ---------------------------------------------------------------------------
# Baseline-only path
# ---------------------------------------------------------------------------

def _run_baseline_only(
    run_dir: Path,
    run_id: str,
    model_path: str,
    mock: bool,
) -> None:
    """Run inference + eval on base model only (no fine-tuning)."""
    config = {
        "run_id": run_id,
        "model_path": model_path,
        "mock": mock,
        "baseline_only": True,
    }
    with open(run_dir / "config.json", "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2, ensure_ascii=False)

    iter_dir = run_dir / "iter_00"
    iter_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("[baseline] Running inference on base model (no LoRA)")
    print(f"{'='*60}")

    inference_dir = step_inference(None, 0, iter_dir, model_path, mock)
    eval_results = step_evaluate(inference_dir, 0)

    f = eval_results["f"]
    print(f"\n{'='*60}")
    print(f"[baseline] Scores: xguard={f[0]:.4f}, orbench={f[1]:.4f}, ifeval={f[2]:.4f}")
    print(f"{'='*60}")

    report_path = run_dir / "baseline_report.md"
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(f"# Baseline Report (no fine-tuning)\n\n")
        fh.write(f"- Model: `{model_path}`\n")
        fh.write(f"- XGuard: {f[0]:.4f}\n")
        fh.write(f"- OrBench: {f[1]:.4f}\n")
        fh.write(f"- IFEval: {f[2]:.4f}\n")
    print(f"\n[report] Saved to {report_path}")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_loop(
    max_iters: int = DEFAULT_MAX_ITERS,
    T_max: int = DEFAULT_T_MAX,
    max_length: int = DEFAULT_MAX_LENGTH,
    model_path: str = DEFAULT_MODEL_PATH,
    quality_threshold: float = 4.0,
    seed: int = DEFAULT_SEED,
    mock: bool = False,
    resume_from: Optional[int] = None,
    num_epochs: int = 1,
    baseline_only: bool = False,
    iter_retries: int = 2,
):
    """
    Main iterative fine-tuning loop (synchronous entry point).

    Args:
        max_iters: Maximum number of iterations
        T_max: Total token budget per iteration
        model_path: Path to base model
        quality_threshold: Filter out responses scoring below this threshold
        seed: Random seed
        mock: If True, skip actual training and inference (local testing)
        resume_from: Resume from a specific iteration (loads saved state)
        iter_retries: Max retries per iteration on transient failure
    """
    run_id = _timestamp()
    run_dir = RESULTS_DIR / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    log_path = run_dir / "run.log"
    log_file = open(log_path, "w", encoding="utf-8")
    _orig_stdout, _orig_stderr = sys.stdout, sys.stderr
    sys.stdout = _Tee(_orig_stdout, log_file)
    sys.stderr = _Tee(_orig_stderr, log_file)

    try:
        _run_loop_body(
            run_id=run_id,
            run_dir=run_dir,
            max_iters=max_iters,
            T_max=T_max,
            max_length=max_length,
            model_path=model_path,
            quality_threshold=quality_threshold,
            seed=seed,
            mock=mock,
            resume_from=resume_from,
            num_epochs=num_epochs,
            baseline_only=baseline_only,
            iter_retries=iter_retries,
        )
    finally:
        sys.stdout, sys.stderr = _orig_stdout, _orig_stderr
        log_file.close()
        print(f"[log] Saved to {log_path}", file=_orig_stdout)


def _run_loop_body(
    run_id: str,
    run_dir: Path,
    max_iters: int,
    T_max: int,
    max_length: int,
    model_path: str,
    quality_threshold: float,
    seed: int,
    mock: bool,
    resume_from: Optional[int],
    num_epochs: int = 1,
    baseline_only: bool = False,
    iter_retries: int = 2,
) -> None:
    """Inner loop body (synchronous); stdout/stderr are already teed to run.log."""
    print(f"{'#'*60}")
    print(f"# Iterative Fine-tuning Loop")
    print(f"# Run ID: {run_id}")
    if baseline_only:
        print(f"# Mode: BASELINE ONLY (no fine-tuning)")
    else:
        print(f"# Max iterations: {max_iters}")
        print(f"# Token budget: {T_max:,}")
    print(f"# Max length: {max_length} (slided cache key)")
    print(f"# Model: {model_path}")
    print(f"# Quality Threshold: {quality_threshold}")
    print(f"# Mock: {mock}")
    print(f"# Output: {run_dir}")
    print(f"# Log: {run_dir / 'run.log'}")
    print(f"{'#'*60}")

    if baseline_only:
        _run_baseline_only(run_dir, run_id, model_path, mock)
        return

    archive_path = run_dir / "pareto_archive.json"
    history_path = run_dir / "full_history.json"

    archive = ParetoArchive(archive_path)
    history = ParetoArchive.load_history(history_path)

    baseline_record, baseline_eval, baseline_iter_dir = _get_or_create_baseline(
        run_dir, model_path, max_length, mock
    )
    if not history or history[0].get("iter_label") != "baseline":
        archive.add(baseline_record)
        history.insert(0, baseline_record)
        archive.save()
        ParetoArchive.save_history(history, history_path)

    w = {"xguard": 1.0, "orbench": 0.0, "ifeval": 0.0}
    dims: Dict[str, dict] = {}

    start_iter = resume_from if resume_from is not None else 0
    if resume_from is not None and resume_from > 0 and history:
        last = history[-1]
        if "next_w" in last:
            w = last["next_w"]
        if "next_dims" in last:
            dims = last["next_dims"]
        elif "next_bw" in last:
            old_bw = last["next_bw"]
            old_fc = last.get("next_focus_criteria", {})
            dims = {}
            for dim_name in old_bw:
                dims[dim_name] = {"bucket_weights": old_bw[dim_name], "focus_criteria": {"include": old_fc, "exclude": {}}}
        print(f"[resume] Resuming from iteration {start_iter} with w={w}")
    elif resume_from is None or resume_from == 0:
        w, dims = asyncio.run(_advise_from_baseline(
            archive, history, baseline_record, baseline_eval, baseline_iter_dir, T_max
        ))

    config = {
        "run_id": run_id,
        "max_iters": max_iters,
        "T_max": T_max,
        "max_length": max_length,
        "model_path": model_path,
        "quality_threshold": quality_threshold,
        "seed": seed,
        "mock": mock,
        "num_epochs": num_epochs,
        "initial_w": w,
    }
    with open(run_dir / "config.json", "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2, ensure_ascii=False)

    total_t0 = time.time()

    step_ensure_slided(model_path, max_length, mock)

    for iteration in range(start_iter, max_iters):
        iter_t0 = time.time()
        iter_dir = run_dir / f"iter_{iteration:02d}"
        iter_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'#'*60}")
        print(f"# ITERATION {iteration}/{max_iters-1}")
        print(f"# Distribution: {w}")
        print(f"# Dims: {json.dumps(dims, ensure_ascii=False, default=str)[:200]}")
        print(f"{'#'*60}")

        with open(iter_dir / "distribution.json", "w") as fh:
            json.dump({"w": w, "dims": dims, "iteration": iteration}, fh, indent=2, default=str)

        succeeded = False
        for attempt in range(iter_retries + 1):
            try:
                w_next, dims_next = _run_single_iteration(
                    w=w, dims=dims, quality_threshold=quality_threshold,
                    T_max=T_max, max_length=max_length,
                    iteration=iteration, iter_dir=iter_dir, run_dir=run_dir,
                    seed=seed, model_path=model_path, mock=mock,
                    num_epochs=num_epochs,
                    archive=archive, history=history, history_path=history_path,
                )
                _clear_checkpoint(iter_dir)
                succeeded = True
                break
            except Exception as e:
                if attempt < iter_retries:
                    wait = 2 ** attempt * 5
                    print(f"\n[WARN] Iteration {iteration} attempt {attempt+1}/{iter_retries+1} failed: {e}")
                    print(f"[WARN] Retrying in {wait}s...")
                    traceback.print_exc()
                    time.sleep(wait)
                else:
                    print(f"\n[ERROR] Iteration {iteration} failed after {iter_retries+1} attempts: {e}")
                    traceback.print_exc()
                    archive.save()
                    ParetoArchive.save_history(history, history_path)

        if not succeeded:
            break

        iter_elapsed = time.time() - iter_t0
        print(f"\n[iter {iteration}] Completed in {_format_duration(iter_elapsed)}")

        w = w_next
        dims = dims_next

    total_elapsed = time.time() - total_t0

    # ---------------------------------------------------------------------------
    # Final report
    # ---------------------------------------------------------------------------
    finetune_iters = sum(1 for h in history if h.get("iter", -1) >= 0)
    print(f"\n{'#'*60}")
    print(f"# FINAL REPORT")
    print(f"# Total time: {_format_duration(total_elapsed)}")
    print(f"# Fine-tuning iterations completed: {finetune_iters}")
    print(f"# Total records (incl. baseline): {len(history)}")
    print(f"# Pareto frontier size: {archive.size()}")
    print(f"{'#'*60}")

    print(f"\n## Pareto Frontier")
    print(archive.to_markdown_table())

    print(f"\n## Full History")
    print(ParetoArchive.history_to_markdown(history))

    report_path = run_dir / "final_report.md"
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(f"# Iterative Fine-tuning Report\n\n")
        fh.write(f"- Run ID: `{run_id}`\n")
        fh.write(f"- Total time: {_format_duration(total_elapsed)}\n")
        fh.write(f"- Fine-tuning iterations: {finetune_iters}\n")
        fh.write(f"- Token budget: {T_max:,}\n")
        fh.write(f"- Model: `{model_path}`\n\n")
        fh.write(f"## Pareto Frontier\n\n{archive.to_markdown_table()}\n\n")
        fh.write(f"## Full History\n\n{ParetoArchive.history_to_markdown(history)}\n")
    print(f"\n[report] Saved to {report_path}")


def _run_single_iteration(
    w: Dict[str, float],
    dims: Dict[str, dict],
    quality_threshold: float,
    T_max: int,
    max_length: int,
    iteration: int,
    iter_dir: Path,
    run_dir: Path,
    seed: int,
    model_path: str,
    mock: bool,
    num_epochs: int,
    archive: ParetoArchive,
    history: List[Dict[str, Any]],
    history_path: Path,
) -> Tuple[Dict[str, float], Dict[str, dict]]:
    """Execute one iteration with checkpoint support. Returns (w_next, dims_next)."""
    cp = _load_checkpoint(iter_dir)
    last_step = cp.get("last_completed_step", "")

    # 1. Construct training data
    if last_step < "construct_data":
        data_path = step_construct_data(
            w, dims, quality_threshold, T_max, max_length,
            iteration, iter_dir, run_dir, seed, model_path,
        )
        _save_checkpoint(iter_dir, "construct_data", data_path=str(data_path))
    else:
        data_path = Path(cp["data_path"])
        print(f"[checkpoint] Skipping construct_data (already done)")

    # 2. Fine-tune
    if last_step < "finetune":
        lora_path = step_finetune(data_path, iteration, iter_dir, model_path, max_length, mock, num_epochs=num_epochs, grad_accum=1)
        _save_checkpoint(iter_dir, "finetune", lora_path=str(lora_path))
    else:
        lora_path = Path(cp["lora_path"])
        print(f"[checkpoint] Skipping finetune (already done)")

    # 3. Inference
    if last_step < "inference":
        inference_dir = step_inference(lora_path, iteration, iter_dir, model_path, mock)
        _save_checkpoint(iter_dir, "inference", inference_dir=str(inference_dir))
    else:
        inference_dir = Path(cp["inference_dir"])
        print(f"[checkpoint] Skipping inference (already done)")

    # 4. Evaluate
    if last_step < "evaluate":
        eval_results = step_evaluate(inference_dir, iteration)
        _save_checkpoint(iter_dir, "evaluate")
    else:
        eval_results_path = iter_dir / "eval_results.json"
        with open(eval_results_path, "r", encoding="utf-8") as fh:
            eval_results = json.load(fh)
        eval_results["f"] = tuple(eval_results["f"])
        print(f"[checkpoint] Skipping evaluate (already done)")

    # 5-6. Pareto + LLM advisor (async call)
    w_next, dims_next, record, accepted = asyncio.run(
        step_pareto_and_advise(
            archive, history, w, dims, eval_results, iteration, T_max, iter_dir
        )
    )

    w_next = _validate_distribution(w_next)
    record["next_w"] = w_next
    record["next_dims"] = dims_next

    archive.save()
    ParetoArchive.save_history(history, history_path)

    return w_next, dims_next


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Iterative fine-tuning loop with Pareto frontier + LLM advisor"
    )
    parser.add_argument("--max-iters", type=int, default=DEFAULT_MAX_ITERS,
                        help=f"Max iterations (default: {DEFAULT_MAX_ITERS})")
    parser.add_argument("--T-max", type=int, default=DEFAULT_T_MAX,
                        help=f"Token budget per iteration (default: {DEFAULT_T_MAX:,})")
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH,
                        help=f"Slided window + training max_length (default: {DEFAULT_MAX_LENGTH})")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH,
                        help="Path to base model")
    parser.add_argument("--quality-threshold", type=float, default=0.0,
                        help="Drop training rows with response_score below this threshold")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--mock", action="store_true",
                        help="Mock mode: skip training and inference (local testing)")
    parser.add_argument("--resume-from", type=int, default=None,
                        help="Resume from a specific iteration")
    parser.add_argument("--num-epochs", type=int, default=1,
                        help="Number of training epochs per iteration (default: 1)")
    parser.add_argument("--baseline-only", action="store_true",
                        help="Skip fine-tuning; run inference + eval on base model only")
    parser.add_argument("--iter-retries", type=int, default=2,
                        help="Max retries per iteration on transient failure (default: 2)")

    args = parser.parse_args()

    run_loop(
        max_iters=args.max_iters,
        T_max=args.T_max,
        max_length=args.max_length,
        model_path=args.model_path,
        quality_threshold=args.quality_threshold,
        seed=args.seed,
        mock=args.mock,
        resume_from=args.resume_from,
        num_epochs=args.num_epochs,
        baseline_only=args.baseline_only,
        iter_retries=args.iter_retries,
    )
