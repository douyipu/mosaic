"""
Distribution Agent for dynamic data allocation with Pydantic AI.

The tools are split into two layers:
- `eval_diagnose`: quantitative slice-level diagnosis
- `eval_examples`: inspect L2/L3 details for qualitative analysis
- `train_check`: check the available training pool

The final output is a unified `DistributionPlan`
(`w` + per-dimension `DimPlan` + rationale).

Install with:
    uv pip install pydantic-ai
"""

import asyncio
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
from pydantic import BaseModel, Field

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.tools import Tool

deepseek_model = OpenAIChatModel(
    "deepseek-chat",
    provider=OpenAIProvider(
        base_url="https://api.deepseek.com",
        api_key=os.environ.get("DEEPSEEK_API_KEY"),
    ),
)

# ---------------------------------------------------------------------------
# 数据模型
# ---------------------------------------------------------------------------


class FocusCriteria(BaseModel):
    """Focus conditions for one dimension, expressed with include/exclude.

    The allowed keys in `include` depend on the dimension:
      xguard:  pc_patterns, safety, leak_types, pressure_range, concealment_range
      orbench: refusal_type, help_level, friction
      ifeval:  inst_complexity, failed_constraint
    """

    include: dict[str, Any] = Field(default_factory=dict)
    exclude: dict[str, Any] = Field(default_factory=dict)


class DimPlan(BaseModel):
    """Sampling plan for one dimension."""

    bucket_weights: dict[str, float] = Field(
        default_factory=dict,
        description="Bucket weights, e.g. {'complexity_4': 0.35, 'complexity_5': 0.25, ...}; must sum to 1",
    )
    focus_criteria: FocusCriteria = Field(
        default_factory=FocusCriteria,
        description="L2/L3 focus criteria",
    )


class DistributionPlan(BaseModel):
    """Final distribution plan returned by the agent."""

    w: dict[str, float] = Field(
        description="Macro weights over {xguard, orbench, ifeval}; must sum to 1"
    )
    dims: dict[str, DimPlan] = Field(
        default_factory=dict,
        description="Detailed per-dimension plans",
    )
    rationale: str = Field(default="", description="Decision rationale")


# ---------------------------------------------------------------------------
# 工具调用日志
# ---------------------------------------------------------------------------


def _args_brief(args: dict) -> str:
    parts = []
    for k, v in args.items():
        if isinstance(v, dict) and len(str(v)) > 40:
            parts.append(f"{k}=<dict>")
        elif isinstance(v, list) and len(str(v)) > 40:
            parts.append(f"{k}=<list[{len(v)}]>")
        else:
            parts.append(f"{k}={v!r}")
    return ", ".join(parts[:4])


def _serialize_for_log(obj: Any) -> Any:
    if obj is None:
        return None
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "__dict__"):
        return str(obj)[:500]
    return obj


# ---------------------------------------------------------------------------
# Agent 上下文
# ---------------------------------------------------------------------------

VALID_DIMENSIONS = {"xguard", "orbench", "ifeval"}


@dataclass
class AgentContext:
    eval_results_path: Path
    target_score: float = 3.2
    token_budget: int = 1_000_000
    trace_log_path: Path | None = None
    _cache: dict = field(default_factory=dict)

    def _log_tool_call(
        self, tool_name: str, args: dict, result_summary: str, result_full: Any = None
    ):
        print(f"  [tool] {tool_name}({_args_brief(args)}) -> {result_summary}")
        if self.trace_log_path:
            entry = {
                "tool": tool_name,
                "args": args,
                "result_summary": result_summary,
                "result_full": _serialize_for_log(result_full),
            }
            try:
                with open(self.trace_log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
            except Exception as e:
                print(f"  [tool] failed to write trace log: {e}")

    def get_eval_data(self) -> dict:
        if "eval_data" not in self._cache:
            with open(self.eval_results_path, "r", encoding="utf-8") as f:
                self._cache["eval_data"] = json.load(f)
        return self._cache["eval_data"]

    def get_train_pool(self, dimension: str):
        """Load a training pool lazily and cache it. Returns a pandas DataFrame or None."""
        cache_key = f"train_pool_{dimension}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            import pandas as pd
        except ImportError:
            self._cache[cache_key] = None
            return None

        project_root = Path(__file__).resolve().parent.parent
        dataset_dir = project_root / "dataset"
        finetune_data = project_root / "packages" / "finetune" / "data"

        source_map = {
            "xguard": (
                dataset_dir / "train" / "xguard-train.parquet",
                finetune_data / "xguard-train-slided-4096.parquet",
            ),
            "orbench": (
                dataset_dir / "train" / "orbench.parquet",
                finetune_data / "orbench-slided-4096.parquet",
            ),
            "ifeval": (
                dataset_dir / "train" / "tulu-3.parquet",
                finetune_data / "tulu-3-slided-4096.parquet",
            ),
        }
        if dimension not in source_map:
            self._cache[cache_key] = None
            return None

        raw_path, slided_path = source_map[dimension]
        path = slided_path if slided_path.exists() else raw_path
        if not path.exists():
            self._cache[cache_key] = None
            return None

        try:
            df = pd.read_parquet(path)
            rename_map = {}
            if "score" in df.columns and "_score" not in df.columns:
                rename_map["score"] = "_score"
            if "slice" in df.columns and "_slice" not in df.columns:
                rename_map["slice"] = "_slice"
            if rename_map:
                df = df.rename(columns=rename_map)
            if "_win_tokens" in df.columns:
                df["_token_estimate"] = df["_win_tokens"].fillna(0).astype(int)
            elif "_orig_tokens" in df.columns:
                df["_token_estimate"] = df["_orig_tokens"].fillna(0).astype(int)
            self._cache[cache_key] = df
            return df
        except Exception:
            self._cache[cache_key] = None
            return None


# ---------------------------------------------------------------------------
# 工具：Eval 诊断层
# ---------------------------------------------------------------------------


def _is_fail(sample: dict, dimension: str) -> bool:
    rc = sample.get("response_channel", {})
    if dimension == "xguard":
        return rc.get("l2", {}).get("safety", "unknown") != "safe"
    return float(rc.get("score", 0)) < 3.0


def _build_breakdown(samples: list, dimension: str) -> dict[str, Any]:
    """Build a dimension-specific diagnostic breakdown."""
    if dimension == "xguard":
        p_dist: dict[int, int] = {}
        c_dist: dict[int, int] = {}
        leak_types: dict[str, int] = {}
        pc_fail: dict[str, list] = {}

        for s in samples:
            pc = s.get("prompt_channel", {})
            rc = s.get("response_channel", {})
            p = pc.get("l2", {}).get("pressure", 0)
            c = pc.get("l2", {}).get("concealment", 0)
            p_dist[p] = p_dist.get(p, 0) + 1
            c_dist[c] = c_dist.get(c, 0) + 1

            pc_key = f"P{p}_C{c}"
            if pc_key not in pc_fail:
                pc_fail[pc_key] = [0, 0]
            pc_fail[pc_key][1] += 1
            if rc.get("l2", {}).get("safety", "unknown") != "safe":
                pc_fail[pc_key][0] += 1

            for lt, val in rc.get("l3", {}).get("safety_leaks", {}).items():
                if val:
                    leak_types[lt] = leak_types.get(lt, 0) + 1

        high_risk = [
            f"{k}({f}/{t})" for k, (f, t) in pc_fail.items() if t >= 3 and f / t > 0.3
        ]
        return {
            "pressure_dist": p_dist,
            "concealment_dist": c_dist,
            "leak_types": leak_types,
            "high_risk_combos": high_risk,
        }

    if dimension == "orbench":
        r_dist: dict[str, int] = {}
        h_dist: dict[str, int] = {}
        f_dist: dict[str, int] = {}
        combo_fail: dict[str, list] = {}

        for s in samples:
            l2 = s.get("response_channel", {}).get("l2", {})
            r = l2.get("refusal_type", "unknown")
            h = l2.get("help_level", "unknown")
            f = l2.get("friction", "unknown")
            r_dist[r] = r_dist.get(r, 0) + 1
            h_dist[h] = h_dist.get(h, 0) + 1
            f_dist[f] = f_dist.get(f, 0) + 1

            combo = f"{r}_{h}_{f}"
            if combo not in combo_fail:
                combo_fail[combo] = [0, 0]
            combo_fail[combo][1] += 1
            if s["response_channel"]["score"] < 3.0:
                combo_fail[combo][0] += 1

        high_risk = [
            f"{k}({fail}/{tot})"
            for k, (fail, tot) in combo_fail.items()
            if tot >= 3 and fail / tot > 0.3
        ]
        return {
            "refusal_type_dist": r_dist,
            "help_level_dist": h_dist,
            "friction_dist": f_dist,
            "high_risk_combos": high_risk,
        }

    if dimension == "ifeval":
        fc_dist: dict[str, int] = {}
        r_strict_sum = 0.0
        r_loose_sum = 0.0
        r_count = 0

        for s in samples:
            rc = s.get("response_channel", {})
            for cid in rc.get("diag", {}).get("failed_constraints", []):
                fc_dist[cid] = fc_dist.get(cid, 0) + 1
            l2 = rc.get("l2", {})
            if "r_strict" in l2 and "r_loose" in l2:
                r_strict_sum += l2["r_strict"]
                r_loose_sum += l2["r_loose"]
                r_count += 1

        return {
            "failed_constraints_dist": fc_dist,
            "avg_r_strict": round(r_strict_sum / r_count, 3) if r_count else 0.0,
            "avg_r_loose": round(r_loose_sum / r_count, 3) if r_count else 0.0,
            "top_failed_constraints": [
                f"{cid}({cnt})"
                for cid, cnt in sorted(fc_dist.items(), key=lambda x: -x[1])[:10]
                if cnt >= 2
            ],
        }

    return {}


FOCUS_KEYS_BY_DIM: dict[str, list[str]] = {
    "xguard": [
        "pc_patterns",
        "safety",
        "leak_types",
        "pressure_range",
        "concealment_range",
    ],
    "orbench": ["refusal_type", "help_level", "friction"],
    "ifeval": ["inst_complexity", "failed_constraint"],
}


def _reliability_level(n: int) -> str:
    if n < 10:
        return "low"
    if n < 30:
        return "medium"
    return "high"


def _score_ci(scores: list[float], confidence: float = 0.95) -> tuple[float, float]:
    """Bootstrap confidence interval (percentile method, 2000 resamples)."""
    import random

    n = len(scores)
    if n < 2:
        mean = scores[0] if scores else 0.0
        return (mean, mean)
    rng = random.Random(42)
    means = sorted(sum(rng.choices(scores, k=n)) / n for _ in range(2000))
    alpha = (1 - confidence) / 2
    lo = means[int(2000 * alpha)]
    hi = means[int(2000 * (1 - alpha)) - 1]
    return (round(lo, 3), round(hi, 3))


async def eval_diagnose(
    ctx: RunContext[AgentContext],
    dimension: str,
    slice_key: str,
) -> dict[str, Any]:
    """
    Quantitatively diagnose one slice and return a shared header plus a
    dimension-specific breakdown.

    The result also includes:
    - reliability: low / medium / high
    - available_focus_keys: valid include keys for focus_criteria

    Args:
        dimension: xguard | orbench | ifeval
        slice_key: slice label. xguard: complexity_1~5,
            orbench: boundary_1~4, ifeval: complexity_1~5
    """
    if dimension not in VALID_DIMENSIONS:
        result = {
            "error": f"Unknown dimension {dimension}; expected one of {sorted(VALID_DIMENSIONS)}"
        }
        ctx.deps._log_tool_call(
            "eval_diagnose",
            {"dimension": dimension, "slice_key": slice_key},
            "error",
            result,
        )
        return result

    data = ctx.deps.get_eval_data()
    details = data.get(dimension, {}).get("details", [])
    samples = [
        s for s in details if s.get("prompt_channel", {}).get("slice") == slice_key
    ]

    if not samples:
        result = {
            "dimension": dimension,
            "slice_key": slice_key,
            "total": 0,
            "note": "No matching samples",
            "available_focus_keys": FOCUS_KEYS_BY_DIM.get(dimension, []),
        }
        ctx.deps._log_tool_call(
            "eval_diagnose",
            {"dimension": dimension, "slice_key": slice_key},
            "0 samples",
            result,
        )
        return result

    n = len(samples)
    fail_count = sum(1 for s in samples if _is_fail(s, dimension))
    scores = [s["response_channel"]["score"] for s in samples]
    avg_score = sum(scores) / n
    ci_lo, ci_hi = _score_ci(scores)

    reliability = _reliability_level(n)
    result = {
        "dimension": dimension,
        "slice_key": slice_key,
        "total": n,
        "avg_score": round(avg_score, 3),
        "score_95ci": [ci_lo, ci_hi],
        "fail_rate": round(fail_count / n * 100, 1),
        "gap_to_target": round(ctx.deps.target_score - avg_score, 3),
        "reliability": reliability,
        "breakdown": _build_breakdown(samples, dimension),
        "available_focus_keys": FOCUS_KEYS_BY_DIM.get(dimension, []),
    }
    if reliability != "high":
        ci_width = ci_hi - ci_lo
        result["warning"] = (
            f"Only {n} samples are available (95% CI width={ci_width:.2f}). "
            f"This slice is {'highly unstable' if reliability == 'low' else 'not yet stable enough'}, "
            f"so {'do not' if reliability == 'low' else 'only cautiously'} change its weight."
        )

    ctx.deps._log_tool_call(
        "eval_diagnose",
        {"dimension": dimension, "slice_key": slice_key},
        f"{n} samples, fail_rate={result['fail_rate']}%, reliability={reliability}",
        result,
    )
    return result


def _sample_to_anonymous(s: dict, dimension: str) -> dict[str, Any]:
    """Convert a sample into an anonymous record for qualitative analysis."""
    rc = s.get("response_channel", {})
    pc = s.get("prompt_channel", {})
    record: dict[str, Any] = {
        "score": float(rc.get("score", 0)),
        "status": "failed" if _is_fail(s, dimension) else "passed",
        "l2": {},
        "l3": {},
    }

    if dimension == "xguard":
        record["l2"] = {
            "pressure": pc.get("l2", {}).get("pressure", 0),
            "concealment": pc.get("l2", {}).get("concealment", 0),
            "safety": rc.get("l2", {}).get("safety", "unknown"),
        }
        record["l3"] = {
            "pressure": pc.get("l3", {}).get("pressure", {}),
            "concealment": pc.get("l3", {}).get("concealment", {}),
            "safety_leaks": rc.get("l3", {}).get("safety_leaks", {}),
        }
    elif dimension == "orbench":
        record["l2"] = {
            "boundary_level": pc.get("l2", {}).get("boundary_level", 0),
            "refusal_type": rc.get("l2", {}).get("refusal_type", "unknown"),
            "help_level": rc.get("l2", {}).get("help_level", "unknown"),
            "friction": rc.get("l2", {}).get("friction", "unknown"),
        }
        record["l3"] = {
            k: rc.get("l3", {}).get(k, {}) for k in ("refusal", "help", "friction")
        }
    elif dimension == "ifeval":
        record["l2"] = {
            "inst_complexity": pc.get("l2", {}).get("inst_complexity", 0),
            "r_strict": rc.get("l2", {}).get("r_strict", 0),
            "r_loose": rc.get("l2", {}).get("r_loose", 0),
        }
        record["l3"] = rc.get("l3", {})
        record["diag"] = {
            "failed_constraints": rc.get("diag", {}).get("failed_constraints", [])
        }

    return record


async def eval_examples(
    ctx: RunContext[AgentContext],
    dimension: str,
    slice_key: str,
    condition: str = "failed",
    score_range: list[int] | None = None,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """
    Inspect sample-level L2/L3 details for qualitative analysis.

    Args:
        dimension: xguard | orbench | ifeval
        slice_key: slice label
        condition: failed | passed | all | borderline
        score_range: optional score filter, e.g. [2, 3]
        limit: number of examples to return
    """
    if dimension not in VALID_DIMENSIONS:
        ctx.deps._log_tool_call(
            "eval_examples", {"dimension": dimension}, "unknown dimension", []
        )
        return []

    data = ctx.deps.get_eval_data()
    details = data.get(dimension, {}).get("details", [])
    samples = [
        s for s in details if s.get("prompt_channel", {}).get("slice") == slice_key
    ]

    if condition == "borderline":
        score_range = score_range or [2, 3]

    if score_range and len(score_range) == 2:
        lo, hi = score_range
        samples = [s for s in samples if lo <= s["response_channel"]["score"] <= hi]
    elif condition == "failed":
        samples = [s for s in samples if _is_fail(s, dimension)]
    elif condition == "passed":
        samples = [s for s in samples if not _is_fail(s, dimension)]

    samples = samples[:limit]
    result = [_sample_to_anonymous(s, dimension) for s in samples]
    ctx.deps._log_tool_call(
        "eval_examples",
        {
            "dimension": dimension,
            "slice_key": slice_key,
            "condition": condition,
            "score_range": score_range,
            "limit": limit,
        },
        f"returned {len(result)} samples",
        result,
    )
    return result


# ---------------------------------------------------------------------------
# 工具：Train 可执行层
# ---------------------------------------------------------------------------


async def train_check(
    ctx: RunContext[AgentContext],
    dimension: str,
    slice_key: str = "",
) -> dict[str, Any]:
    """
    Query the available sample count and token estimate for one training pool,
    optionally restricted to a slice.

    Args:
        dimension: xguard | orbench | ifeval
        slice_key: optional slice key, e.g. complexity_4
    """
    if dimension not in VALID_DIMENSIONS:
        result = {"error": f"Unknown dimension {dimension}"}
        ctx.deps._log_tool_call(
            "train_check", {"dimension": dimension}, "error", result
        )
        return result

    pool = ctx.deps.get_train_pool(dimension)
    if pool is None:
        result = {
            "dimension": dimension,
            "available": False,
            "note": "Training pool unavailable (missing parquet files or pandas).",
        }
        ctx.deps._log_tool_call(
            "train_check",
            {"dimension": dimension, "slice_key": slice_key},
            "unavailable",
            result,
        )
        return result

    df = pool
    if slice_key and "_slice" in df.columns:
        df = df[df["_slice"] == slice_key]

    total_tokens = (
        int(df["_token_estimate"].sum()) if "_token_estimate" in df.columns else 0
    )
    slice_dist = df["_slice"].value_counts().to_dict() if "_slice" in df.columns else {}

    result = {
        "dimension": dimension,
        "slice_key": slice_key or "(all)",
        "available": True,
        "total_samples": len(df),
        "estimated_tokens": total_tokens,
        "slice_distribution": slice_dist,
    }
    ctx.deps._log_tool_call(
        "train_check",
        {"dimension": dimension, "slice_key": slice_key},
        f"{len(df)} samples, ~{total_tokens:,} tokens",
        result,
    )
    return result


# ---------------------------------------------------------------------------
# Agent 定义
# ---------------------------------------------------------------------------

system_prompt = """You are an expert data-mixture optimization agent. Your job is to propose the next training distribution from evaluation results.

## Available tools

### eval_diagnose(dimension, slice_key)
Returns avg_score, score_95ci, fail_rate, gap_to_target, breakdown, reliability, and available_focus_keys.
- score_95ci: 95% bootstrap confidence interval [lo, hi]
- reliability: low (<10 samples) / medium (10-29) / high (>=30)
  - low: do not make decisions based on this slice
  - medium: adjust conservatively
- available_focus_keys lists the allowed focus keys for that dimension

### eval_examples(dimension, slice_key, condition, score_range, limit)
Inspect L2/L3 labels without exposing sample IDs.
- condition: "failed" | "passed" | "borderline" | "all"
- score_range: optional, e.g. [2, 3] for borderline cases

### train_check(dimension, slice_key)
Query available samples and token estimates in the training pool.

## Recommended workflow

1. Read the evaluation overview and iteration history first.
2. Use `eval_diagnose` on slices with the largest gap or failure rate.
3. Use `eval_examples(condition="borderline")` to inspect near-miss failures.
4. Use `train_check` before allocating budget.
5. Return a valid `DistributionPlan`.

## Output format

- **w**: macro weights over the three objectives; values must be >= 0 and sum to 1
- **dims**: one `DimPlan` per dimension
  - **bucket_weights**: slice weights inside that dimension; must sum to 1
  - **focus_criteria**: include / exclude patterns at the L2/L3 level
- **rationale**: concise explanation of the decision

Example:
```json
{
  "xguard": {
    "bucket_weights": {"complexity_3": 0.15, "complexity_4": 0.40, "complexity_5": 0.45},
    "focus_criteria": {
      "include": {"pc_patterns": ["P4_C3", "P5_C3"], "safety": ["partial_leak"]},
      "exclude": {}
    }
  }
}
```

## Constraints

1. Do not invent new data; only reweight what exists in the pool.
2. Prioritize safety-critical slices when there is clear leakage.
3. Respect reliability and confidence intervals:
   - low reliability: avoid meaningful reweighting
   - medium reliability: only adjust conservatively and explain why
   - wider CIs imply more conservative changes
4. Do not over-sacrifice the other objectives.
5. If the prompt includes the previous round and deltas, analyze what changed before proposing the next distribution.
"""

distribution_agent = Agent(
    deepseek_model,
    system_prompt=system_prompt,
    output_type=DistributionPlan,
    deps_type=AgentContext,
    tools=[
        Tool(eval_diagnose),
        Tool(eval_examples),
        Tool(train_check),
    ],
)


# ---------------------------------------------------------------------------
# get_next_distribution — run_loop 调用的主接口
# ---------------------------------------------------------------------------


async def get_next_distribution(
    frontier_summary: str,
    history_table: str,
    current_record: dict[str, Any],
    T_max: int = 1_000_000,
    eval_details: str = "",
    source_stats: dict[str, Any] | None = None,
    prev_record: dict[str, Any] | None = None,
    n_samples: int = 1,
    temperature: float = 0.0,
    aggregate_method: str = "median",
    max_retries: int = 3,
    **kwargs,
) -> tuple[dict[str, float], dict[str, dict], dict[str, Any]]:
    """
    Use the agent to analyze evaluation results and decide the next training distribution.

    Returns:
        (w, dims, extra_info)
        - w: {"xguard": 0.6, "orbench": 0.3, "ifeval": 0.1}
        - dims: {"xguard": {"bucket_weights": {...}, "focus_criteria": {...}}, ...}
        - extra_info: {"rationale": str}
    """
    import asyncio

    from dotenv import load_dotenv

    load_dotenv()

    eval_results_path = None
    if source_stats and "_eval_results_path" in source_stats:
        eval_results_path = source_stats["_eval_results_path"]

    if not eval_results_path:
        results_dir = Path(__file__).parent.parent / "results"
        if results_dir.exists():
            runs = sorted(results_dir.glob("run_*"), reverse=True)
            for run_dir in runs:
                eval_file = run_dir / "iter_00" / "eval_results.json"
                if eval_file.exists():
                    eval_results_path = eval_file
                    break

    if not eval_results_path or not Path(eval_results_path).exists():
        print(
            "[distribution_agent] warning: eval_results.json not found, falling back to the default distribution"
        )
        return {"xguard": 1.0, "orbench": 0.0, "ifeval": 0.0}, {}, {}

    print(f"[distribution_agent] analyzing evaluation file: {eval_results_path}")

    trace_log_path = None
    if source_stats and "_trace_log_path" in source_stats:
        trace_log_path = Path(source_stats["_trace_log_path"])

    last_exception = None
    for attempt in range(max_retries):
        try:
            plan = await run_agent(
                eval_results_path=str(eval_results_path),
                frontier_summary=frontier_summary,
                history_table=history_table,
                current_record=current_record,
                prev_record=prev_record,
                eval_details=eval_details,
                target_score=3.2,
                token_budget=T_max,
                trace_log_path=trace_log_path,
            )

            print("[distribution_agent] agent decision completed")
            print(f"[distribution_agent] rationale: {plan.rationale[:200]}...")

            dims_dict = {
                dim_name: dim_plan.model_dump()
                for dim_name, dim_plan in plan.dims.items()
            }

            extra_info = {"rationale": plan.rationale}
            return plan.w, dims_dict, extra_info

        except Exception as e:
            last_exception = e
            if attempt < max_retries - 1:
                wait_time = 2**attempt
                print(
                    f"[distribution_agent] agent run failed (attempt {attempt + 1}/{max_retries}): {e}"
                )
                print(f"[distribution_agent] retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
            else:
                break

    print(
        f"[distribution_agent] agent run failed after {max_retries} attempts: {last_exception}"
    )
    print("[distribution_agent] falling back to the default distribution")
    import traceback

    traceback.print_exc()
    return {"xguard": 1.0, "orbench": 0.0, "ifeval": 0.0}, {}, {}


# ---------------------------------------------------------------------------
# 运行入口
# ---------------------------------------------------------------------------


def _build_delta_section(current_record: dict, prev_record: dict | None) -> str:
    """Build a delta section so the agent can compare the previous and current rounds."""
    if not prev_record:
        return ""

    lines = ["## Previous-round review (analyze this before proposing the next step)\n"]

    prev_w = prev_record.get("w", {})
    prev_f = prev_record.get("f", (0, 0, 0))
    curr_f = current_record.get("f", (0, 0, 0))

    lines.append(f"Previous distribution: {json.dumps(prev_w, ensure_ascii=False)}")

    dim_names = ["xguard", "orbench", "ifeval"]
    for i, dim in enumerate(dim_names):
        prev_score = prev_f[i] if i < len(prev_f) else 0
        curr_score = curr_f[i] if i < len(curr_f) else 0
        delta = curr_score - prev_score
        arrow = "↑" if delta > 0.01 else ("↓" if delta < -0.01 else "→")
        lines.append(
            f"  {dim}: {prev_score:.4f} -> {curr_score:.4f} ({arrow}{abs(delta):.4f})"
        )

    prev_dims = prev_record.get("dims", {})
    if prev_dims:
        lines.append("\nPrevious per-dimension plan:")
        for dim, plan in prev_dims.items():
            bw = plan.get("bucket_weights", {})
            fc = plan.get("focus_criteria", {}).get("include", {})
            if bw:
                lines.append(
                    f"  {dim} bucket_weights: {json.dumps(bw, ensure_ascii=False)}"
                )
            if fc:
                lines.append(
                    f"  {dim} focus_include: {json.dumps(fc, ensure_ascii=False)}"
                )

    lines.append(
        "\n**Answer this first**: what improved or regressed after the previous change, and should the strategy be continued or corrected?"
    )
    return "\n".join(lines)


async def run_agent(
    eval_results_path: str,
    frontier_summary: str = "",
    history_table: str = "",
    current_record: dict[str, Any] | None = None,
    prev_record: dict[str, Any] | None = None,
    eval_details: str = "",
    target_score: float = 3.2,
    token_budget: int = 1_000_000,
    trace_log_path: Path | None = None,
) -> DistributionPlan:
    if trace_log_path:
        trace_log_path.parent.mkdir(parents=True, exist_ok=True)
        trace_log_path.write_text("", encoding="utf-8")

    ctx = AgentContext(
        eval_results_path=Path(eval_results_path),
        target_score=target_score,
        token_budget=token_budget,
        trace_log_path=trace_log_path,
    )

    record = current_record or {}
    iteration = record.get("iter", 0)
    w_str = json.dumps(record.get("w", {}), ensure_ascii=False)
    f = record.get("f", (0, 0, 0))
    accepted = record.get("accepted", False)

    delta_section = _build_delta_section(record, prev_record)

    user_prompt = f"""You are working on iteration {iteration}.

## Pareto frontier
{frontier_summary if frontier_summary else "(none)"}

## Full iteration history
{history_table if history_table else "(none)"}

{delta_section}

## Current round
- Current distribution w: {w_str}
- Scores: xguard={f[0] if len(f) > 0 else 0:.4f}, orbench={f[1] if len(f) > 1 else 0:.4f}, ifeval={f[2] if len(f) > 2 else 0:.4f}
- Accepted by frontier: {"yes" if accepted else "no"}

## Evaluation overview
{eval_details if eval_details else "(none)"}

## Task
1. Review the previous round first if previous-round data is available.
2. Identify the dimensions and slices with the largest gap or failure rate.
3. Use `eval_diagnose` for deeper analysis and respect the reliability field.
4. Use `eval_examples(condition="borderline")` to inspect near-miss failures.
5. Use `train_check` before allocating budget.
6. Return a valid `DistributionPlan`.

Target score: {target_score}
Token budget: {token_budget:,}
"""

    result = await distribution_agent.run(user_prompt, deps=ctx)
    return result.output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


async def main():
    import sys

    if len(sys.argv) < 2:
        print("Usage: uv run python scripts/distribution_agent.py <eval_results.json>")
        print(
            "Example: uv run python scripts/distribution_agent.py results/run_xxx/iter_00/eval_results.json"
        )
        sys.exit(1)

    eval_path = sys.argv[1]

    print("=" * 60)
    print("Running Distribution Agent...")
    print("=" * 60)

    plan = await run_agent(eval_path, current_record={"iter": 0})

    print("\n" + "=" * 60)
    print("Agent Decision:")
    print("=" * 60)
    print(f"\nRationale:\n{plan.rationale}")
    print(f"\nMacro Distribution (w):\n{json.dumps(plan.w, indent=2)}")
    for dim_name, dim_plan in plan.dims.items():
        print(f"\n[{dim_name}]")
        print(f"  Bucket Weights: {json.dumps(dim_plan.bucket_weights, indent=2)}")
        print(
            f"  Focus Criteria: {json.dumps(dim_plan.focus_criteria.model_dump(), indent=2)}"
        )


if __name__ == "__main__":
    asyncio.run(main())
