#!/usr/bin/env python3
"""
Pareto frontier archive for multi-objective optimization.

Maintains a set of non-dominated solutions across iterations.
Each record contains: data distribution `w`, three objective scores `f`,
iteration number, and optional metadata.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def dominates(a: tuple, b: tuple) -> bool:
    """Return True if `a` Pareto-dominates `b` (all >= and at least one >)."""
    at_least_one_better = False
    for ai, bi in zip(a, b):
        if ai < bi:
            return False
        if ai > bi:
            at_least_one_better = True
    return at_least_one_better


class ParetoArchive:
    """Maintain a set of non-dominated (Pareto-optimal) configurations."""

    OBJ_NAMES = ("xguard_score", "orbench_score", "ifeval_score")

    def __init__(self, archive_path: Optional[Path] = None):
        self.records: List[Dict[str, Any]] = []
        self.archive_path = archive_path
        if archive_path and archive_path.exists():
            self._load(archive_path)

    # ------------------------------------------------------------------
    # Core
    # ------------------------------------------------------------------

    def add(self, record: Dict[str, Any]) -> bool:
        """
        Try to add a record to the archive.

        A record is accepted if:
          - It dominates at least one existing member, OR
          - It is not dominated by any existing member (new non-dominated point)

        Returns True if accepted into the frontier.

        record must contain:
          - "f": (f1, f2, f3)  — three objective values (higher is better)
          - "w": dict           — data distribution
          - "iter": int         — iteration number
        """
        new_f = tuple(record["f"])

        # Check if new record is dominated by or duplicates any existing
        for existing in self.records:
            existing_f = tuple(existing["f"])
            if dominates(existing_f, new_f):
                return False
            if existing_f == new_f:
                return False

        # Remove any existing records dominated by the new one
        self.records = [
            r for r in self.records
            if not dominates(new_f, tuple(r["f"]))
        ]

        self.records.append(record)

        if self.archive_path:
            self._save(self.archive_path)

        return True

    def get_frontier(self) -> List[Dict[str, Any]]:
        """Return all non-dominated records."""
        return list(self.records)

    def size(self) -> int:
        return len(self.records)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def to_markdown_table(self) -> str:
        """Generate a human-readable markdown table of the frontier."""
        lines = []
        lines.append("| Iter | XGuard | OrBench | IFEval | Distribution |")
        lines.append("|------|--------|---------|--------|--------------|")
        for r in sorted(self.records, key=lambda x: x.get("iter", 0)):
            f = r["f"]
            w = r.get("w", {})
            w_str = ", ".join(f"{k}={v:.2f}" for k, v in w.items()) if isinstance(w, dict) else str(w)
            iter_display = r.get("iter_label", r.get("iter", "?"))
            lines.append(f"| {iter_display} | {f[0]:.4f} | {f[1]:.4f} | {f[2]:.4f} | {w_str} |")
        return "\n".join(lines)

    def summary_for_llm(self) -> str:
        """Generate a structured text summary for the LLM advisor."""
        parts = ["## Current Pareto Frontier\n"]
        if not self.records:
            parts.append("(empty — no iterations completed yet)\n")
            return "\n".join(parts)

        parts.append(self.to_markdown_table())
        parts.append("")

        # Best per objective
        for i, name in enumerate(self.OBJ_NAMES):
            best = max(self.records, key=lambda r: r["f"][i])
            parts.append(f"- Best {name}: {best['f'][i]:.4f} (iter {best.get('iter', '?')})")

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.records, fh, ensure_ascii=False, indent=2)

    def _load(self, path: Path):
        with open(path, "r", encoding="utf-8") as fh:
            self.records = json.load(fh)

    def save(self, path: Optional[Path] = None):
        p = path or self.archive_path
        if p:
            self._save(p)

    # ------------------------------------------------------------------
    # History helpers (all iterations, including dominated ones)
    # ------------------------------------------------------------------

    @staticmethod
    def load_history(path: Path) -> List[Dict[str, Any]]:
        """Load full iteration history (all attempts, not just frontier)."""
        if not path.exists():
            return []
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    @staticmethod
    def save_history(history: List[Dict[str, Any]], path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(history, fh, ensure_ascii=False, indent=2)

    @staticmethod
    def history_to_markdown(history: List[Dict[str, Any]]) -> str:
        """Generate markdown table from full history."""
        lines = []
        lines.append("| Iter | XGuard | OrBench | IFEval | Accepted | Distribution |")
        lines.append("|------|--------|---------|--------|----------|--------------|")
        for r in history:
            f = r["f"]
            w = r.get("w", {})
            w_str = ", ".join(f"{k}={v:.2f}" for k, v in w.items()) if isinstance(w, dict) else str(w)
            acc = "[OK]" if r.get("accepted", False) else "[NO]"
            iter_display = r.get("iter_label", r.get("iter", "?"))
            lines.append(f"| {iter_display} | {f[0]:.4f} | {f[1]:.4f} | {f[2]:.4f} | {acc} | {w_str} |")
        return "\n".join(lines)
