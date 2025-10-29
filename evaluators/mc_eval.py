# mc_eval.py
from __future__ import annotations
import json
import pandas as pd
import numpy as np
import ast
import logging
from typing import Any, List, Optional

from errors import EvaluationError
from .base import BaseEvaluator

logger = logging.getLogger(__name__)

LETTER_SPACE = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _parse_options_cell(cell) -> list[str]:
    """Parse 'options' cell that may be stringified list or separated by '||'."""
    if isinstance(cell, list):
        return [str(x) for x in cell]
    if pd.isna(cell):
        return []
    s = str(cell).strip()
    try:
        val = ast.literal_eval(s)
        if isinstance(val, list):
            return [str(x) for x in val]
    except Exception:
        pass
    if "||" in s:
        return [part.strip() for part in s.split("||") if part.strip()]
    return []
 

def _letter_to_index(x: Optional[str]) -> Optional[int]:
    """Convert 'A'/'B'/'C' or 'A)' into an index."""
    if x is None or pd.isna(x):
        return None
    s = str(x).strip()
    if not s:
        return None
    c = s[0].upper()
    return LETTER_SPACE.index(c) if c in LETTER_SPACE else None


class MultipleChoiceEvaluator(BaseEvaluator):
    """Evaluator for multiple-choice benchmarks."""

    def compute(
        self,
        meta: dict[str, Any],
        df: pd.DataFrame,
        output_json_path: str,
    ) -> dict[str, Any]:
        """Compute accuracy, confusion matrix, and optional per-category breakdown."""
        if df.empty:
            raise EvaluationError("Empty dataframe provided to MultipleChoiceEvaluator.")

        df["_opts"] = df.get("options", "").apply(_parse_options_cell)
        df["_k"] = df["_opts"].apply(len)
        df["_t_idx"] = df.get("true_answer", "").apply(_letter_to_index)
        df["_p_idx"] = df.get("model_answer", "").apply(_letter_to_index)

        is_corr = pd.to_numeric(df.get("is_correct", 0), errors="coerce").fillna(0).astype(int)
        accuracy = float(is_corr.mean())

        out: dict[str, Any] = {
            "count": int(len(df)),
            "accuracy": round(accuracy, 6),
        }

        # Confusion matrix
        tp_mask = df["_t_idx"].notna() & df["_p_idx"].notna()
        if tp_mask.any():
            max_k = int(df["_k"].max()) if df["_k"].notna().any() else 0
            labels = [LETTER_SPACE[i] for i in range(max_k)]
            cm = np.zeros((max_k, max_k), dtype=int)
            for _, r in df[tp_mask].iterrows():
                t, p = int(r["_t_idx"]), int(r["_p_idx"])
                if 0 <= t < max_k and 0 <= p < max_k:
                    cm[t, p] += 1
            out["confusion_matrix"] = {"labels": labels, "matrix": cm.tolist()}

        # Per-category accuracy
        if "category" in df.columns:
            grp = df.groupby("category")["is_correct"].mean().reset_index()
            out["per_category"] = [
                {
                    "category": str(c),
                    "count": int((df["category"] == c).sum()),
                    "accuracy": round(float(a), 6),
                }
                for c, a in zip(grp["category"], grp["is_correct"])
            ]

        result = {"metadata": meta, "out": out}

        try:
            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=4)
            logger.info("✅ MC evaluation results saved to %s", output_json_path)
        except OSError as e:
            logger.error("Failed to write MC evaluation results: %s", e)
            raise EvaluationError(f"Could not save output file: {e}") from e

        return result
