# string_eval.py
from __future__ import annotations
import json
import pandas as pd
import logging
from typing import Any
from errors import EvaluationError
from .base import BaseEvaluator

logger = logging.getLogger(__name__)


class StringBasedEvaluator(BaseEvaluator):
    """Evaluator for string-based tasks using 0/1 correctness."""

    def compute(
        self,
        meta: dict[str, Any],
        df: pd.DataFrame,
        output_json_path: str,
    ) -> dict[str, Any]:
        """Compute accuracy and optional per-category breakdown."""
        if df.empty:
            raise EvaluationError("Empty dataframe passed to StringBasedEvaluator.")

        df["is_correct"] = pd.to_numeric(df["is_correct"], errors="coerce").fillna(0)
        accuracy = float(df["is_correct"].mean())

        out = {"count": int(len(df)), "accuracy": round(accuracy, 4)}

        if "category" in df.columns:
            grp = df.groupby("category")["is_correct"].mean().reset_index()
            out["per_category"] = [
                {
                    "category": str(c),
                    "count": int((df["category"] == c).sum()),
                    "accuracy": round(float(a), 4),
                }
                for c, a in zip(grp["category"], grp["is_correct"])
            ]

        result = {"metadata": meta, "out": out}

        try:
            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=4)
            logger.info("✅ String-based evaluation saved to %s", output_json_path)
        except OSError as e:
            logger.error("Failed to save string-based evaluation: %s", e)
            raise EvaluationError(f"Could not save output file: {e}") from e

        return result
