# string_eval.py
from __future__ import annotations
import json
import pandas as pd
import logging
from typing import Any
from errors import EvaluationError
from .base import BaseEvaluator

logger = logging.getLogger(__name__)

class AccuracyEvaluator(BaseEvaluator):
    """Evaluator for boolean tasks (computes accuracy)."""

    def compute(
        self,
        meta: dict[str, Any],
        df: pd.DataFrame,
        output_json_path: str,
    ) -> dict[str, Any]:
        """Compute accuracy and coverage stats."""
        if df.empty:
            raise EvaluationError("Empty dataframe passed to AccuracyEvaluator.")

        if "is_correct" not in df.columns:
            raise EvaluationError("Missing required column: 'is_correct'.")

        s = pd.to_numeric(df["is_correct"], errors="coerce")

        valid = int(s.notna().sum())
        invalid = int(s.size-valid)


        accuracy = float(s.mean(skipna=True)) if valid > 0 else None

        out = {
            "type": "accuracy",
            "valid_count": valid,
            "invalid_count":invalid,
            "metrics": {
                "accuracy": round(accuracy, 4) if accuracy is not None else None
            },
        }

        result = {"metadata": meta, "out": out}

        try:
            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=4)
            logger.info("✅ Accuracy evaluation saved to %s", output_json_path)
        except OSError as e:
            logger.error("Failed to save accuracy evaluation: %s", e)
            raise EvaluationError(f"Could not save output file: {e}") from e

        return result
