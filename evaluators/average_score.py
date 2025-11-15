# score_eval.py
from __future__ import annotations
import json
import pandas as pd
import logging
from typing import Any
from errors import EvaluationError
from .base import BaseEvaluator

logger = logging.getLogger(__name__)

class ScoreEvaluator(BaseEvaluator):
    """Evaluator for numeric scores (e.g., 0–10). Computes average and std."""

    def compute(
        self,
        meta: dict[str, Any],
        df: pd.DataFrame,
        output_json_path: str,
    ) -> dict[str, Any]:
        """Compute mean, std, and validity stats."""
        if df.empty:
            raise EvaluationError("Empty dataframe passed to ScoreEvaluator.")

        if "score" not in df.columns:
            raise EvaluationError("Missing required column: 'score'.")

        s = pd.to_numeric(df["score"], errors="coerce")
        valid = int(s.notna().sum())
        invalid = int(s.size-valid)


        avg = float(s.mean(skipna=True)) if valid > 0 else None
        std = float(s.std(skipna=True, ddof=1)) if valid > 1 else None

        out = {
            "type": "score_avg",
            "valid_count": valid,
            "invalid_count":invalid,
            "metrics": {
                "avg": round(avg, 4) if avg is not None else None,
                "std": round(std, 4) if std is not None else None,
            },
        }

        result = {"metadata": meta, "out": out}

        try:
            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=4)
            logger.info("✅ Score-based evaluation saved to %s", output_json_path)
        except OSError as e:
            logger.error("Failed to save score-based evaluation: %s", e)
            raise EvaluationError(f"Could not save output file: {e}") from e

        return result
