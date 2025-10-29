# prompt_eval.py
from __future__ import annotations
import json
import pandas as pd
import logging
from typing import Any
from errors import EvaluationError
from .base import BaseEvaluator

logger = logging.getLogger(__name__)


class PromptBasedEvaluator(BaseEvaluator):
    """Evaluator for 1–10 LLM-as-a-judge scoring benchmarks."""

    def compute(
        self,
        meta: dict[str, Any],
        df: pd.DataFrame,
        output_json_path: str,
    ) -> dict[str, Any]:
        """Compute mean, std, and percentiles for LLM-based numeric scores."""
        if df.empty:
            raise EvaluationError("Empty dataframe passed to PromptBasedEvaluator.")

        if "score" not in df.columns:
            raise EvaluationError("PromptBasedEvaluator requires a 'score' column.")

        s = pd.to_numeric(df["score"], errors="coerce").dropna()
        if len(s) == 0:
            stats = {"count": 0, "average_score": 0.0, "std_score": 0.0,
                     "percentiles": {"p25": 0.0, "p50": 0.0, "p75": 0.0}}
        else:
            stats = {
                "count": int(len(s)),
                "average_score": round(float(s.mean()), 6),
                "std_score": round(float(s.std(ddof=0)), 6),
                "percentiles": {
                    "p25": round(float(s.quantile(0.25)), 6),
                    "p50": round(float(s.quantile(0.5)), 6),
                    "p75": round(float(s.quantile(0.75)), 6),
                },
            }

        result = {"metadata": meta, "out": stats}

        try:
            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=4)
            logger.info("✅ Prompt-based evaluation saved to %s", output_json_path)
        except OSError as e:
            logger.error("Failed to save prompt-based evaluation: %s", e)
            raise EvaluationError(f"Could not save output file: {e}") from e

        return result
