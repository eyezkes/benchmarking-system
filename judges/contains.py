# contains_judge.py
from __future__ import annotations
from typing import Any
import logging
import pandas as pd
import re
import unicodedata

from .base import BaseJudge
from utils import validate_required_columns
from errors import EvaluationError

logger = logging.getLogger(__name__)

def _normalize_text(s: str) -> str:
    """Normalize text for case-insensitive comparison (Unicode-safe)."""
    if s is None:
        return ""
    # Unicode normalize (NFKC) + lower + basic punctuation cleanup
    s = unicodedata.normalize("NFKC", str(s)).casefold()
    s = re.sub(r"[^\w\s’']", " ", s)  # remove punctuation, keep apostrophes
    s = re.sub(r"\s+", " ", s).strip()
    return s


class Contains(BaseJudge):
    """Judge that checks if the true answer appears in the model's answer (for with_true_answer type tasks)."""

    def check_single_answer(
        self,
        model_answer: str,
        true_answer: str,
    ) -> int:
        """Return 1 if true_answer (normalized) is contained in model_answer."""
        if not true_answer:
            raise EvaluationError("true_answer cannot be empty for Contains judge.")

        model_norm = _normalize_text(model_answer)
        truth_norm = _normalize_text(true_answer)

        result = 1 if truth_norm in model_norm else 0

        logger.debug(
            "Contains check: true='%s' in model='%s' → %d",
            truth_norm,
            model_norm,
            result,
        )
        return result

    def check_answers(
        self,
        meta: dict[str, Any],
        df: pd.DataFrame,
        output_csv_path: str,
    ) -> tuple[dict[str, Any], pd.DataFrame]:
        """
        Evaluate multiple string-based answers and mark if model output contains the correct answer.
        """
        required_cols = ["model_answer", "true_answer"]
        validate_required_columns(df, required_cols)

        df["is_correct"] = df.apply(
            lambda r: self.check_single_answer(r["model_answer"], r["true_answer"]),
            axis=1,
        )

        # Add judge metadata
        meta["judge"] = {
            "type": "Contains",
            "judge_model": None,
            "model_params": None,
            "eval_prompt": None,
            "invalid_count":0
        }

        # Save results
        df.to_csv(output_csv_path, index=False)
        logger.info("✅ Contains check complete. Results saved to %s", output_csv_path)
        return meta, df
