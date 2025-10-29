# mc_judge.py
from __future__ import annotations
from typing import Any
import logging
import pandas as pd

from .base import BaseJudge
from utils import validate_required_columns
from errors import EvaluationError

logger = logging.getLogger(__name__)


class MultipleChoiceJudge(BaseJudge):
    """Judge that compares letter answers for multiple-choice benchmarks."""

    def check_single_answer(self, model_answer: str, true_answer: str) -> int:
        """Return 1 if letters match (case-insensitive), else 0."""
        if true_answer is None:
            raise EvaluationError("true_answer cannot be None for MultipleChoiceJudge.")

        model_letter = str(model_answer).strip().upper()
        true_letter = str(true_answer).strip().upper()

        result = 1 if model_letter == true_letter else 0
        logger.debug("MC check: model=%s, true=%s → %d", model_letter, true_letter, result)
        return result

    def check_answers(
        self,
        meta: dict[str, Any],
        df: pd.DataFrame,
        output_csv_path: str,
    ) -> tuple[dict[str, Any], pd.DataFrame]:
        """Evaluate all answers for a multiple-choice dataset."""
        required_cols = ["model_answer", "true_answer"]
        validate_required_columns(df, required_cols)

        df["is_correct"] = df.apply(
            lambda r: self.check_single_answer(r["model_answer"], r["true_answer"]),
            axis=1,
        )

        meta["judge"] = {"judge_model": self._model_name()}

        df.to_csv(output_csv_path, index=False)
        logger.info("✅ MC evaluation complete. Results saved to %s", output_csv_path)
        return meta, df
