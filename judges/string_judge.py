# string_judge.py
from __future__ import annotations
from typing import Any
import logging
import pandas as pd

from .base import BaseJudge
from utils import validate_required_columns
from errors import EvaluationError, ModelError

logger = logging.getLogger(__name__)


class StringBasedJudge(BaseJudge):
    """Judge that uses an LLM to compare free-form text answers."""

    def check_single_answer(self, model_answer: str, true_answer: str) -> int:
        """Return 1 if LLM judges them equivalent, else 0."""
        if true_answer is None:
            raise EvaluationError("true_answer cannot be None for StringBasedJudge.")
        if self.model is None:
            raise EvaluationError("StringBasedJudge requires a model instance (LLM).")

        prompt = f"""Compare these two answers. Do they convey the same meaning?

Expected Answer: {true_answer}
Model's Answer: {model_answer}

Consider:
- Synonyms and paraphrasing are acceptable
- Grammatical differences are acceptable
- The core meaning must be the same

Does the Model's Answer match the Expected Answer?
Answer with only: True or False

Answer:"""

        system_prompt = (
            "You are an objective evaluation assistant that judges model answers "
            "against true answers. Always stay consistent, neutral, and concise."
        )

        try:
            raw_response = self.model.generate(prompt, system_prompt)
        except Exception as exc:
            logger.error("LLM call failed: %s", exc)
            raise ModelError(f"StringBasedJudge model call failed: {exc}") from exc

        if isinstance(raw_response, tuple):
            raw_response = raw_response[0]

        response_text = str(raw_response).strip().lower()
        if "true" in response_text:
            result = 1
        elif "false" in response_text:
            result = 0
        else:
            logger.warning("Unclear response from model: %r", response_text)
            result = 0

        logger.debug("String-based check: %s → %d", response_text, result)
        return result

    def check_answers(
        self, meta: dict[str, Any], df: pd.DataFrame, output_csv_path: str
    ) -> tuple[dict[str, Any], pd.DataFrame]:
        """Run string-based evaluation on all rows."""
        if self.model is None:
            raise EvaluationError("StringBasedJudge requires a model instance (LLM).")

        required_cols = ["model_answer", "true_answer"]
        validate_required_columns(df, required_cols)

        df["is_correct"] = df.apply(
            lambda r: self.check_single_answer(r["model_answer"], r["true_answer"]),
            axis=1,
        )

        meta["judge"] = {"judge_model": self._model_name()}
        df.to_csv(output_csv_path, index=False)
        logger.info("✅ String-based evaluation complete. Saved to %s", output_csv_path)
        return meta, df
