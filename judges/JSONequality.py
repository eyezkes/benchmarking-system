# json_equality_judge.py
from __future__ import annotations
from typing import Any
import json
import logging
import pandas as pd

from .base import BaseJudge
from utils import validate_required_columns
from errors import EvaluationError

logger = logging.getLogger(__name__)

class JSONEquals(BaseJudge):
    """
    Judge that checks whether model_answer and true_answer
    are JSON-equivalent (key order ignored, spacing ignored).
    """

    def _parse_json(self, text: str) -> Any:
        """Try to parse JSON string; raise EvaluationError if invalid."""
        if text is None:
            raise EvaluationError("Cannot compare None as JSON.")

        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise EvaluationError(f"Invalid JSON: {text[:100]}... ({e})")

    def check_single_answer(
        self,
        model_answer: str,
        true_answer: str,
    ) -> int:
        """Return 1 if both JSON objects are equivalent, else 0."""
        model_json = self._parse_json(model_answer)
        true_json = self._parse_json(true_answer)

        equal = self._json_equal(model_json, true_json)
        logger.debug("JSON equality check: %s", equal)
        return 1 if equal else 0

    def _json_equal(self, a: Any, b: Any) -> bool:
        """Deep equality comparison ignoring dict key order."""
        # Type mismatch
        if type(a) != type(b):
            return False

        # Dicts: compare recursively
        if isinstance(a, dict):
            if set(a.keys()) != set(b.keys()):
                return False
            return all(self._json_equal(a[k], b[k]) for k in a)

        # Lists: same length, element-wise equality (order matters)
        if isinstance(a, list):
            if len(a) != len(b):
                return False
            return all(self._json_equal(x, y) for x, y in zip(a, b))

        # Other types: direct comparison
        return a == b

    def check_answers(
        self,
        meta: dict[str, Any],
        df: pd.DataFrame,
        output_csv_path: str,
    ) -> tuple[dict[str, Any], pd.DataFrame]:
        """Evaluate multiple JSON-based answers."""
        required_cols = ["model_answer", "true_answer"]
        validate_required_columns(df, required_cols)

        def safe_check(row):
            try:
                return self.check_single_answer(row["model_answer"], row["true_answer"])
            except EvaluationError as e:
                logger.warning("Row failed JSON equality: %s", e)
                return 0  # invalid JSON or mismatch counts as incorrect

        df["is_correct"] = df.apply(safe_check, axis=1)

        meta["judge"] = {
            "type": "JSONEquality",
            "judge_model": None,
            "model_params": None,
            "eval_prompt": None,
        }

        df.to_csv(output_csv_path, index=False)
        logger.info("✅ JSON equality check complete. Results saved to %s", output_csv_path)
        return meta, df
