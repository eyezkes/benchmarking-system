# prompt_based_score10.py
from __future__ import annotations
from typing import Any
import json
import logging
import pandas as pd

from .base import BaseJudge
from utils import validate_required_columns
from errors import EvaluationError, ModelError

logger = logging.getLogger(__name__)

class PromptBasedScore(BaseJudge):
    """
    LLM-as-a-Judge (score mode, fixed 0..10 scale).
    - No rationale
    - Invalid / unparsable → NaN
    - Records valid_count in meta["evaluation"]
    """

    def __init__(self, model, eval_prompt: str) -> None:
        super().__init__(model=model)
        if not eval_prompt or not eval_prompt.strip():
            raise ValueError("eval_prompt must be a non-empty string")
        self.eval_prompt = eval_prompt.strip()

    def _build_user_message(self, question, model_answer, true_answer) -> str:
        fmt = (
            'Return STRICT JSON with exactly this format:\n'
            '{\n'
            '  "score": number  // integer or float between 0 and 10\n'
            '}\n'
            'No code fences, no markdown, no extra text.'
        )
        parts = []
        if question:
            parts.append(f"Question:\n{question}")
        if true_answer is not None:
            parts.append(f"Reference (ground truth):\n{true_answer}")
        parts.append(f"Model Answer:\n{model_answer}")
        ctx = "\n\n".join(parts)
        return f"{self.eval_prompt}\n\n{ctx}\n\n{fmt}"

    def _parse_llm_json(self, text: str) -> dict:
        s = text.strip()
        if s.startswith("```"):
            j = s.find("```", 3)
            if j != -1:
                s = s[3:j].strip()
        try:
            data = json.loads(s)
        except Exception as e:
            raise EvaluationError(f"Invalid JSON: {e}")
        if not isinstance(data, dict) or "score" not in data:
            raise EvaluationError('JSON must contain key "score"')
        return data

    def check_single_answer(
        self,
        question: str | None = None,
        model_answer: str = "",
        true_answer: str | None = None,
        prompt: str | None = None,
    ):
        if self.model is None:
            raise EvaluationError("Model required for PromptBasedScore10.")
        rubric = (prompt or self.eval_prompt).strip()
        if not rubric:
            raise EvaluationError("Empty eval prompt.")

        user_msg = self._build_user_message(question, model_answer, true_answer).replace(self.eval_prompt, rubric, 1)

        try:
            out = self.model.generate(user_msg, timeout=None)
        except Exception as e:
            raise ModelError(f"LLM call failed: {e}")

        data = self._parse_llm_json(out)
        try:
            raw = float(data["score"])
        except Exception:
            raise EvaluationError('"score" must be numeric')
        
        if raw < 0.0 or raw > 10.0:
            raise EvaluationError(f"Score {raw} out of expected 0–10 range")

        return {"score": raw}

    def check_answers(
        self,
        meta: dict[str, Any],
        df: pd.DataFrame,
        output_csv_path: str,
        **kwargs: Any,
    ):
        required_cols = ["model_answer"]
        validate_required_columns(df, required_cols)

        scores = []
        for _, r in df.iterrows():
            try:
                res = self.check_single_answer(
                    question=r.get("question"),
                    model_answer=r.get("model_answer", ""),
                    true_answer=r.get("true_answer"),
                    prompt=kwargs.get("eval_prompt_override"),
                )
                scores.append(res["score"])
            except (EvaluationError, ModelError) as e:
                logger.warning("PromptBasedScore10 row failed: %s", e)
                scores.append(float("nan"))  # NaN for invalid

        df["score"] = scores

        meta["judge"] = {
            "type": "Prompt-based score(0-10)",
            "mode": "SCORE_0_10",
            "judge_model": getattr(self.model, "get_name", lambda: None)(),
            "model_params": getattr(self.model, "get_params", lambda: {})(),
            "eval_prompt": self.eval_prompt,
        }

        df.to_csv(output_csv_path, index=False)
        logger.info("✅ PromptBasedScore10 done.")
        return meta, df
