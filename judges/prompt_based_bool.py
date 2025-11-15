# prompt_based_boolean.py
from __future__ import annotations
from typing import Any
import json
import logging
import pandas as pd

from .base import BaseJudge
from utils import validate_required_columns
from errors import EvaluationError, ModelError

logger = logging.getLogger(__name__)

class PromptBasedBoolean(BaseJudge):
    """
    LLM-as-a-Judge (boolean mode)

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
            '  "passed": true or false\n'
            '}\n'
            'No code fences, no markdown, no additional commentary and "True,1,T,False" etc. is not allowed. only "true" or "false" .'
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
        if not isinstance(data, dict) or "passed" not in data:
            raise EvaluationError('JSON must contain key "passed"')
        return data

    def check_single_answer(
        self,
        question: str | None = None,
        model_answer: str = "",
        true_answer: str | None = None,
        prompt: str | None = None,
    ):
        if self.model is None:
            raise EvaluationError("Model required for PromptBasedBoolean.")
        rubric = (prompt or self.eval_prompt).strip()
        if not rubric:
            raise EvaluationError("Empty eval prompt.")

        user_msg = self._build_user_message(question, model_answer, true_answer).replace(self.eval_prompt, rubric, 1)
        try:
            out = self.model.generate(user_msg, timeout=None)
        except Exception as e:
            raise ModelError(f"LLM call failed: {e}")

        data = self._parse_llm_json(out)

        # Strict type check: must be a real JSON boolean (true/false), not "true"/1/etc.
        val = data["passed"]
        if not isinstance(val, bool):
            raise EvaluationError('"passed" must be a JSON boolean (true or false)')
        return {"passed": val}

    def check_answers(
        self,
        meta: dict[str, Any],
        df: pd.DataFrame,
        output_csv_path: str,
        **kwargs: Any,
    ):
        required_cols = ["model_answer"]
        validate_required_columns(df, required_cols)

        results = []
        for _, r in df.iterrows():
            try:
                res = self.check_single_answer(
                    question=r.get("question"),
                    model_answer=r.get("model_answer", ""),
                    true_answer=r.get("true_answer"),
                    prompt=kwargs.get("eval_prompt_override"),
                )
                results.append(res["passed"])
            except (EvaluationError, ModelError) as e:
                logger.warning("PromptBasedBoolean row failed: %s", e)
                results.append(float("nan"))  # NaN for invalid

        df["is_correct"] = results


        meta["judge"] = {
            "type": "Prompt-based Boolean",
            "judge_model": getattr(self.model, "get_name", lambda: None)(),
            "model_params": getattr(self.model, "get_params", lambda: {})(),
            "eval_prompt": self.eval_prompt,
        }

        df.to_csv(output_csv_path, index=False)
        logger.info("✅ PromptBasedBoolean done.")
        return meta, df
