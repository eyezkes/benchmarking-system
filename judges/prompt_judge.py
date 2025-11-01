# prompt_judge.py
from __future__ import annotations
from typing import Any
import logging
import pandas as pd
import re

from .base import BaseJudge
from utils import validate_required_columns
from errors import EvaluationError, ModelError

logger = logging.getLogger(__name__)


class PromptBasedJudge(BaseJudge):


    def check_single_answer(self, model_answer: str, eval_prompt: str) -> tuple[int, str]:
        """Return (score, reasoning) for how well model_answer fits eval_prompt."""
        if eval_prompt is None:
            raise EvaluationError("eval_prompt cannot be None for PromptBasedJudge.")
        if self.model is None:
            raise EvaluationError("PromptBasedJudge requires a model instance (LLM).")

        user_prompt = f"""
Evaluate how well the following answer satisfies the evaluation criteria.

Evaluation Prompt (criteria):
{eval_prompt}

Model's Answer:
{model_answer}

Scoring Rules:
- 10 = Perfectly satisfies the evaluation criteria
- 7–9 = Mostly satisfies with minor issues
- 4–6 = Partially satisfies the criteria
- 1–3 = Weakly satisfies or largely off-target
- 0 = Completely fails to satisfy

Provide your evaluation in this EXACT format:
Score: [integer between 0 and 10]
Reasoning: [brief explanation of your scoring decision]
"""

        system_content = (
            "You are an impartial evaluation assistant that scores model answers "
            "based on explicit criteria. Be concise, neutral, and consistent."
        )

        try:
            raw_response = self.model.generate(user_prompt, system_content)
        except Exception as exc:
            logger.error("Model evaluation failed: %s", exc)
            raise ModelError(f"PromptBasedJudge generation failed: {exc}") from exc

        if isinstance(raw_response, tuple):
            raw_response = raw_response[0]

        response_text = str(raw_response).strip()
        score_match = re.search(r"Score:\s*(\d+)", response_text)
        reasoning_match = re.search(r"Reasoning:\s*(.*)", response_text, re.DOTALL)

        score: int | None = None
        reasoning = ""

        if score_match:
            try:
                score = int(score_match.group(1))
                score = max(0, min(score, 10))
            except ValueError:
                score = None
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()

        if score is None:
            logger.warning("Could not parse score from response: %r", response_text)
            score = 0

        logger.debug("Prompt-based score=%s reasoning=%s", score, reasoning[:50])
        return score, reasoning

    def check_answers(
        self,
        meta: dict[str, Any],
        df: pd.DataFrame,
        output_csv_path: str,
        eval_prompt: str,
    ) -> tuple[dict[str, Any], pd.DataFrame]:
        """Evaluate all answers with the same eval_prompt."""
        if self.model is None:
            raise EvaluationError("PromptBasedJudge requires a model instance (LLM).")

        required_cols = ["model_answer"]
        validate_required_columns(df, required_cols)

        results = df.apply(
            lambda r: self.check_single_answer(r["model_answer"], eval_prompt), axis=1
        )

        df["score"] = [r[0] for r in results]
        df["reasoning"] = [r[1] for r in results]

        # enrich meta about judge run
        meta["judge"] = {
            "judge_model": self.model.get_name(),
            "model_params":self.model.get_params(),
            "eval_prompt": eval_prompt,
        }

        df.to_csv(output_csv_path, index=False)
        logger.info("✅ Prompt-based evaluation complete. Saved to %s", output_csv_path)
        return meta, df
