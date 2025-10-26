from typing import Any
import pandas as pd
import re
from .base import BaseJudge  
from utils import *

class PromptBasedJudge(BaseJudge):

    def check_single_answer(self, model_answer: str, eval_prompt: str):
        """
        Evaluate how well the model's answer fulfills the given prompt.
        Returns a (score, reasoning) tuple.
        """

        if eval_prompt is None:
            raise ValueError("prompt cannot be None for PromptBasedJudge.")
        if self.model is None:
            raise ValueError("PromptBasedJudge requires a model instance (LLM).")

        # 🔹 User prompt for evaluation
        user_prompt = f"""
Evaluate how well the model's answer fulfills the following prompt.

Prompt:
{eval_prompt}

Model's Answer:
{model_answer}

Scoring Rules:
- 10 = Perfectly meets the evaluation prompt with completeness, clarity, and accuracy
- 7-9 = Mostly meets the evaluation prompt but may have minor issues
- 4-6 = Partially meets the evaluation prompt; noticeable gaps
- 1-3 = Poorly meets the evaluation prompt or largely incorrect
- 0 = Completely fails to meet the evalution prompt

Provide your evaluation in this EXACT format:
Score: [integer between 1 and 10]
Reasoning: [brief explanation of your scoring decision]
"""

        # 🔹 System prompt for evaluator role
        system_content = """You are an impartial evaluation assistant that scores model answers based on how well they satisfy a given prompt.
Be concise, consistent, and neutral. Follow the requested format strictly.
"""

        # 🔹 Generate LLM response
        raw_response = self.model.generate(user_prompt, system_content)

        # Some models may return tuple (text, metadata)
        if isinstance(raw_response, tuple):
            raw_response = raw_response[0]

        response_text = str(raw_response).strip()

        # --- Parse score and reasoning ---
        score_match = re.search(r"Score:\s*(\d+)", response_text)
        reasoning_match = re.search(r"Reasoning:\s*(.*)", response_text, re.DOTALL)

        score = None
        reasoning = ""

        if score_match:
            try:
                score = int(score_match.group(1))
                score = max(1, min(score, 10))  # clamp 1–10
            except ValueError:
                score = None

        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()

        return score, reasoning


    def check_answers(self, meta: dict[str, Any], df: pd.DataFrame,output_csv_path:str, eval_prompt:str):
        """
        Evaluate each row in a CSV file using the LLM model.
        Requires columns: ['model_answer', 'prompt'].
        Outputs columns: ['score', 'reasoning'].
        """

        if self.model is None:
            raise ValueError("PromptBasedJudge requires a model instance (LLM).")

        required_cols = ["model_answer"]
        validate_required_columns(df,required_cols)

        # 🔹 Apply evaluation for each row
        results = df.apply(
            lambda r: self.check_single_answer(r["model_answer"], eval_prompt),
            axis=1
        )

        # Unpack tuples into two separate columns
        df["score"] = [r[0] for r in results]
        df["reasoning"] = [r[1] for r in results]



        meta["judge"]={"judge_model":self._model_name(),
                     "eval_prompt":eval_prompt}
        

        results_df = pd.DataFrame(df)
        results_df.to_csv(output_csv_path, index=False)
        return meta,df
