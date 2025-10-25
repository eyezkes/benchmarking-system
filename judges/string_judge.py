from typing import Any
import pandas as pd
from .base import BaseJudge  
from utils import *

class StringBasedJudge(BaseJudge):

    def check_single_answer(self, model_answer: str, true_answer: str):

        if true_answer is None:
            raise ValueError("true_answer cannot be None for StringBasedJudge.")
        if self.model is None:
            raise ValueError("StringBasedJudge requires a model instance (LLM).")

        # 🔹 LLM için prompt oluştur
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
        
        system_prompt="""You are an objective evaluation assistant that judges model answers against true answers.
Always stay consistent, neutral, and concise."""

        raw_response = self.model.generate(prompt,system_prompt)

        if isinstance(raw_response, tuple):
            raw_response = raw_response[0]

        response_text = str(raw_response).strip().lower()

        if "true" in response_text:
            return 1
        elif "false" in response_text:
            return 0


    def check_answers(self,meta: dict[str, Any], df:pd.DataFrame , output_csv_path: str):
        """
        CSV içindeki tüm satırları model ile değerlendirir.
        'model_answer' ve 'true_answer' sütunlarını kullanır.
        """
        if self.model is None:
            raise ValueError("StringBasedJudge requires a model instance (LLM).")

        required_cols = ["model_answer", "true_answer"]
        validate_required_columns(df,required_cols)

        df["is_correct"] = df.apply(
            lambda r: self.check_single_answer(r["model_answer"], r["true_answer"]),
            axis=1
        )

        df.to_csv(output_csv_path, index=False)
        print(f"✅ String-based evaluation saved to: {output_csv_path}")

        meta["judge"]={"judge_model":self._model_name()}
        

        return meta,df