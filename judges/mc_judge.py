from typing import Any
import pandas as pd
from .base import BaseJudge  
from utils import *

class MultipleChoiceJudge(BaseJudge):

    def check_single_answer(self, model_answer: str, true_answer: str):

        if true_answer is None:
            raise ValueError("true_answer cannot be None for MultipleChoiceJudge.")
        
        model_letter = str(model_answer).strip().upper()
        true_letter = str(true_answer).strip().upper()

        return True if model_letter == true_letter else False

    def check_answers(self, meta: dict[str, Any],df: pd.DataFrame, output_csv_path: str):

        required_cols = ["model_answer", "true_answer"]
        validate_required_columns(required_cols,df)

        df["is_correct"] = df.apply(
            lambda r: self.check_single_answer(r["model_answer"], r["true_answer"]),
            axis=1
        ).astype(int)

        df.to_csv(output_csv_path, index=False)
        print(f"✅ Evaluation complete. Results saved to: {output_csv_path}")

        return df
