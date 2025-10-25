import pandas as pd
from typing import Dict, Any
from .base import BaseEvaluator

class StringBasedEvaluator(BaseEvaluator):
    """String-based evaluation: accuracy (0/1 üzerinden)."""

    def compute(self, df: pd.DataFrame) -> Dict[str, Any]:
        # 0/1 to float
        df["is_correct"] = pd.to_numeric(df["is_correct"], errors="coerce").fillna(0)
        accuracy = float(df["is_correct"].mean())

        result = {
            "count": int(len(df)),
            "accuracy": round(accuracy, 4)
        }

        # category varsa ayrıca kategori bazlı ortalama
        if "category" in df.columns:
            grp = df.groupby("category")["is_correct"].mean().reset_index()
            result["per_category"] = [
                {
                    "category": str(c),
                    "count": int((df["category"] == c).sum()),
                    "accuracy": round(float(a), 4),
                }
                for c, a in zip(grp["category"], grp["is_correct"])
            ]

        return result
