from typing import Dict, Any
import pandas as pd
from .base import BaseEvaluator

class PromptBasedEvaluator(BaseEvaluator):
    """
    Beklenen kolonlar:
      - score_1_10 (1..10)
    """
    required_cols = ["score"]

    def compute(self, df: pd.DataFrame) -> Dict[str, Any]:

        s = pd.to_numeric(df["score"], errors="coerce").dropna()
        if len(s) == 0:
            avg = std = p25 = p50 = p75 = 0.0
        else:
            avg = float(s.mean())
            std = float(s.std(ddof=0))  # population std
            p25, p50, p75 = [float(s.quantile(q)) for q in (0.25, 0.5, 0.75)]

        out: Dict[str, Any] = {
            "count": int(len(s)),
            "average_score": round(avg, 6),
            "std_score": round(std, 6),
            "percentiles": {"p25": round(p25, 6), "p50": round(p50, 6), "p75": round(p75, 6)},
        }

        if "category" in df.columns and len(s) > 0:
            grp = df.dropna(subset=["score_1_10"]).groupby("category")["score_1_10"].mean().reset_index()
            out["per_category"] = [
                {
                    "category": str(c),
                    "count": int((df["category"] == c).sum()),
                    "avg": round(float(a), 6),
                }
                for c, a in zip(grp["category"], grp["score_1_10"])
            ]

        return out
