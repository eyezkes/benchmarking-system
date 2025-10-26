import json
from typing import Dict, Any
import pandas as pd

from utils import validate_required_columns
from .base import BaseEvaluator

class PromptBasedEvaluator(BaseEvaluator):
    """
    Beklenen kolonlar:
      - score_1_10 (1..10)
    """
  #  required_cols = ["score"]

    def compute(self, meta,df: pd.DataFrame,output_json_path:str):

     #   validate_required_columns(df,required_cols)

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

        result: Dict[str, Any] = {
            "metadata": meta,
            "out": out,
        }
        json_string = json.dumps(result, ensure_ascii=False, indent=4)
        try:
            with open(output_json_path, 'w', encoding='utf-8') as f:
            # Okunabilirliği artırmak için 'indent=4' kullanılması önerilir.
            # ensure_ascii=False, Türkçe/özel karakterlerin düzgün kaydedilmesini sağlar.
                f.write(json_string)
                print(f"saved: {output_json_path}")
        
        except IOError as e:
            print(f"Error ({output_json_path}): {e}")

        return json_string



