import json
import pandas as pd
from typing import Dict, Any
from .base import BaseEvaluator

class StringBasedEvaluator(BaseEvaluator):
    """String-based evaluation: accuracy (0/1 üzerinden)."""

    def compute(self,meta, df: pd.DataFrame,output_json_path:str) :
        # 0/1 to float
        df["is_correct"] = pd.to_numeric(df["is_correct"], errors="coerce").fillna(0)
        accuracy = float(df["is_correct"].mean())

        out = {
            "count": int(len(df)),
            "accuracy": round(accuracy, 4)
        }

        # category varsa ayrıca kategori bazlı ortalama
        if "category" in df.columns:
            grp = df.groupby("category")["is_correct"].mean().reset_index()
            out["per_category"] = [
                {
                    "category": str(c),
                    "count": int((df["category"] == c).sum()),
                    "accuracy": round(float(a), 4),
                }
                for c, a in zip(grp["category"], grp["is_correct"])
            ]

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

        

        
