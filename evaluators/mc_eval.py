import json
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import ast

LETTER_SPACE = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

def _parse_options_cell(cell) -> List[str]:
    """
    options alanı string olarak Python listesi gibi geliyor:
      "['37.00°C, 310.15K', '37.50°C, 311.65K', ...]"
    Bunu listeye çevirir. Boş/parse edilemezse [] döner.
    """
    if isinstance(cell, list):
        return [str(x) for x in cell]
    if pd.isna(cell):
        return []
    s = str(cell).strip()
    try:
        val = ast.literal_eval(s)
        if isinstance(val, list):
            return [str(x) for x in val]
    except Exception:
        pass
    # Son çare: '||' ile ayrılmış olabilir
    if "||" in s:
        return [part.strip() for part in s.split("||") if part.strip()]
    return []

def _letter_to_index(x: Optional[str]) -> Optional[int]:
    """
    'A', 'B', 'C' ... veya 'A)' gibi formları indekse çevirir.
    Uyuşmazsa None döner.
    """
    if x is None or pd.isna(x):
        return None
    s = str(x).strip()
    if not s:
        return None
    c = s[0].upper()
    if c in LETTER_SPACE:
        return LETTER_SPACE.index(c)
    return None

class MultipleChoiceEvaluator:
    """
    Girdi varsayımları:
      - df['task_type'] == 'mc' satırlar değerlendirilecek
      - df['options']: string olarak Python listesi (örn: "['A', 'B', 'C']")
      - df['true_answer']: harf ('A','B',...)
      - df['model_answer']: harf ('A','B',...)
      - df['is_correct'] (opsiyonel): 0/1; yoksa harflerden türetilir

    Çıktı:
      {
        "count": int,
        "accuracy": float,
        "confusion_matrix": { "labels": ["A","B","C",...], "matrix": [[...], ...] },
        "per_category": [...]  # (category kolonu varsa)
      }
    """

    def compute(self,meta, df: pd.DataFrame,output_json_path:str):

        if df.empty:
            return {"count": 0, "accuracy": 0.0}

        # options'u parse et + her satır için şık sayısını bul
        df["_opts"] = df.get("options", "").apply(_parse_options_cell)
        df["_k"] = df["_opts"].apply(len)

        # harfleri indekslere çevir
        df["_t_idx"] = df.get("true_answer").apply(_letter_to_index) if "true_answer" in df.columns else None
        df["_p_idx"] = df.get("model_answer").apply(_letter_to_index) if "model_answer" in df.columns else None


        is_corr = pd.to_numeric(df["is_correct"], errors="coerce").fillna(0).astype(int)


        # accuracy hesapla (NaN'leri hariç bırak)
        valid_mask = is_corr.notna()
        accuracy = float(is_corr[valid_mask].mean()) if valid_mask.any() else 0.0

        out: Dict[str, Any] = {
            "count": int(len(df)),
            "accuracy": round(accuracy, 6),
        }

        # Confusion matrix: yalnızca t_idx ve p_idx geçerliyse
        tp_mask = df["_t_idx"].notna() & df["_p_idx"].notna()
        if tp_mask.any():
            max_k = int(df["_k"].max()) if df["_k"].notna().any() else 0
            labels = [LETTER_SPACE[i] for i in range(max_k)]
            cm = np.zeros((max_k, max_k), dtype=int)

            for _, r in df[tp_mask].iterrows():
                t = int(r["_t_idx"])
                p = int(r["_p_idx"])
                # Güvenlik: bazı satırlarda şık sayıları farklı olabilir
                if t < max_k and p < max_k and t >= 0 and p >= 0:
                    cm[t, p] += 1

            out["confusion_matrix"] = {"labels": labels, "matrix": cm.tolist()}

        # (opsiyonel) kategori kırılımı
        if "category" in df.columns:
            grp = pd.DataFrame({"category": df["category"], "is_correct": is_corr})
            grp = grp.dropna(subset=["is_correct"])
            if not grp.empty:
                per_cat = grp.groupby("category")["is_correct"].mean().reset_index()
                out["per_category"] = [
                    {
                        "category": str(c),
                        "count": int((df["category"] == c).sum()),
                        "accuracy": round(float(a), 6),
                    }
                    for c, a in zip(per_cat["category"], per_cat["is_correct"])
                ]

        # temizlik (yan kolonları dışarı sızdırma)
        for col in ["_opts", "_k", "_t_idx", "_p_idx"]:
            if col in df.columns:
                df.drop(columns=[col], inplace=True, errors="ignore")

        
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
