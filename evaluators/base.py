from __future__ import annotations
from typing import Dict, Any, List
from abc import ABC, abstractmethod
import pandas as pd

class BaseEvaluator(ABC):
    """Tüm evaluator stratejileri için temel arayüz."""
    required_cols: List[str] = []



    @abstractmethod
    def compute(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Alt sınıflar metrik hesaplayıp dict döndürmeli."""
        ...
