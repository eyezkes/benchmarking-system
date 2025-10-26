from __future__ import annotations
from typing import Dict, Any, List
from abc import ABC, abstractmethod
import pandas as pd

class BaseEvaluator(ABC):




    @abstractmethod
    def compute(self, meta:dict[str, Any] ,df: pd.DataFrame,output_json_path:str) :
        pass
