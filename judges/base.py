from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional
import pandas as pd
from model import Model
from task import Task

class BaseJudge(ABC):
        
    def __init__(self, model: Optional[Model] = None):

        self.model = model

    def _model_name(self) -> str:

        return getattr(self.model, "name", lambda: type(self.model).__name__)()

    @abstractmethod
    def check_single_answer(self, question: Optional[str] = None, model_answer: str = "", true_answer: Optional[str] = None, prompt: Optional[str] = None):
        pass

    @abstractmethod
    def check_answers(self,meta: dict[str, Any],df:pd.DataFrame,output_csv_path:str):
        pass