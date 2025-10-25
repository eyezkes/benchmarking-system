from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional
import pandas as pd
from model import Model
from task import Task

class BaseJudge(ABC):
        
    def __init__(self,task:Task, model: Optional[Model] = None):
        self.task=task
        self.model = model

    def _model_name(self) -> str:

        return getattr(self.model, "name", lambda: type(self.model).__name__)()


    def _path(self, filename: str) -> Path:
        return self.task.get_path(filename)

    @abstractmethod
    def check_single_answer(self, question: Optional[str] = None, model_answer: str = "", true_answer: Optional[str] = None, prompt: Optional[str] = None):
        pass

    @abstractmethod
    def check_answers(self,meta: dict[str, Any],df:pd.DataFrame,output_csv_path:str):
        pass