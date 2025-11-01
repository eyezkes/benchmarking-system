# base.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Optional
import logging
import pandas as pd

from model import Model

logger = logging.getLogger(__name__)


class BaseJudge(ABC):
    """Abstract base class for all judge types."""

    def __init__(self, model: Optional[Model] = None) -> None:
        """Initialize with an optional LLM model."""
        self.model = model

    @abstractmethod
    def check_single_answer(
        self,
        question: Optional[str] = None,
        model_answer: str = "",
        true_answer: Optional[str] = None,
        prompt: Optional[str] = None,
    ):
        """Compare a single answer and return a judgment (type depends on subclass)."""
        pass

    @abstractmethod
    def check_answers(
        self,
        meta: dict[str, Any],
        df: pd.DataFrame,
        output_csv_path: str,
        **kwargs: Any,
    ):
        """Evaluate multiple answers in a dataset."""
        pass
