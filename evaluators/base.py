# base.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class BaseEvaluator(ABC):
    """Abstract base class for all evaluator types."""

    @abstractmethod
    def compute(
        self,
        meta: dict[str, Any],
        df: pd.DataFrame,
        output_json_path: str,
    ) -> dict[str, Any]:
        """Compute evaluation metrics and save results to a JSON file."""
        pass
