# task.py
from __future__ import annotations

import datetime as _dt
import logging
import uuid
from enum import StrEnum
from pathlib import Path

logger = logging.getLogger(__name__)


class TaskType(StrEnum):
    """Enumeration for supported benchmark task types."""
    MULTIPLE_CHOICE = "multiple_choice"
    STRING_BASED    = "string_based"
    PROMPT_BASED    = "prompt_based"


class Task:
    """
    Represents the specification of a benchmark task (not the run itself).

    Attributes:
        id: Short unique identifier (timestamp + short UUID).
        type: TaskType of this benchmark.
        dataset_path: Path to the dataset file.
        sample_size: Number of samples to take from dataset.
        seed: Random seed (default 42).
        prompt_template: Optional user-instruction part appended to the question
                         (and options for MCQ) in the user prompt.
    """

    def __init__(
        self,
        id: str,
        type: TaskType,
        created_at: _dt.datetime,
        dataset_path: str | Path,
        sample_size: int,
        prompt_template: str | None = None,
        seed: int = 42,
    ) -> None:
        if not id or not isinstance(id, str):
            raise ValueError("Task.id must be a non-empty string")
        if not isinstance(type, TaskType):
            raise ValueError("Task.type must be a TaskType")
        if not isinstance(created_at, _dt.datetime):
            raise ValueError("Task.created_at must be a datetime")
        if not dataset_path:
            raise ValueError("dataset_path must be provided")
        if not isinstance(sample_size, int) or sample_size <= 0:
            raise ValueError("sample_size must be a positive integer")
        if not isinstance(seed, int):
            raise ValueError("seed must be an integer")

        self.id = id
        self.type = type
        self.created_at = created_at
        self.dataset_path = str(dataset_path)
        self.sample_size = sample_size
        self.prompt_template = prompt_template
        self.seed = seed

        logger.info(
            "Created Task(id=%s, type=%s, dataset=%s, sample=%d)",
            self.id,
            self.type,
            self.dataset_path,
            self.sample_size,
        )

    @staticmethod
    def _default_prompt_template(task_type: TaskType) -> str:
        """
        Default user-instruction snippet for each task type.
        This text is appended after question/options in the Runner.
        """
        if task_type == TaskType.MULTIPLE_CHOICE:
            return "Answer only with a single option letter (A, B, C, ...)."
        if task_type == TaskType.STRING_BASED:
            return "Answer clearly in a few sentences."
        if task_type == TaskType.PROMPT_BASED:
            # Daha serbest kullanım için boş veya çok kısa bırakılabilir.
            return ""
        raise ValueError(f"Unknown TaskType: {task_type}")

    @staticmethod
    def new(
        task_type: TaskType,
        dataset_path: str | Path,
        sample_size: int,
        prompt_template: str | None = None,
        seed: int = 42,
    ) -> "Task":
        """Create a new Task with timestamp-based id and given dataset/specs."""
        if not isinstance(task_type, TaskType):
            raise ValueError("task_type must be a TaskType")

        created = _dt.datetime.now()
        timestamp = created.strftime("%Y%m%d%H%M%S")
        short_uuid = uuid.uuid4().hex[:8]
        task_id = f"{timestamp}-{short_uuid}"

        if prompt_template is None:
            prompt_template = Task._default_prompt_template(task_type)

        return Task(
            id=task_id,
            type=task_type,
            created_at=created,
            dataset_path=dataset_path,
            sample_size=sample_size,
            prompt_template=prompt_template,
            seed=seed,
        )
