from __future__ import annotations

import datetime as _dt
import logging
from enum import StrEnum
from pathlib import Path
import uuid

logger = logging.getLogger(__name__)


class TaskType(StrEnum):
    """Enumeration for supported benchmark task types."""
    MULTIPLE_CHOICE = "multiple_choice"
    STRING_BASED    = "string_based"
    PROMPT_BASED    = "prompt_based"


class Task:
    """Represents a single benchmark task/run with its own output directory.

    Attributes:
        id: Short string identifier (timestamp + short UUID).
        type: TaskType of this run (MC / String / Prompt).
        created_at: Local creation datetime (naive, local time).
        output_dir: Path to this task's output directory under ./outputs/{id}.
        system_prompt: (Optional) Custom system prompt for PromptBased tasks.
    """

    def __init__(
        self,
        id: str,
        type: TaskType,
        created_at: _dt.datetime,
        system_prompt: str | None = None,   # 🔹 NEW
    ) -> None:
        if not id or not isinstance(id, str):
            raise ValueError("Task.id must be a non-empty string")
        if not isinstance(type, TaskType):
            raise ValueError("Task.type must be a TaskType")
        if not isinstance(created_at, _dt.datetime):
            raise ValueError("Task.created_at must be a datetime")

        self.id = id
        self.type = type
        self.created_at = created_at
        self.system_prompt = system_prompt   # 🔹 NEW FIELD

        self.output_dir = Path("outputs") / self.id
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            logger.error("Failed to create output directory %s: %s", self.output_dir, exc)
            raise

        logger.info(
            "Created Task(id=%s, type=%s, has_system_prompt=%s) at %s",
            self.id,
            self.type,
            bool(self.system_prompt),
            self.created_at.isoformat(),
        )

    @staticmethod
    def new(task_type: TaskType, system_prompt: str | None = None) -> "Task": 
        """Create a new Task with a local timestamp-based id.

        Uses local time (as you preferred) and an 8-char UUID suffix.

        Args:
            task_type: One of TaskType.
            system_prompt: Optional custom system prompt for PromptBased tasks.

        Returns:
            Task: Newly created task instance ready to use.
        """
        if not isinstance(task_type, TaskType):
            raise ValueError("task_type must be a TaskType")

        created = _dt.datetime.now()
        timestamp_str = created.strftime("%Y%m%d%H%M%S")
        short_uuid = uuid.uuid4().hex[:8]
        task_id = f"{timestamp_str}-{short_uuid}"

        return Task(id=task_id, type=task_type, created_at=created, system_prompt=system_prompt)

    def get_path(self, filename: str) -> Path:
        """Return a full path inside this task's output directory."""
        if not filename or not isinstance(filename, str):
            raise ValueError("filename must be a non-empty string")
        p = Path(filename)
        if p.is_absolute():
            raise ValueError("filename must be relative, not absolute")

        full = self.output_dir / p
        logger.debug("Resolved path for task %s: %s", self.id, full)
        return full
