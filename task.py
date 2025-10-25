
import datetime
from enum import StrEnum
from pathlib import Path
import uuid
class TaskType(StrEnum):
    MULTIPLE_CHOICE = "multiple_choice"
    STRING_BASED    = "string_based"
    PROMPT_BASED    = "prompt_based"

class Task:
    def __init__(self, id: str, type: TaskType, created_at: datetime.datetime):
        self.id = id
        self.type = type
        self.created_at = created_at
        self.output_dir = Path(f"outputs/{self.id}")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def new(task_type: TaskType) -> "Task":
        created = datetime.datetime.now()
        timestamp_str = created.strftime("%Y%m%d%H%M%S")
        short_uuid = str(uuid.uuid4())[:8]

        return Task(
            id=f"{timestamp_str}-{short_uuid}",
            type=task_type,
            created_at=created
        )
    
    def get_path(self, filename: str) -> Path:
        """
        Returns a full file path inside this task's output directory.

        Example:
            task.get_path("run_results.csv")
            -> Path("outputs/20251025183200-abc12345/run_results.csv")
        """
        return self.output_dir / filename