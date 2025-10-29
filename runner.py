# runner.py
from __future__ import annotations

import logging
import time
import random
from pathlib import Path
from typing import Optional, Iterable, Any

import pandas as pd

from errors import EvaluationError, ModelError
from task import Task, TaskType
from utils import sample_dataset

logger = logging.getLogger(__name__)


class Runner:
    """Coordinates dataset sampling and model runs for a given Task."""

    def __init__(self, model: Any) -> None:
        """
        Args:
            model: Object exposing `.name() -> str` and
                   `.generate(prompt: str, system_content: str) -> str`.
        """
        if not hasattr(model, "generate"):
            raise ValueError("model must provide a .generate(prompt, system_content) method")
        self.model = model

    def _model_name(self) -> str:
        """Return the model name or class name if missing."""
        try:
            return getattr(self.model, "name", lambda: type(self.model).__name__)()
        except Exception:
            return type(self.model).__name__

        
    def _model_params(self) -> dict:
        try:
            # 1) get_params() varsa onu kullan
            if hasattr(self.model, "get_params") and callable(getattr(self.model, "get_params")):
                return dict(self.model.get_params())
            # 2) Yoksa .params bir dict ise onu kullan
            if hasattr(self.model, "params") and isinstance(getattr(self.model, "params"), dict):
                return dict(getattr(self.model, "params"))
            return {}
        except Exception:
            logger.warning("Could not retrieve model params")
            return {}

    
    # === NEW ===
    @staticmethod
    def _default_system_prompt() -> str:
        return (
            "You are a knowledgeable and reliable AI assistant.\n"
            "Answer questions accurately, clearly, and concisely.\n"
            "Avoid unnecessary explanations or reasoning unless explicitly requested."
        )

    # === NEW ===
    def _resolve_system_prompt(self, task: Task) -> tuple[str, str]:
        """
        Decide which system prompt to use.

        Returns:
            (system_content, source)
              source in {'task', 'default'}
        """
        # Sadece PROMPT_BASED için task.system_prompt'u etkinleştir.
        if task.type == TaskType.PROMPT_BASED:
            sys_from_task = getattr(task, "system_prompt", None)
            if sys_from_task is not None:
                if not isinstance(sys_from_task, str):
                    raise EvaluationError("task.system_prompt must be a string when provided.")
                if not sys_from_task.strip():
                    raise EvaluationError("task.system_prompt cannot be empty/whitespace.")
                return sys_from_task, "task"

        # Diğer tüm durumlarda varsayılan
        return self._default_system_prompt(), "default"

    def run(
        self,
        task: Task,
        path: str | Path,
        sample_size: int,
        seed: Optional[int] = None,
        measure_k: int = 5,
    ) -> tuple[dict, pd.DataFrame]:
        """Run the model over a sampled dataset and write results CSV into task folder.

        Args:
            task: Task instance (controls output directory and metadata).
            path: Dataset path (CSV/Parquet/JSON/JSONL supported by utils.sample_dataset).
            sample_size: Number of rows to sample.
            seed: Optional random seed for deterministic sampling and latency sampling.
            measure_k: Number of rows for which latency will be measured (<= sample size).

        Returns:
            (run_meta, results_df)
              - run_meta: dict with timing/model/dataset metadata
              - results_df: DataFrame with columns:
                    question_id, question, options, model_answer, true_answer

        Raises:
            ValueError: For invalid arguments.
            ModelError: If the underlying model fails.
            EvaluationError: For invalid task type or prompt construction issues.
        """
        if not isinstance(task, Task):
            raise ValueError("task must be a Task")
        if sample_size <= 0:
            raise ValueError("sample_size must be > 0")
        if measure_k < 0:
            raise ValueError("measure_k must be >= 0")

        logger.info(
            "Starting run: task_id=%s type=%s dataset=%s sample_size=%d measure_k=%d seed=%s model=%s",
            task.id, task.type, str(path), sample_size, measure_k, seed, self._model_name()
        )

        # ---- Sample data
        sampled_df = sample_dataset(path, sample_size, seed)
        if sampled_df.empty:
            raise EvaluationError("Sampled dataset is empty")

        results: list[dict] = []
        n = len(sampled_df)
        k = min(measure_k, n)
        rng = random.Random(seed)
        measure_idx = set(rng.sample(range(n), k)) if k > 0 else set()

        times_ms: list[float] = []

        # === NEW === system prompt seçimi (bir kez belirleyip tüm satırlarda kullanıyoruz)
        system_content, system_source = self._resolve_system_prompt(task)

        for row_i, (_, row) in enumerate(sampled_df.iterrows()):
            # Required columns
            if "question" not in row:
                raise EvaluationError("Dataset row missing required 'question' column")

            question = row["question"]
            question_id = row.get("question_id", None)

            # ---- Build prompt by task type
            t_type = task.type
            if t_type == TaskType.MULTIPLE_CHOICE:
                options = row.get("options", None)
                if options is None:
                    raise EvaluationError("MULTIPLE_CHOICE row missing 'options'")

                # Render options A), B), ...
                prompt = f"{question}\nOptions:\n"
                try:
                    for i, opt in enumerate(options):
                        prompt += f"{chr(65 + i)}) {opt}\n"
                except TypeError:
                    raise EvaluationError("'options' must be an iterable of strings")

                prompt += "\nAnswer only with a single letter (A, B, C, or D)."

            elif t_type == TaskType.STRING_BASED:
                prompt = f"{question}\nAnswer in few sentences, directly and clearly"

            elif t_type == TaskType.PROMPT_BASED:
                # PromptBased’te soru ham verilir; davranışı system prompt yönlendirir
                prompt = f"{question}\n"

            else:
                raise EvaluationError(f"Unknown task type: {task.type}")

            # ---- Generate answer (measure latency for selected indices)
            try:
                if row_i in measure_idx:
                    t0 = time.perf_counter()
                    model_answer = self.model.generate(prompt, system_content)
                    t1 = time.perf_counter()
                    times_ms.append((t1 - t0) * 1000.0)
                else:
                    model_answer = self.model.generate(prompt, system_content)
            except ModelError:
                # Pass through with context
                raise
            except Exception as exc:
                logger.error("Model generation failed at row %d: %s", row_i, exc)
                raise ModelError(f"Generation failed at row {row_i}: {exc}") from exc

            # ---- True answer / options (if present)
            true_answer = row["answer"] if "answer" in row else None
            options = row["options"] if "options" in row else None

            results.append(
                {
                    "question_id": question_id,
                    "question": question,
                    "options": options,
                    "model_answer": model_answer,
                    "true_answer": true_answer,
                }
            )

        # ---- Build results DataFrame and save
        results_df = pd.DataFrame(results)
        filename = f"run_{task.id}.csv"
        out_path = task.get_path(filename)
        results_df.to_csv(out_path, index=False)
        logger.info("Wrote results to %s", out_path)

        # ---- Run-level metadata
        p = Path(path)
        avg_ms = (sum(times_ms) / len(times_ms)) if times_ms else None


        run_meta = {
            "id": task.id,
            "task_type": task.type,
            "run": {
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
                "dataset": {"name": p.name},
                "sample": {"size": sample_size, "seed": seed},
                "model": {
                    "name": self._model_name(),
                    "params": self._model_params(),
                    "system_prompt" : system_content
                },
                "latency": {
                    "measured_count": len(times_ms),
                    "avg_ms": round(avg_ms, 2) if avg_ms is not None else None,
                },
            },
        }

        logger.info(
            "Run finished: task_id=%s measured=%d avg_ms=%s ",
            task.id, len(times_ms), run_meta["run"]["latency"]["avg_ms"]
        )
        return run_meta, results_df
