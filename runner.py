# runner.py
from __future__ import annotations

import logging
import random
import time
import uuid
from pathlib import Path
from typing import Any

import pandas as pd

from errors import EvaluationError, ModelError
from task import Task, TaskType
from utils import sample_dataset

logger = logging.getLogger(__name__)


class Runner:

    def __init__(self, model: Any) -> None:
        if not hasattr(model, "generate"):
            raise ValueError("model must provide a .generate(prompt) method")
        self.model = model
        self._current_run_id: str | None = None
        self._current_run_dir: Path | None = None

    # ---------- helpers ----------

    @staticmethod
    def _default_system_prompt() -> str:
        return (
            "You are a knowledgeable and reliable AI assistant.\n"
            "Answer questions accurately, clearly, and concisely.\n"
            "Avoid unnecessary explanations or reasoning unless explicitly requested."
        )

    @staticmethod
    def _new_run_id() -> str:
        return time.strftime("%Y%m%d%H%M%S", time.localtime()) + "-" + uuid.uuid4().hex[:8]

    def get_path(self, filename: str) -> Path:
        """
        Resolve a path under the MOST RECENT run directory:
            outputs/runs/{last_run_id}/{filename}
        """
        if not filename or not isinstance(filename, str):
            raise ValueError("filename must be a non-empty string")
        if self._current_run_dir is None:
            raise RuntimeError("No active run directory. Call runner.run(...) first.")
        p = Path(filename)
        if p.is_absolute():
            raise ValueError("filename must be relative, not absolute")
        out = self._current_run_dir / p
        out.parent.mkdir(parents=True, exist_ok=True)
        return out

    # ---------- main ----------

    def run(self, task: Task, measure_k: int = 5) -> tuple[dict, pd.DataFrame]:
        if not isinstance(task, Task):
            raise ValueError("task must be a Task")
        if task.sample_size <= 0:
            raise ValueError("task.sample_size must be > 0")
        if measure_k < 0:
            raise ValueError("measure_k must be >= 0")

        dataset_path = task.dataset_path
        sample_size = task.sample_size
        seed = task.seed

        logger.info(
            "Starting run: dataset=%s sample_size=%d seed=%s model=%s",
            dataset_path, sample_size, seed, self.model.get_name()
        )

        # ---- sample dataset
        sampled_df = sample_dataset(dataset_path, sample_size, seed)
        if sampled_df.empty:
            raise EvaluationError("Sampled dataset is empty")

        n = len(sampled_df)
        k = min(measure_k, n)
        rng = random.Random(seed)
        measure_idx = set(rng.sample(range(n), k)) if k > 0 else set()
        times_ms: list[float] = []

        rows: list[dict] = []

        # ---- inference
        for i, (_, row) in enumerate(sampled_df.iterrows()):
            if "question" not in row:
                raise EvaluationError("Dataset row missing required 'question'")

            question = row["question"]
            ttype = task.type

            # Task.prompt_template yalnızca yönerge/snippet, soru/options burada kurulur
            instruction = task.prompt_template or ""

            if ttype == TaskType.MULTIPLE_CHOICE:
                opts = row.get("options")
                if opts is None:
                    raise EvaluationError("MULTIPLE_CHOICE row missing 'options'")

                prompt = f"{question}\nOptions:\n"
                for j, opt in enumerate(opts):
                    prompt += f"{chr(65 + j)}) {opt}\n"
                if instruction:
                    prompt += "\n" + instruction

            elif ttype == TaskType.STRING_BASED:
                prompt = f"{question}\n"
                if instruction:
                    prompt += instruction

            elif ttype == TaskType.PROMPT_BASED:
                prompt = f"{question}\n"
                if instruction:
                    prompt += instruction

            else:
                raise EvaluationError(f"Unknown task type: {ttype}")

            try:
                if i in measure_idx:
                    t0 = time.perf_counter()
                    ans = self.model.generate(prompt)
                    t1 = time.perf_counter()
                    times_ms.append((t1 - t0) * 1000.0)
                else:
                    ans = self.model.generate(prompt)
            except Exception as exc:
                logger.error("Model generation failed at row %d: %s", i, exc)
                raise ModelError(f"Generation failed at row {i}: {exc}") from exc

            rows.append({
                "question_id": row.get("question_id"),
                "question": question,
                "options": row.get("options"),
                "true_answer": row.get("answer"),
                "model_answer": ans,
            })

        results_df = pd.DataFrame(rows)

        # ---- run folder: outputs/runs/{run_id}/
        run_id = self._new_run_id()
        run_dir = Path("outputs") / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        self._current_run_id = run_id
        self._current_run_dir = run_dir

        # ---- save results CSV
        run_csv = run_dir / f"run_{run_id}.csv"
        results_df.to_csv(run_csv, index=False)

        avg_ms = (sum(times_ms) / len(times_ms)) if times_ms else None

        meta = {
            "run_id": run_id,
            "task_id": task.id,
            "task_type": task.type,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),

            # --- task/dataset specs ---
            "dataset_path": str(task.dataset_path),
            "dataset_name": Path(task.dataset_path).name,
            "sample_size": task.sample_size,
            "seed": task.seed,
            "system_prompt": self.model.get_system_prompt(),
            "user prompt":task.prompt_template,  

            # --- model info ---
            "model_name": self.model.get_name(),
            "model_params": self.model.get_params(),

            # --- latency summary ---
            "measured_count": len(times_ms),
            "latency_ms_avg": round(avg_ms, 2) if avg_ms is not None else None,
        }

        logger.info("Run finished: run_id=%s avg_ms=%s", run_id, meta["latency_ms_avg"])
        return meta, results_df
