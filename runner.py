import datetime
from os import PathLike
import time
import uuid
from task import Task
from utils import *
import random
from typing import Iterable, Optional
import pandas as pd
from model import Model
from task import *



class Runner:
    def __init__(self, model: Model):
        self.model = model

    def _model_name(self) -> str:

        return getattr(self.model, "name", lambda: type(self.model).__name__)()



    def run(
        self,
        task: Task,
        path: PathLike,
        sample_size: int,
        seed: Optional[int] = None,
        measure_k: int = 25,           # 👈 kaç tanesinde süre ölçülecek
    ):
        sampled_df = sample_dataset(path, sample_size, seed)
        results = []
        t_id = task.id
        t_type = task.type

        # --- rastgele ölçüm indekslerini seç ---
        n = len(sampled_df)
        k = min(measure_k, n)
        rng = random.Random(seed if seed is not None else None)
        measure_idx = set(rng.sample(range(n), k)) if k > 0 else set()

        times_ms = []  # sadece generate süresi (ms)

        for row_i, (_, row) in enumerate(sampled_df.iterrows()):
            question = row["question"]
            question_id = row.get("question_id")

            # --- Prompt construction by benchmark type (AYNI BIRAKILDI) ---
            system_content = """You are a knowledgeable and reliable AI assistant.
Answer questions accurately, clearly, and concisely.
Avoid unnecessary explanations or reasoning unless explicitly requested."""
            if t_type == TaskType.MULTIPLE_CHOICE:
                options = row["options"]
                prompt = f"{question}\nOptions:\n"
                for i, opt in enumerate(options):
                    prompt += f"{chr(65+i)}) {opt}\n"
                prompt += "\nAnswer only with a single letter (A, B, C, or D)."
            elif t_type == TaskType.STRING_BASED:
                prompt = f"{question}\nAnswer in few sentences, directly and clearly"
            elif t_type == TaskType.PROMPT_BASED:
                prompt = f"{question}\n"
            else:
                raise ValueError(f"Unknown task type: {task.type}")

            # --- Get model answer (seçilenlerde süre ölç) ---
            if row_i in measure_idx:
                t0 = time.perf_counter()
                model_answer = self.model.generate(prompt, system_content)
                t1 = time.perf_counter()
                times_ms.append((t1 - t0) * 1000.0)
            else:
                model_answer = self.model.generate(prompt, system_content)

            # --- True answer (if exists) ---
            true_answer = row["answer"] if "answer" in row else None
            options = row["options"] if "options" in row else None

            results.append({
                "question_id": question_id,
                "question": question,
                "options": options,
                "model_answer": model_answer,
                "true_answer": true_answer
            })

        # Convert to DataFrame for next stage
        results_df = pd.DataFrame(results)
        filename = f"run_{task.id}.csv"
        results_df.to_csv(task.get_path(filename), index=False)

        # ----- run-level metadata (dict) -----
        p = Path(path)
        # ortalama süre (ms)
        avg_ms = (sum(times_ms) / len(times_ms)) if times_ms else None

        run_meta = {
            "id": t_id,
            "task_type": t_type,
            "run": {
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
                "dataset": {
                    "name": p.name,
                },
                "sample": {
                    "size": sample_size,
                    "seed": seed,
                },
                "model": {
                    "name": self._model_name(),
                    # "params": self.model_params,
                },
                "latency": { 
                    "measured_count": len(times_ms),
                    "avg_ms": round(avg_ms, 2),
                }
            }
        }

        print("done")
        return run_meta, results_df