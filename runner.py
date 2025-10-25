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
    
    def generate_run_id(self):
    # Tarih ve saat kısmı (Okunabilirlik)
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    # Kısa bir UUID parçasını ekleme (Benzersizlik garantisi)
        short_uuid = str(uuid.uuid4())[:8] 
    
        return f"{timestamp_str}-{short_uuid}"


    def run(self,
        task:Task,
        path: PathLike,            
        sample_size: int,
        seed: Optional[int] = None,
        ):

        sampled_df= sample_dataset(path,sample_size,seed)
        results = []
        t_id=task.id
        t_type=task.type

        for _, row in sampled_df.iterrows():

            question = row["question"]
            question_id = row.get("question_id")

            # --- Prompt construction by benchmark type ---
            system_content="""You are a knowledgeable and reliable AI assistant.
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

            elif t_type== TaskType.PROMPT_BASED:
                prompt = f"{question}\n"

            else:
                raise ValueError(f"Unknown task type: {task.type}")

            # --- Get model answer ---
            model_answer = self.model.generate(prompt,system_content)

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
        filename=f"run_{task.id}.csv"
        results_df.to_csv(task.get_path(filename), index=False)

                # ----- run-level metadata (dict) -----
        p = Path(path)
        run_meta = {
        "id": t_id,
        "task_type": t_type,  # <-- Önceki elemandan sonra virgül olmalı
    
    # Yeni bir anahtar ("run_details" veya sadece "run") ile alt sözlük açılıyor
        "run": { 
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
        
            "dataset": {
                "name": p.name,
            # Virgül eklenmemişse buraya eklenmeli
            },
        
            "sample": {
                "size": sample_size,
                "seed": seed,
            # Virgül eklenmemişse buraya eklenmeli
            },
        
            "model": {
                "name": self._model_name(),
            # "params": self.model_params, 
            }
        } 
    }      
          

        print("done")
        return run_meta, results_df