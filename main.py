

from typing import Any, Dict
from utils import *
from runner import Runner
from model import Model
from judges import *
from logging_conf import setup_logging
from task import *
from evaluators import *
from dotenv import load_dotenv
import os

def run_benchmark_pipeline(
    *,
    task: Task,
    model_under_test: Model,
    judge: BaseJudge,
    evaluator: BaseEvaluator,
    measure_k: int = 25,
) -> Dict[str, Any]:
    """
    Core pipeline used by the backend.

    Steps:
      1. Run the model on the given Task via Runner.
      2. Apply a Judge to compute is_correct/score columns.
      3. Apply an Evaluator to summarize results to JSON.
      4. Return metadata and output file paths for the frontend/backend.

    Args:
        task:
            Task specification (type, dataset_path, sample_size, prompts...).
        model_under_test:
            Model instance to be evaluated.
        judge:
            (Equals, Contains, JSONEquals,
            PromptBasedScore, PromptBasedBoolean).
        evaluator:
            evaluator (e.g., ScoreEvaluator).
        measure_k:
            How many rows to use for latency measurement.

    Returns:
        dict with:
            - "meta": final metadata dict
            - "results_csv": path to raw run CSV
            - "judged_csv": path to judged CSV
            - "eval_json": path to evaluation JSON
    """
    runner = Runner(model_under_test)

    # 1) Run base model and get answers
    meta, df = runner.run(task=task, measure_k=measure_k)
    run_id = meta["run_id"]

    # Runner already saved run_{run_id}.csv; path is:
    results_csv = runner.get_path(f"run_{run_id}.csv")

    judged_csv = None
    eval_json = None

    judged_csv = runner.get_path(f"judge_{run_id}.csv")
    meta, df = judge.check_answers(meta, df, str(judged_csv))


    eval_json = runner.get_path(f"eval_{run_id}.json")
    evaluator.compute(meta, df, str(eval_json))

    return {
        "meta": meta,
        "results_csv": str(results_csv),
        "judged_csv": str(judged_csv) ,
        "eval_json": str(eval_json) ,
    }





def main():
    setup_logging()
    load_dotenv()
    api_key=os.getenv("OPENAI_API_KEY")
    model_high = Model(model_name="gpt-5-nano", api_key=api_key,system_prompt="Answer questions as a judge", params={"reasoning_effort": "high" })
    model_minimal=Model(model_name="gpt-5-nano", api_key=api_key,system_prompt="Pretend as a crazy man",params={"reasoning_effort": "minimal" })
    model_judge=Model(model_name="gpt-4.1",api_key=api_key)
    task1=Task.new(TaskType.WITH_TRUE_ANSWER,"contains_test.csv",10,prompt_template="Answer questions with waffling")

    judge=Contains()
    judge2=PromptBasedBoolean(model_judge,eval_prompt="Is model answer contains the true answer? Be strict it is important only accept if it contains the same answer like it written(lower upper case OK). Even if they meant same thing dont accept")
    judge3=PromptBasedBoolean(model_high, eval_prompt="Is model answer contains the true answer? Be strict it is important only accept if it contains the same answer like it written(lower-upper case and singular plural is OK). Even if they meant same thing dont accept")
    acc_eval=AccuracyEvaluator()
    run_benchmark_pipeline(task=task1,model_under_test=model_minimal,judge=judge2,evaluator=acc_eval)
    run_benchmark_pipeline(task=task1,model_under_test=model_minimal,judge=judge3,evaluator=acc_eval)



if __name__ == "__main__":
    main()