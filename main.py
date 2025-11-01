

from utils import *
from runner import Runner
from model import Model
from judges import *
from logging_conf import setup_logging
from task import *
from evaluators import *

def run_evaluation(task:Task,model,judge_model=None):
    runner=Runner(model)
    meta,df=runner.run(task)
    run_id=meta["run_id"]
    if (task.type==TaskType.MULTIPLE_CHOICE):
        judge=MultipleChoiceJudge()
        filename=f"judge_{run_id}.csv"        
        meta,df= judge.check_answers(meta,df,runner.get_path(filename))
        eval=MultipleChoiceEvaluator()
        filename=f"eval_{run_id}.json"
        out=eval.compute(meta,df,runner.get_path(filename))
        return out
    if (task.type==TaskType.STRING_BASED):
        judge=StringBasedJudge(judge_model)
        filename=f"judge_{run_id}.csv"        
        meta,df= judge.check_answers(meta,df,runner.get_path(filename))
        eval=StringBasedEvaluator()
        filename=f"eval_{run_id}.json"
        out=eval.compute(meta,df,runner.get_path(filename))
        return out
    if (task.type==TaskType.PROMPT_BASED):
        judge=PromptBasedJudge(judge_model)
        filename=f"judge_{run_id}.csv"        
        meta,df=judge.check_answers(meta,df,runner.get_path(filename),task.eval_prompt)
        eval=PromptBasedEvaluator()
        filename=f"eval_{run_id}.json"  
        out=eval.compute(meta,df,runner.get_path(filename))
        return out






def main():

    setup_logging()

if __name__ == "__main__":
    main()