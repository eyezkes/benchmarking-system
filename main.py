

from utils import *
from runner import Runner
from model import Model
from judges import *
from logging_conf import setup_logging
from task import *
from evaluators import *

def run_evaluation(task:Task,judge_model,meta,df):
    if (task.type==TaskType.MULTIPLE_CHOICE):
        judge=MultipleChoiceJudge()
        filename=f"judge_{task.id}.csv"        
        meta,df= judge.check_answers(meta,df,task.get_path(filename))
        eval=MultipleChoiceEvaluator()
        filename=f"eval_{task.id}.json"
        out=eval.compute(meta,df,task.get_path(filename))
        return out
    if (task.type==TaskType.STRING_BASED):
        judge=StringBasedJudge(judge_model)
        filename=f"judge_{task.id}.csv"        
        meta,df= judge.check_answers(meta,df,task.get_path(filename))
        eval=StringBasedEvaluator()
        filename=f"eval_{task.id}.json"
        out=eval.compute(meta,df,task.get_path(filename))
        return out
    if (task.type==TaskType.PROMPT_BASED):
        judge=PromptBasedJudge(judge_model)
        filename=f"judge_{task.id}.csv"        
        meta,df=judge.check_answers(meta,df,task.get_path(filename),task.system_prompt)
        eval=PromptBasedEvaluator()
        filename=f"eval_{task.id}.json"  
        out=eval.compute(meta,df,task.get_path(filename))
        return out






def main():

    print("hi")

if __name__ == "__main__":
    main()