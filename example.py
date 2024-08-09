import os
import numpy as np
from ioh import get_problem, logger
import re
from misc import aoc_logger, correct_aoc, OverBudgetException
from llamea import LLaMEA

# Execution code starts here
api_key = os.getenv("OPENAI_API_KEY")
ai_model = "codellama:7b" # gpt-4-turbo or gpt-3.5-turbo gpt-4o llama3:70b
experiment_name ="pop5-8"


def evaluateBBOB(code, algorithm_name, algorithm_name_long, details=False):
    auc_mean = 0
    auc_std = 0
    detailed_aucs = [0,0,0,0,0]
    exec(code, globals())
    
    error = ""
    l2 = aoc_logger(budget, upper=1e2, triggers=[logger.trigger.ALWAYS])
    aucs = []
    detail_aucs = []
    algorithm = None
    for dim in [5]:
        budget = 2000 * dim
        for fid in np.arange(1,25):
            for iid in [1,2,3]: #, 4, 5]
                problem = get_problem(fid, iid, dim)
                problem.attach_logger(l2)

                for rep in range(3):
                    np.random.seed(rep)
                    try:
                        algorithm = globals()[algorithm_name](budget=budget, dim=dim)
                        algorithm(problem)
                    except OverBudgetException:
                        pass
                    
                    auc = correct_aoc(problem, l2, budget)
                    aucs.append(auc)
                    detail_aucs.append(auc)
                    l2.reset(problem)
                    problem.reset()
            if fid == 5:
                detailed_aucs[0] = np.mean(detail_aucs)
                detail_aucs = []
            if fid == 9:
                detailed_aucs[1] = np.mean(detail_aucs)
                detail_aucs = []
            if fid == 14:
                detailed_aucs[2] = np.mean(detail_aucs)
                detail_aucs = []
            if fid == 19:
                detailed_aucs[3] = np.mean(detail_aucs)
                detail_aucs = []
            if fid == 24:
                detailed_aucs[4] = np.mean(detail_aucs)
                detail_aucs = []

    auc_mean = np.mean(aucs)
    auc_std = np.std(aucs)

    i = 0
    while os.path.exists(f"currentexp/aucs-{algorithm_name}-{i}.npy"):
        i+=1
    np.save(f"currentexp/aucs-{algorithm_name}-{i}.npy", aucs)

    feedback = f"The algorithm {algorithm_name_long} got an average Area over the convergence curve (AOCC, 1.0 is the best) score of {auc_mean:0.2f} with standard deviation {auc_std:0.2f}."
    if details:
        feedback = (f"{feedback}\nThe mean AOCC score of the algorithm {algorithm_name_long} on Separable functions was {detailed_aucs[0]:.02f}, "
                    f"on functions with low or moderate conditioning {detailed_aucs[1]:.02f}, "
                    f"on functions with high conditioning and unimodal {detailed_aucs[2]:.02f}, "
                    f"on Multi-modal functions with adequate global structure {detailed_aucs[3]:.02f}, "
                    f"and on Multi-modal functions with weak global structure {detailed_aucs[4]:.02f}")


    print(algorithm_name_long, algorithm, auc_mean, auc_std)
    return feedback, auc_mean, error


task_prompt = """
The optimization algorithm should handle a wide range of tasks, which is evaluated on the BBOB test suite of 24 noiseless functions. Your task is to write the optimization algorithm in Python code. The code should contain an `__init__(self, budget, dim)` function and the function `def __call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.
The func() can only be called as many times as the budget allows, not more. Each of the optimization functions has a search space between -5.0 (lower bound) and 5.0 (upper bound). The dimensionality can be varied.
Give an excellent and novel heuristic algorithm to solve this task and also give it a name. Give the response in the format:
# Name: <name>
# Code: <code>
"""

for experiment_i in range(5):
    es = LLaMEA(evaluateBBOB, n_parents=10, n_offspring=50, api_key=api_key, task_prompt=task_prompt, experiment_name=experiment_name, model=ai_model, elitism=True, budget=1000)
    print(es.run())