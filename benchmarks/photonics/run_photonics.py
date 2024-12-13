import os
import numpy as np
import ioh
from nevergrad import benchmark

from ioh import wrap_problem, get_problem, logger
import re
from misc import aoc_logger, correct_aoc, OverBudgetException
from llamea import LLaMEA
import time

from benchmarks.photonics.problem import objective_f, cost_minibragg, upper_lower_bound

# Execution code starts here
api_key = os.getenv("OPENAI_API_KEY")
ai_model = "gemini-1.5-flash" #"gpt-4o-2024-05-13"  # gpt-4-turbo or gpt-3.5-turbo gpt-4o llama3:70b gpt-4o-2024-05-13, gemini-1.5-flash gpt-4-turbo-2024-04-09
experiment_name = "photonics"
if "gemini" in ai_model:
    api_key = os.environ["GEMINI_API_KEY"]

import numpy as np


def evaluate(
    solution, explogger = None
):
    
    code = solution.solution
    algorithm_name = solution.name
    exec(code, globals())


    lb, ub = upper_lower_bound()

    dim = 10
    budget = 50000
    error = ""
    algorithm = None

    def cost_minibragg_wrapper(x):
        return objective_f(np.array(x))

    wrap_problem(cost_minibragg_wrapper, "cost_minibragg", ioh.ProblemClass.REAL, dimension=dim, instance=0, lb=lb, ub=ub)

    # perform a small run to check for any code errors
    l2_temp = aoc_logger(10, upper=1e2, triggers=[logger.trigger.ALWAYS])
    problem = get_problem("cost_minibragg")
    problem.attach_logger(l2_temp)
    
    try:
        algorithm = globals()[algorithm_name](budget=10, dim=dim)
        algorithm(problem)
    except OverBudgetException:
        pass

    # last but not least, perform the final validation
    l2 = aoc_logger(budget, upper=1e2, triggers=[logger.trigger.ALWAYS])
    problem = get_problem("cost_minibragg")
    problem.attach_logger(l2)
    
    try:
        algorithm = globals()[algorithm_name](budget=budget, dim=dim)
        algorithm(problem)
    except OverBudgetException:
        pass
    auc = correct_aoc(problem, l2, budget)
    
    feedback = f"The algorithm {algorithm_name} got an Area over the convergence curve (AOCC, 1.0 is the best) score of {auc:0.2f}."
    print(algorithm_name, algorithm, auc)

    solution.set_scores(auc, feedback)
    return solution


role_prompt = "You are a highly skilled computer scientist in the field of natural computing. Your task is to design novel metaheuristic algorithms to solve black box optimization problems."
task_prompt = """
The optimization algorithm should handle the optimization of a Bragg mirror. We will optimize the thicknesses of a layer-stack of two alterning dielectric materials, in total 10 layers. We want to minimize one minus the reflectance of the layer-stack.
Your task is to write the optimization algorithm in Python code. The code should contain an `__init__(self, budget, dim)` function and the function `def __call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.
The func() can only be called as many times as the budget allows, not more. Each of the optimization functions has a search space between 0.0 (lower bound) and 214.28 (upper bound). The dimensionality is 10.
An example of such code (a simple random search), is as follows:
```python
import numpy as np

class RandomSearch:
    def __init__(self, budget=50000, dim=10):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None
        for i in range(self.budget):
            x = np.random.uniform(0.0, 214.28)
            
            f = func(x)
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = x
            
        return self.f_opt, self.x_opt
```

Give an excellent and novel heuristic algorithm to solve this task and also give it a one-line description, describing the main idea. Give the response in the format:
# Description: <short-description>
# Code: <code>
"""

feedback_prompts = [
    f"Either refine or redesign to improve the solution (and give it a distinct one-line description)."
]


#test run
from scipy.optimize import differential_evolution

lb, ub = upper_lower_bound()
bounds = [(lb,ub)] * 10

def cost_minibragg_wrapper(x):
    return objective_f(np.array(x))

dim = 10
wrap_problem(cost_minibragg_wrapper, "cost_minibragg", ioh.ProblemClass.REAL, dimension=dim, instance=0, lb=lb, ub=ub)

# perform a small run to check for any code errors
l2_temp = aoc_logger(10, upper=1e2, triggers=[logger.trigger.ALWAYS])
problem = get_problem("cost_minibragg")
problem.attach_logger(l2_temp)

budget = 50000
print("Doing small run")
try:
    differential_evolution(problem, bounds, maxiter=10000)
except OverBudgetException:
    pass

print("Doing large run")

# last but not least, perform the final validation
l2 = aoc_logger(budget, upper=1e2, triggers=[logger.trigger.ALWAYS])

problem = get_problem("cost_minibragg")
problem.attach_logger(l2)

try:
    differential_evolution(problem, bounds, maxiter=10000)
except OverBudgetException:
    pass
auc = correct_aoc(problem, l2, budget)

a = aaaa

for experiment_i in [1]:
    es = LLaMEA(
        evaluate,
        n_parents=1,
        n_offspring=1,
        role_prompt=role_prompt,
        task_prompt=task_prompt,
        mutation_prompts=feedback_prompts,
        api_key=api_key,
        experiment_name=experiment_name,
        model=ai_model,
        elitism=True,
        HPO=False,
    )
    print(es.run())
