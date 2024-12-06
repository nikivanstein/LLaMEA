import os
import numpy as np
import ioh
from nevergrad import benchmark

from ioh import wrap_problem, get_problem, logger
import re
from misc import aoc_logger, correct_aoc, OverBudgetException
from llamea import LLaMEA
import time

# Execution code starts here
api_key = os.getenv("OPENAI_API_KEY")
ai_model = "codellama:7b" #"gpt-4o-2024-05-13"  # gpt-4-turbo or gpt-3.5-turbo gpt-4o llama3:70b gpt-4o-2024-05-13, gemini-1.5-flash gpt-4-turbo-2024-04-09
experiment_name = "photonics"
if "gemini" in ai_model:
    api_key = os.environ["GEMINI_API_KEY"]

from itertools import product
from ConfigSpace import Configuration, ConfigurationSpace

import numpy as np
from smac import Scenario
from smac import AlgorithmConfigurationFacade


"""
Name=photonics, dimension=80, shape-of-search-space=(2, 40), constraints=None, specialConstraints=[]/False, fully_continuous=True, is_tuningFalse, real_world=False, noisy=False, domain=:[[  2.5   2.5   2.5   2.5   2.5   2.5   2.5   2.5   2.5   2.5   2.5   2.5
2.5   2.5   2.5   2.5   2.5   2.5   2.5   2.5   2.5   2.5   2.5   2.5
2.5   2.5   2.5   2.5   2.5   2.5   2.5   2.5   2.5   2.5   2.5   2.5
2.5   2.5   2.5   2.5]
[105.  105.  105.  105.  105.  105.  105.  105.  105.  105.  105.  105.
105.  105.  105.  105.  105.  105.  105.  105.  105.  105.  105.  105.
105.  105.  105.  105.  105.  105.  105.  105.  105.  105.  105.  105.
105.  105.  105.  105. ]], fully-bounded=True, number-of-objectives=1 [10, 10, 10, 10, 10, 100, 100, 100, 100, 100, 1000, 1000]
Name=ultrasmall_photonics, dimension=20, shape-of-search-space=(2, 10), constraints=None, specialConstraints=[]/False, fully_continuous=True, is_tuningFalse, real_world=False, noisy=False, domain=:[[  2.5   2.5   2.5   2.5   2.5   2.5   2.5   2.5   2.5   2.5]
[105.  105.  105.  105.  105.  105.  105.  105.  105.  105. ]], fully-bounded=True, number-of-objectives=1 [10, 10, 10, 10, 10, 100, 100, 100, 100, 100, 1000, 1000]
Name=verysmall_photonics, dimension=20, shape-of-search-space=(1, 20), constraints=None, specialConstraints=[]/False, fully_continuous=True, is_tuningFalse, real_world=False, noisy=False, domain=:[[105. 105. 105. 105. 105. 105. 105. 105. 105. 105. 105. 105. 105. 105.
105. 105. 105. 105. 105. 105.]], fully-bounded=True, number-of-objectives=1 [10, 10, 10, 10, 10, 100, 100, 100, 100, 100, 1000, 1000]
Name=small_photonics, dimension=40, shape-of-search-space=(2, 20), constraints=None, specialConstraints=[]/False, fully_continuous=True, is_tuningFalse, real_world=False, noisy=False, domain=:[[  2.5   2.5   2.5   2.5   2.5   2.5   2.5   2.5   2.5   2.5   2.5   2.5
2.5   2.5   2.5   2.5   2.5   2.5   2.5   2.5]
[105.  105.  105.  105.  105.  105.  105.  105.  105.  105.  105.  105.
105.  105.  105.  105.  105.  105.  105.  105. ]], fully-bounded=True, number-of-objectives=1 [10, 10, 10, 10, 10, 100, 100, 100, 100, 100, 1000, 1000]


"""


# Perhaps first test class RandomSearch:
#     def __init__(self, budget=10000, dim=10):
#         self.budget = budget
#         self.dim = dim

#     def __call__(self, func):
#         self.f_opt = np.Inf
#         self.x_opt = None
#         for i in range(self.budget):
#             x = np.random.uniform(2.5, 105)
            
#             f = func(x)
#             if f < self.f_opt:
#                 self.f_opt = f
#                 self.x_opt = x
            
#         return self.f_opt, self.x_opt

def evaluate(
    solution, explogger = None
):
    
    code = solution.solution
    algorithm_name = solution.name
    exec(code, globals())

    photonics_f = list(benchmark.registry["photonics"]())
    #lets take only the first for now
    f = photonics_f[0].function
    dim = f.dimension
    budget = 1000
    error = ""
    algorithm = None


    def photonics_wrapper(x):
        return f(np.array(x))

    wrap_problem(photonics_wrapper, "photonics", ioh.ProblemClass.REAL, dimension=dim, instance=0, lb=2.5, ub=105.)

    # perform a small run to check for any code errors
    l2_temp = aoc_logger(10, upper=1e2, triggers=[logger.trigger.ALWAYS])
    problem = get_problem("photonics")
    problem.attach_logger(l2_temp)
    
    try:
        algorithm = globals()[algorithm_name](budget=10, dim=f.dimension)
        algorithm(problem)
    except OverBudgetException:
        pass

    # now optimize the hyper-parameters
    def get_performance(config: Configuration, instance: str, seed: int = 0):
        np.random.seed(seed)
        fid, iid = instance.split(",")
        fid = int(fid[1:])
        iid = int(iid[:-1])
        problem = get_problem("photonics")
        l2 = aoc_logger(budget, upper=1e2, triggers=[logger.trigger.ALWAYS])
        problem.attach_logger(l2)
        try:
            algorithm = globals()[algorithm_name](
                budget=budget, dim=dim, **dict(config)
            )
            algorithm(problem)
        except OverBudgetException:
            pass
        except Exception as e:
            print(problem.state, budget, e)
        auc = correct_aoc(problem, l2, budget)
        return 1 - auc
    
    if "_configspace" not in solution.keys():
        # No HPO possible, evaluate only the default
        incumbent = {}
        error = "The configuration space was not properly formatted or not present in your answer. The evaluation was done on the default configuration."
    else:
        configuration_space = solution.configspace
        scenario = Scenario(
            configuration_space,
            name=str(int(time.time())) + "-" + algorithm_name,
            deterministic=True,
            n_trials=200,
            output_directory="smac3_output" if explogger is None else explogger.dirname + "/smac"
            #n_workers=10
        )
        smac = AlgorithmConfigurationFacade(scenario, get_performance)
        incumbent = smac.optimize()

    # last but not least, perform the final validation
    l2 = aoc_logger(budget, upper=1e2, triggers=[logger.trigger.ALWAYS])
    problem = get_problem("photonics")
    problem.attach_logger(l2)
    
    try:
        algorithm = globals()[algorithm_name](budget=budget, dim=dim,**dict(incumbent))
        algorithm(problem)
    except OverBudgetException:
        pass
    auc = correct_aoc(problem, l2, budget)
    
    dict_hyperparams = dict(incumbent)
    feedback = f"The algorithm {algorithm_name} got an Area over the convergence curve (AOCC, 1.0 is the best) score of {auc:0.2f} with optimal hyperparameters {dict_hyperparams}."
    print(algorithm_name, algorithm, auc)

    solution.add_metadata("incumbent", dict_hyperparams)
    solution.set_scores(auc, feedback)
    
    return solution


role_prompt = "You are a highly skilled computer scientist in the field of natural computing. Your task is to design novel metaheuristic algorithms to solve black box optimization problems."
task_prompt = """
The optimization algorithm should handle photonic problems with a variable dimensionality and a lower bound of 2.5 and upper bound of 105, Your task is to write the optimization algorithm in Python code. The code should contain an `__init__(self, budget, dim)` function with optional additional arguments and the function `def __call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.
The func() can only be called as many times as the budget allows, not more. Each of the optimization functions has a search space between 2.5 (lower bound) and 105.0 (upper bound). The dimensionality can be varied.
An example of such code (a simple random search), is as follows:
```python
import numpy as np

class RandomSearch:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None
        for i in range(self.budget):
            x = np.random.uniform(2.5, 105)
            
            f = func(x)
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = x
            
        return self.f_opt, self.x_opt
```

In addition, any hyper-parameters the algorithm uses will be optimized by SMAC, for this, provide a Configuration space as Python dictionary (without the dim and budget parameters) and include all hyper-parameters in the __init__ function header.
An example configuration space is as follows:

```python
{
    "float_parameter": (0.1, 1.5),
    "int_parameter": (2, 10), 
    "categoral_parameter": ["mouse", "cat", "dog"]
}
```

Give an excellent and novel heuristic algorithm including its configuration space to solve this task and also give it a one-line description, describing the main idea. Give the response in the format:
# Description: <short-description>
# Code: <code>
# Space: <configuration_space>
"""

feedback_prompts = [
    f"Either refine or redesign to improve the solution (and give it a distinct one-line description)."
]

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
        HPO=True,
    )
    print(es.run())
