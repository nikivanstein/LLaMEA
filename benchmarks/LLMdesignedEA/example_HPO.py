import os
import numpy as np
from ioh import get_problem, wrap_problem, logger
import ioh
import re
from misc import aoc_logger, correct_aoc, OverBudgetException, budget_logger
from llamea import LLaMEA
import time
from benchmarks.LLMdesignedEA.GNBG.GNBG_instances import load_problem

# Execution code starts here
api_key = os.getenv("OPENAI_API_KEY")
ai_model = "codellama3:7b"#"gpt-4o-2024-05-13"  # gpt-4-turbo or gpt-3.5-turbo gpt-4o llama3:70b gpt-4o-2024-05-13, gemini-1.5-flash gpt-4-turbo-2024-04-09
experiment_name = "test"
if "gemini" in ai_model:
    api_key = os.environ["GEMINI_API_KEY"]

from itertools import product
from ConfigSpace import Configuration, ConfigurationSpace

import numpy as np
from smac import Scenario
from smac import AlgorithmConfigurationFacade


def evaluateWithHPO(
    solution, explogger = None
):
    fitness_mean = 0
    code = solution.solution
    algorithm_name = solution.name
    exec(code, globals())


    error = ""
    algorithm = None


    # now optimize the hyper-parameters
    def get_fitness(config: Configuration, instance: str, seed: int = 0):
        np.random.seed(seed)
        fid = int(instance)

        gnbg = load_problem(fid, seed)
        dim = gnbg.Dimension
        budget = gnbg.MaxEvals


        wrap_problem(gnbg.fitness, f"gnbg{fid}",
            ioh.ProblemClass.REAL,
            dimension=dim)

        problem = get_problem(f"gnbg{fid}")

        bl = budget_logger(budget, triggers=[logger.trigger.ALWAYS])
        problem.attach_logger(bl)
        try:
            algorithm = globals()[algorithm_name](
                budget=budget, dim=dim, **dict(config)
            )
            algorithm(problem)
        except OverBudgetException:
            pass
        except Exception as e:
            print(problem.state, budget, e)
        # After running the algorithm, the best fitness value is stored in gnbg.BestFoundResult.
        # The best found position is stored in gnbg.BestFoundPosition.
        # The function evaluation number where the algorithm reached the acceptance threshold is stored in gnbg.AcceptanceReachPoint. If the algorithm did not reach the acceptance threshold, it is set to infinity.
        # For visualizing the convergence behavior, the history of the objective values is stored in gnbg.FEhistory, however it needs to be processed as follows:

        fitness = gnbg.BestFoundResult
        return fitness
    
    def get_fitness_all(config: Configuration):
        np.random.seed(0)
        all_f = []
        all_x = []
        for seed in range(31):
            for fid in range(1,25):
                gnbg = load_problem(fid, seed)
                dim = gnbg.Dimension
                budget = gnbg.MaxEvals


                wrap_problem(gnbg.fitness, f"gnbg{fid}",
                    ioh.ProblemClass.REAL,
                    dimension=dim)

                problem = get_problem(f"gnbg{fid}")

                bl = budget_logger(budget, triggers=[logger.trigger.ALWAYS])
                problem.attach_logger(bl)
                try:
                    algorithm = globals()[algorithm_name](
                        budget=budget, dim=dim, **dict(config)
                    )
                    algorithm(problem)
                except OverBudgetException:
                    pass
                except Exception as e:
                    print(problem.state, budget, e)
                # After running the algorithm, the best fitness value is stored in gnbg.BestFoundResult.
                # The best found position is stored in gnbg.BestFoundPosition.
                # The function evaluation number where the algorithm reached the acceptance threshold is stored in gnbg.AcceptanceReachPoint. If the algorithm did not reach the acceptance threshold, it is set to infinity.
                # For visualizing the convergence behavior, the history of the objective values is stored in gnbg.FEhistory, however it needs to be processed as follows:
                all_f.append(gnbg.BestFoundResult)
                all_x.append(gnbg.BestFoundPosition)
                problem.reset()
        return all_f, all_x

    args = list(range(1, 25))
    np.random.shuffle(args)
    inst_feats = {str(arg): [arg[0]] for idx, arg in enumerate(args)}
    error = ""
    
    if "_configspace" not in solution.keys():
        # No HPO possible, evaluate only the default
        incumbent = {}
        error = "The configuration space was not properly formatted or not present in your answer. The evaluation was done on the default configuration."
    else:
        configuration_space = solution.configspace
        scenario = Scenario(
            configuration_space,
            name=str(int(time.time())) + "-" + algorithm_name,
            deterministic=False,
            min_budget=12,
            max_budget=100,
            n_trials=2000,
            instances=args,
            instance_features=inst_feats,
            output_directory="smac3_output" if explogger is None else explogger.dirname + "/smac"
            #n_workers=10
        )
        smac = AlgorithmConfigurationFacade(scenario, get_fitness, logging_level=30)
        incumbent = smac.optimize()

    # last but not least, perform the final validation
    
    all_f, all_x = get_fitness_all(incumbent)
    final_f = np.mean(all_f)

    dict_hyperparams = dict(incumbent)
    feedback = f"The algorithm {algorithm_name} got an average fitness (0.0 is the best) score of {final_f:0.2f} with optimal hyperparameters {dict_hyperparams}."
    print(algorithm_name, algorithm, final_f)

    solution.add_metadata("all_f", all_f)
    solution.add_metadata("all_x", all_x)
    solution.add_metadata("incumbent", dict_hyperparams)
    solution.set_scores(final_f, feedback)
    
    return solution


role_prompt = "You are a highly skilled computer scientist in the field of natural computing. Your task is to design novel metaheuristic algorithms to solve black box optimization problems."
task_prompt = """
The optimization algorithm should handle a wide range of tasks, which is evaluated on the GNBG test suite of 24 functions. Your task is to write the optimization algorithm in Python code. The code should contain an `__init__(self, budget, dim)` function with optional additional arguments and the function `def __call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.
The func() can only be called as many times as the budget allows, not more. Each of the optimization functions has a search space between -100.0 (lower bound) and 100.0 (upper bound). The dimensionality will be varied.
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
            x = np.random.uniform(-100, 100)
            
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
        evaluateWithHPO,
        n_offspring=20,
        n_parents=10,
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
