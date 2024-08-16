import os
import numpy as np
from ioh import get_problem, logger
import re
from misc import aoc_logger, correct_aoc, OverBudgetException
from llamea import LLaMEA

# Execution code starts here
api_key = os.getenv("OPENAI_API_KEY")
ai_model = "codellama:7b"  # gpt-4-turbo or gpt-3.5-turbo gpt-4o llama3:70b
experiment_name = "HPO"


def evaluateBBOBWithHPO(
    code, algorithm_name, algorithm_name_long, configuration_space=None, explogger=None
):
    """
    Evaluates an optimization algorithm on the BBOB (Black-Box Optimization Benchmarking) suite and computes
    the Area Over the Convergence Curve (AOCC) to measure performance. In addddition, if a configuration space is provided, it
    applies Hyper-parameter optimization with SMAC first.

    Parameters:
    -----------
    code : str
        A string of Python code to be executed within the global context. This can be used to define or import
        additional components required for the evaluation.
    algorithm_name : str
        The name of the algorithm function (as a string) that will be evaluated. This function should be
        accessible in the global namespace.
    algorithm_name_long : str
        A longer, more descriptive name for the algorithm, used in logging and feedback.
    configuration_space : object, optional
        If provided, this object represents the configuration space for Hyperparameter Optimization (HPO).
    explogger : object, optional
        An experimental logger object used to log the results of the evaluation. If not provided, logging will be skipped.

    Returns:
    --------
    feedback : str
        A formatted string summarizing the algorithm's performance in terms of AOCC (Area Over the Convergence Curve).
    auc_mean : float
        The mean AOCC score across all evaluated problems and repetitions.
    error : str
        A placeholder for error messages, currently not implemented or utilized.

    Functionality:
    --------------
    - Executes the provided `code` string in the global context, allowing for dynamic inclusion of necessary components.
    - Iterates over a predefined set of dimensions (currently only 5), function IDs (1 to 24), and instance IDs (1 to 3).
    - For each problem, the specified algorithm is instantiated and executed with a defined budget.
    - AOCC is computed for each run, and the results are aggregated across all runs, problems, and repetitions.
    - The function handles cases where the algorithm exceeds its budget using an `OverBudgetException`.
    - Logs the results if an `explogger` is provided.
    - The function returns a feedback string, the mean AOCC score, and an error placeholder.

    Notes:
    ------
    - The budget for each algorithm run is set to 10,000.
    - The function currently only evaluates a single dimension (5), but this can be extended.
    - Hyperparameter Optimization (HPO) with SMAC is mentioned but not implemented.
    - The AOCC score is a metric where 1.0 is the best possible outcome, indicating optimal convergence.

    Example:
    --------
    feedback, auc_mean, error = evaluateBBOB("import numpy as np", "MyAlgorithm", "My Custom Algorithm")
    """
    auc_mean = 0
    auc_std = 0

    exec(code, globals())

    if configuration_space != None:
        # implement HPO with SMAC
        pass

    budget = 10000
    error = ""
    l2 = aoc_logger(budget, upper=1e2, triggers=[logger.trigger.ALWAYS])
    aucs = []
    detail_aucs = []
    algorithm = None
    for dim in [5]:
        for fid in np.arange(1, 25):
            for iid in [1, 2, 3]:  # , 4, 5]
                problem = get_problem(fid, iid, dim)
                problem.attach_logger(l2)

                for rep in range(3):
                    np.random.seed(rep)
                    try:
                        # Todo insert hyper-parameters here as well after HPO
                        algorithm = globals()[algorithm_name](budget=budget, dim=dim)
                        algorithm(problem)
                    except OverBudgetException:
                        pass

                    auc = correct_aoc(problem, l2, budget)
                    aucs.append(auc)
                    detail_aucs.append(auc)
                    l2.reset(problem)
                    problem.reset()

    auc_mean = np.mean(aucs)
    auc_std = np.std(aucs)
    if explogger != None:
        explogger.log_aucs(aucs)
    feedback = f"The algorithm {algorithm_name_long} got an average Area over the convergence curve (AOCC, 1.0 is the best) score of {auc_mean:0.2f} with standard deviation {auc_std:0.2f}."
    print(algorithm_name_long, algorithm, auc_mean, auc_std)
    return feedback, auc_mean, error


role_prompt = "You are a highly skilled computer scientist in the field of natural computing. Your task is to design novel metaheuristic algorithms to solve black box optimization problems."
task_prompt = """
The optimization algorithm should handle a wide range of tasks, which is evaluated on the BBOB test suite of 24 noiseless functions. Your task is to write the optimization algorithm in Python code. The code should contain an `__init__(self, budget, dim)` function with optional additional arguments and the function `def __call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.
The func() can only be called as many times as the budget allows, not more. Each of the optimization functions has a search space between -5.0 (lower bound) and 5.0 (upper bound). The dimensionality can be varied.
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
            x = np.random.uniform(func.bounds.lb, func.bounds.ub)
            
            f = func(x)
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = x
            
        return self.f_opt, self.x_opt
```

In addition, any hyper-parameters the algorithm uses will be optimized by SMAC, for this, provide a Configuration space in json format (without the dim and budget parameters) and include all hyper-parameters in the __init__ function header.
An example configuration space is as follows:

```json
{
    "float_parameter": (0.1, 1.5),
    "int_parameter": (2, 10), 
    "categoral_parameter": ["mouse", "cat", "dog"]
}
```

Give an excellent and novel heuristic algorithm including its configuration space to solve this task and also give it a name. Give the response in the format:
# Name: <name>
# Code: <code>
# Space: <configuration_space>
"""

feedback_prompt = (
    f"Either refine or redesign to improve the solution (and give it a distinct name). Give the response in the format:\n"
    f"# Name: <name>\n"
    f"# Code: <code>\n"
    f"# Space: <configuration_space>"
)

for experiment_i in [1]:
    es = LLaMEA(
        evaluateBBOBWithHPO,
        role_prompt=role_prompt,
        task_prompt=task_prompt,
        api_key=api_key,
        experiment_name=experiment_name,
        model=ai_model,
        elitism=True,
        HPO=True,
    )
    print(es.run())
