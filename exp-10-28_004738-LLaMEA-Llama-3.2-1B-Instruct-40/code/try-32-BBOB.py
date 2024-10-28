import random
import numpy as np

class BBOB:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.funcs = self.generate_functions()

    def generate_functions(self):
        functions = []
        for _ in range(24):
            func = lambda x: np.random.uniform(-5.0, 5.0)
            functions.append(func)
        return functions

    def __call__(self, func):
        return func(np.random.uniform(-5.0, 5.0))

def f(x):
    return x**2 + 0.5*x + 0.1

def f_prime(x):
    return 2*x + 0.5

def f_double_prime(x):
    return 2

def f_double_prime_prime(x):
    return 4

def bbo_opt(func, x0, bounds, budget):
    x = x0
    for _ in range(budget):
        x = func(x)
        if x < bounds[0]:
            x = bounds[0]
        elif x > bounds[1]:
            x = bounds[1]
        if random.random() < 0.4:
            x = random.uniform(bounds[0], bounds[1])
        if random.random() < 0.2:
            x = random.uniform(bounds[0], bounds[1])
        if random.random() < 0.4:
            x = random.uniform(bounds[0], bounds[1])
    return x

# Exception Occurred: Traceback (most recent call last)
# File "/root/LLaMEA/llamea/llamea.py", line 187, in initialize_single
# new_individual = self.evaluate_fitness(new_individual)
# ^
# File "/root/LLaMEA/mutation_exp.py", line 52, in evaluateBBOB
# algorithm(problem)
# File "<string>", line 17, in __call__
# TypeError: __call__(): incompatible function arguments. The following argument types are supported:
#    1. (self: ioh.iohcpp.problem.RealSingleObjective, arg0: List[float]) -> float
#    2. (self: ioh.iohcpp.problem.RealSingleObjective, arg0: List[List[float]]) -> List[float]
# Invoked with: <RealSingleObjectiveProblem 1. Sphere (iid=1 dim=5)>, -4.521232642195706