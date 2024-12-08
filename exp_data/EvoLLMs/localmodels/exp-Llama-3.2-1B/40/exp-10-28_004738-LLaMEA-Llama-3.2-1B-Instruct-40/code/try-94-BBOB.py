import random
import numpy as np
from scipy.optimize import differential_evolution

class BBOB:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.funcs = self.generate_functions()

    def generate_functions(self):
        functions = []
        for _ in range(24):
            func = lambda x: np.random.uniform(-5.0, 5.0, self.dim)
            functions.append(func)
        return functions

    def __call__(self, func, x0, bounds, budget):
        return func(x0)

def f(x):
    return x**2 + 0.5*x + 0.1

def f_prime(x):
    return 2*x + 0.5

def f_double_prime(x):
    return 2

def f_double_prime_prime(x):
    return 4

def bbo_opt(func, x0, bounds, budget):
    # Use differential evolution with probability 0.4 to refine the search
    for _ in range(budget):
        res = differential_evolution(func, bounds, x0=x0)
        if random.random() < 0.4:
            x0 = res.x
    return x0

# One-line description: Evolutionary Algorithm for Multi-Dimensional Optimization
# Code: 