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

    def __call__(self, func, x0, bounds, budget):
        new_individual = x0
        for _ in range(budget):
            x = new_individual
            if random.random() < 0.4:
                x = func(x)
            if random.random() < 0.2:
                x = np.random.uniform(bounds[0], bounds[1])
            if random.random() < 0.2:
                x = np.random.uniform(bounds[0], bounds[1])
            if random.random() < 0.4:
                x = func(x)
            new_individual = x
        return new_individual

def f(x):
    return x**2 + 0.5*x + 0.1

def f_prime(x):
    return 2*x + 0.5

def f_double_prime(x):
    return 2

def f_double_prime_prime(x):
    return 4

def bbo_opt(func, x0, bounds, budget):
    return self.__call__(func, x0, bounds, budget)

# Description: Black Box Optimization using BBOB
# Code: 