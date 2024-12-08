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
            func = lambda x: random.uniform(-5.0, 5.0)
            functions.append(func)
        return functions

    def __call__(self, func):
        return func(random.uniform(-5.0, 5.0))

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
        if random.random() < 0.4:  # Refine strategy with probability 0.4
            x = random.uniform(bounds[0], bounds[1])
        if random.random() < 0.2:  # Refine strategy with probability 0.2
            x = random.uniform(bounds[0], bounds[1])
        if random.random() < 0.1:  # Refine strategy with probability 0.1
            x = random.uniform(bounds[0], bounds[1])
    return x

# Novel Metaheuristic Algorithm: BBOB-ES
# Code: 