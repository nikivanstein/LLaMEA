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
            func = lambda x: np.random.uniform(-5.0, 5.0, self.dim)
            functions.append(func)
        return functions

    def __call__(self, func):
        return func(np.random.uniform(-5.0, 5.0, self.dim))

def f(x):
    return x**2 + 0.5*x + 0.1

def f_prime(x):
    return 2*x + 0.5

def f_double_prime(x):
    return 2

def f_double_prime_prime(x):
    return 4

def bbo_opt(func, x0, bounds, budget, alpha=0.4, beta=0.4, mu=0.1, sigma=0.01, rho=0.9):
    x = x0
    for _ in range(budget):
        x = func(x)
        if x < bounds[0]:
            x = bounds[0]
        elif x > bounds[1]:
            x = bounds[1]
        if random.random() < alpha:
            x = random.uniform(bounds[0], bounds[1])
        if random.random() < beta:
            x = random.uniform(bounds[0], bounds[1])
        if random.random() < rho:
            x = x + sigma * (x - x) / (1 - rho)
        if random.random() < mu:
            x = x - sigma * (x - x) / (1 - rho)
        if random.random() < sigma:
            x = x + random.uniform(-sigma, sigma)
        if random.random() < sigma:
            x = x + random.uniform(-sigma, sigma)
    return x

# One-line description: Evolutionary Algorithm for Multi-Dimensional Optimization
# Code: 