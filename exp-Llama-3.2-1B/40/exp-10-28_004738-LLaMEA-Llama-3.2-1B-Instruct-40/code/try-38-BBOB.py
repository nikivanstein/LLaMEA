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
    population = [x0]
    for _ in range(budget):
        new_individual = population[-1]
        fitness = func(new_individual)
        if fitness < bounds[0]:
            new_individual = bounds[0]
        elif fitness > bounds[1]:
            new_individual = bounds[1]
        if random.random() < 0.4:
            new_individual = random.uniform(bounds[0], bounds[1])
        if random.random() < 0.2:
            new_individual = random.uniform(bounds[0], bounds[1])
        if random.random() < 0.4:
            new_individual = random.uniform(bounds[0], bounds[1])
        population.append(new_individual)
    return population

# Description: Black Box Optimization using BBOB
# Code: 